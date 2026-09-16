"""Ingestão das guias de ITBI pagas do município de São Paulo.

A Secretaria Municipal da Fazenda publica uma planilha por ano, com uma aba
por mês e uma linha por guia paga: cadastro (SQL), endereço, valor declarado,
valor venal de referência, área construída, uso e padrão do IPTU.

É a única fonte pública com preço efetivamente transacionado por endereço,
mas não traz construtora, corretor, comprador, dormitórios nem aluguel — a
Prefeitura omite razão social e nomes por sigilo fiscal.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urljoin

import pandas as pd
import requests

from gjurema.config import HTTP_TIMEOUT, ITBI_SP_INDEX_URL, RAW_DIR

logger = logging.getLogger(__name__)

DOWNLOAD_HEADERS = {"User-Agent": "GJurema/0.1 (+pipeline de precificacao imobiliaria)"}

COLUMNS = {
    "N° do Cadastro (SQL)": "sql",
    "Nome do Logradouro": "logradouro",
    "Número": "numero",
    "Complemento": "complemento",
    "CEP": "cep",
    "Natureza de Transação": "natureza",
    "Valor de Transação (declarado pelo contribuinte)": "valor_transacao",
    "Data de Transação": "data",
    "Valor Venal de Referência": "valor_venal",
    "Proporção Transmitida (%)": "proporcao_pct",
    "Valor Venal de Referência (proporcional)": "valor_venal_proporcional",
    "Base de Cálculo adotada": "base_calculo",
    "Tipo de Financiamento": "tipo_financiamento",
    "Valor Financiado": "valor_financiado",
    "Área Construída (m2)": "area_construida",
    "Descrição do uso (IPTU)": "uso",
    "Padrão (IPTU)": "padrao",
}

# A planilha de 2024 é publicada sem linha de cabeçalho: a primeira guia vira
# o nome das colunas. O layout da guia é estável desde 2017, então a ordem
# abaixo reconstrói os nomes por posição.
POSITIONAL_HEADER = (
    "N° do Cadastro (SQL)",
    "Nome do Logradouro",
    "Número",
    "Complemento",
    "Bairro",
    "Referência",
    "CEP",
    "Natureza de Transação",
    "Valor de Transação (declarado pelo contribuinte)",
    "Data de Transação",
    "Valor Venal de Referência",
    "Proporção Transmitida (%)",
    "Valor Venal de Referência (proporcional)",
    "Base de Cálculo adotada",
    "Tipo de Financiamento",
    "Valor Financiado",
    "Cartório de Registro",
    "Matrícula do Imóvel",
    "Situação do SQL",
    "Área do Terreno (m2)",
    "Testada (m)",
    "Fração Ideal",
    "Área Construída (m2)",
    "Uso (IPTU)",
    "Descrição do uso (IPTU)",
    "Padrão (IPTU)",
    "Descrição do padrão (IPTU)",
    "ACC (IPTU)",
)

# Sem estas colunas a guia não descreve uma venda: a planilha é inutilizável.
REQUIRED_COLUMNS = (
    "sql",
    "logradouro",
    "numero",
    "cep",
    "natureza",
    "valor_transacao",
    "data",
    "area_construida",
    "uso",
)

SHEET_PATTERN = re.compile(r"^(JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)-(\d{4})$")
MONTHS = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}

# Recortes com volume suficiente para estatística por prédio e distrito.
SEGMENTS = {
    "APARTAMENTO EM CONDOMÍNIO (EXIGE FRAÇÃO IDEAL)": "Apartamento",
    "RESIDÊNCIA": "Casa",
    "ESCRITÓRIO/CONSULTÓRIO EM CONDOMÍNIO (UNID AUTÔNOMA EXIGE FRAÇÃO IDEAL)": "Sala comercial",
    "LOJA": "Loja",
}

# Guias que não são compra e venda (dação, integralização de capital, cisão)
# têm valor pactuado fora de mercado e distorcem a mediana.
PURCHASE_NATURE = "compra e venda"

MIN_AREA_M2 = 15.0
MIN_VALUE = 20_000.0
# Corte de outliers do R$/m²: erro de digitação e permuta simbólica são comuns.
PRICE_M2_RANGE = (500.0, 60_000.0)


def _download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.info("ITBI já em cache: %s", destination.name)
        return destination
    logger.info("Baixando ITBI de %s", url)
    response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def workbook_urls(index_url: str = ITBI_SP_INDEX_URL) -> list[str]:
    """Lista os XLSX publicados na página de guias de ITBI pagas."""
    response = requests.get(index_url, headers=DOWNLOAD_HEADERS, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    links = re.findall(r'href="([^"]+)"', response.text)

    urls: list[str] = []
    for link in links:
        absolute = urljoin(index_url, link.replace(" ", "%20"))
        name = unquote(absolute).lower()
        if "itbi" not in name or ".ods" in name:
            continue
        if not (name.endswith(".xlsx") or "xls-xlsx" in name):
            continue
        if absolute not in urls:
            urls.append(absolute)
    return urls


def _file_name(url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", unquote(url).lower().rsplit("/", 1)[-1]).strip("-")
    return f"{slug}.xlsx"


def _covers(url: str, years: list[int]) -> bool:
    """Ano no nome do arquivo evita baixar planilhas fora do recorte pedido.

    As planilhas correntes são nomeadas pela data de publicação e trazem
    também o ano anterior, então o ano do nome cobre Y-1 e Y.
    """
    marcados = {int(match) for match in re.findall(r"(20\d{2})", unquote(url))}
    if not marcados:
        return True
    return any(year in {marcado - 1, marcado} for marcado in marcados for year in years)


def parse_workbook(path: Path) -> pd.DataFrame:
    """Lê as abas mensais de uma planilha anual e devolve as guias em formato tidy."""
    workbook = pd.ExcelFile(path)
    frames = []
    for sheet in workbook.sheet_names:
        match = SHEET_PATTERN.match(sheet.strip().upper())
        if match is None:
            continue
        raw = _read_sheet(workbook, sheet)
        available = {source: target for source, target in COLUMNS.items() if source in raw.columns}
        frame = raw[list(available)].rename(columns=available)
        faltando = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if faltando:
            raise ValueError(f"{path.name}/{sheet}: colunas obrigatórias ausentes: {faltando}")
        frame["competencia"] = pd.Timestamp(int(match.group(2)), MONTHS[match.group(1)], 1)
        frames.append(frame)

    if not frames:
        raise ValueError(f"Nenhuma aba mensal encontrada em {path}")

    tidy = pd.concat(frames, ignore_index=True)
    # Planilhas antigas omitem colunas opcionais; ausente vira nulo para o
    # filtro distinguir "não informado" de zero.
    for column in COLUMNS.values():
        if column not in tidy.columns:
            tidy[column] = pd.NA
    return tidy


def _read_sheet(workbook: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    raw = workbook.parse(sheet_name=sheet)
    if any(column in raw.columns for column in COLUMNS):
        return raw
    # Coluna final vazia é descartada na leitura, então o layout pode vir
    # truncado à direita; mais colunas que o layout significa outra planilha.
    if len(raw.columns) > len(POSITIONAL_HEADER):
        return raw
    logger.info("%s: aba sem cabeçalho, colunas reconstruídas por posição", sheet)
    names = list(POSITIONAL_HEADER[: len(raw.columns)])
    return workbook.parse(sheet_name=sheet, header=None, names=names)


def _strip_accents(values: pd.Series) -> pd.Series:
    normalized = values.fillna("").astype(str).str.upper().str.strip()
    return normalized.map(
        lambda text: unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    )


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica os filtros de mercado e deriva preço por m², prédio e ágio."""
    frame = raw.copy()
    for column in COLUMNS.values():
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in ("valor_transacao", "valor_venal", "valor_venal_proporcional",
                   "valor_financiado", "area_construida", "proporcao_pct", "padrao"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["data"] = pd.to_datetime(frame["data"], errors="coerce")
    frame["segmento"] = frame["uso"].map(SEGMENTS)
    natureza = frame["natureza"].fillna("").astype(str).str.lower()

    frame = frame[
        natureza.str.contains(PURCHASE_NATURE)
        & frame["segmento"].notna()
        & frame["data"].notna()
        # Transmissão parcial não tem preço comparável ao do imóvel inteiro.
        # Proporção ausente em planilha antiga: a guia é tratada como imóvel inteiro.
        & frame["proporcao_pct"].fillna(100).ge(100)
        & frame["area_construida"].ge(MIN_AREA_M2)
        & frame["valor_transacao"].ge(MIN_VALUE)
    ].copy()

    frame["preco_m2"] = frame["valor_transacao"] / frame["area_construida"]
    frame = frame[frame["preco_m2"].between(*PRICE_M2_RANGE)]

    # O campo "Bairro" da guia é preenchido livremente e recebe "TORRE 1",
    # "BLOCO B" e afins; o CEP é a chave geográfica confiável.
    frame = frame.drop(columns=["bairro"], errors="ignore")
    frame["logradouro"] = _strip_accents(frame["logradouro"])
    # Complemento vem como "AP 71" numa planilha e como 71 em outra: sem tipo
    # único o Parquet não fecha.
    for column in ("complemento", "sql", "uso", "natureza"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["numero"] = pd.to_numeric(frame["numero"], errors="coerce").fillna(0).astype(int)
    cep = pd.to_numeric(frame["cep"], errors="coerce")
    frame["cep"] = cep.fillna(0).astype("int64").astype(str).str.zfill(8)
    # O SQL identifica a unidade; o endereço é o que agrupa unidades do mesmo prédio.
    frame["predio_id"] = frame["logradouro"] + "|" + frame["numero"].astype(str) + "|" + frame["cep"]

    financiado = frame["valor_financiado"]
    frame["financiado"] = financiado.gt(0).astype("boolean").where(financiado.notna())
    venal = frame["valor_venal_proporcional"].where(frame["valor_venal_proporcional"] > 0)
    frame["agio_venal"] = frame["valor_transacao"] / venal - 1

    frame["ano_mes"] = frame["data"].dt.to_period("M").dt.to_timestamp()
    # Planilhas de anos diferentes republicam guias retificadas da mesma venda.
    frame = frame.drop_duplicates(subset=["sql", "data", "valor_transacao"])
    return frame.sort_values("data").reset_index(drop=True)


def load_transactions(
    years: list[int] | None = None,
    force_download: bool = False,
    index_url: str = ITBI_SP_INDEX_URL,
) -> pd.DataFrame:
    """Baixa, normaliza e concatena as guias de ITBI dos anos disponíveis."""
    frames = []
    for url in workbook_urls(index_url):
        if years is not None and not _covers(url, years):
            continue
        path = _download(url, RAW_DIR / "itbi_sp" / _file_name(url), force=force_download)
        try:
            raw = parse_workbook(path)
        except Exception as exc:  # pragma: no cover - planilha fora do padrão
            logger.warning("Planilha ignorada (%s): %s", path.name, exc)
            continue
        if years is not None:
            raw = raw[raw["competencia"].dt.year.isin(years)]
        if raw.empty:
            continue
        logger.info("%s: %s guias", path.name, len(raw))
        frames.append(raw)

    if not frames:
        raise RuntimeError("Nenhuma planilha de ITBI disponível")

    transactions = clean(pd.concat(frames, ignore_index=True))
    if years is not None:
        transactions = transactions[transactions["data"].dt.year.isin(years)]
    logger.info("ITBI-SP: %s transações após limpeza", len(transactions))
    return transactions.reset_index(drop=True)
