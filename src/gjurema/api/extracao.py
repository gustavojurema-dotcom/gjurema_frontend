"""Leitura de contratos e fichas de imóvel (PDF, Word, imagem, texto).

A extração é um rascunho do cadastro: cada campo reconhecido volta com o
trecho de onde saiu, para o cliente conferir antes de gravar na carteira.
Nada é inventado — o que não aparece no documento volta vazio.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

PDF_SUFFIXES = (".pdf",)
DOCX_SUFFIXES = (".docx",)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
TEXT_SUFFIXES = (".txt", ".csv", ".md")
SUPPORTED_SUFFIXES = PDF_SUFFIXES + DOCX_SUFFIXES + IMAGE_SUFFIXES + TEXT_SUFFIXES

MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}

MOEDA = r"R?\$?\s*([\d][\d\.\s]*,\d{2}|[\d][\d\.]*\d)"
AREA = r"([\d]{1,5}(?:[.,]\d{1,3})?)\s*m\s*[²2]"


class UnsupportedFile(ValueError):
    """Formato sem leitor disponível."""


def _suffix(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def extract_text(filename: str, content: bytes) -> str:
    """Texto bruto do arquivo; imagens passam por OCR."""
    suffix = _suffix(filename)
    if suffix in TEXT_SUFFIXES:
        return content.decode("utf-8", errors="ignore")
    if suffix in PDF_SUFFIXES:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in DOCX_SUFFIXES:
        import docx

        documento = docx.Document(io.BytesIO(content))
        paragrafos = [p.text for p in documento.paragraphs]
        celulas = [c.text for t in documento.tables for r in t.rows for c in r.cells]
        return "\n".join(paragrafos + celulas)
    if suffix in IMAGE_SUFFIXES:
        import pytesseract
        from PIL import Image

        try:
            return pytesseract.image_to_string(Image.open(io.BytesIO(content)), lang="por")
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(Image.open(io.BytesIO(content)))
    raise UnsupportedFile(f"formato {suffix or filename} não suportado")


def _to_number(texto: str) -> float | None:
    limpo = re.sub(r"[^\d,.]", "", texto)
    if not limpo:
        return None
    if "," in limpo:
        limpo = limpo.replace(".", "").replace(",", ".")
    elif limpo.count(".") > 1 or re.fullmatch(r"\d{1,3}(\.\d{3})+", limpo):
        limpo = limpo.replace(".", "")
    try:
        return float(limpo)
    except ValueError:
        return None


def _find_money(texto: str, rotulos: tuple[str, ...]) -> float | None:
    for rotulo in rotulos:
        achado = re.search(rf"{rotulo}[^\n\r:]*[:\s]\s*{MOEDA}", texto, re.IGNORECASE)
        if achado:
            valor = _to_number(achado.group(1))
            if valor:
                return valor
    return None


def _find_date(texto: str, rotulos: tuple[str, ...]) -> str | None:
    for rotulo in rotulos:
        janela = re.search(rf"{rotulo}[^\n\r]{{0,80}}", texto, re.IGNORECASE)
        if not janela:
            continue
        trecho = janela.group(0)
        numerica = re.search(r"(\d{2})[/-](\d{2})[/-](\d{4})", trecho)
        if numerica:
            dia, mes, ano = (int(g) for g in numerica.groups())
            return date(ano, mes, dia).isoformat()
        iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", trecho)
        if iso:
            return iso.group(0)
        extenso = re.search(r"(\d{1,2})\s+de\s+([a-zçã]+)\s+de\s+(\d{4})", trecho, re.IGNORECASE)
        if extenso and extenso.group(2).lower() in MESES:
            return date(int(extenso.group(3)), MESES[extenso.group(2).lower()], int(extenso.group(1))).isoformat()
    return None


def _find_area(texto: str, rotulos: tuple[str, ...]) -> float | None:
    for rotulo in rotulos:
        achado = re.search(rf"{rotulo}[^\n\r]{{0,40}}?{AREA}", texto, re.IGNORECASE)
        if achado:
            return _to_number(achado.group(1))
    return None


def parse_fields(texto: str) -> dict:
    """Campos da carteira reconhecidos no documento (sem inferir o ausente)."""
    campos: dict = {}

    endereco = re.search(
        r"((?:Rua|Avenida|Av\.?|Alameda|Pra[cç]a|Travessa|Estrada|Rodovia)\s+[^\n,;]{3,60},?\s*n?[º°]?\s*\d{1,6})",
        texto,
        re.IGNORECASE,
    )
    if endereco:
        campos["endereco"] = re.sub(r"\s+", " ", endereco.group(1)).replace(" ,", ",").strip()

    cep = re.search(r"\b(\d{5})-?(\d{3})\b", texto)
    if cep:
        campos["cep"] = f"{cep.group(1)}-{cep.group(2)}"

    bairro = re.search(r"bairro[:\s]+([^\n,;]{2,40})", texto, re.IGNORECASE)
    if bairro:
        campos["bairro"] = bairro.group(1).strip()

    unidade = re.search(
        r"\b(?:unidade|apartamento|apto|ap|conjunto|cj)\b\.?\s*n?[º°]?\s*[:\-]?\s*(\d{1,5}[A-Za-z]?)",
        texto,
        re.IGNORECASE,
    )
    if unidade:
        campos["unidade"] = f"Unidade {unidade.group(1)}"

    matricula = re.search(r"matr[ií]cula\s*n?[º°]?\s*([\d\.]{3,15})", texto, re.IGNORECASE)
    if matricula:
        campos["matricula"] = matricula.group(1)

    empreendimento = re.search(r"(?:edif[ií]cio|empreendimento|condom[ií]nio)\s+([^\n,;]{3,60})", texto, re.IGNORECASE)
    if empreendimento:
        campos["nome"] = empreendimento.group(1).strip()

    construtora = re.search(r"(?:construtora|incorporadora)[:\s]+([^\n,;]{3,60})", texto, re.IGNORECASE)
    if construtora:
        campos["construtora"] = construtora.group(1).strip()

    vagas = re.search(r"(\d{1,2})\s*vagas?", texto, re.IGNORECASE)
    if vagas:
        campos["vagas"] = int(vagas.group(1))

    quartos = re.search(r"(\d{1,2})\s*(?:dormit[óo]rios?|quartos?)", texto, re.IGNORECASE)
    if quartos:
        campos["quartos"] = f"{quartos.group(1)} dorm."

    preco = _find_money(
        texto,
        (
            "valor (?:total )?(?:da |de )?(?:compra|venda|aquisi[cç][ãa]o)",
            "pre[cç]o (?:total|de venda)",
            "valor do im[óo]vel",
        ),
    )
    if preco:
        campos["preco"] = preco

    aluguel = _find_money(texto, ("aluguel", "loca[cç][ãa]o", "valor locat[íi]cio"))
    if aluguel:
        campos["aluguelMensal"] = aluguel

    condominio = _find_money(texto, ("condom[íi]nio",))
    if condominio:
        campos["condominioMensal"] = condominio

    iptu = _find_money(texto, ("iptu",))
    if iptu:
        campos["iptuAnual"] = iptu

    corretagem = _find_money(texto, ("corretagem", "comiss[ãa]o"))
    if corretagem:
        campos["corretagem"] = corretagem

    privativa = _find_area(texto, ("[áa]rea privativa", "privativa"))
    if privativa:
        campos["priv"] = privativa
    comum = _find_area(texto, ("[áa]rea comum", "comum"))
    if comum:
        campos["comum"] = comum
    total = _find_area(texto, ("[áa]rea total", "[áa]rea constru[íi]da", "total"))
    if total:
        campos["total"] = total

    compra = _find_date(texto, ("data (?:da )?(?:compra|venda|aquisi[cç][ãa]o)", "assinatura", "escritura"))
    if compra:
        campos["dataCompra"] = compra

    return campos


def parse_documents(arquivos: list[tuple[str, bytes]]) -> dict:
    """Consolida vários documentos: o primeiro valor reconhecido prevalece."""
    campos: dict = {}
    lidos: list[dict] = []
    for nome, conteudo in arquivos:
        try:
            texto = extract_text(nome, conteudo)
        except UnsupportedFile as exc:
            lidos.append({"arquivo": nome, "erro": str(exc), "campos": []})
            continue
        except Exception as exc:  # pragma: no cover - dependente do arquivo
            logger.warning("Falha lendo %s: %s", nome, exc)
            lidos.append({"arquivo": nome, "erro": f"não foi possível ler o arquivo: {exc}", "campos": []})
            continue
        encontrados = parse_fields(texto)
        lidos.append(
            {
                "arquivo": nome,
                "caracteres": len(texto),
                "campos": sorted(encontrados),
                "erro": None if texto.strip() else "documento sem texto legível (imagem sem OCR?)",
            }
        )
        for chave, valor in encontrados.items():
            campos.setdefault(chave, valor)
    return {"campos": campos, "arquivos": lidos}
