"""Ingestão das séries históricas do FipeZAP.

A planilha oficial traz, por cidade, um painel mensal com número-índice,
variações, preço médio de venda (R$/m²), preço médio de locação (R$/m²) e
rental yield, sempre abertos por tipologia (Total, 1D, 2D, 3D, 4D).
Este módulo baixa a planilha e a normaliza em formato tidy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from gjurema.config import FIPEZAP_URL, HTTP_TIMEOUT, RAW_DIR

logger = logging.getLogger(__name__)

NON_CITY_SHEETS = {"Resumo", "Aux", "Índice FipeZAP"}

# (coluna inicial, segmento, métrica) para o bloco residencial da planilha.
# Cada bloco ocupa cinco colunas: Total, 1D, 2D, 3D, 4D.
RESIDENTIAL_BLOCKS = [
    (2, "venda", "indice"),
    (7, "venda", "var_mensal"),
    (12, "venda", "var_12m"),
    (17, "venda", "preco_m2"),
    (22, "locacao", "indice"),
    (27, "locacao", "var_mensal"),
    (32, "locacao", "var_12m"),
    (37, "locacao", "preco_m2"),
    (42, "rentabilidade", "yield_mensal"),
]

DATE_COLUMN = 1
FIRST_DATA_ROW = 4
TYPOLOGY_ORDER = ["Total", "1D", "2D", "3D", "4D"]


def download(destination: Path | None = None, force: bool = False) -> Path:
    """Baixa a planilha de séries históricas do FipeZAP."""
    destination = destination or RAW_DIR / "fipezap-serieshistoricas.xlsx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.info("FipeZAP já em cache: %s", destination)
        return destination
    logger.info("Baixando FipeZAP de %s", FIPEZAP_URL)
    response = requests.get(FIPEZAP_URL, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _parse_city_sheet(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    body = raw.iloc[FIRST_DATA_ROW:].copy()
    dates = pd.to_datetime(body.iloc[:, DATE_COLUMN], errors="coerce")
    body = body[dates.notna()]
    dates = dates[dates.notna()]

    frames = []
    for start, segment, metric in RESIDENTIAL_BLOCKS:
        block = body.iloc[:, start : start + len(TYPOLOGY_ORDER)].copy()
        block.columns = TYPOLOGY_ORDER
        block = block.apply(pd.to_numeric, errors="coerce")
        block["date"] = dates.values
        melted = block.melt(id_vars="date", var_name="typology", value_name="value")
        melted["segment"] = segment
        melted["metric"] = metric
        frames.append(melted)

    tidy = pd.concat(frames, ignore_index=True)
    tidy["city"] = city
    return tidy.dropna(subset=["value"])


def load_panel(path: Path | None = None, force_download: bool = False) -> pd.DataFrame:
    """Devolve o painel tidy: city, date, segment, typology, metric, value."""
    path = path or download(force=force_download)
    workbook = pd.ExcelFile(path)
    cities = [s for s in workbook.sheet_names if s not in NON_CITY_SHEETS]

    frames = []
    for city in cities:
        raw = workbook.parse(sheet_name=city, header=None)
        if raw.shape[1] <= RESIDENTIAL_BLOCKS[-1][0]:
            logger.warning("Aba %s com layout inesperado — ignorada", city)
            continue
        frames.append(_parse_city_sheet(raw, city))

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["city", "segment", "metric", "typology", "date"]).reset_index(drop=True)


def wide_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivota o painel para uma linha por cidade × tipologia × mês."""
    selected = panel[panel["metric"].isin({"preco_m2", "indice", "var_12m", "yield_mensal"})]
    wide = selected.pivot_table(
        index=["city", "typology", "date"],
        columns=["segment", "metric"],
        values="value",
        aggfunc="first",
    )
    wide.columns = [f"{segment}_{metric}" for segment, metric in wide.columns]
    return wide.reset_index()
