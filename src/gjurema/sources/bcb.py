"""Séries macroeconômicas do Banco Central (SGS)."""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from gjurema.config import BCB_SGS_URL, HTTP_TIMEOUT, SGS_SERIES

logger = logging.getLogger(__name__)


CHUNK_YEARS = 8
MAX_ATTEMPTS = 4


def _request_window(code: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Consulta uma janela do SGS, tolerando respostas HTML intermitentes."""
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(BCB_SGS_URL.format(code=code), params=params, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise requests.RequestException(f"SGS {code} {params}: {exc}") from exc
            logger.debug("Retentando SGS %s (%s/%s): %s", code, attempt, MAX_ATTEMPTS, exc)
            time.sleep(2 * attempt)
    return []


def fetch_series(code: int, start: str = "01/01/2008") -> pd.DataFrame:
    """Baixa uma série do SGS e devolve colunas date/value.

    A API rejeita janelas longas de séries diárias (HTTP 406), então a
    consulta é fatiada em blocos de poucos anos.
    """
    begin = pd.to_datetime(start, dayfirst=True)
    today = pd.Timestamp.today().normalize()

    records: list[dict] = []
    window_start = begin
    while window_start <= today:
        window_end = min(window_start + pd.DateOffset(years=CHUNK_YEARS) - pd.Timedelta(days=1), today)
        records.extend(_request_window(code, window_start, window_end))
        window_start = window_end + pd.Timedelta(days=1)

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["data"], format="%d/%m/%Y")
    frame["value"] = pd.to_numeric(frame["valor"], errors="coerce")
    return frame[["date", "value"]].dropna().drop_duplicates("date").reset_index(drop=True)


def load_macro(start: str = "01/01/2008") -> pd.DataFrame:
    """Consolida as séries macro relevantes em frequência mensal."""
    frames = []
    for name, code in SGS_SERIES.items():
        try:
            series = fetch_series(code, start=start)
        except requests.RequestException as exc:  # pragma: no cover - rede
            logger.warning("Falha ao baixar série %s (%s): %s", name, code, exc)
            continue
        if series.empty:
            continue
        monthly = (
            series.set_index("date")["value"]
            .resample("MS")
            .mean()
            .rename(name)
        )
        frames.append(monthly)

    if not frames:
        return pd.DataFrame(columns=["date"])

    macro = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date"})
    macro = macro.sort_values("date").ffill()
    return macro
