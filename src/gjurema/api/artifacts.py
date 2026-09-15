"""Leitura dos artefatos de São Paulo servidos pela API.

Os arquivos são regravados pelo pipeline enquanto a API está no ar; o cache
é invalidado pela data de modificação e a publicação atômica do pipeline
garante que nunca se leia um arquivo pela metade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gjurema import pipeline_sp
from gjurema.config import ARTIFACTS_DIR
from gjurema.models import price_sp


def _version(*paths: Path) -> tuple[float, ...]:
    return tuple(path.stat().st_mtime if path.exists() else 0.0 for path in paths)


@dataclass
class MarketData:
    """Snapshot dos artefatos de mercado carregado em memória."""

    transactions: pd.DataFrame
    city_series: pd.DataFrame
    bairro_series: pd.DataFrame
    predio_series: pd.DataFrame
    liquidity: pd.DataFrame
    appreciation: pd.DataFrame
    buildings: pd.DataFrame
    indices: pd.DataFrame
    metrics: dict
    model: price_sp.HedonicPriceModel | None

    @property
    def bairros(self) -> list[str]:
        return sorted(self.bairro_series["bairro"].dropna().unique().tolist())

    @property
    def segmentos(self) -> list[str]:
        return sorted(self.city_series["segmento"].dropna().unique().tolist())

    def bairro_level(self, bairro: str, segmento: str | None = None) -> float | None:
        """Último R$/m² mediano observado no bairro."""
        frame = self.bairro_series[self.bairro_series["bairro"] == bairro]
        if segmento:
            frame = frame[frame["segmento"] == segmento]
        if frame.empty:
            return None
        return float(frame.sort_values("ano").iloc[-1]["preco_m2"])

    def predio_level(self, predio_id: str) -> float | None:
        frame = self.predio_series[self.predio_series["predio_id"] == predio_id]
        if frame.empty:
            return None
        return float(frame.sort_values("ano").iloc[-1]["preco_m2"])


PATHS = (
    pipeline_sp.TRANSACTIONS_PATH,
    pipeline_sp.SERIES_CITY_PATH,
    pipeline_sp.SERIES_BAIRRO_PATH,
    pipeline_sp.SERIES_PREDIO_PATH,
    pipeline_sp.LIQUIDITY_PATH,
    pipeline_sp.APPRECIATION_PATH,
    pipeline_sp.BUILDINGS_PATH,
    pipeline_sp.INDICES_PATH,
    pipeline_sp.METRICS_PATH,
    ARTIFACTS_DIR / price_sp.META_FILE,
)

_cache: tuple[tuple[float, ...], MarketData] | None = None


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load() -> MarketData:
    model = None
    if (ARTIFACTS_DIR / price_sp.META_FILE).exists():
        model = price_sp.HedonicPriceModel.load(ARTIFACTS_DIR)
    metrics = (
        json.loads(pipeline_sp.METRICS_PATH.read_text()) if pipeline_sp.METRICS_PATH.exists() else {}
    )
    return MarketData(
        transactions=_read(pipeline_sp.TRANSACTIONS_PATH),
        city_series=_read(pipeline_sp.SERIES_CITY_PATH),
        bairro_series=_read(pipeline_sp.SERIES_BAIRRO_PATH),
        predio_series=_read(pipeline_sp.SERIES_PREDIO_PATH),
        liquidity=_read(pipeline_sp.LIQUIDITY_PATH),
        appreciation=_read(pipeline_sp.APPRECIATION_PATH),
        buildings=_read(pipeline_sp.BUILDINGS_PATH),
        indices=_read(pipeline_sp.INDICES_PATH),
        metrics=metrics,
        model=model,
    )


def market_data() -> MarketData:
    """Artefatos em cache, recarregados quando o pipeline os republica."""
    global _cache
    version = _version(*PATHS)
    if _cache is None or _cache[0] != version:
        _cache = (version, _load())
    return _cache[1]


def available() -> bool:
    return pipeline_sp.TRANSACTIONS_PATH.exists()
