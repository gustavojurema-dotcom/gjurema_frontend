"""Índices de comparação (BCB/SGS) para a aba de rentabilidade.

O produto compara a valorização do imóvel com o custo de oportunidade do
investidor, então as séries entram como retorno acumulado desde o ano-base:
taxas mensais são compostas e o dólar é comparado pelo nível médio do mês.
"""

from __future__ import annotations

import logging

import pandas as pd

from gjurema.sources import bcb

logger = logging.getLogger(__name__)

RATE_SERIES = {"cdi": 4391, "ipca": 433, "selic": 4390, "igpm": 189}
LEVEL_SERIES = {"dolar": 3698}
LABELS = {
    "cdi": "CDI",
    "ipca": "IPCA",
    "selic": "Selic",
    "igpm": "IGP-M",
    "dolar": "Dólar",
}


def _year_end_factor(monthly: pd.Series) -> pd.Series:
    """Fator acumulado até dezembro de cada ano, a partir de taxas mensais em %."""
    factors = (1 + monthly / 100).cumprod()
    return factors.groupby(factors.index.year).last()


def annual_accumulated(start_year: int, end_year: int) -> pd.DataFrame:
    """Retorno acumulado por ano (%) com base 0 no ano inicial."""
    rows: list[dict] = []
    for name, code in {**RATE_SERIES, **LEVEL_SERIES}.items():
        try:
            series = bcb.fetch_series(code, start=f"01/01/{start_year}")
        except Exception as exc:  # pragma: no cover - rede
            logger.warning("Índice %s indisponível: %s", name, exc)
            continue
        if series.empty:
            continue

        monthly = series.set_index("date")["value"].resample("MS").mean().dropna()
        if name in RATE_SERIES:
            yearly = _year_end_factor(monthly)
        else:
            yearly = monthly.groupby(monthly.index.year).last() / monthly.iloc[0]

        base = yearly.loc[start_year] if start_year in yearly.index else 1.0
        for year, value in yearly.items():
            if start_year <= year <= end_year:
                rows.append(
                    {
                        "indice": name,
                        "nome": LABELS[name],
                        "ano": int(year),
                        "acumulado_pct": float(value / base - 1) * 100,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["indice", "ano"]).reset_index(drop=True)
