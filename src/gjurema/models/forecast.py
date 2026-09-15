"""Projeção de valorização por cidade × tipologia (tendência amortecida).

O MVP usa um Holt com tendência amortecida implementado em numpy — leve,
sem dependências extras e adequado às séries mensais suavizadas do FipeZAP.
As camadas seguintes do roadmap substituem isto por Prophet/LSTM com
variáveis exógenas (SELIC, emprego, população).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _holt_damped(series: np.ndarray, horizon: int, alpha: float, beta: float, phi: float) -> np.ndarray:
    level = series[0]
    trend = series[1] - series[0] if series.size > 1 else 0.0
    for value in series[1:]:
        previous_level = level
        level = alpha * value + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - previous_level) + (1 - beta) * phi * trend

    damping = np.cumsum(phi ** np.arange(1, horizon + 1))
    return level + damping * trend


def project_prices(
    features: pd.DataFrame,
    horizon_months: int = 12,
    alpha: float = 0.3,
    beta: float = 0.1,
    phi: float = 0.92,
    min_history: int = 24,
) -> pd.DataFrame:
    """Projeta o preço por m² e a valorização esperada no horizonte."""
    rows = []
    for (city, typology), group in features.groupby(["city", "typology"]):
        history = group.sort_values("date")["venda_preco_m2"].dropna()
        if len(history) < min_history:
            continue
        log_history = np.log(history.to_numpy())
        projection = np.exp(_holt_damped(log_history, horizon_months, alpha, beta, phi))
        last_price = float(history.iloc[-1])
        rows.append(
            {
                "city": city,
                "typology": typology,
                "preco_m2_atual": last_price,
                "preco_m2_projetado": float(projection[-1]),
                "valorizacao_projetada_pct": float(projection[-1] / last_price - 1) * 100,
                "horizonte_meses": horizon_months,
            }
        )
    return pd.DataFrame(rows).sort_values("valorizacao_projetada_pct", ascending=False).reset_index(drop=True)


def backtest(
    features: pd.DataFrame,
    horizon_months: int = 12,
    **kwargs: float,
) -> dict[str, float]:
    """Erro médio da projeção quando treinada até `horizon_months` atrás."""
    errors = []
    for _, group in features.groupby(["city", "typology"]):
        history = group.sort_values("date")["venda_preco_m2"].dropna()
        if len(history) < 24 + horizon_months:
            continue
        train = np.log(history.iloc[:-horizon_months].to_numpy())
        actual = float(history.iloc[-1])
        projection = _holt_damped(
            train,
            horizon_months,
            kwargs.get("alpha", 0.3),
            kwargs.get("beta", 0.1),
            kwargs.get("phi", 0.92),
        )
        predicted = float(np.exp(projection[-1]))
        errors.append(abs(predicted - actual) / actual)
    if not errors:
        return {"mape_pct": float("nan"), "n": 0}
    return {"mape_pct": float(np.mean(errors) * 100), "n": len(errors)}
