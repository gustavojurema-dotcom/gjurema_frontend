"""Camada 2 — feature engineering do painel cidade × tipologia × mês."""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = (3, 6, 12)


def _add_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    price = frame["venda_preco_m2"]
    for window in MOMENTUM_WINDOWS:
        frame[f"venda_var_{window}m"] = price.pct_change(window)
    frame["venda_preco_m2_lag1"] = price.shift(1)
    frame["venda_preco_m2_lag12"] = price.shift(12)
    frame["locacao_preco_m2_lag1"] = frame["locacao_preco_m2"].shift(1)
    frame["volatilidade_12m"] = price.pct_change().rolling(12).std()
    return frame


def build_feature_table(
    wide_panel: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    socio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combina FipeZAP, macro do BCB e socioeconômicos do IBGE.

    O resultado é a base de treino do modelo de preço justo: uma linha por
    cidade × tipologia × mês com preço de venda e locação por m², momentum,
    yield implícito e contexto macro/socioeconômico.
    """
    frame = wide_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["venda_preco_m2"])

    frame = pd.concat(
        [_add_momentum(group) for _, group in frame.groupby(["city", "typology"], sort=False)],
        ignore_index=True,
    )

    frame["yield_bruto_anual"] = (frame["locacao_preco_m2"] * 12) / frame["venda_preco_m2"]
    frame["preco_relativo_cidade"] = frame.groupby(["date", "typology"])["venda_preco_m2"].transform(
        lambda s: s / s.median()
    )
    frame["preco_relativo_tipologia"] = frame.groupby(["date", "city"])["venda_preco_m2"].transform(
        lambda s: s / s.median()
    )

    if macro is not None and not macro.empty:
        macro = macro.copy()
        macro["date"] = pd.to_datetime(macro["date"])
        frame = frame.merge(macro, on="date", how="left")

    if socio is not None and not socio.empty:
        columns = [c for c in ("city", "uf", "population", "gdp_per_capita") if c in socio.columns]
        frame = frame.merge(socio[columns], on="city", how="left")
        if "population" in frame:
            frame["log_population"] = np.log1p(frame["population"])
        if "gdp_per_capita" in frame:
            frame["log_gdp_per_capita"] = np.log1p(frame["gdp_per_capita"])

    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["month_index"] = (frame["year"] - frame["year"].min()) * 12 + frame["month"]
    return frame.sort_values(["city", "typology", "date"]).reset_index(drop=True)


def latest_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    """Último mês disponível por cidade × tipologia."""
    return (
        features.sort_values("date")
        .groupby(["city", "typology"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def city_market_index(features: pd.DataFrame) -> pd.DataFrame:
    """Índice de mercado por cidade: aquecimento de preço, yield e volatilidade.

    Normaliza cada componente em ranking percentual e combina:
    valorização 12m (50%), yield bruto (30%) e estabilidade (20%).
    O FipeZAP publica locação para um subconjunto das cidades; onde o yield
    não existe o componente recebe o valor neutro 0,5 em vez de propagar NaN.
    """
    snapshot = latest_snapshot(features)
    snapshot = snapshot[snapshot["typology"] == "Total"].copy()

    appreciation = snapshot["venda_var_12m"].rank(pct=True).fillna(0.5)
    yield_rank = snapshot["yield_bruto_anual"].rank(pct=True).fillna(0.5)
    stability = (1 - snapshot["volatilidade_12m"].rank(pct=True)).fillna(0.5)

    snapshot["indice_mercado"] = 0.5 * appreciation + 0.3 * yield_rank + 0.2 * stability
    snapshot["tem_dados_locacao"] = snapshot["locacao_preco_m2"].notna()
    columns = [
        "city",
        "date",
        "venda_preco_m2",
        "locacao_preco_m2",
        "venda_var_12m",
        "yield_bruto_anual",
        "volatilidade_12m",
        "tem_dados_locacao",
        "indice_mercado",
    ]
    return snapshot[columns].sort_values("indice_mercado", ascending=False).reset_index(drop=True)
