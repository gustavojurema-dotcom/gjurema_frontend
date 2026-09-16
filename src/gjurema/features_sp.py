"""Agregados de mercado de São Paulo a partir das guias de ITBI.

Do transacional saem três recortes usados pelo produto: séries de R$/m²
(cidade, bairro e prédio), liquidez observada (guias por mês) e valorização
acumulada. Medianas com poucas observações são instáveis, então todo recorte
carrega o número de transações que o sustenta.
"""

from __future__ import annotations

import pandas as pd

# Abaixo disso a mediana do mês responde a uma única venda atípica.
MIN_TRANSACTIONS = 5


def monthly_series(transactions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mediana mensal de R$/m² por recorte, com a contagem que a sustenta."""
    grouped = transactions.groupby([*keys, "ano_mes"], dropna=True, observed=True)
    series = grouped.agg(
        preco_m2=("preco_m2", "median"),
        valor_mediano=("valor_transacao", "median"),
        area_mediana=("area_construida", "median"),
        transacoes=("preco_m2", "size"),
        financiado_pct=("financiado", "mean"),
        agio_venal_mediano=("agio_venal", "median"),
    ).reset_index()
    series["financiado_pct"] = series["financiado_pct"] * 100
    return series.sort_values([*keys, "ano_mes"]).reset_index(drop=True)


def annual_series(transactions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Versão anual das séries — é o recorte que o gráfico de evolução usa."""
    frame = transactions.copy()
    frame["ano_mes"] = frame["data"].dt.year
    series = monthly_series(frame, keys).rename(columns={"ano_mes": "ano"})
    return series[series["transacoes"] >= MIN_TRANSACTIONS].reset_index(drop=True)


def liquidity(transactions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Giro observado: transações por ano e meses por transação no recorte."""
    months = transactions["ano_mes"].nunique()
    if months == 0:
        return pd.DataFrame(columns=[*keys, "transacoes", "transacoes_ano", "giro_meses"])

    grouped = transactions.groupby(keys, dropna=True, observed=True)
    frame = grouped.agg(
        transacoes=("preco_m2", "size"),
        preco_m2=("preco_m2", "median"),
    ).reset_index()
    frame["transacoes_ano"] = frame["transacoes"] / (months / 12)
    frame["giro_meses"] = months / frame["transacoes"]
    return frame.sort_values("transacoes", ascending=False).reset_index(drop=True)


def appreciation(series: pd.DataFrame, key: str, years: int = 5) -> pd.DataFrame:
    """Valorização acumulada do R$/m² entre o primeiro e o último ano do recorte."""
    if series.empty:
        return pd.DataFrame(columns=[key, "ano_inicial", "ano_final", "preco_m2", "valorizacao_pct"])

    last_year = int(series["ano"].max())
    window = series[series["ano"] >= last_year - years]
    frames = []
    for name, group in window.groupby(key, observed=True):
        group = group.sort_values("ano")
        first, last = group.iloc[0], group.iloc[-1]
        if first["ano"] == last["ano"]:
            continue
        frames.append(
            {
                key: name,
                "ano_inicial": int(first["ano"]),
                "ano_final": int(last["ano"]),
                "preco_m2": float(last["preco_m2"]),
                "transacoes": int(group["transacoes"].sum()),
                "valorizacao_pct": float(last["preco_m2"] / first["preco_m2"] - 1) * 100,
            }
        )
    ranking = pd.DataFrame(frames)
    if ranking.empty:
        return ranking
    return ranking.sort_values("valorizacao_pct", ascending=False).reset_index(drop=True)


def building_directory(transactions: pd.DataFrame) -> pd.DataFrame:
    """Catálogo de prédios com endereço, volume e preço mediano mais recente."""
    grouped = transactions.sort_values("data").groupby("predio_id", observed=True)
    directory = grouped.agg(
        logradouro=("logradouro", "last"),
        numero=("numero", "last"),
        cep=("cep", "last"),
        bairro=("bairro", "last"),
        segmento=("segmento", "last"),
        transacoes=("preco_m2", "size"),
        preco_m2=("preco_m2", "median"),
        area_mediana=("area_construida", "median"),
        ultima_transacao=("data", "max"),
    ).reset_index()
    directory["endereco"] = (
        directory["logradouro"] + ", " + directory["numero"].astype(str)
    )
    return directory.sort_values("transacoes", ascending=False).reset_index(drop=True)


def market_index(liquidez: pd.DataFrame, valorizacao: pd.DataFrame, key: str) -> pd.DataFrame:
    """Índice 0–1 por recorte combinando valorização e liquidez relativas."""
    frame = valorizacao.merge(liquidez[[key, "transacoes_ano", "giro_meses"]], on=key, how="left")
    if frame.empty:
        return frame

    def rank(column: str) -> pd.Series:
        return frame[column].rank(pct=True).fillna(0.5)

    frame["indice_mercado"] = 0.6 * rank("valorizacao_pct") + 0.4 * rank("transacoes_ano")
    return frame.sort_values("indice_mercado", ascending=False).reset_index(drop=True)
