"""Precificação de um imóvel a partir do modelo hedônico e dos comparáveis.

O ITBI descreve área construída total, então a comparação com o mercado é
sempre por m² total; o m² privativo aparece só como referência do contrato.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from gjurema.api.artifacts import MarketData
from gjurema.models import price_sp

DEFAULT_PADRAO = 3.0


def normalize(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", str(text).upper().strip())
    return stripped.encode("ascii", "ignore").decode()


def resolve_predio(data: MarketData, endereco: str) -> str | None:
    """Encontra o prédio do ITBI pelo logradouro e número do contrato."""
    if data.buildings.empty or not endereco:
        return None
    match = re.match(r"^(.*?),\s*(\d+)", endereco)
    if not match:
        return None
    street, number = normalize(match.group(1)), int(match.group(2))
    # O ITBI grava o logradouro sem o tipo ("Rua", "Avenida").
    core = re.sub(r"^(RUA|AVENIDA|AV|ALAMEDA|PRACA|TRAVESSA|ESTRADA|RODOVIA)\s+", "", street)
    candidates = data.buildings[
        (data.buildings["numero"] == number)
        & data.buildings["logradouro"].str.contains(re.escape(core), na=False)
    ]
    if candidates.empty:
        return None
    return str(candidates.sort_values("transacoes", ascending=False).iloc[0]["predio_id"])


def price(
    data: MarketData,
    *,
    bairro: str,
    segmento: str,
    area_total_m2: float,
    padrao: float = DEFAULT_PADRAO,
    referencia: pd.Timestamp | None = None,
    predio_id: str | None = None,
) -> dict:
    """Preço justo por m² e total, com a faixa de referência do modelo."""
    if data.model is None:
        raise RuntimeError("modelo hedônico de São Paulo ainda não foi treinado")

    referencia = pd.Timestamp(referencia or data.transactions["data"].max())
    bairro_m2 = data.bairro_level(bairro, segmento) or data.bairro_level(bairro)
    predio_m2 = data.predio_level(predio_id) if predio_id else None

    row = price_sp.design_row(
        segmento=segmento,
        bairro=bairro,
        area_m2=area_total_m2,
        padrao=padrao,
        data=referencia,
        origin=pd.Timestamp(data.model.origin),
        preco_m2_predio=predio_m2,
        preco_m2_bairro=bairro_m2,
    )
    interval = data.model.predict_interval(row).iloc[0]

    comparaveis = data.transactions[
        (data.transactions["bairro"] == bairro) & (data.transactions["segmento"] == segmento)
    ]
    return {
        "preco_justo_m2": float(interval["preco_justo_m2"]),
        "preco_justo_m2_p10": float(interval["preco_justo_m2_p10"]),
        "preco_justo_m2_p90": float(interval["preco_justo_m2_p90"]),
        "valor_justo": float(interval["preco_justo_m2"]) * area_total_m2,
        "valor_justo_p10": float(interval["preco_justo_m2_p10"]) * area_total_m2,
        "valor_justo_p90": float(interval["preco_justo_m2_p90"]) * area_total_m2,
        "mediana_bairro_m2": bairro_m2,
        "mediana_predio_m2": predio_m2,
        "transacoes_bairro": int(len(comparaveis)),
        "referencia": referencia.date().isoformat(),
        "mape_pct": float(data.model.metrics.get("mape_pct", 0.0)),
    }


def price_portfolio_item(data: MarketData, item: dict) -> dict:
    """Aplica a precificação a um imóvel da carteira do cliente."""
    predio_id = resolve_predio(data, item.get("endereco", ""))
    estimativa = price(
        data,
        bairro=item["bairro"],
        segmento=item.get("segmento", "Apartamento"),
        area_total_m2=float(item["total"]),
        padrao=float(item.get("padrao", DEFAULT_PADRAO)),
        predio_id=predio_id,
    )
    pago = float(item["preco"])
    estimativa.update(
        {
            "predio_id": predio_id,
            "preco_pago": pago,
            "preco_m2_pago_total": pago / float(item["total"]),
            "preco_m2_pago_privativo": pago / float(item["priv"]),
            "delta_pct": (estimativa["valor_justo"] / pago - 1) * 100,
        }
    )
    return estimativa
