"""API que serve o painel GJurema sobre os dados do ITBI-SP.

Cada endpoint declara a origem do dado que devolve (`fonte`): o painel
distingue visualmente o que vem da base pública do que vem do contrato do
cliente, e essa distinção é do dado, não do layout.

Execução:
    uvicorn gjurema.api.app:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from gjurema.api import artifacts, pricing
from gjurema.config import CARTEIRA_PATH

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="GJurema · Inteligência de dados imobiliários", version="0.2.0")


def _data() -> artifacts.MarketData:
    if not artifacts.available():
        raise HTTPException(
            status_code=503,
            detail="Artefatos do ITBI-SP ausentes: rode `python -m gjurema.pipeline_sp all`.",
        )
    return artifacts.market_data()


def _carteira() -> list[dict]:
    if not CARTEIRA_PATH.exists():
        return []
    return json.loads(CARTEIRA_PATH.read_text())


def _item(item_id: str) -> dict:
    for item in _carteira():
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"imóvel {item_id} não está na carteira")


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


@app.get("/api/meta")
def meta() -> dict:
    """Cobertura da base, bairros/segmentos disponíveis e qualidade do modelo."""
    data = _data()
    cobertura = data.metrics.get("cobertura", {})
    return {
        "fonte": "ITBI público · Prefeitura de São Paulo",
        "cobertura": cobertura,
        "modelo": data.metrics.get("preco_m2", {}),
        "bairros": data.bairros,
        "segmentos": data.segmentos,
    }


@app.get("/api/carteira")
def carteira() -> dict:
    """Carteira do cliente já precificada contra o mercado do ITBI."""
    data = _data()
    itens = []
    for item in _carteira():
        estimativa = pricing.price_portfolio_item(data, item)
        itens.append({**item, "precificacao": estimativa})
    return {"fonte": "Contrato do cliente + ITBI público", "itens": itens}


@app.get("/api/precificacao")
def precificacao(
    bairro: str,
    area_total: float = Query(gt=0),
    segmento: str = "Apartamento",
    padrao: float = pricing.DEFAULT_PADRAO,
    preco: float | None = Query(default=None, gt=0),
) -> dict:
    """Preço justo de um imóvel qualquer, dentro ou fora da carteira."""
    data = _data()
    if bairro not in data.bairros:
        raise HTTPException(status_code=404, detail=f"bairro {bairro} sem transações na base")
    estimativa = pricing.price(
        data, bairro=bairro, segmento=segmento, area_total_m2=area_total, padrao=padrao
    )
    if preco:
        estimativa["preco_pago"] = preco
        estimativa["delta_pct"] = (estimativa["valor_justo"] / preco - 1) * 100
    return {"fonte": "ITBI público", **estimativa}


@app.get("/api/serie")
def serie(
    bairro: str | None = None,
    segmento: str | None = None,
    predio_id: str | None = None,
    carteira_id: str | None = None,
) -> dict:
    """Séries anuais de R$/m² do prédio, do bairro e da cidade."""
    data = _data()
    if carteira_id:
        item = _item(carteira_id)
        bairro = bairro or item["bairro"]
        segmento = segmento or item.get("segmento")
        predio_id = predio_id or pricing.resolve_predio(data, item.get("endereco", ""))

    cidade = data.city_series
    if segmento:
        cidade = cidade[cidade["segmento"] == segmento]
    cidade = cidade.groupby("ano", as_index=False).agg(
        preco_m2=("preco_m2", "median"), transacoes=("transacoes", "sum")
    )

    bairro_frame = pd.DataFrame()
    if bairro:
        bairro_frame = data.bairro_series[data.bairro_series["bairro"] == bairro]
        if segmento:
            bairro_frame = bairro_frame[bairro_frame["segmento"] == segmento]
        bairro_frame = bairro_frame.groupby("ano", as_index=False).agg(
            preco_m2=("preco_m2", "median"), transacoes=("transacoes", "sum")
        )

    predio_frame = pd.DataFrame()
    if predio_id:
        predio_frame = data.predio_series[data.predio_series["predio_id"] == predio_id]

    return {
        "fonte": "ITBI público",
        "bairro": bairro,
        "segmento": segmento,
        "predio_id": predio_id,
        "cidade": _records(cidade),
        "bairro_serie": _records(bairro_frame),
        "predio_serie": _records(predio_frame[["ano", "preco_m2", "transacoes"]])
        if not predio_frame.empty
        else [],
    }


@app.get("/api/indices")
def indices(desde: int | None = None) -> dict:
    """Retorno acumulado dos índices de comparação (BCB/SGS)."""
    data = _data()
    frame = data.indices
    if not frame.empty and desde:
        base = frame[frame["ano"] == desde].set_index("indice")["acumulado_pct"]
        frame = frame[frame["ano"] >= desde].copy()
        frame["acumulado_pct"] = [
            (1 + row.acumulado_pct / 100) / (1 + base.get(row.indice, 0.0) / 100) * 100 - 100
            for row in frame.itertuples()
        ]
    return {"fonte": "BCB/SGS", "series": _records(frame)}


@app.get("/api/rentabilidade")
def rentabilidade(carteira_id: str) -> dict:
    """Valorização do imóvel (ITBI) contra os índices, desde a compra."""
    data = _data()
    item = _item(carteira_id)
    compra = pd.Timestamp(item["dataCompra"])

    predio_id = pricing.resolve_predio(data, item.get("endereco", ""))
    fonte_serie = "prédio"
    frame = data.predio_series[data.predio_series["predio_id"] == predio_id] if predio_id else pd.DataFrame()
    if frame.empty:
        fonte_serie = "bairro"
        frame = data.bairro_series[data.bairro_series["bairro"] == item["bairro"]]
        frame = frame[frame["segmento"] == item.get("segmento", "Apartamento")]
        frame = frame.groupby("ano", as_index=False).agg(preco_m2=("preco_m2", "median"))

    frame = frame[frame["ano"] >= compra.year].sort_values("ano")
    imovel = []
    if not frame.empty:
        base = float(frame.iloc[0]["preco_m2"])
        imovel = [
            {"ano": int(row.ano), "acumulado_pct": float(row.preco_m2 / base - 1) * 100}
            for row in frame.itertuples()
        ]

    return {
        "fonte": f"ITBI público · série do {fonte_serie}",
        "imovel": item["nome"],
        "ano_base": int(compra.year),
        "serie": imovel,
        "indices": indices(desde=int(compra.year))["series"],
    }


@app.get("/api/composicao")
def composicao(dim: str = "tipo") -> dict:
    """Composição da carteira do cliente por tipo, bairro, quartos ou construtora."""
    campos = {
        "tipo": "tipo",
        "bairro": "bairro",
        "quartos": "quartos",
        "cidade": "cidade",
        "construtora": "construtora",
    }
    if dim not in campos:
        raise HTTPException(status_code=400, detail=f"dimensão {dim} não suportada")

    agregado: dict[str, list[float]] = {}
    for item in _carteira():
        chave = str(item.get(campos[dim], "—")).split(" · ")[0]
        atual = agregado.setdefault(chave, [0, 0.0])
        atual[0] += 1
        atual[1] += float(item["preco"])

    return {
        "fonte": "Dados do cliente",
        "dimensao": dim,
        "itens": [{"rotulo": k, "ativos": int(v[0]), "valor": v[1]} for k, v in agregado.items()],
    }


@app.get("/api/ranking/valorizacao")
def ranking_valorizacao(limit: int = 12, minimo_transacoes: int = 200) -> dict:
    """Bairros por valorização acumulada do R$/m² (janela da base)."""
    data = _data()
    frame = data.appreciation
    if not frame.empty:
        frame = frame[frame["transacoes"] >= minimo_transacoes].head(limit)
    return {"fonte": "ITBI público", "itens": _records(frame)}


@app.get("/api/ranking/vendas")
def ranking_vendas(limit: int = 12, ano: int | None = None) -> dict:
    """Bairros por volume de transações registradas."""
    data = _data()
    frame = data.transactions
    if ano:
        frame = frame[frame["data"].dt.year == ano]
    contagem = (
        frame.groupby("bairro", as_index=False)
        .agg(transacoes=("preco_m2", "size"), preco_m2=("preco_m2", "median"))
        .sort_values("transacoes", ascending=False)
        .head(limit)
    )
    return {"fonte": "ITBI público", "ano": ano, "itens": _records(contagem)}


@app.get("/api/liquidez")
def liquidez(limit: int = 12, segmento: str | None = None) -> dict:
    """Giro observado por bairro e segmento: transações por ano e meses por venda."""
    data = _data()
    frame = data.liquidity
    if segmento and not frame.empty:
        frame = frame[frame["segmento"] == segmento]
    return {"fonte": "ITBI · liquidez observada", "itens": _records(frame.head(limit))}


@app.get("/api/mercado")
def mercado() -> dict:
    """Indicadores do último mês fechado da base (visão construtora/corretor)."""
    data = _data()
    frame = data.transactions
    ultimo = frame["ano_mes"].max()
    mes = frame[frame["ano_mes"] == ultimo]
    anterior = frame[frame["ano_mes"] == ultimo - pd.DateOffset(months=1)]
    por_mes = (
        frame.groupby("ano_mes", as_index=False)
        .agg(transacoes=("preco_m2", "size"), preco_m2=("preco_m2", "median"))
        .tail(12)
    )
    por_segmento = (
        mes.groupby("segmento", as_index=False)
        .agg(transacoes=("preco_m2", "size"), preco_m2=("preco_m2", "median"))
        .sort_values("transacoes", ascending=False)
    )
    variacao = (len(mes) / len(anterior) - 1) * 100 if len(anterior) else None
    return {
        "fonte": "ITBI público",
        "mes": pd.Timestamp(ultimo).date().isoformat(),
        "transacoes_mes": int(len(mes)),
        "variacao_mes_pct": variacao,
        "preco_m2_mediano": float(mes["preco_m2"].median()),
        "ticket_mediano": float(mes["valor_transacao"].median()),
        "financiado_pct": float(mes["financiado"].mean() * 100),
        "agio_venal_mediano_pct": float(mes["agio_venal"].median() * 100),
        "serie_mensal": _records(por_mes),
        "por_segmento": _records(por_segmento),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
