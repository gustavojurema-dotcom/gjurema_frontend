"""API que serve o painel GJurema sobre os dados do ITBI-SP.

Cada endpoint declara a origem do dado que devolve (`fonte`): o painel
distingue visualmente o que vem da base pública do que vem do contrato do
cliente, e essa distinção é do dado, não do layout.

Execução:
    uvicorn gjurema.api.app:app --reload
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from gjurema.api import artifacts, extracao, pricing
from gjurema.api import carteira as carteira_store

STATIC_DIR = Path(__file__).parent / "static"

# Carteira, contrato e matrícula são dados do cliente: quando a API sai do
# localhost, `GJUREMA_API_TOKEN` passa a ser exigido nesses endpoints.
TOKEN_ENV = "GJUREMA_API_TOKEN"
TOKEN_HEADER = "X-GJurema-Token"

# Precificação roda o modelo a cada chamada: janela simples evita que um
# cliente monopolize o processo.
RATE_LIMIT = int(os.getenv("GJUREMA_RATE_LIMIT", "120"))
RATE_WINDOW_S = 60.0

ARQUIVOS = File(...)

app = FastAPI(title="GJurema · Inteligência de dados imobiliários", version="0.2.0")

_hits: dict[str, deque[float]] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    agora = time.monotonic()
    origem = request.client.host if request.client else "desconhecido"
    janela = _hits.setdefault(origem, deque())
    while janela and agora - janela[0] > RATE_WINDOW_S:
        janela.popleft()
    if len(janela) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": f"limite de {RATE_LIMIT} chamadas por minuto atingido"},
        )
    janela.append(agora)
    return await call_next(request)


def require_private(
    x_gjurema_token: str | None = Header(default=None, alias=TOKEN_HEADER),
) -> None:
    """Protege os dados do cliente quando um token está configurado."""
    esperado = os.getenv(TOKEN_ENV, "")
    if esperado and x_gjurema_token != esperado:
        raise HTTPException(status_code=401, detail="token da carteira ausente ou inválido")


def _data() -> artifacts.MarketData:
    if not artifacts.available():
        raise HTTPException(
            status_code=503,
            detail="Artefatos do ITBI-SP ausentes: rode `python -m gjurema.pipeline_sp all`.",
        )
    return artifacts.market_data()


def _carteira() -> list[dict]:
    return carteira_store.load()


def _item(item_id: str) -> dict:
    for item in _carteira():
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"imóvel {item_id} não está na carteira")


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _annual(frame: pd.DataFrame) -> pd.DataFrame:
    """Série anual de R$/m² direto das transações.

    Mediana de medianas não é mediana: quando o recorte não fixa o segmento,
    o agregado por segmento não pode ser reaproveitado.
    """
    if frame.empty:
        return pd.DataFrame(columns=["ano", "preco_m2", "transacoes"])
    return (
        frame.assign(ano=frame["data"].dt.year)
        .groupby("ano", as_index=False)
        .agg(preco_m2=("preco_m2", "median"), transacoes=("preco_m2", "size"))
    )


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


@app.get("/api/carteira", dependencies=[Depends(require_private)])
def carteira() -> dict:
    """Carteira do cliente já precificada contra o mercado do ITBI."""
    data = _data()
    itens = []
    for item in _carteira():
        estimativa = pricing.price_portfolio_item(data, item)
        itens.append({**item, "precificacao": estimativa})
    return {"fonte": "Contrato do cliente + ITBI público", "itens": itens}


@app.get("/api/carteira/campos")
def carteira_campos() -> dict:
    """Esquema do cadastro: o painel monta o formulário a partir daqui."""
    return {"fonte": "Dados do cliente", "campos": list(carteira_store.CAMPOS)}


@app.post("/api/carteira", status_code=201, dependencies=[Depends(require_private)])
def carteira_criar(payload: dict) -> dict:
    """Cadastra um imóvel na carteira e já o precifica contra o ITBI."""
    try:
        item = carteira_store.create(payload)
    except carteira_store.ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"fonte": "Dados do cliente", "item": item}


@app.patch("/api/carteira/{item_id}", dependencies=[Depends(require_private)])
def carteira_atualizar(item_id: str, payload: dict) -> dict:
    """Atualiza campos do imóvel — inclusive o aluguel vigente."""
    try:
        item = carteira_store.update(item_id, payload)
    except carteira_store.ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"imóvel {item_id} não está na carteira") from exc
    return {"fonte": "Dados do cliente", "item": item}


@app.delete("/api/carteira/{item_id}", dependencies=[Depends(require_private)])
def carteira_remover(item_id: str) -> dict:
    try:
        carteira_store.delete(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"imóvel {item_id} não está na carteira") from exc
    return {"removido": item_id}


@app.post("/api/carteira/extrair", dependencies=[Depends(require_private)])
async def carteira_extrair(arquivos: list[UploadFile] = ARQUIVOS) -> dict:
    """Rascunho de cadastro a partir de contratos em PDF, Word ou imagem.

    Devolve só o que foi reconhecido no documento: o cliente confere e
    completa antes de gravar.
    """
    lidos = [(arquivo.filename or "arquivo", await arquivo.read()) for arquivo in arquivos]
    resultado = extracao.parse_documents(lidos)
    return {"fonte": "Documento do cliente", **resultado}


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

    if segmento:
        cidade = data.city_series[data.city_series["segmento"] == segmento]
        cidade = cidade[["ano", "preco_m2", "transacoes"]]
    else:
        cidade = _annual(data.transactions)

    bairro_frame = pd.DataFrame()
    if bairro:
        if segmento:
            bairro_frame = data.bairro_series[
                (data.bairro_series["bairro"] == bairro)
                & (data.bairro_series["segmento"] == segmento)
            ][["ano", "preco_m2", "transacoes"]]
        else:
            bairro_frame = _annual(data.transactions[data.transactions["bairro"] == bairro])

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


@app.get("/api/rentabilidade", dependencies=[Depends(require_private)])
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


@app.get("/api/rentabilidade/mensal", dependencies=[Depends(require_private)])
def rentabilidade_mensal(carteira_id: str, meses: int = Query(default=36, ge=6, le=240)) -> dict:
    """Rendimento mensal do imóvel (aluguel + valorização) contra os índices.

    O ITBI só tem série anual de R$/m², então o valor do imóvel é
    interpolado mês a mês entre os anos observados.
    """
    data = _data()
    item = _item(carteira_id)
    area = float(item["total"])
    aluguel = item.get("aluguelMensal")

    predio_id = pricing.resolve_predio(data, item.get("endereco", ""))
    fonte_serie = "prédio"
    frame = data.predio_series[data.predio_series["predio_id"] == predio_id] if predio_id else pd.DataFrame()
    if frame.empty:
        fonte_serie = "bairro"
        frame = data.bairro_series[data.bairro_series["bairro"] == item["bairro"]]
        frame = frame[frame["segmento"] == item.get("segmento", "Apartamento")]
        frame = frame.groupby("ano", as_index=False).agg(preco_m2=("preco_m2", "median"))

    serie: list[dict] = []
    if not frame.empty:
        anual = frame.sort_values("ano").set_index(pd.to_datetime(frame.sort_values("ano")["ano"], format="%Y"))
        mensal = anual["preco_m2"].resample("MS").interpolate("linear").dropna().tail(meses)
        anterior = None
        for competencia, preco_m2 in mensal.items():
            valor = float(preco_m2) * area
            valorizacao = None if anterior is None else (valor / anterior - 1) * 100
            rendimento = float(aluguel) / valor * 100 if aluguel else None
            serie.append(
                {
                    "ano_mes": competencia.date().isoformat(),
                    "valor_imovel": valor,
                    "aluguel_pct": rendimento,
                    "valorizacao_pct": valorizacao,
                    "total_pct": None
                    if rendimento is None and valorizacao is None
                    else (rendimento or 0.0) + (valorizacao or 0.0),
                }
            )
            anterior = valor

    indices_frame = data.indices_monthly
    if not indices_frame.empty and serie:
        inicio = pd.Timestamp(serie[0]["ano_mes"])
        indices_frame = indices_frame[indices_frame["ano_mes"] >= inicio]

    return {
        "fonte": f"ITBI público · série do {fonte_serie} + BCB/SGS",
        "imovel": item["nome"],
        "aluguel_mensal": float(aluguel) if aluguel else None,
        "serie": serie,
        "indices": _records(indices_frame),
    }


@app.get("/api/composicao", dependencies=[Depends(require_private)])
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
def ranking_vendas(limit: int = 12, ano: int | None = None, mes: str | None = None) -> dict:
    """Bairros por volume de transações registradas, no período pedido."""
    data = _data()
    frame = data.transactions
    if mes:
        try:
            competencia = pd.Timestamp(mes).to_period("M").to_timestamp()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"mês inválido: {mes}") from exc
        frame = frame[frame["ano_mes"] == competencia]
    if ano:
        frame = frame[frame["data"].dt.year == ano]
    contagem = (
        frame.groupby("bairro", as_index=False)
        .agg(transacoes=("preco_m2", "size"), preco_m2=("preco_m2", "median"))
        .sort_values("transacoes", ascending=False)
        .head(limit)
    )
    return {"fonte": "ITBI público", "ano": ano, "mes": mes, "itens": _records(contagem)}


@app.get("/api/agio")
def agio(limit: int = 10, ano: int | None = None, minimo_transacoes: int = 30) -> dict:
    """Ágio mediano sobre o valor venal de referência, por bairro."""
    data = _data()
    frame = data.transactions.dropna(subset=["agio_venal"])
    if ano:
        frame = frame[frame["data"].dt.year == ano]
    agregado = (
        frame.groupby("bairro", as_index=False)
        .agg(agio_pct=("agio_venal", "median"), transacoes=("agio_venal", "size"))
        .query("transacoes >= @minimo_transacoes")
        .sort_values("agio_pct", ascending=False)
        .head(limit)
    )
    agregado["agio_pct"] = agregado["agio_pct"] * 100
    return {"fonte": "ITBI público", "ano": ano, "itens": _records(agregado)}


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


@app.get("/favicon.svg")
@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
