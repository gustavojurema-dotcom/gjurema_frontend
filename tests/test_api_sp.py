import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gjurema import features_sp
from gjurema.api import app as api_app
from gjurema.api import artifacts, pricing
from gjurema.models import price_sp

CARTEIRA = [
    {
        "id": "u702",
        "nome": "Verve Pinheiros",
        "unidade": "Unidade 702",
        "bairro": "PINHEIROS",
        "endereco": "Rua Pinheiros 0, 100",
        "segmento": "Apartamento",
        "tipo": "Apartamento",
        "quartos": "2 dorm.",
        "construtora": "MPD",
        "priv": 71.0,
        "total": 134.0,
        "padrao": 4,
        "preco": 1_071_500,
        "dataCompra": "2022-06-01",
        "corretagem": 53_253.55,
        "matricula": "157.690",
        "fracao": 0.005,
        "fluxo": [["Ato", 100_000]],
    }
]


@pytest.fixture(scope="module")
def market(sp_transactions) -> artifacts.MarketData:
    bairro_series = features_sp.annual_series(sp_transactions, ["bairro", "segmento"])
    model = price_sp.train(sp_transactions, params={"n_estimators": 60, "max_depth": 4})
    return artifacts.MarketData(
        transactions=sp_transactions,
        city_series=features_sp.annual_series(sp_transactions, ["segmento"]),
        bairro_series=bairro_series,
        predio_series=features_sp.annual_series(sp_transactions, ["predio_id"]),
        liquidity=features_sp.liquidity(sp_transactions, ["bairro", "segmento"]),
        appreciation=features_sp.appreciation(
            features_sp.annual_series(sp_transactions, ["bairro"]), "bairro"
        ),
        buildings=features_sp.building_directory(sp_transactions),
        indices=pd.DataFrame(
            {
                "indice": ["cdi"] * 3 + ["ipca"] * 3,
                "nome": ["CDI"] * 3 + ["IPCA"] * 3,
                "ano": [2022, 2023, 2024] * 2,
                "acumulado_pct": [0.0, 13.0, 26.5, 0.0, 4.6, 9.2],
            }
        ),
        metrics={"cobertura": {"transacoes": len(sp_transactions), "inicio": "2022-01-01"}},
        model=model,
    )


@pytest.fixture
def client(market, tmp_path, monkeypatch) -> TestClient:
    carteira_path = tmp_path / "carteira.json"
    carteira_path.write_text(json.dumps(CARTEIRA))
    monkeypatch.setattr(api_app, "CARTEIRA_PATH", carteira_path)
    monkeypatch.setattr(artifacts, "available", lambda: True)
    monkeypatch.setattr(artifacts, "market_data", lambda: market)
    return TestClient(api_app.app)


def test_meta_expoe_cobertura_bairros_e_segmentos(client):
    corpo = client.get("/api/meta").json()
    assert corpo["fonte"].startswith("ITBI público")
    assert "PINHEIROS" in corpo["bairros"]
    assert corpo["segmentos"] == ["Apartamento", "Sala comercial"]
    assert corpo["cobertura"]["transacoes"] > 0


def test_carteira_precifica_cada_imovel_do_cliente(client):
    item = client.get("/api/carteira").json()["itens"][0]
    estimativa = item["precificacao"]

    assert item["nome"] == "Verve Pinheiros"
    assert estimativa["predio_id"] is not None  # endereço casado com o ITBI
    assert estimativa["valor_justo_p10"] < estimativa["valor_justo"] < estimativa["valor_justo_p90"]
    assert estimativa["preco_m2_pago_privativo"] == pytest.approx(1_071_500 / 71.0)
    assert estimativa["delta_pct"] == pytest.approx(
        (estimativa["valor_justo"] / 1_071_500 - 1) * 100
    )


def test_precificacao_de_imovel_fora_da_carteira(client):
    corpo = client.get(
        "/api/precificacao", params={"bairro": "PINHEIROS", "area_total": 70, "preco": 1_000_000}
    ).json()
    assert corpo["valor_justo"] == pytest.approx(corpo["preco_justo_m2"] * 70)
    assert corpo["transacoes_bairro"] > 0
    assert corpo["delta_pct"] == pytest.approx((corpo["valor_justo"] / 1_000_000 - 1) * 100)


def test_precificacao_rejeita_bairro_sem_transacoes(client):
    resposta = client.get("/api/precificacao", params={"bairro": "MOEMA", "area_total": 70})
    assert resposta.status_code == 404


def test_precificacao_exige_area_positiva(client):
    assert client.get("/api/precificacao", params={"bairro": "PINHEIROS", "area_total": 0}).status_code == 422


def test_serie_devolve_predio_bairro_e_cidade(client):
    corpo = client.get("/api/serie", params={"carteira_id": "u702"}).json()
    assert corpo["bairro"] == "PINHEIROS"
    assert corpo["predio_id"] is not None
    assert len(corpo["predio_serie"]) >= 2
    assert {"ano", "preco_m2", "transacoes"} <= set(corpo["cidade"][0])
    assert len(corpo["cidade"]) == len(corpo["bairro_serie"])


def test_serie_sem_filtro_de_carteira(client):
    corpo = client.get("/api/serie", params={"segmento": "Apartamento"}).json()
    assert corpo["predio_serie"] == [] and corpo["bairro_serie"] == []
    assert corpo["cidade"]


def test_rentabilidade_compara_imovel_e_indices_desde_a_compra(client):
    corpo = client.get("/api/rentabilidade", params={"carteira_id": "u702"}).json()
    assert corpo["ano_base"] == 2022
    assert corpo["serie"][0]["acumulado_pct"] == pytest.approx(0.0)
    assert min(p["ano"] for p in corpo["indices"]) == 2022
    cdi = [p for p in corpo["indices"] if p["indice"] == "cdi"]
    assert cdi[0]["acumulado_pct"] == pytest.approx(0.0)
    assert cdi[-1]["acumulado_pct"] == pytest.approx(26.5)


def test_rentabilidade_rebaseia_indices_no_ano_da_compra(client, market, monkeypatch):
    corpo = client.get("/api/indices", params={"desde": 2023}).json()["series"]
    cdi = {p["ano"]: p["acumulado_pct"] for p in corpo if p["indice"] == "cdi"}
    assert cdi[2023] == pytest.approx(0.0)
    assert cdi[2024] == pytest.approx((1.265 / 1.13 - 1) * 100)


def test_imovel_fora_da_carteira_retorna_404(client):
    assert client.get("/api/rentabilidade", params={"carteira_id": "zzz"}).status_code == 404


def test_composicao_agrega_a_carteira_do_cliente(client):
    corpo = client.get("/api/composicao", params={"dim": "bairro"}).json()
    assert corpo["fonte"] == "Dados do cliente"
    assert corpo["itens"] == [{"rotulo": "PINHEIROS", "ativos": 1, "valor": 1_071_500.0}]
    assert client.get("/api/composicao", params={"dim": "corretor"}).status_code == 400


def test_rankings_e_liquidez(client):
    valorizacao = client.get("/api/ranking/valorizacao", params={"minimo_transacoes": 10}).json()
    assert valorizacao["itens"] and "valorizacao_pct" in valorizacao["itens"][0]

    vendas = client.get("/api/ranking/vendas", params={"limit": 2}).json()["itens"]
    assert len(vendas) == 2
    assert vendas[0]["transacoes"] >= vendas[1]["transacoes"]

    liquidez = client.get("/api/liquidez", params={"segmento": "Apartamento", "limit": 5}).json()
    assert all(item["segmento"] == "Apartamento" for item in liquidez["itens"])
    assert liquidez["itens"][0]["giro_meses"] > 0


def test_ranking_de_vendas_filtra_o_mes(client, market):
    ultimo = market.transactions["ano_mes"].max()
    corpo = client.get("/api/ranking/vendas", params={"mes": ultimo.date().isoformat()}).json()
    do_mes = int((market.transactions["ano_mes"] == ultimo).sum())
    assert sum(item["transacoes"] for item in corpo["itens"]) == do_mes
    assert do_mes < len(market.transactions)
    assert client.get("/api/ranking/vendas", params={"mes": "mes-que-vem"}).status_code == 400


def test_agio_devolve_mediana_percentual_por_bairro(client, market):
    corpo = client.get("/api/agio", params={"minimo_transacoes": 5}).json()
    itens = corpo["itens"]
    assert itens and itens == sorted(itens, key=lambda i: -i["agio_pct"])
    esperado = market.transactions[market.transactions["bairro"] == itens[0]["bairro"]]
    assert itens[0]["agio_pct"] == pytest.approx(esperado["agio_venal"].median() * 100)


def test_serie_sem_segmento_usa_mediana_das_transacoes(client, market):
    corpo = client.get("/api/serie", params={"bairro": "PINHEIROS"}).json()
    transacoes = market.transactions[market.transactions["bairro"] == "PINHEIROS"]
    ano = corpo["bairro_serie"][0]["ano"]
    esperado = transacoes[transacoes["data"].dt.year == ano]["preco_m2"].median()
    assert corpo["bairro_serie"][0]["preco_m2"] == pytest.approx(esperado)


def test_mercado_resume_o_ultimo_mes_fechado(client, market):
    corpo = client.get("/api/mercado").json()
    ultimo = market.transactions["ano_mes"].max()
    esperado = int((market.transactions["ano_mes"] == ultimo).sum())

    assert corpo["mes"] == ultimo.date().isoformat()
    assert corpo["transacoes_mes"] == esperado
    assert 0 <= corpo["financiado_pct"] <= 100
    assert len(corpo["serie_mensal"]) == 12
    assert {p["segmento"] for p in corpo["por_segmento"]} == {"Apartamento", "Sala comercial"}


def test_carteira_exige_token_quando_configurado(client, monkeypatch):
    monkeypatch.setenv(api_app.TOKEN_ENV, "segredo")
    assert client.get("/api/carteira").status_code == 401
    assert client.get("/api/rentabilidade", params={"carteira_id": "u702"}).status_code == 401
    liberado = client.get("/api/carteira", headers={api_app.TOKEN_HEADER: "segredo"})
    assert liberado.status_code == 200
    # a base pública continua aberta
    assert client.get("/api/meta").status_code == 200


def test_rate_limit_responde_429(client, monkeypatch):
    monkeypatch.setattr(api_app, "RATE_LIMIT", 3)
    api_app._hits.clear()
    codigos = [client.get("/api/meta").status_code for _ in range(4)]
    assert codigos == [200, 200, 200, 429]
    api_app._hits.clear()


def test_index_serve_o_painel_html(client):
    resposta = client.get("/")
    assert resposta.status_code == 200
    assert "GJUREMA" in resposta.text
    # o painel consome a API em vez de carregar dados embutidos
    assert 'api("/meta")' in resposta.text
    assert "const CARTEIRA=[{" not in resposta.text.replace(" ", "")


def test_build_incompleto_responde_503(tmp_path, monkeypatch):
    # geração parcial: só as transações foram publicadas
    transacoes = tmp_path / "sp_transacoes.parquet"
    transacoes.write_bytes(b"")
    monkeypatch.setattr(artifacts, "REQUIRED_PATHS", (transacoes, tmp_path / "sp_metrics.json"))
    assert artifacts.available() is False


def test_endpoints_respondem_503_sem_artefatos(monkeypatch):
    monkeypatch.setattr(artifacts, "available", lambda: False)
    cliente = TestClient(api_app.app)
    for rota in ("/api/meta", "/api/mercado", "/api/liquidez"):
        resposta = cliente.get(rota)
        assert resposta.status_code == 503
        assert "pipeline_sp" in resposta.json()["detail"]


def test_resolve_predio_ignora_tipo_de_logradouro(market):
    assert pricing.resolve_predio(market, "Avenida Pinheiros 0, 100") is not None
    assert pricing.resolve_predio(market, "Rua Inexistente, 999") is None
    assert pricing.resolve_predio(market, "sem número") is None
