"""Camada 4 — dashboard GJurema (Streamlit).

Execute com:
    streamlit run src/gjurema/dashboard/app.py
"""

from __future__ import annotations

import json
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from gjurema.config import ARTIFACTS_DIR, PROCESSED_DIR
from gjurema.models import price as price_model
from gjurema.models import returns as returns_model

st.set_page_config(page_title="GJurema — Inteligência Imobiliária", layout="wide", page_icon="🏙️")

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
MARKET_INDEX_PATH = PROCESSED_DIR / "indice_mercado.parquet"
FORECAST_PATH = PROCESSED_DIR / "valorizacao_projetada.parquet"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_table(path_name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / path_name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model() -> price_model.FairPriceModel:
    return price_model.FairPriceModel.load(ARTIFACTS_DIR)


def brl(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", ".")


def md(text: str) -> str:
    """Escapa cifrões para o markdown do Streamlit não interpretar como LaTeX."""
    return text.replace("$", r"\$")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


if not FEATURES_PATH.exists():
    st.error("Base não encontrada. Rode `python -m gjurema.pipeline all` antes de abrir o dashboard.")
    st.stop()

features = load_features()
market_index = load_table("indice_mercado.parquet")
projections = load_table("valorizacao_projetada.parquet")
snapshot = features.sort_values("date").groupby(["city", "typology"], as_index=False).tail(1)
reference_date = features["date"].max().strftime("%m/%Y")

st.sidebar.title("GJurema")
st.sidebar.caption("Gestão Patrimonial · MVP do modelo de precificação")
PAGES = [
    "Preço justo e oportunidade",
    "Rentabilidade por cidade",
    "Séries históricas",
    "Simulador de ROI",
    "Qualidade do modelo",
]
page = st.sidebar.radio("Navegação", PAGES)
st.sidebar.info(f"Dados FipeZAP até {reference_date} · Macro: BCB/SGS · Socioeconômico: IBGE")

if page == "Preço justo e oportunidade":
    st.title("Preço justo e score de oportunidade")

    left, right = st.columns([1, 2])
    with left:
        city = st.selectbox("Cidade", sorted(features["city"].unique()), index=0)
        typology = st.selectbox("Tipologia", ["Total", "1D", "2D", "3D", "4D"], index=2)
        area = st.number_input("Área útil (m²)", min_value=15.0, max_value=1000.0, value=70.0, step=5.0)
        asking_price = st.number_input("Preço pedido (R$)", min_value=50_000.0, value=850_000.0, step=10_000.0)

    row = snapshot[(snapshot["city"] == city) & (snapshot["typology"] == typology)]
    if row.empty:
        st.warning("Sem dados de FipeZAP para essa combinação de cidade e tipologia.")
        st.stop()

    model = load_model()
    interval = model.predict_interval(row).iloc[0]
    observed_m2 = float(row["venda_preco_m2"].iloc[0])

    city_index = 0.5
    if not market_index.empty and city in set(market_index["city"]):
        observed_index = float(market_index.loc[market_index["city"] == city, "indice_mercado"].iloc[0])
        if not math.isnan(observed_index):
            city_index = observed_index

    score = price_model.opportunity_score(
        asking_price=asking_price,
        area_m2=area,
        fair_price_m2=float(interval["preco_justo_m2"]),
        market_index=city_index,
    )

    with right:
        a, b, c = st.columns(3)
        a.metric("Preço justo (modelo)", brl(score["preco_justo_total"]), f"{score['desvio_pct']:+.1f}% vs. pedido")
        b.metric("Preço justo por m²", brl(interval["preco_justo_m2"]))
        c.metric("Score de oportunidade", f"{score['score_oportunidade']:.0f}/100")

        st.caption(
            md(
                f"Faixa de incerteza: {brl(interval['preco_justo_m2_p10'] * area)} – "
                f"{brl(interval['preco_justo_m2_p90'] * area)} · "
                f"Preço médio FipeZAP observado: {brl(observed_m2)}/m² ({brl(observed_m2 * area)})"
            )
        )

        verdict = (
            "Ativo abaixo do preço justo estimado — oportunidade."
            if score["desconto_pct"] > 5
            else "Ativo acima do preço justo estimado — atenção."
            if score["desconto_pct"] < -5
            else "Ativo precificado dentro da faixa justa."
        )
        if score["desconto_pct"] > 5:
            st.success(verdict)
        else:
            st.info(verdict)

        if not projections.empty:
            projection = projections[(projections["city"] == city) & (projections["typology"] == typology)]
            if not projection.empty:
                st.metric(
                    "Valorização projetada (12 meses)",
                    f"{float(projection['valorizacao_projetada_pct'].iloc[0]):+.1f}%",
                )

    st.subheader("Contribuição das variáveis no modelo")
    importances = model.feature_importances().head(8).reset_index()
    importances.columns = ["variável", "importância"]
    st.plotly_chart(
        px.bar(importances, x="importância", y="variável", orientation="h").update_yaxes(autorange="reversed"),
        width="stretch",
    )

elif page == "Rentabilidade por cidade":
    st.title("Cap rate e yield por cidade")

    typology = st.selectbox("Tipologia", ["Total", "1D", "2D", "3D", "4D"], index=0)
    reference_area = st.slider("Área de referência (m²)", 30, 200, 70, step=5)

    table = snapshot[snapshot["typology"] == typology].dropna(subset=["venda_preco_m2", "locacao_preco_m2"]).copy()
    table["preco_imovel"] = table["venda_preco_m2"] * reference_area
    table["aluguel_mensal"] = table["locacao_preco_m2"] * reference_area
    table["yield_bruto"] = table.apply(
        lambda r: returns_model.gross_yield(r["aluguel_mensal"], r["preco_imovel"]), axis=1
    )
    table["yield_liquido"] = table.apply(
        lambda r: returns_model.net_yield(r["aluguel_mensal"], r["preco_imovel"]), axis=1
    )
    table["cap_rate"] = table["yield_liquido"]
    table["payback_anos"] = table.apply(
        lambda r: returns_model.payback_years(r["preco_imovel"], r["aluguel_mensal"]), axis=1
    )

    ranked = table.sort_values("yield_liquido", ascending=False)
    st.plotly_chart(
        px.bar(
            ranked.head(20),
            x="city",
            y=["yield_bruto", "yield_liquido"],
            barmode="group",
            labels={"value": "yield anual", "city": "cidade", "variable": "indicador"},
        ),
        width="stretch",
    )

    columns = [
        "city",
        "venda_preco_m2",
        "locacao_preco_m2",
        "preco_imovel",
        "aluguel_mensal",
        "cap_rate",
        "yield_bruto",
        "yield_liquido",
        "payback_anos",
        "venda_var_12m",
    ]
    display = ranked[columns].rename(
        columns={
            "city": "cidade",
            "venda_preco_m2": "venda R$/m²",
            "locacao_preco_m2": "aluguel R$/m²",
            "preco_imovel": "preço imóvel",
            "aluguel_mensal": "aluguel mensal",
            "yield_bruto": "yield bruto",
            "yield_liquido": "yield líquido",
            "payback_anos": "payback (anos)",
            "venda_var_12m": "valorização 12m",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "venda R$/m²": "{:,.0f}",
                "aluguel R$/m²": "{:,.1f}",
                "preço imóvel": "{:,.0f}",
                "aluguel mensal": "{:,.0f}",
                "cap_rate": "{:.2%}",
                "yield bruto": "{:.2%}",
                "yield líquido": "{:.2%}",
                "payback (anos)": "{:.0f}",
                "valorização 12m": "{:.2%}",
            }
        ),
        width="stretch",
        height=520,
    )

elif page == "Séries históricas":
    st.title("Séries históricas FipeZAP")

    cities = st.multiselect(
        "Cidades",
        sorted(features["city"].unique()),
        default=["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Florianópolis"],
    )
    typology = st.selectbox("Tipologia", ["Total", "1D", "2D", "3D", "4D"], index=0)
    metric = st.radio("Métrica", ["venda_preco_m2", "locacao_preco_m2", "yield_bruto_anual"], horizontal=True)

    series = features[(features["city"].isin(cities)) & (features["typology"] == typology)]
    if series.empty:
        st.warning("Selecione ao menos uma cidade com dados.")
    else:
        st.plotly_chart(
            px.line(series, x="date", y=metric, color="city", labels={"date": "mês", metric: metric}),
            width="stretch",
        )

    if not projections.empty:
        st.subheader("Valorização projetada para 12 meses")
        top = projections[projections["typology"] == typology].head(15)
        st.plotly_chart(
            px.bar(top, x="city", y="valorizacao_projetada_pct", labels={"valorizacao_projetada_pct": "% projetado"}),
            width="stretch",
        )

elif page == "Simulador de ROI":
    st.title("Simulador de retorno do investimento")

    left, right = st.columns(2)
    with left:
        city = st.selectbox("Cidade", sorted(features["city"].unique()))
        typology = st.selectbox("Tipologia", ["Total", "1D", "2D", "3D", "4D"], index=2)
        area = st.number_input("Área útil (m²)", min_value=15.0, value=70.0, step=5.0)
        row = snapshot[(snapshot["city"] == city) & (snapshot["typology"] == typology)]
        default_price = float(row["venda_preco_m2"].iloc[0] * area) if not row.empty else 700_000.0

        observed_rent_m2 = float(row["locacao_preco_m2"].iloc[0]) if not row.empty else float("nan")
        rent_is_estimated = math.isnan(observed_rent_m2)
        if rent_is_estimated:
            benchmark_yield = float(snapshot["yield_bruto_anual"].median())
            default_rent = default_price * benchmark_yield / 12
        else:
            default_rent = observed_rent_m2 * area

        purchase_price = st.number_input("Preço de compra (R$)", min_value=50_000.0, value=round(default_price, -3))
        monthly_rent = st.number_input("Aluguel mensal estimado (R$)", min_value=300.0, value=round(default_rent, -1))
        if rent_is_estimated:
            st.warning(
                f"O FipeZAP não publica locação para {city} ({typology}). "
                "O aluguel acima é uma estimativa pelo yield mediano das cidades com dados — ajuste manualmente."
            )
    with right:
        down_payment_rate = st.slider("Entrada (% do valor)", 20, 100, 100, step=5) / 100
        financing_rate = st.slider("Juros do financiamento (% a.a.)", 8.0, 16.0, 11.0, step=0.5) / 100
        appreciation = st.slider("Valorização esperada (% a.a.)", 0.0, 15.0, 5.0, step=0.5) / 100
        vacancy = st.slider("Vacância (% do ano)", 0.0, 30.0, 8.0, step=1.0) / 100
        selic = st.slider("SELIC de referência (% a.a.)", 5.0, 16.0, 10.5, step=0.25) / 100

    assumptions = returns_model.Assumptions(
        vacancy_rate=vacancy,
        appreciation_rate=appreciation,
        selic_rate=selic,
    )
    simulation = returns_model.simulate(
        purchase_price,
        monthly_rent,
        assumptions=assumptions,
        down_payment_rate=down_payment_rate,
        financing_rate=financing_rate,
    )

    a, b, c, d = st.columns(4)
    a.metric("Cap rate", pct(simulation["cap_rate"]))
    b.metric("Yield bruto", pct(simulation["yield_bruto"]))
    c.metric("Yield líquido", pct(simulation["yield_liquido"]))
    payback = simulation["payback_anos"]
    d.metric("Payback", "—" if payback == float("inf") else f"{payback:.0f} anos")

    horizons = pd.DataFrame(
        [
            {
                "horizonte": f"{years} anos",
                "TIR": data["tir"],
                "VPL @ SELIC": data["vpl_selic"],
                "capital investido": data["capital_investido"],
                "retorno total": data["retorno_total"],
                "renda fixa equivalente": data["renda_fixa_equivalente"],
            }
            for years, data in simulation["horizontes"].items()
        ]
    )
    st.subheader("Retorno por horizonte")
    st.dataframe(
        horizons.style.format(
            {
                "TIR": "{:.2%}",
                "VPL @ SELIC": "R$ {:,.0f}",
                "capital investido": "R$ {:,.0f}",
                "retorno total": "R$ {:,.0f}",
                "renda fixa equivalente": "R$ {:,.0f}",
            }
        ),
        width="stretch",
    )

    st.subheader("Cenários em 10 anos")
    scenario_table = pd.DataFrame(
        returns_model.scenarios(
            purchase_price,
            monthly_rent,
            base=assumptions,
            down_payment_rate=down_payment_rate,
            financing_rate=financing_rate,
        )
    ).T
    scenario_table.index.name = "cenário"
    st.dataframe(
        scenario_table.reset_index().style.format(
            {"tir": "{:.2%}", "vpl_selic": "R$ {:,.0f}", "valorizacao_aa": "{:.1%}", "vacancia": "{:.1%}"}
        ),
        width="stretch",
    )

else:
    st.title("Qualidade do modelo")
    if not METRICS_PATH.exists():
        st.warning("Rode `python -m gjurema.pipeline train` para gerar as métricas.")
        st.stop()

    metrics = json.loads(METRICS_PATH.read_text())
    fair = metrics["fair_price"]
    a, b, c = st.columns(3)
    a.metric("R² (log preço) — validação", f"{fair['r2_log']:.3f}")
    b.metric("Erro percentual médio", f"{fair['mape_pct']:.2f}%")
    c.metric("Backtest da projeção 12m", f"{metrics['forecast_backtest']['mape_pct']:.2f}%")
    st.caption(
        f"Validação temporal a partir de {fair['valid_start']} · "
        f"{fair['n_train']} observações de treino e {fair['n_valid']} de validação."
    )

    importances = pd.Series(metrics["top_features"]).sort_values(ascending=True)
    st.plotly_chart(
        px.bar(importances, orientation="h", labels={"value": "importância", "index": "variável"}),
        width="stretch",
    )
    st.json(metrics)
