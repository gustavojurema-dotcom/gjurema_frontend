import pandas as pd
import pytest

from gjurema import features_sp


def test_monthly_series_agrega_preco_liquidez_e_agio(sp_transactions):
    series = features_sp.monthly_series(sp_transactions, ["segmento"])
    apartamento = series[series["segmento"] == "Apartamento"]

    assert set(apartamento["ano_mes"]) == set(sp_transactions["ano_mes"])
    assert apartamento["transacoes"].sum() == (sp_transactions["segmento"] == "Apartamento").sum()
    assert apartamento["financiado_pct"].between(0, 100).all()
    assert apartamento["preco_m2"].gt(0).all()
    assert apartamento["agio_venal_mediano"].notna().all()


def test_annual_series_descarta_recortes_com_poucas_observacoes(sp_transactions):
    series = features_sp.annual_series(sp_transactions, ["bairro", "segmento"])
    assert series["transacoes"].min() >= features_sp.MIN_TRANSACTIONS
    assert set(series["ano"]) <= set(sp_transactions["data"].dt.year)
    assert series["ano"].dtype.kind == "i"


def test_liquidity_deriva_giro_a_partir_dos_meses_cobertos(sp_transactions):
    liquidez = features_sp.liquidity(sp_transactions, ["bairro", "segmento"])
    meses = sp_transactions["ano_mes"].nunique()
    linha = liquidez.iloc[0]

    assert linha["transacoes_ano"] == pytest.approx(linha["transacoes"] / (meses / 12))
    assert linha["giro_meses"] == pytest.approx(meses / linha["transacoes"])
    assert liquidez["transacoes"].is_monotonic_decreasing


def test_liquidity_sem_transacoes():
    vazio = pd.DataFrame(columns=["ano_mes", "bairro", "segmento", "preco_m2"])
    assert features_sp.liquidity(vazio, ["bairro"]).empty


def test_appreciation_compara_primeiro_e_ultimo_ano():
    series = pd.DataFrame(
        {
            "bairro": ["PINHEIROS"] * 3 + ["SANTA CECILIA"] * 2 + ["LAPA"],
            "ano": [2022, 2023, 2024, 2023, 2024, 2024],
            "preco_m2": [10_000, 11_000, 12_500, 8_000, 7_200, 9_000],
            "transacoes": [50, 60, 70, 30, 30, 10],
        }
    )
    ranking = features_sp.appreciation(series, "bairro", years=5)

    assert ranking["bairro"].tolist() == ["PINHEIROS", "SANTA CECILIA"]  # LAPA só tem um ano
    assert ranking.iloc[0]["valorizacao_pct"] == pytest.approx(25.0)
    assert ranking.iloc[1]["valorizacao_pct"] == pytest.approx(-10.0)
    assert ranking.iloc[0]["ano_inicial"] == 2022 and ranking.iloc[0]["ano_final"] == 2024


def test_appreciation_respeita_a_janela_de_anos():
    series = pd.DataFrame(
        {
            "bairro": ["PINHEIROS"] * 3,
            "ano": [2018, 2023, 2024],
            "preco_m2": [5_000, 11_000, 12_500],
            "transacoes": [40, 60, 70],
        }
    )
    ranking = features_sp.appreciation(series, "bairro", years=1)
    assert ranking.iloc[0]["ano_inicial"] == 2023


def test_building_directory_agrupa_unidades_do_mesmo_endereco(sp_transactions):
    predios = features_sp.building_directory(sp_transactions)
    assert len(predios) == sp_transactions["predio_id"].nunique()
    assert predios["transacoes"].sum() == len(sp_transactions)
    assert predios.iloc[0]["endereco"].count(",") == 1
    assert predios["ultima_transacao"].max() == sp_transactions["data"].max()


def test_market_index_combina_valorizacao_e_liquidez():
    valorizacao = pd.DataFrame(
        {"bairro": ["A", "B"], "valorizacao_pct": [30.0, 5.0], "preco_m2": [10_000, 8_000]}
    )
    liquidez = pd.DataFrame(
        {"bairro": ["A", "B"], "transacoes_ano": [10.0, 90.0], "giro_meses": [1.2, 0.1]}
    )
    indice = features_sp.market_index(liquidez, valorizacao, "bairro")
    assert indice["indice_mercado"].between(0, 1).all()
    assert indice.iloc[0]["bairro"] == "A"
