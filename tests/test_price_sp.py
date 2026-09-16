import numpy as np
import pandas as pd
import pytest

from gjurema.models import price_sp

FAST_PARAMS = {"n_estimators": 60, "max_depth": 4}


@pytest.fixture(scope="module")
def modelo(sp_transactions) -> price_sp.HedonicPriceModel:
    return price_sp.train(sp_transactions, params=FAST_PARAMS)


def test_build_design_usa_apenas_historico_anterior_a_venda(sp_transactions):
    design = price_sp.build_design(sp_transactions)
    primeiro = design.groupby("predio_id", observed=True).head(1)
    assert primeiro["preco_m2_predio_anterior"].isna().all()

    predio = design[design["predio_id"] == design.iloc[0]["predio_id"]].sort_values("data")
    esperado = predio["preco_m2"].iloc[:2].mean()
    assert predio["preco_m2_predio_anterior"].iloc[2] == pytest.approx(esperado)
    assert design["month_index"].min() == 0


def test_design_row_alinha_features_com_o_treino():
    row = price_sp.design_row(
        segmento="Apartamento",
        bairro="PINHEIROS",
        area_m2=70,
        padrao=4,
        data=pd.Timestamp("2024-06-15"),
        origin=pd.Timestamp("2022-01-10"),
        preco_m2_predio=None,
        preco_m2_bairro=15_000.0,
    )
    assert list(row.columns) == price_sp.CATEGORICAL_FEATURES + price_sp.NUMERIC_FEATURES
    assert row.iloc[0]["month_index"] == 29
    assert row.iloc[0]["log_area"] == pytest.approx(np.log(70))
    assert np.isnan(row.iloc[0]["preco_m2_predio_anterior"])


def test_train_valida_no_periodo_mais_recente(sp_transactions, modelo):
    assert modelo.metrics["valid_start"] > str(sp_transactions["data"].min().date())
    assert modelo.metrics["n_train"] > modelo.metrics["n_valid"] > 0
    assert modelo.metrics["mape_pct"] < 25
    assert modelo.metrics["r2_log"] > 0.5


def test_predict_interval_ordenado_e_no_nivel_do_mercado(sp_transactions, modelo):
    row = price_sp.design_row(
        segmento="Apartamento",
        bairro="PINHEIROS",
        area_m2=70,
        padrao=4,
        data=sp_transactions["data"].max(),
        origin=pd.Timestamp(modelo.origin),
        preco_m2_bairro=15_000.0,
    )
    faixa = modelo.predict_interval(row).iloc[0]
    assert faixa["preco_justo_m2_p10"] < faixa["preco_justo_m2"] < faixa["preco_justo_m2_p90"]
    assert 8_000 < faixa["preco_justo_m2"] < 25_000


def test_predict_aceita_bairro_desconhecido(modelo, sp_transactions):
    row = price_sp.design_row(
        segmento="Apartamento",
        bairro="BAIRRO INEXISTENTE",
        area_m2=70,
        padrao=3,
        data=sp_transactions["data"].max(),
        origin=pd.Timestamp(modelo.origin),
    )
    assert float(modelo.predict(row)[0]) > 0


def test_save_load_preserva_previsao(tmp_path, modelo, sp_transactions):
    modelo.save(tmp_path)
    recarregado = price_sp.HedonicPriceModel.load(tmp_path)
    amostra = price_sp.build_design(sp_transactions).tail(20)

    assert recarregado.categories == modelo.categories
    assert recarregado.origin == modelo.origin
    np.testing.assert_allclose(recarregado.predict(amostra), modelo.predict(amostra), rtol=1e-5)


def test_feature_importances_cobre_todas_as_features(modelo):
    importancias = modelo.feature_importances()
    assert set(importancias.index) == set(modelo.features)
    assert importancias.is_monotonic_decreasing
