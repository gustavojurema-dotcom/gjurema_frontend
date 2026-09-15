import numpy as np
import pytest

from gjurema import features as feature_layer
from gjurema.models import price


@pytest.fixture
def trained_model(synthetic_panel, synthetic_macro, synthetic_socio):
    table = feature_layer.build_feature_table(synthetic_panel, synthetic_macro, synthetic_socio)
    return price.train(table, params={"n_estimators": 80, "max_depth": 3}), table


def test_leaky_features_are_excluded(trained_model):
    model, _ = trained_model
    assert not price.LEAKY_FEATURES & set(model.features)


def test_model_fits_synthetic_series(trained_model):
    model, _ = trained_model
    assert model.metrics["r2_log"] > 0.8
    assert model.metrics["mape_pct"] < 15


def test_predict_interval_brackets_point_estimate(trained_model):
    model, table = trained_model
    sample = feature_layer.latest_snapshot(table)
    interval = model.predict_interval(sample)
    assert (interval["preco_justo_m2_p10"] <= interval["preco_justo_m2"]).all()
    assert (interval["preco_justo_m2"] <= interval["preco_justo_m2_p90"]).all()


def test_model_roundtrip(tmp_path, trained_model):
    model, table = trained_model
    model.save(tmp_path)
    restored = price.FairPriceModel.load(tmp_path)
    sample = feature_layer.latest_snapshot(table)
    np.testing.assert_allclose(model.predict(sample), restored.predict(sample), rtol=1e-6)


def test_opportunity_score_rewards_discount():
    cheap = price.opportunity_score(asking_price=800_000, area_m2=70, fair_price_m2=14_000)
    expensive = price.opportunity_score(asking_price=1_100_000, area_m2=70, fair_price_m2=14_000)
    assert cheap["desconto_pct"] > 0 > expensive["desconto_pct"]
    assert cheap["score_oportunidade"] > expensive["score_oportunidade"]
    assert cheap["preco_justo_total"] == 980_000


def test_opportunity_score_handles_missing_market_index():
    neutral = price.opportunity_score(asking_price=800_000, area_m2=70, fair_price_m2=14_000)
    missing = price.opportunity_score(
        asking_price=800_000, area_m2=70, fair_price_m2=14_000, market_index=float("nan")
    )
    assert missing["score_oportunidade"] == pytest.approx(neutral["score_oportunidade"])


def test_opportunity_score_rejects_invalid_area():
    with pytest.raises(ValueError):
        price.opportunity_score(asking_price=500_000, area_m2=0, fair_price_m2=10_000)
