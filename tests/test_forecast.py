from gjurema import features as feature_layer
from gjurema.models import forecast


def test_projection_covers_all_series(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    projections = forecast.project_prices(table, horizon_months=12)
    assert len(projections) == table.groupby(["city", "typology"]).ngroups
    assert projections["valorizacao_projetada_pct"].notna().all()


def test_projection_follows_upward_trend(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    projections = forecast.project_prices(table, horizon_months=12)
    assert (projections["preco_m2_projetado"] > projections["preco_m2_atual"]).all()


def test_backtest_reports_error(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    result = forecast.backtest(table, horizon_months=12)
    assert result["n"] > 0
    assert result["mape_pct"] < 20
