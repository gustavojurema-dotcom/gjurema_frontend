import pandas as pd

from gjurema import features as feature_layer


def test_build_feature_table_adds_momentum_and_macro(synthetic_panel, synthetic_macro, synthetic_socio):
    table = feature_layer.build_feature_table(synthetic_panel, synthetic_macro, synthetic_socio)

    assert {"venda_var_12m", "venda_preco_m2_lag12", "yield_bruto_anual", "log_population"} <= set(table.columns)
    assert table["selic_meta_aa"].notna().all()
    assert table.groupby(["city", "typology"]).size().nunique() == 1


def test_momentum_is_computed_within_each_series(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    first_rows = table.groupby(["city", "typology"]).head(1)
    assert first_rows["venda_preco_m2_lag1"].isna().all()


def test_latest_snapshot_has_one_row_per_series(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    snapshot = feature_layer.latest_snapshot(table)
    assert len(snapshot) == table.groupby(["city", "typology"]).ngroups
    assert snapshot["date"].nunique() == 1


def test_city_market_index_is_bounded(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    index = feature_layer.city_market_index(table)
    assert index["indice_mercado"].between(0, 1).all()
    assert index["indice_mercado"].is_monotonic_decreasing


def test_city_market_index_survives_missing_rent(synthetic_panel):
    panel = synthetic_panel.copy()
    panel.loc[panel["city"] == "Cidade B", "locacao_preco_m2"] = None
    index = feature_layer.city_market_index(feature_layer.build_feature_table(panel))

    assert index["indice_mercado"].notna().all()
    assert index.loc[index["city"] == "Cidade B", "tem_dados_locacao"].eq(False).all()


def test_yield_matches_rent_over_price(synthetic_panel):
    table = feature_layer.build_feature_table(synthetic_panel)
    row = table.iloc[0]
    expected = row["locacao_preco_m2"] * 12 / row["venda_preco_m2"]
    assert row["yield_bruto_anual"] == pd.Series([expected]).iloc[0]
