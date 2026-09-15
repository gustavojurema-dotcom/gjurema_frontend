import pandas as pd

from gjurema.sources import fipezap, ibge


def _fake_sheet() -> pd.DataFrame:
    """Reproduz o layout da planilha FipeZAP com duas datas."""
    width = 56
    rows = [[None] * width for _ in range(6)]
    rows[3][1] = "Data"
    for index, (date, sale, rent) in enumerate(
        [(pd.Timestamp("2024-01-01"), 10_000.0, 40.0), (pd.Timestamp("2024-02-01"), 10_100.0, 41.0)]
    ):
        row = rows[4 + index]
        row[1] = date
        for offset, typology_factor in enumerate([1.0, 0.9, 0.95, 1.05, 1.2]):
            row[2 + offset] = 100 + offset  # número-índice venda
            row[17 + offset] = sale * typology_factor  # preço médio venda
            row[22 + offset] = 90 + offset  # número-índice locação
            row[37 + offset] = rent * typology_factor  # preço médio locação
            row[42 + offset] = 0.004  # yield mensal
    return pd.DataFrame(rows)


def test_parse_city_sheet_produces_tidy_rows():
    tidy = fipezap._parse_city_sheet(_fake_sheet(), "Cidade A")

    assert set(tidy.columns) == {"date", "typology", "value", "segment", "metric", "city"}
    assert set(tidy["segment"]) == {"venda", "locacao", "rentabilidade"}
    sale_total = tidy[(tidy["metric"] == "preco_m2") & (tidy["segment"] == "venda") & (tidy["typology"] == "Total")]
    assert sale_total["value"].tolist() == [10_000.0, 10_100.0]


def test_wide_prices_pivots_segments():
    tidy = fipezap._parse_city_sheet(_fake_sheet(), "Cidade A")
    wide = fipezap.wide_prices(tidy)

    assert {"venda_preco_m2", "locacao_preco_m2"} <= set(wide.columns)
    assert len(wide) == len(tidy["date"].unique()) * len(fipezap.TYPOLOGY_ORDER)


def test_city_uf_covers_fipezap_sheets():
    workbook_cities = set(ibge.CITY_UF)
    assert {"São Paulo", "São José", "Santa Maria", "Campo Grande"} <= workbook_cities


def test_normalize_removes_accents_and_case():
    assert ibge._normalize("São José dos Campos") == "sao jose dos campos"
