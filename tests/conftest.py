import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Painel sintético no formato de `fipezap.wide_prices`."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    rows = []
    for city, base in [("Cidade A", 8_000.0), ("Cidade B", 5_000.0)]:
        for typology, factor in [("Total", 1.0), ("2D", 0.95), ("3D", 1.1)]:
            trend = np.linspace(0, 0.6, len(dates))
            noise = np.sin(np.arange(len(dates)) / 6) * 0.01
            sale = base * factor * (1 + trend + noise)
            rows.append(
                pd.DataFrame(
                    {
                        "city": city,
                        "typology": typology,
                        "date": dates,
                        "venda_preco_m2": sale,
                        "locacao_preco_m2": sale * 0.004,
                        "venda_indice": sale / sale[0] * 100,
                        "rentabilidade_yield_mensal": 0.004,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def synthetic_macro() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "selic_meta_aa": np.linspace(14, 10, len(dates)),
            "ipca_mensal": np.full(len(dates), 0.4),
            "igpm_mensal": np.full(len(dates), 0.5),
            "incc_mensal": np.full(len(dates), 0.3),
            "credito_imobiliario_taxa_aa": np.linspace(9, 11, len(dates)),
        }
    )


@pytest.fixture
def synthetic_socio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Cidade A", "Cidade B"],
            "uf": ["SP", "MG"],
            "population": [12_000_000, 2_500_000],
            "gdp_per_capita": [65_000, 40_000],
        }
    )
