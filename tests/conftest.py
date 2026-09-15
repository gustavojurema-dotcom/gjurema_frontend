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


@pytest.fixture(scope="session")
def sp_transactions() -> pd.DataFrame:
    """Guias de ITBI sintéticas já no formato de `itbi_sp.clean`."""
    rng = np.random.default_rng(7)
    meses = pd.date_range("2022-01-01", periods=36, freq="MS")
    bairros = {"PINHEIROS": 14_000.0, "SANTA CECILIA": 9_000.0, "LAPA": 7_500.0}
    segmentos = {"Apartamento": 1.0, "Sala comercial": 0.85}

    linhas = []
    contador = 0
    for mes in meses:
        tendencia = 1 + 0.005 * ((mes.year - 2022) * 12 + mes.month - 1)
        for bairro, nivel in bairros.items():
            for segmento, fator in segmentos.items():
                for predio in range(3):
                    for _ in range(3):
                        contador += 1
                        area = float(rng.uniform(35, 140))
                        preco_m2 = nivel * fator * tendencia * (1 + predio * 0.05) * rng.normal(1, 0.06)
                        valor = preco_m2 * area
                        financiado = bool(rng.random() < 0.45)
                        linhas.append(
                            {
                                "sql": f"{contador:010d}",
                                "logradouro": f"RUA {bairro} {predio}",
                                "numero": 100 + predio,
                                "cep": f"0{predio}{list(bairros).index(bairro)}000000"[:8],
                                "bairro": bairro,
                                "segmento": segmento,
                                "data": mes + pd.Timedelta(days=int(rng.integers(0, 27))),
                                "valor_transacao": valor,
                                "valor_venal_proporcional": valor / 1.2,
                                "valor_financiado": valor * 0.6 if financiado else 0.0,
                                "area_construida": area,
                                "padrao": 3 + predio % 2,
                                "preco_m2": preco_m2,
                                "financiado": financiado,
                                "agio_venal": 0.2,
                                "ano_mes": mes,
                                "predio_id": f"RUA {bairro} {predio}|{100 + predio}|0{predio}000000",
                            }
                        )
    return pd.DataFrame(linhas).sort_values("data").reset_index(drop=True)


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
