"""Configurações centrais do MVP GJurema."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("GJUREMA_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FIPEZAP_URL = "https://downloads.fipe.org.br/indices/fipezap/fipezap-serieshistoricas.xlsx"
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
IBGE_AGREGADOS_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
IBGE_LOCALIDADES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades"
# Página da Secretaria Municipal da Fazenda com as guias de ITBI pagas por ano.
ITBI_SP_INDEX_URL = "https://prefeitura.sp.gov.br/web/fazenda/w/acesso_a_informacao/31501"

# Carteira do cliente (contratos): dado privado, fora do ITBI.
CARTEIRA_PATH = Path(os.environ.get("GJUREMA_CARTEIRA", DATA_DIR / "carteira.json"))

HTTP_TIMEOUT = 120

# Séries do Sistema Gerenciador de Séries Temporais do Banco Central.
SGS_SERIES = {
    "selic_meta_aa": 432,
    "selic_mensal": 4390,
    "ipca_mensal": 433,
    "igpm_mensal": 189,
    "incc_mensal": 192,
    "credito_imobiliario_taxa_aa": 25497,
}

# Ano de referência dos agregados municipais do IBGE (população e PIB).
IBGE_POPULATION_AGGREGATE = 6579
IBGE_POPULATION_VARIABLE = 9324
IBGE_GDP_AGGREGATE = 5938
IBGE_GDP_VARIABLE = 37

TYPOLOGIES = ["Total", "1D", "2D", "3D", "4D"]

# Custos operacionais padrão usados pelo motor de rentabilidade quando o
# usuário não informa valores próprios.
DEFAULT_OPERATING_ASSUMPTIONS = {
    "vacancy_rate": 0.08,
    "iptu_rate_on_value": 0.008,
    "condo_monthly_rate_on_rent": 0.20,
    "maintenance_rate_on_rent": 0.05,
    "management_rate_on_rent": 0.08,
    "income_tax_rate_on_rent": 0.15,
    "transaction_cost_rate": 0.05,
}


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
