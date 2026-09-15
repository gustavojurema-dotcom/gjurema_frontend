"""Pipeline de São Paulo: ITBI → agregados → modelo hedônico.

Uso:
    python -m gjurema.pipeline_sp build --anos 2019 2020 2021 2022 2023 2024 2025 2026
    python -m gjurema.pipeline_sp train
    python -m gjurema.pipeline_sp all
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from gjurema import artifacts_io, features_sp
from gjurema.config import ARTIFACTS_DIR, PROCESSED_DIR, ensure_dirs
from gjurema.models import price_sp
from gjurema.sources import cep as cep_source
from gjurema.sources import indices as indices_source
from gjurema.sources import itbi_sp

logger = logging.getLogger(__name__)

TRANSACTIONS_PATH = PROCESSED_DIR / "sp_transacoes.parquet"
SERIES_CITY_PATH = PROCESSED_DIR / "sp_serie_cidade.parquet"
SERIES_BAIRRO_PATH = PROCESSED_DIR / "sp_serie_bairro.parquet"
SERIES_PREDIO_PATH = PROCESSED_DIR / "sp_serie_predio.parquet"
LIQUIDITY_PATH = PROCESSED_DIR / "sp_liquidez.parquet"
APPRECIATION_PATH = PROCESSED_DIR / "sp_valorizacao.parquet"
BUILDINGS_PATH = PROCESSED_DIR / "sp_predios.parquet"
INDICES_PATH = PROCESSED_DIR / "sp_indices.parquet"
METRICS_PATH = ARTIFACTS_DIR / "sp_metrics.json"


def build(anos: list[int] | None = None, force_download: bool = False, skip_cep: bool = False) -> pd.DataFrame:
    """Baixa o ITBI, resolve bairros por CEP e materializa os agregados."""
    ensure_dirs()
    transactions = itbi_sp.load_transactions(years=anos, force_download=force_download)

    ceps = transactions["cep"].unique().tolist()
    # `skip_cep` roda offline: aproveita o que já foi resolvido antes em vez de
    # esvaziar a base, porque o bairro é obrigatório nos agregados.
    bairros = cep_source.load_cache() if skip_cep else cep_source.resolve(ceps)
    transactions["bairro"] = transactions["cep"].map(bairros).replace("", pd.NA)
    faltando = transactions["bairro"].isna().mean()
    logger.info("Bairro resolvido para %.1f%% das transações", (1 - faltando) * 100)
    transactions = transactions.dropna(subset=["bairro"])

    artifacts_io.write_parquet(transactions, TRANSACTIONS_PATH)
    artifacts_io.write_parquet(features_sp.annual_series(transactions, ["segmento"]), SERIES_CITY_PATH)
    artifacts_io.write_parquet(
        features_sp.annual_series(transactions, ["bairro", "segmento"]), SERIES_BAIRRO_PATH
    )
    artifacts_io.write_parquet(features_sp.annual_series(transactions, ["predio_id"]), SERIES_PREDIO_PATH)
    artifacts_io.write_parquet(features_sp.liquidity(transactions, ["bairro", "segmento"]), LIQUIDITY_PATH)

    bairro_series = features_sp.annual_series(transactions, ["bairro"])
    artifacts_io.write_parquet(features_sp.appreciation(bairro_series, "bairro"), APPRECIATION_PATH)
    artifacts_io.write_parquet(features_sp.building_directory(transactions), BUILDINGS_PATH)

    periodo = transactions["data"].dt.year
    indices = indices_source.annual_accumulated(int(periodo.min()), int(periodo.max()))
    if not indices.empty:
        artifacts_io.write_parquet(indices, INDICES_PATH)

    logger.info("ITBI-SP: %s transações, %s bairros", len(transactions), transactions["bairro"].nunique())
    return transactions


def train(transactions: pd.DataFrame | None = None) -> price_sp.HedonicPriceModel:
    """Treina e persiste o modelo hedônico de R$/m²."""
    ensure_dirs()
    if transactions is None:
        transactions = pd.read_parquet(TRANSACTIONS_PATH)

    model = price_sp.train(transactions)
    model.save(ARTIFACTS_DIR)

    metrics = {
        "preco_m2": model.metrics,
        "top_features": model.feature_importances().head(10).round(4).to_dict(),
        "cobertura": {
            "transacoes": int(len(transactions)),
            "bairros": int(transactions["bairro"].nunique()),
            "predios": int(transactions["predio_id"].nunique()),
            "inicio": str(transactions["data"].min().date()),
            "fim": str(transactions["data"].max().date()),
        },
    }
    artifacts_io.write_json(metrics, METRICS_PATH)
    logger.info("Modelo SP treinado: %s", metrics["preco_m2"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline ITBI-SP da GJurema")
    parser.add_argument("command", choices=["build", "train", "all"])
    parser.add_argument("--anos", nargs="*", type=int, default=None, help="anos a ingerir")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--skip-cep",
        action="store_true",
        help="não consulta o ViaCEP: usa apenas os CEPs já resolvidos em cache",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    transactions = None
    if args.command in {"build", "all"}:
        transactions = build(anos=args.anos, force_download=args.force_download, skip_cep=args.skip_cep)
    if args.command in {"train", "all"}:
        train(transactions)


if __name__ == "__main__":
    main()
