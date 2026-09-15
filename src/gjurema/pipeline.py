"""Orquestração do pipeline: ingestão → features → treino → artefatos.

Uso:
    python -m gjurema.pipeline build     # baixa e consolida as fontes
    python -m gjurema.pipeline train     # treina o modelo de preço justo
    python -m gjurema.pipeline all       # build + train
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from gjurema import features as feature_layer
from gjurema.config import ARTIFACTS_DIR, PROCESSED_DIR, ensure_dirs
from gjurema.models import forecast, price
from gjurema.sources import bcb, fipezap, ibge

logger = logging.getLogger(__name__)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PANEL_PATH = PROCESSED_DIR / "fipezap_panel.parquet"
MARKET_INDEX_PATH = PROCESSED_DIR / "indice_mercado.parquet"
FORECAST_PATH = PROCESSED_DIR / "valorizacao_projetada.parquet"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def build(force_download: bool = False, skip_ibge: bool = False) -> pd.DataFrame:
    """Baixa FipeZAP/BCB/IBGE e materializa a tabela de features."""
    ensure_dirs()

    panel = fipezap.load_panel(force_download=force_download)
    logger.info("FipeZAP: %s linhas, %s cidades", len(panel), panel["city"].nunique())
    panel.to_parquet(PANEL_PATH, index=False)

    wide = fipezap.wide_prices(panel)
    macro = bcb.load_macro()
    logger.info("BCB: %s meses de séries macro", len(macro))

    socio = pd.DataFrame()
    if not skip_ibge:
        try:
            socio = ibge.load_socioeconomic(sorted(panel["city"].unique()))
            logger.info("IBGE: %s municípios resolvidos", len(socio))
        except Exception as exc:  # pragma: no cover - rede
            logger.warning("IBGE indisponível, seguindo sem socioeconômicos: %s", exc)

    table = feature_layer.build_feature_table(wide, macro=macro, socio=socio)
    table.to_parquet(FEATURES_PATH, index=False)

    feature_layer.city_market_index(table).to_parquet(MARKET_INDEX_PATH, index=False)
    forecast.project_prices(table).to_parquet(FORECAST_PATH, index=False)
    logger.info("Features gravadas em %s (%s linhas)", FEATURES_PATH, len(table))
    return table


def train(table: pd.DataFrame | None = None) -> price.FairPriceModel:
    """Treina e persiste o modelo de preço justo."""
    ensure_dirs()
    if table is None:
        table = pd.read_parquet(FEATURES_PATH)

    model = price.train(table)
    model.save(ARTIFACTS_DIR)

    metrics = {
        "fair_price": model.metrics,
        "forecast_backtest": forecast.backtest(table),
        "top_features": model.feature_importances().head(10).round(4).to_dict(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    logger.info("Modelo treinado: %s", metrics["fair_price"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de dados e modelos GJurema")
    parser.add_argument("command", choices=["build", "train", "all"])
    parser.add_argument("--force-download", action="store_true", help="ignora o cache local do FipeZAP")
    parser.add_argument("--skip-ibge", action="store_true", help="não consulta a API do IBGE")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    table = None
    if args.command in {"build", "all"}:
        table = build(force_download=args.force_download, skip_ibge=args.skip_ibge)
    if args.command in {"train", "all"}:
        train(table)


if __name__ == "__main__":
    main()
