"""Camada 3 — modelo de preço justo por m² (XGBoost).

O modelo aprende o preço de venda por m² a partir de cidade, tipologia,
contexto macroeconômico (SELIC, IPCA, IGP-M, crédito imobiliário),
socioeconômico (população e PIB per capita) e do nível de preço observado
doze meses antes. O alvo é modelado em log para estabilizar a variância.

A incerteza é estimada empiricamente: os resíduos da janela de validação
temporal definem os quantis usados como intervalo de confiança.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

TARGET = "venda_preco_m2"
CATEGORICAL_FEATURES = ["city", "typology", "uf"]
NUMERIC_FEATURES = [
    "month_index",
    "month",
    "venda_preco_m2_lag12",
    "selic_meta_aa",
    "ipca_mensal",
    "igpm_mensal",
    "incc_mensal",
    "credito_imobiliario_taxa_aa",
    "log_population",
    "log_gdp_per_capita",
]

# Features derivadas do próprio preço corrente vazariam o alvo.
LEAKY_FEATURES = {
    "venda_preco_m2_lag1",
    "venda_var_3m",
    "venda_var_6m",
    "venda_var_12m",
    "preco_relativo_cidade",
    "preco_relativo_tipologia",
    "yield_bruto_anual",
    "locacao_preco_m2",
    "locacao_preco_m2_lag1",
    "volatilidade_12m",
}

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "random_state": 42,
    "enable_categorical": True,
    "tree_method": "hist",
}


@dataclass
class FairPriceModel:
    """Modelo de preço justo treinado, com metadados de avaliação."""

    model: XGBRegressor
    features: list[str]
    categorical_features: list[str]
    residual_quantiles: dict[str, float]
    metrics: dict[str, float]
    categories: dict[str, list[str]] = field(default_factory=dict)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        for column in self.features:
            if column not in data.columns:
                data[column] = np.nan
        for column in self.categorical_features:
            data[column] = pd.Categorical(data[column], categories=self.categories.get(column))
        return data[self.features]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Preço justo previsto (R$/m²), corrigido pelo viés da validação."""
        raw = self.model.predict(self._prepare(frame))
        return np.exp(raw + self.residual_quantiles["p50"])

    def predict_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Preço justo com banda de incerteza derivada dos resíduos."""
        point = self.predict(frame)
        lower = float(np.exp(self.residual_quantiles["p10"] - self.residual_quantiles["p50"]))
        upper = float(np.exp(self.residual_quantiles["p90"] - self.residual_quantiles["p50"]))
        return pd.DataFrame(
            {
                "preco_justo_m2": point,
                "preco_justo_m2_p10": point * lower,
                "preco_justo_m2_p90": point * upper,
            },
            index=frame.index,
        )

    def feature_importances(self) -> pd.Series:
        importances = pd.Series(self.model.feature_importances_, index=self.features)
        return importances.sort_values(ascending=False)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(directory / "fair_price.json")
        metadata = {
            "features": self.features,
            "categorical_features": self.categorical_features,
            "residual_quantiles": self.residual_quantiles,
            "metrics": self.metrics,
            "categories": self.categories,
        }
        path = directory / "fair_price_meta.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path) -> FairPriceModel:
        metadata = json.loads((directory / "fair_price_meta.json").read_text())
        model = XGBRegressor(**DEFAULT_PARAMS)
        model.load_model(directory / "fair_price.json")
        return cls(
            model=model,
            features=metadata["features"],
            categorical_features=metadata["categorical_features"],
            residual_quantiles=metadata["residual_quantiles"],
            metrics=metadata["metrics"],
            categories=metadata["categories"],
        )


def select_features(frame: pd.DataFrame) -> list[str]:
    numeric = [c for c in NUMERIC_FEATURES if c in frame.columns and c not in LEAKY_FEATURES]
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    return categorical + numeric


def train(
    features: pd.DataFrame,
    validation_months: int = 12,
    params: dict | None = None,
) -> FairPriceModel:
    """Treina o modelo com validação temporal nos últimos meses do painel."""
    data = features.dropna(subset=[TARGET]).copy()
    data["date"] = pd.to_datetime(data["date"])

    columns = select_features(data)
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]
    categories = {c: sorted(data[c].dropna().astype(str).unique().tolist()) for c in categorical}
    for column in categorical:
        data[column] = pd.Categorical(data[column].astype(str), categories=categories[column])

    cutoff = data["date"].max() - pd.DateOffset(months=validation_months)
    train_set = data[data["date"] <= cutoff]
    valid_set = data[data["date"] > cutoff]
    if valid_set.empty:
        train_set, valid_set = data, data

    y_train = np.log(train_set[TARGET])
    y_valid = np.log(valid_set[TARGET])

    model = XGBRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(train_set[columns], y_train)

    predicted = model.predict(valid_set[columns])
    residuals = y_valid.to_numpy() - predicted

    metrics = {
        "r2_log": float(r2_score(y_valid, predicted)),
        "mae_log": float(mean_absolute_error(y_valid, predicted)),
        "mape_pct": float(np.mean(np.abs(np.expm1(residuals))) * 100),
        "n_train": int(len(train_set)),
        "n_valid": int(len(valid_set)),
        "valid_start": str(valid_set["date"].min().date()),
    }
    residual_quantiles = {
        "p10": float(np.quantile(residuals, 0.10)),
        "p50": float(np.quantile(residuals, 0.50)),
        "p90": float(np.quantile(residuals, 0.90)),
    }

    return FairPriceModel(
        model=model,
        features=columns,
        categorical_features=categorical,
        residual_quantiles=residual_quantiles,
        metrics=metrics,
        categories=categories,
    )


def opportunity_score(
    asking_price: float,
    area_m2: float,
    fair_price_m2: float,
    market_index: float = 0.5,
    liquidity: float = 0.5,
) -> dict[str, float]:
    """Camada 4 — traduz o desvio de preço em um score de oportunidade.

    Combina o desconto sobre o preço justo (60%), o aquecimento do mercado
    local (25%) e a liquidez histórica da tipologia na cidade (15%).
    O score é normalizado para 0–100.
    """
    if area_m2 <= 0:
        raise ValueError("area_m2 deve ser positiva")
    # Cidades sem dado de locação não têm índice de mercado; usa-se o neutro.
    market_index = market_index if np.isfinite(market_index) else 0.5
    liquidity = liquidity if np.isfinite(liquidity) else 0.5
    fair_value = fair_price_m2 * area_m2
    deviation = (asking_price - fair_value) / fair_value
    discount = -deviation

    discount_component = float(np.clip(discount / 0.30, -1, 1))
    score = 100 * (0.60 * (discount_component + 1) / 2 + 0.25 * market_index + 0.15 * liquidity)

    return {
        "preco_justo_total": fair_value,
        "desvio_pct": deviation * 100,
        "desconto_pct": discount * 100,
        "score_oportunidade": float(np.clip(score, 0, 100)),
    }
