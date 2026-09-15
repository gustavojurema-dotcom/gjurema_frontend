"""Modelo hedônico de preço por m² para São Paulo (base ITBI).

Diferente do modelo nacional, que trabalha com o índice mensal da cidade,
aqui cada linha é uma venda registrada: o preço é explicado por localização
(bairro e prédio), área, padrão construtivo, segmento e tempo.

O nível de preço do prédio entra como média histórica *anterior* à venda,
para não usar a própria transação na explicação dela mesma.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from gjurema import artifacts_io

TARGET = "preco_m2"
CATEGORICAL_FEATURES = ["segmento", "bairro"]
NUMERIC_FEATURES = [
    "log_area",
    "padrao",
    "month_index",
    "mes",
    "preco_m2_predio_anterior",
    "preco_m2_bairro_anterior",
]

DEFAULT_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "enable_categorical": True,
    "tree_method": "hist",
}

MODEL_FILE = "sp_price.json"
META_FILE = "sp_price_meta.json"


def build_design(transactions: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features da venda, todas observáveis antes dela."""
    frame = transactions.sort_values("data").copy()
    frame["log_area"] = np.log(frame["area_construida"])
    frame["mes"] = frame["data"].dt.month
    origin = frame["data"].min()
    frame["month_index"] = (
        (frame["data"].dt.year - origin.year) * 12 + frame["data"].dt.month - origin.month
    )

    for key, column in (("predio_id", "preco_m2_predio_anterior"), ("bairro", "preco_m2_bairro_anterior")):
        grouped = frame.groupby(key, observed=True)[TARGET]
        frame[column] = grouped.transform(lambda s: s.shift().expanding().mean())

    return frame


def design_row(
    *,
    segmento: str,
    bairro: str,
    area_m2: float,
    padrao: float,
    data: pd.Timestamp,
    origin: pd.Timestamp,
    preco_m2_predio: float | None = None,
    preco_m2_bairro: float | None = None,
) -> pd.DataFrame:
    """Monta a linha de features para precificar um imóvel fora da base."""
    data = pd.Timestamp(data)
    origin = pd.Timestamp(origin)
    return pd.DataFrame(
        [
            {
                "segmento": segmento,
                "bairro": bairro,
                "log_area": float(np.log(area_m2)),
                "padrao": float(padrao),
                "month_index": (data.year - origin.year) * 12 + data.month - origin.month,
                "mes": data.month,
                "preco_m2_predio_anterior": preco_m2_predio if preco_m2_predio else np.nan,
                "preco_m2_bairro_anterior": preco_m2_bairro if preco_m2_bairro else np.nan,
            }
        ]
    )


@dataclass
class HedonicPriceModel:
    """Modelo hedônico treinado, com intervalo empírico de resíduos."""

    model: XGBRegressor
    features: list[str]
    categorical_features: list[str]
    residual_quantiles: dict[str, float]
    metrics: dict[str, float]
    categories: dict[str, list[str]] = field(default_factory=dict)
    origin: str = ""

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        for column in self.features:
            if column not in data.columns:
                data[column] = np.nan
        for column in self.categorical_features:
            data[column] = pd.Categorical(data[column].astype(str), categories=self.categories.get(column))
        return data[self.features]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(self._prepare(frame))
        return np.exp(raw + self.residual_quantiles["p50"])

    def predict_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
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
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        artifacts_io.write_with(directory / MODEL_FILE, lambda temp: self.model.save_model(temp))
        artifacts_io.write_json(
            {
                "features": self.features,
                "categorical_features": self.categorical_features,
                "residual_quantiles": self.residual_quantiles,
                "metrics": self.metrics,
                "categories": self.categories,
                "origin": self.origin,
            },
            directory / META_FILE,
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> HedonicPriceModel:
        metadata = json.loads((directory / META_FILE).read_text())
        model = XGBRegressor(**DEFAULT_PARAMS)
        model.load_model(directory / MODEL_FILE)
        return cls(
            model=model,
            features=metadata["features"],
            categorical_features=metadata["categorical_features"],
            residual_quantiles=metadata["residual_quantiles"],
            metrics=metadata["metrics"],
            categories=metadata["categories"],
            origin=metadata.get("origin", ""),
        )


def train(
    transactions: pd.DataFrame,
    validation_months: int = 3,
    params: dict | None = None,
) -> HedonicPriceModel:
    """Treina com validação temporal: valida nas vendas mais recentes."""
    data = build_design(transactions).dropna(subset=[TARGET])
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    categories = {c: sorted(data[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
    for column in CATEGORICAL_FEATURES:
        data[column] = pd.Categorical(data[column].astype(str), categories=categories[column])

    cutoff = data["data"].max() - pd.DateOffset(months=validation_months)
    train_set = data[data["data"] <= cutoff]
    valid_set = data[data["data"] > cutoff]
    if train_set.empty or valid_set.empty:
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
        "mape_pct": float(np.mean(np.abs(np.expm1(-residuals))) * 100),
        "n_train": int(len(train_set)),
        "n_valid": int(len(valid_set)),
        "valid_start": str(valid_set["data"].min().date()),
    }
    residual_quantiles = {
        "p10": float(np.quantile(residuals, 0.10)),
        "p50": float(np.quantile(residuals, 0.50)),
        "p90": float(np.quantile(residuals, 0.90)),
    }

    return HedonicPriceModel(
        model=model,
        features=columns,
        categorical_features=CATEGORICAL_FEATURES,
        residual_quantiles=residual_quantiles,
        metrics=metrics,
        categories=categories,
        origin=str(data["data"].min().date()),
    )
