from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quantstrat.models.base import ModelResult
from quantstrat.models.sklearn_training import train_validate_predict_sklearn


def train_validate_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any] | None = None,
    random_seed: int = 42,
    weight_column: str | None = None,
) -> ModelResult:
    config = config or {}

    def model(candidate: dict[str, Any]):
        return RandomForestRegressor(
            n_estimators=candidate.get("n_estimators", config.get("n_estimators", 100)),
            max_depth=candidate.get("max_depth", config.get("max_depth", 6)),
            min_samples_leaf=candidate.get("min_samples_leaf", config.get("min_samples_leaf", 100)),
            max_features=candidate.get("max_features", config.get("max_features", "sqrt")),
            max_samples=candidate.get("max_samples", config.get("max_samples")),
            n_jobs=config.get("n_jobs", -1),
            random_state=random_seed,
        )

    return train_validate_predict_sklearn(
        model_name="random_forest",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
