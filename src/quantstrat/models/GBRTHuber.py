from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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
        return GradientBoostingRegressor(
            loss="huber",
            learning_rate=candidate.get("learning_rate", config.get("learning_rate", 0.05)),
            n_estimators=int(candidate.get("n_estimators", config.get("n_estimators", 100))),
            max_depth=int(candidate.get("max_depth", config.get("max_depth", 2))),
            min_samples_leaf=int(
                candidate.get("min_samples_leaf", config.get("min_samples_leaf", 100))
            ),
            max_features=candidate.get("max_features", config.get("max_features")),
            random_state=random_seed,
        )

    return train_validate_predict_sklearn(
        model_name="gbrt_huber",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
