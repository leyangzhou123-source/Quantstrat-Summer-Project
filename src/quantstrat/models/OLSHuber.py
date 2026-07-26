from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import SGDRegressor

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
        return SGDRegressor(
            loss="huber",
            penalty=None,
            epsilon=candidate.get("epsilon", config.get("epsilon", 0.1)),
            alpha=0.0,
            fit_intercept=config.get("fit_intercept", True),
            learning_rate=config.get("learning_rate", "invscaling"),
            eta0=candidate.get("eta0", config.get("eta0", 0.001)),
            max_iter=config.get("max_iter", 2000),
            tol=config.get("tol", 1e-4),
            random_state=random_seed,
        )

    return train_validate_predict_sklearn(
        model_name="ols_huber",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
