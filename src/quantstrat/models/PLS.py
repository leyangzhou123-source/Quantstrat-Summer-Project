from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cross_decomposition import PLSRegression

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
        n_components = min(
            int(candidate.get("n_components", config.get("n_components", 5))),
            len(features),
            max(1, len(train) - 1),
        )
        return PLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=config.get("max_iter", 500),
            tol=config.get("tol", 1e-6),
        )

    return train_validate_predict_sklearn(
        model_name="pls",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
