from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet

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
        return ElasticNet(
            alpha=candidate.get("alpha", config.get("alpha", 1.0)),
            l1_ratio=candidate.get("l1_ratio", config.get("l1_ratio", 0.5)),
            fit_intercept=config.get("fit_intercept", True),
            max_iter=config.get("max_iter", 1000),
            tol=config.get("tol", 1e-4),
            selection=config.get("selection", "cyclic"),
            random_state=random_seed,
        )

    return train_validate_predict_sklearn(
        model_name="elastic_net",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
