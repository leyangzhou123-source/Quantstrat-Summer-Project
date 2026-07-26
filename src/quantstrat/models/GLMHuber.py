from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

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
        return make_pipeline(
            SplineTransformer(
                n_knots=int(candidate.get("n_knots", config.get("n_knots", 3))),
                degree=int(config.get("degree", 3)),
                include_bias=False,
            ),
            SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                alpha=candidate.get("alpha", config.get("alpha", 1e-4)),
                l1_ratio=candidate.get("l1_ratio", config.get("l1_ratio", 0.5)),
                epsilon=candidate.get("epsilon", config.get("epsilon", 0.1)),
                max_iter=config.get("max_iter", 2000),
                tol=config.get("tol", 1e-4),
                random_state=random_seed,
            ),
        )

    return train_validate_predict_sklearn(
        model_name="glm_huber",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
