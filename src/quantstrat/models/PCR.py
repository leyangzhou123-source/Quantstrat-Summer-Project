from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from quantstrat.models.base import ModelResult
from quantstrat.models.jax_linear_models import train_validate_predict_jax_pcr
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
    if config.get("backend") == "jax":
        return train_validate_predict_jax_pcr(
            train=train,
            validation=validation,
            test=test,
            target=target,
            features=features,
            config=config,
            weight_column=weight_column,
        )

    def model(candidate: dict[str, Any]):
        n_components = min(
            candidate.get("n_components", config.get("n_components", 25)),
            len(features),
            max(1, len(train) - 1),
        )
        return make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=random_seed),
            Ridge(alpha=candidate.get("alpha", config.get("alpha", 1.0))),
        )

    return train_validate_predict_sklearn(
        model_name="pcr",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
