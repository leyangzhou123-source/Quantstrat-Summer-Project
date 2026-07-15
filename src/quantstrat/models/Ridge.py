from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult


def train_validate_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any] | None = None,
    random_seed: int = 42,
) -> ModelResult:
    config = config or {}
    model = make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=config.get("alpha", 1.0),
            fit_intercept=config.get("fit_intercept", True),
            random_state=random_seed,
        ),
    )
    x_train = train[features].fillna(0.0)
    y_train = train[target].to_numpy(dtype=float)
    model.fit(x_train, y_train)

    validation_prediction = pd.Series(
        model.predict(validation[features].fillna(0.0)),
        index=validation.index,
        name="forecast",
    )
    test_prediction = pd.Series(
        model.predict(test[features].fillna(0.0)),
        index=test.index,
        name="forecast",
    )
    return ModelResult(
        model_name="ridge",
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": out_of_sample_r2(validation[target], validation_prediction),
        },
    )
