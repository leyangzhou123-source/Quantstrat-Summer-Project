from __future__ import annotations

import pandas as pd

from quantstrat.models.registry import ModelAdapter


def fit_and_predict(
    model: ModelAdapter,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
) -> pd.DataFrame:
    fitted = model.fit(train=train, target=target, features=features)
    predictions = test.copy()
    predictions["forecast"] = fitted.predict(test, features=features)
    predictions["model"] = model.name
    return predictions

