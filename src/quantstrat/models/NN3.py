from __future__ import annotations

from typing import Any

import pandas as pd

from quantstrat.models.NeuralNetwork import train_validate_predict_depth
from quantstrat.models.base import ModelResult


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
    return train_validate_predict_depth(
        3, train, validation, test, target, features, config, random_seed, weight_column
    )
