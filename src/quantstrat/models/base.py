from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    predictions: pd.Series
    validation_metrics: dict[str, float]
