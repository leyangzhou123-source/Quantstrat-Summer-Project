from __future__ import annotations

import numpy as np
import pandas as pd


def error_difference(
    actual: pd.Series,
    forecast_a: pd.Series,
    forecast_b: pd.Series,
) -> pd.Series:
    return (actual - forecast_a) ** 2 - (actual - forecast_b) ** 2


def simple_dm_stat(error_diff: pd.Series) -> float:
    values = error_diff.dropna().to_numpy(dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))

