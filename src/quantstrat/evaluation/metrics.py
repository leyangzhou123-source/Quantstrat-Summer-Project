from __future__ import annotations

import numpy as np
import pandas as pd


def out_of_sample_r2(actual: pd.Series, forecast: pd.Series, benchmark: float | pd.Series = 0.0) -> float:
    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    benchmark_values = np.asarray(benchmark, dtype=float)
    model_sse = np.sum((actual_values - forecast_values) ** 2)
    benchmark_sse = np.sum((actual_values - benchmark_values) ** 2)
    return 1.0 - model_sse / benchmark_sse


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    excess = returns.dropna()
    if excess.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))

