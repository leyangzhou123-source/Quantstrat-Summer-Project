from __future__ import annotations

import numpy as np
import pandas as pd


def out_of_sample_r2(
    actual: pd.Series,
    forecast: pd.Series,
    benchmark: float | pd.Series = 0.0,
    weights: pd.Series | None = None,
) -> float:
    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    benchmark_values = np.asarray(benchmark, dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(forecast_values)
    if benchmark_values.ndim:
        valid &= np.isfinite(benchmark_values)
        benchmark_values = benchmark_values[valid]
    if weights is not None:
        weight_values = weights.to_numpy(dtype=float)
        valid &= np.isfinite(weight_values) & (weight_values > 0)
    if not valid.any():
        return float("nan")
    actual_values = actual_values[valid]
    forecast_values = forecast_values[valid]
    model_errors = (actual_values - forecast_values) ** 2
    benchmark_errors = (actual_values - benchmark_values) ** 2
    if weights is not None:
        weight_values = weight_values[valid]
        weight_values = weight_values / weight_values.mean()
        model_errors = model_errors * weight_values
        benchmark_errors = benchmark_errors * weight_values
    model_sse = np.sum(model_errors)
    benchmark_sse = np.sum(benchmark_errors)
    if not np.isfinite(model_sse) or not np.isfinite(benchmark_sse) or benchmark_sse <= 0:
        return float("nan")
    return 1.0 - model_sse / benchmark_sse


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    excess = returns.dropna()
    if excess.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))
