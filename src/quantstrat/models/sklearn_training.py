from __future__ import annotations

from collections.abc import Callable
import gc
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult


def _sample_weights(frame: pd.DataFrame, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in frame.columns:
        return None
    weights = frame[weight_column].to_numpy(dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    weights = np.where(valid, weights, 0.0)
    return weights / weights[valid].mean()


def _fit_params_for_weights(estimator: RegressorMixin, sample_weight: np.ndarray | None) -> dict[str, Any]:
    if sample_weight is None:
        return {}
    if isinstance(estimator, Pipeline):
        final_name, final_estimator = estimator.steps[-1]
        if "sample_weight" in signature(final_estimator.fit).parameters:
            return {f"{final_name}__sample_weight": sample_weight}
        return {}
    if "sample_weight" in signature(estimator.fit).parameters:
        return {"sample_weight": sample_weight}
    return {}


def _candidate_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("validation_grid")
    if not grid:
        return [{}]
    candidates = [{}]
    for key, values in grid.items():
        candidates = [
            candidate | {key: value}
            for candidate in candidates
            for value in values
        ]
    return candidates


def _feature_matrix(frame: pd.DataFrame, features: list[str], copy: bool) -> np.ndarray:
    values = frame.loc[:, features].to_numpy(dtype=np.float32, copy=copy)
    if not values.flags.writeable:
        values = values.copy()
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _scale_train_validation_test(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    x_train -= mean
    x_train /= std
    x_validation -= mean
    x_validation /= std
    x_test -= mean
    x_test /= std
    return x_train, x_validation, x_test


def train_validate_predict_sklearn(
    model_name: str,
    estimator: RegressorMixin | Callable[[dict[str, Any]], RegressorMixin],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any] | None = None,
    weight_column: str | None = None,
) -> ModelResult:
    config = config or {}
    x_train = _feature_matrix(train, features, copy=config.get("copy_features", False))
    x_validation = _feature_matrix(validation, features, copy=config.get("copy_features", False))
    x_test = _feature_matrix(test, features, copy=config.get("copy_features", False))
    y_train = train[target].to_numpy(dtype=float)
    y_mean = 0.0
    y_std = 1.0
    if config.get("standardize_target", False):
        y_mean = float(np.nanmean(y_train))
        y_std = float(np.nanstd(y_train))
        if not np.isfinite(y_std) or y_std == 0.0:
            y_std = 1.0
        y_train = (y_train - y_mean) / y_std
    train_weights = _sample_weights(train, weight_column if config.get("use_sample_weight", True) else None)
    validation_weights = _sample_weights(validation, weight_column if config.get("weighted_validation", True) else None)
    validation_target = validation[target].copy()
    validation_weight_values = (
        validation[weight_column].copy()
        if validation_weights is not None and weight_column
        else None
    )
    if config.get("drop_features_after_matrix", False):
        train.drop(columns=features, inplace=True, errors="ignore")
        validation.drop(columns=features, inplace=True, errors="ignore")
        test.drop(columns=features, inplace=True, errors="ignore")
        gc.collect()
    if config.get("scale_features", True):
        x_train, x_validation, x_test = _scale_train_validation_test(
            x_train, x_validation, x_test
        )

    def predict(estimator: RegressorMixin, x: np.ndarray) -> np.ndarray:
        forecast = np.asarray(estimator.predict(x), dtype=float).reshape(-1)
        if config.get("standardize_target", False):
            forecast = forecast * y_std + y_mean
        clip = config.get("forecast_clip")
        if clip:
            forecast = np.clip(forecast, float(clip[0]), float(clip[1]))
        return forecast

    best_estimator: RegressorMixin | None = None
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    for candidate in _candidate_configs(config):
        if callable(estimator):
            candidate_estimator = estimator(candidate)
        else:
            candidate_estimator = clone(estimator)
        candidate_estimator.fit(
            x_train,
            y_train,
            **_fit_params_for_weights(candidate_estimator, train_weights),
        )
        validation_prediction = pd.Series(
            predict(candidate_estimator, x_validation),
            index=validation.index,
            name="forecast",
        )
        score = out_of_sample_r2(
            validation_target,
            validation_prediction,
            weights=validation_weight_values,
        )
        if score > best_score:
            best_estimator = candidate_estimator
            best_score = score
            best_params = candidate

    if best_estimator is None:
        raise RuntimeError(f"No candidate estimator was fit for {model_name}.")

    test_prediction = pd.Series(
        predict(best_estimator, x_test),
        index=test.index,
        name="forecast",
    )
    validation_prediction = pd.Series(
        predict(best_estimator, x_validation),
        index=validation.index,
        name="forecast",
    )
    return ModelResult(
        model_name=model_name,
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": best_score,
            **{f"best_{key}": value for key, value in best_params.items()},
        },
    )
