from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult
from quantstrat.models.jax_linear_models import train_validate_predict_jax_linear
from quantstrat.models.sklearn_training import (
    _feature_matrix,
    _sample_weights,
    _scale_train_validation_test,
    train_validate_predict_sklearn,
)


def _fit_weighted_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray | None,
    fit_intercept: bool,
    ridge_jitter: float,
) -> np.ndarray:
    if fit_intercept:
        x_train = np.column_stack([np.ones(len(x_train), dtype=x_train.dtype), x_train])
    if weights is not None:
        root_weights = np.sqrt(weights).astype(x_train.dtype, copy=False)
        x_weighted = x_train * root_weights[:, None]
        y_weighted = y_train * root_weights
    else:
        x_weighted = x_train
        y_weighted = y_train

    xtx = x_weighted.T @ x_weighted
    xty = x_weighted.T @ y_weighted
    if ridge_jitter:
        start = 1 if fit_intercept else 0
        xtx[start:, start:] += np.eye(xtx.shape[0] - start) * ridge_jitter
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx, xty, rcond=None)[0]


def _predict_ols(x: np.ndarray, coefficients: np.ndarray, fit_intercept: bool) -> np.ndarray:
    if fit_intercept:
        return coefficients[0] + x @ coefficients[1:]
    return x @ coefficients


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
    model_features = config.get("features", features)
    if config.get("backend") == "jax":
        return train_validate_predict_jax_linear(
            model_name=config.get("model_name", "ols"),
            train=train,
            validation=validation,
            test=test,
            target=target,
            features=model_features,
            config=config,
            weight_column=weight_column,
        )
    if config.get("solver", "normal_equation") != "normal_equation":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=config.get("fit_intercept", True))
        return train_validate_predict_sklearn(
            model_name=config.get("model_name", "ols"),
            estimator=model,
            train=train,
            validation=validation,
            test=test,
            target=target,
            features=model_features,
            config=config,
            weight_column=weight_column,
        )

    x_train = _feature_matrix(train, model_features, copy=config.get("copy_features", False))
    x_validation = _feature_matrix(
        validation, model_features, copy=config.get("copy_features", False)
    )
    x_test = _feature_matrix(test, model_features, copy=config.get("copy_features", False))
    y_train = train[target].to_numpy(dtype=float)
    train_weights = _sample_weights(
        train, weight_column if config.get("use_sample_weight", True) else None
    )
    validation_weights = _sample_weights(
        validation, weight_column if config.get("weighted_validation", True) else None
    )
    validation_target = validation[target].copy()
    validation_weight_values = (
        validation[weight_column].copy()
        if validation_weights is not None and weight_column
        else None
    )
    if config.get("drop_features_after_matrix", False):
        train.drop(columns=model_features, inplace=True, errors="ignore")
        validation.drop(columns=model_features, inplace=True, errors="ignore")
        test.drop(columns=model_features, inplace=True, errors="ignore")
        gc.collect()
    if config.get("scale_features", True):
        x_train, x_validation, x_test = _scale_train_validation_test(x_train, x_validation, x_test)

    coefficients = _fit_weighted_ols(
        x_train=x_train,
        y_train=y_train,
        weights=train_weights,
        fit_intercept=config.get("fit_intercept", True),
        ridge_jitter=config.get("ridge_jitter", 0.0),
    )
    validation_prediction = pd.Series(
        _predict_ols(x_validation, coefficients, config.get("fit_intercept", True)),
        index=validation.index,
        name="forecast",
    )
    test_prediction = pd.Series(
        _predict_ols(x_test, coefficients, config.get("fit_intercept", True)),
        index=test.index,
        name="forecast",
    )
    return ModelResult(
        model_name=config.get("model_name", "ols"),
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": out_of_sample_r2(
                validation_target,
                validation_prediction,
                weights=validation_weight_values,
            )
        },
    )
