from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np
import pandas as pd

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult
from quantstrat.models.sklearn_training import (
    _candidate_configs,
    _feature_matrix,
    _sample_weights,
    _scale_train_validation_test,
)


def _import_jax():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX backend requested but JAX is not installed. On ICRN, install it with "
            '`python -m pip install --user -U "jax[cuda12]"` if CUDA 12 is available, '
            "or `python -m pip install --user -U jax` for CPU-only testing."
        ) from exc
    return jax, jnp


def _prepare_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any],
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, pd.Series, pd.Series | None]:
    x_train = _feature_matrix(train, features, copy=config.get("copy_features", False))
    x_validation = _feature_matrix(validation, features, copy=config.get("copy_features", False))
    x_test = _feature_matrix(test, features, copy=config.get("copy_features", False))
    y_train = train[target].to_numpy(dtype=np.float32)
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
        train.drop(columns=features, inplace=True, errors="ignore")
        validation.drop(columns=features, inplace=True, errors="ignore")
        test.drop(columns=features, inplace=True, errors="ignore")
        gc.collect()
    if config.get("scale_features", True):
        x_train, x_validation, x_test = _scale_train_validation_test(
            x_train, x_validation, x_test
        )
    return (
        x_train,
        x_validation,
        x_test,
        np.nan_to_num(y_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0),
        train_weights.astype(np.float32, copy=False) if train_weights is not None else None,
        validation_target,
        validation_weight_values,
    )


def _jax_weighted_ridge_coefficients(
    x_train: Any,
    y_train: Any,
    weights: Any,
    alpha: float,
    fit_intercept: bool,
    ridge_jitter: float,
    jnp: Any,
) -> Any:
    if fit_intercept:
        ones = jnp.ones((x_train.shape[0], 1), dtype=x_train.dtype)
        x_train = jnp.concatenate([ones, x_train], axis=1)
    root_weights = jnp.sqrt(weights).reshape((-1, 1))
    x_weighted = x_train * root_weights
    y_weighted = y_train * root_weights.reshape((-1,))
    xtx = x_weighted.T @ x_weighted
    xty = x_weighted.T @ y_weighted
    penalty = jnp.eye(xtx.shape[0], dtype=xtx.dtype) * jnp.asarray(alpha + ridge_jitter, dtype=xtx.dtype)
    if fit_intercept:
        penalty = penalty.at[0, 0].set(0.0)
    return jnp.linalg.solve(xtx + penalty, xty)


def _jax_predict(x: Any, coefficients: Any, fit_intercept: bool) -> Any:
    if fit_intercept:
        return coefficients[0] + x @ coefficients[1:]
    return x @ coefficients


def train_validate_predict_jax_linear(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any] | None = None,
    weight_column: str | None = None,
) -> ModelResult:
    config = config or {}
    _, jnp = _import_jax()
    fit_intercept = config.get("fit_intercept", True)
    model_features = config.get("features", features)
    (
        x_train,
        x_validation,
        x_test,
        y_train,
        train_weights,
        validation_target,
        validation_weight_values,
    ) = _prepare_arrays(
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=model_features,
        config=config,
        weight_column=weight_column,
    )
    x_train_device = jnp.asarray(x_train)
    x_validation_device = jnp.asarray(x_validation)
    x_test_device = jnp.asarray(x_test)
    y_train_device = jnp.asarray(y_train)
    weights_device = (
        jnp.asarray(train_weights)
        if train_weights is not None
        else jnp.ones_like(y_train_device, dtype=jnp.float32)
    )

    best_coefficients = None
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    for candidate in _candidate_configs(config):
        alpha = float(candidate.get("alpha", config.get("alpha", 0.0)))
        ridge_jitter = float(config.get("ridge_jitter", 0.0))
        coefficients = _jax_weighted_ridge_coefficients(
            x_train=x_train_device,
            y_train=y_train_device,
            weights=weights_device,
            alpha=alpha if model_name == "ridge" else 0.0,
            fit_intercept=fit_intercept,
            ridge_jitter=ridge_jitter,
            jnp=jnp,
        )
        if not np.all(np.isfinite(np.asarray(coefficients))):
            coefficients = _jax_weighted_ridge_coefficients(
                x_train=x_train_device,
                y_train=y_train_device,
                weights=weights_device,
                alpha=alpha if model_name == "ridge" else 0.0,
                fit_intercept=fit_intercept,
                ridge_jitter=max(ridge_jitter, 1e-4),
                jnp=jnp,
            )
        validation_prediction = pd.Series(
            np.nan_to_num(
                np.asarray(_jax_predict(x_validation_device, coefficients, fit_intercept)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            index=validation.index,
            name="forecast",
        )
        score = out_of_sample_r2(
            validation_target,
            validation_prediction,
            weights=validation_weight_values,
        )
        if best_coefficients is None or score > best_score:
            best_coefficients = coefficients
            best_score = score
            best_params = candidate | {"backend": "jax"}

    if best_coefficients is None:
        raise RuntimeError(f"No candidate estimator was fit for {model_name}.")

    test_prediction = pd.Series(
        np.nan_to_num(
            np.asarray(_jax_predict(x_test_device, best_coefficients, fit_intercept)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        index=test.index,
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


def train_validate_predict_jax_pcr(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any] | None = None,
    weight_column: str | None = None,
) -> ModelResult:
    config = config or {}
    _, jnp = _import_jax()
    (
        x_train,
        x_validation,
        x_test,
        y_train,
        train_weights,
        validation_target,
        validation_weight_values,
    ) = _prepare_arrays(
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
    x_train_device = jnp.asarray(x_train)
    x_validation_device = jnp.asarray(x_validation)
    x_test_device = jnp.asarray(x_test)
    y_train_device = jnp.asarray(y_train)
    weights_device = (
        jnp.asarray(train_weights)
        if train_weights is not None
        else jnp.ones_like(y_train_device, dtype=jnp.float32)
    )

    covariance = x_train_device.T @ x_train_device
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    sorted_vectors = eigenvectors[:, jnp.argsort(eigenvalues)[::-1]]

    best_coefficients = None
    best_components = None
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    for candidate in _candidate_configs(config):
        n_components = min(
            int(candidate.get("n_components", config.get("n_components", 25))),
            len(features),
            max(1, len(train) - 1),
        )
        alpha = float(candidate.get("alpha", config.get("alpha", 1.0)))
        components = sorted_vectors[:, :n_components]
        train_scores = x_train_device @ components
        coefficients = _jax_weighted_ridge_coefficients(
            x_train=train_scores,
            y_train=y_train_device,
            weights=weights_device,
            alpha=alpha,
            fit_intercept=True,
            ridge_jitter=float(config.get("ridge_jitter", 0.0)),
            jnp=jnp,
        )
        validation_prediction = pd.Series(
            np.asarray(_jax_predict(x_validation_device @ components, coefficients, True)),
            index=validation.index,
            name="forecast",
        )
        score = out_of_sample_r2(
            validation_target,
            validation_prediction,
            weights=validation_weight_values,
        )
        if best_coefficients is None or score > best_score:
            best_coefficients = coefficients
            best_components = components
            best_score = score
            best_params = candidate | {"backend": "jax"}

    if best_coefficients is None or best_components is None:
        raise RuntimeError("No candidate estimator was fit for pcr.")

    test_prediction = pd.Series(
        np.asarray(_jax_predict(x_test_device @ best_components, best_coefficients, True)),
        index=test.index,
        name="forecast",
    )
    return ModelResult(
        model_name="pcr",
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": best_score,
            **{f"best_{key}": value for key, value in best_params.items()},
        },
    )
