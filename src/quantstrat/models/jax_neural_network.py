from __future__ import annotations

from collections.abc import Callable
import gc
import os
from typing import Any

import numpy as np
import pandas as pd

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult
from quantstrat.models.sklearn_training import _candidate_configs


def _feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _target_vector(frame: pd.DataFrame, target: str) -> np.ndarray:
    values = frame[target].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_weights(frame: pd.DataFrame, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in frame.columns:
        return None
    weights = frame[weight_column].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    weights = np.where(valid, weights, 0.0).astype(np.float32, copy=False)
    return weights / weights[valid].mean()


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


def _import_jax():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX NN backend requested but JAX is not installed. On ICRN, install it with "
            "`python -m pip install --user -U \"jax[cuda12]\"` if CUDA 12 is available, "
            "or `python -m pip install --user -U jax` for CPU-only testing."
        ) from exc
    return jax, jnp


def _activation(name: str, jnp: Any) -> Callable[[Any], Any]:
    if name == "relu":
        return lambda x: jnp.maximum(x, 0)
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported JAX NN activation: {name!r}")


def train_validate_predict_jax_depth(
    depth: int,
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
    jax, jnp = _import_jax()

    x_train = _feature_matrix(train, features)
    x_validation = _feature_matrix(validation, features)
    x_test = _feature_matrix(test, features)
    y_train = _target_vector(train, target)
    train_weights = _sample_weights(
        train, weight_column if config.get("use_sample_weight", True) else None
    )
    validation_weight_values = (
        validation[weight_column].copy()
        if weight_column and config.get("weighted_validation", True) and weight_column in validation
        else None
    )
    validation_target = validation[target].copy()

    if config.get("scale_features", True):
        x_train, x_validation, x_test = _scale_train_validation_test(
            x_train, x_validation, x_test
        )

    if config.get("drop_features_after_matrix", False):
        train.drop(columns=features, inplace=True, errors="ignore")
        validation.drop(columns=features, inplace=True, errors="ignore")
        test.drop(columns=features, inplace=True, errors="ignore")
        gc.collect()

    target_mean = np.float32(0.0)
    target_std = np.float32(1.0)
    if config.get("standardize_target", True):
        target_mean = np.float32(np.mean(y_train, dtype=np.float64))
        target_std = np.float32(np.std(y_train, dtype=np.float64))
        if not np.isfinite(target_std) or target_std == 0:
            target_std = np.float32(1.0)
        y_train = (y_train - target_mean) / target_std
    if train_weights is None:
        train_weights = np.ones_like(y_train, dtype=np.float32)

    activation_fn = _activation(config.get("activation", "relu"), jnp)
    batch_size = int(config.get("batch_size", 8192))
    max_iter = int(config.get("max_iter", 100))
    n_iter_no_change = int(config.get("n_iter_no_change", 8))
    tol = float(config.get("tol", 1e-5))
    clip_grad_norm = float(config.get("clip_grad_norm", 5.0))
    validation_batch_size = int(config.get("prediction_batch_size", 65536))
    validate_every = max(1, int(config.get("validate_every", 1)))
    steps_per_epoch = max(1, len(x_train) // batch_size)
    input_dim = len(features)

    def hidden_widths(candidate: dict[str, Any]) -> list[int]:
        configured = candidate.get("layer_widths", config.get("layer_widths"))
        if configured:
            return [int(width) for width in configured[:depth]]
        width = int(candidate.get("width", config.get("width", 32)))
        return [max(1, width // (2**layer)) for layer in range(depth)]

    def init_params(key: Any, widths: list[int]) -> list[tuple[Any, Any]]:
        layer_sizes = [input_dim] + widths + [1]
        params = []
        keys = jax.random.split(key, len(layer_sizes) - 1)
        for layer_index, (layer_key, fan_in, fan_out) in enumerate(
            zip(keys, layer_sizes[:-1], layer_sizes[1:])
        ):
            scale = jnp.sqrt(jnp.asarray(2.0 / max(fan_in, 1), dtype=jnp.float32))
            if config.get("zero_output_init", True) and layer_index == len(layer_sizes) - 2:
                weight = jnp.zeros((fan_in, fan_out), dtype=jnp.float32)
            else:
                weight = jax.random.normal(layer_key, (fan_in, fan_out), dtype=jnp.float32) * scale
            bias = jnp.zeros((fan_out,), dtype=jnp.float32)
            params.append((weight, bias))
        return params

    def forward(params: list[tuple[Any, Any]], x: Any) -> Any:
        hidden = x
        for weight, bias in params[:-1]:
            hidden = activation_fn(hidden @ weight + bias)
        output_weight, output_bias = params[-1]
        return (hidden @ output_weight + output_bias).squeeze(-1)

    def loss_fn(params: list[tuple[Any, Any]], x: Any, y: Any, w: Any, alpha: float) -> Any:
        pred = forward(params, x)
        mse = jnp.mean(w * jnp.square(pred - y))
        penalty = sum(jnp.sum(jnp.square(weight)) for weight, _ in params)
        return mse + jnp.asarray(alpha, dtype=jnp.float32) * penalty / x.shape[0]

    def clip_grads(grads: list[tuple[Any, Any]], max_norm: float) -> list[tuple[Any, Any]]:
        norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(grad_w)) + jnp.sum(jnp.square(grad_b)) for grad_w, grad_b in grads)
        )
        scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=jnp.float32) / (norm + 1e-6))
        return [(grad_w * scale, grad_b * scale) for grad_w, grad_b in grads]

    @jax.jit
    def train_step(
        params: list[tuple[Any, Any]],
        moments: list[list[tuple[Any, Any]]],
        x: Any,
        y: Any,
        w: Any,
        lr: float,
        alpha: float,
        step: Any,
    ):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, w, alpha)
        grads = clip_grads(grads, clip_grad_norm)
        beta1 = jnp.asarray(0.9, dtype=jnp.float32)
        beta2 = jnp.asarray(0.999, dtype=jnp.float32)
        eps = jnp.asarray(1e-8, dtype=jnp.float32)
        new_params = []
        new_moments = [[], []]
        step_f = step.astype(jnp.float32)
        for (weight, bias), (grad_w, grad_b), (m_w, v_w), (m_b, v_b) in zip(
            params, grads, moments[0], moments[1]
        ):
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * jnp.square(grad_w)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * jnp.square(grad_b)
            m_w_hat = m_w / (1.0 - beta1**step_f)
            v_w_hat = v_w / (1.0 - beta2**step_f)
            m_b_hat = m_b / (1.0 - beta1**step_f)
            v_b_hat = v_b / (1.0 - beta2**step_f)
            new_params.append(
                (
                    weight - lr * m_w_hat / (jnp.sqrt(v_w_hat) + eps),
                    bias - lr * m_b_hat / (jnp.sqrt(v_b_hat) + eps),
                )
            )
            new_moments[0].append((m_w, v_w))
            new_moments[1].append((m_b, v_b))
        return new_params, new_moments, loss

    @jax.jit
    def predict_batch(params: list[tuple[Any, Any]], x: Any) -> Any:
        return forward(params, x)

    @jax.jit
    def hidden_batch(params: list[tuple[Any, Any]], x: Any) -> Any:
        hidden = x
        for weight, bias in params[:-1]:
            hidden = activation_fn(hidden @ weight + bias)
        return hidden

    def hidden_numpy(params: list[tuple[Any, Any]], x_values: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, int(x_values.shape[0]), validation_batch_size):
            x_chunk = jnp.asarray(x_values[start : start + validation_batch_size])
            chunks.append(np.asarray(hidden_batch(params, x_chunk), dtype=np.float32))
        return np.concatenate(chunks)

    def predict_numpy(params: list[tuple[Any, Any]], x_values: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, int(x_values.shape[0]), validation_batch_size):
            x_chunk = jnp.asarray(x_values[start : start + validation_batch_size])
            chunks.append(np.asarray(predict_batch(params, x_chunk)))
        predictions = np.concatenate(chunks)
        predictions = predictions * target_std + target_mean
        forecast_clip = config.get("forecast_clip")
        if forecast_clip:
            low, high = forecast_clip
            predictions = np.clip(predictions, float(low), float(high))
        return predictions

    def fit_weighted_ridge_output(
        hidden_train: np.ndarray,
        y_values: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        design = np.column_stack(
            [np.ones(len(hidden_train), dtype=np.float32), hidden_train.astype(np.float32, copy=False)]
        )
        root_weights = np.sqrt(weights).astype(np.float32, copy=False)
        weighted_design = design * root_weights[:, None]
        weighted_y = y_values * root_weights
        xtx = weighted_design.T @ weighted_design
        xty = weighted_design.T @ weighted_y
        penalty = np.eye(xtx.shape[0], dtype=np.float32) * np.float32(alpha)
        penalty[0, 0] = 0.0
        try:
            return np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(xtx + penalty, xty, rcond=None)[0]

    def predict_ridge_output(hidden_values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        predictions = coefficients[0] + hidden_values @ coefficients[1:]
        predictions = predictions * target_std + target_mean
        forecast_clip = config.get("forecast_clip")
        if forecast_clip:
            low, high = forecast_clip
            predictions = np.clip(predictions, float(low), float(high))
        return predictions

    if config.get("output_ridge", False):
        best_members: list[tuple[list[tuple[Any, Any]], np.ndarray]] | None = None
        best_score = -np.inf
        best_params: dict[str, Any] = {}
        best_shrinkage = 1.0
        for candidate in _candidate_configs(config):
            widths = hidden_widths(candidate)
            alpha = float(candidate.get("alpha", config.get("output_ridge_alpha", 10000.0)))
            shrinkage_values = candidate.get(
                "output_shrinkage", config.get("output_shrinkage_grid", [1.0])
            )
            if isinstance(shrinkage_values, (int, float)):
                shrinkage_values = [float(shrinkage_values)]
            ensemble_seeds = candidate.get("ensemble_seeds", config.get("ensemble_seeds", [random_seed]))
            if isinstance(ensemble_seeds, int):
                ensemble_seeds = [ensemble_seeds]
            members = []
            validation_forecasts = []
            for seed in ensemble_seeds:
                params = init_params(jax.random.PRNGKey(int(seed)), widths)
                hidden_train = hidden_numpy(params, x_train)
                hidden_validation = hidden_numpy(params, x_validation)
                hidden_mean = hidden_train.mean(axis=0, dtype=np.float64).astype(np.float32)
                hidden_std = hidden_train.std(axis=0, dtype=np.float64).astype(np.float32)
                hidden_std[~np.isfinite(hidden_std) | (hidden_std == 0)] = 1.0
                hidden_train = (hidden_train - hidden_mean) / hidden_std
                hidden_validation = (hidden_validation - hidden_mean) / hidden_std
                coefficients = fit_weighted_ridge_output(
                    hidden_train=hidden_train,
                    y_values=y_train,
                    weights=train_weights,
                    alpha=alpha,
                )
                members.append((params, coefficients, hidden_mean, hidden_std))
                validation_forecasts.append(predict_ridge_output(hidden_validation, coefficients))
            raw_validation_forecast = np.mean(validation_forecasts, axis=0)
            center = float(np.mean(raw_validation_forecast))
            for shrinkage in shrinkage_values:
                validation_prediction = pd.Series(
                    center + float(shrinkage) * (raw_validation_forecast - center),
                    index=validation.index,
                    name="forecast",
                )
                score = out_of_sample_r2(
                    validation_target,
                    validation_prediction,
                    weights=validation_weight_values,
                )
                if best_members is None or (np.isfinite(score) and score > best_score):
                    best_members = members
                    best_score = score
                    best_shrinkage = float(shrinkage)
                    best_params = candidate | {
                        "backend": "jax_output_ridge",
                        "layer_widths": "-".join(str(width) for width in widths),
                        "ensemble_size": len(members),
                        "output_shrinkage": best_shrinkage,
                        "epochs": 0.0,
                        "device": str(jax.devices()[0]),
                    }
        if best_members is None:
            raise RuntimeError(f"No output-ridge candidate was fit for nn{depth}.")
        test_forecasts = []
        for params, coefficients, hidden_mean, hidden_std in best_members:
            hidden_test = (hidden_numpy(params, x_test) - hidden_mean) / hidden_std
            test_forecasts.append(predict_ridge_output(hidden_test, coefficients))
        raw_test_forecast = np.mean(test_forecasts, axis=0)
        test_center = float(np.mean(raw_test_forecast))
        test_prediction = pd.Series(
            test_center + best_shrinkage * (raw_test_forecast - test_center),
            index=test.index,
            name="forecast",
        )
        return ModelResult(
            model_name=f"nn{depth}",
            predictions=test_prediction,
            validation_metrics={
                "oos_r2": best_score,
                **{f"best_{key}": value for key, value in best_params.items()},
            },
        )

    def train_single_member(
        candidate: dict[str, Any],
        widths: list[int],
        seed: int,
    ) -> tuple[list[tuple[Any, Any]], float, int]:
        alpha = float(candidate.get("alpha", config.get("alpha", 1e-4)))
        learning_rate = float(
            candidate.get("learning_rate_init", config.get("learning_rate_init", 1e-3))
        )
        learning_rate_decay = float(config.get("learning_rate_decay", 1.0))
        key = jax.random.PRNGKey(seed)
        rng = np.random.default_rng(seed)
        params = init_params(key, widths)
        zero_weight_moments = [
            (jnp.zeros_like(weight), jnp.zeros_like(weight)) for weight, _ in params
        ]
        zero_bias_moments = [
            (jnp.zeros_like(bias), jnp.zeros_like(bias)) for _, bias in params
        ]
        moments = [zero_weight_moments, zero_bias_moments]
        initial_validation_prediction = pd.Series(
            predict_numpy(params, x_validation),
            index=validation.index,
            name="forecast",
        )
        best_member_params = params
        best_member_score = out_of_sample_r2(
            validation_target,
            initial_validation_prediction,
            weights=validation_weight_values,
        )
        epochs_without_improvement = 0
        step = jnp.asarray(1, dtype=jnp.int32)

        for epoch in range(max_iter):
            shuffled = rng.permutation(len(x_train))[: steps_per_epoch * batch_size]
            current_learning_rate = learning_rate * (learning_rate_decay**epoch)
            for batch_idx in range(steps_per_epoch):
                batch_index = shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                params, moments, _ = train_step(
                    params,
                    moments,
                    jnp.asarray(x_train[batch_index]),
                    jnp.asarray(y_train[batch_index]),
                    jnp.asarray(train_weights[batch_index]),
                    current_learning_rate,
                    alpha,
                    step,
                )
                step = step + 1

            if (epoch + 1) % validate_every != 0 and epoch < max_iter - 1:
                continue
            validation_prediction = pd.Series(
                predict_numpy(params, x_validation),
                index=validation.index,
                name="forecast",
            )
            score = out_of_sample_r2(
                validation_target,
                validation_prediction,
                weights=validation_weight_values,
            )
            improved = np.isfinite(score) and score > best_member_score + tol
            if improved:
                best_member_params = params
                best_member_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if config.get("early_stopping", True) and epochs_without_improvement >= n_iter_no_change:
                break

        if best_member_params is None:
            best_member_params = params
        return best_member_params, best_member_score, epoch + 1

    best_estimator: list[list[tuple[Any, Any]]] | None = None
    best_score = -np.inf
    best_params: dict[str, Any] = {}

    for candidate in _candidate_configs(config):
        widths = hidden_widths(candidate)
        ensemble_seeds = candidate.get("ensemble_seeds", config.get("ensemble_seeds", [random_seed]))
        if isinstance(ensemble_seeds, int):
            ensemble_seeds = [ensemble_seeds]
        member_params = []
        member_epochs = []
        for seed in ensemble_seeds:
            params, _, epochs = train_single_member(candidate, widths, int(seed))
            member_params.append(params)
            member_epochs.append(epochs)
        validation_forecast = np.mean(
            [predict_numpy(params, x_validation) for params in member_params],
            axis=0,
        )
        validation_prediction = pd.Series(
            validation_forecast,
            index=validation.index,
            name="forecast",
        )
        candidate_score = out_of_sample_r2(
            validation_target,
            validation_prediction,
            weights=validation_weight_values,
        )

        if best_estimator is None or (np.isfinite(candidate_score) and candidate_score > best_score):
            best_estimator = member_params
            best_score = candidate_score
            best_params = candidate | {
                "backend": "jax",
                "layer_widths": "-".join(str(width) for width in widths),
                "ensemble_size": len(member_params),
                "epochs": float(np.mean(member_epochs)),
                "device": str(jax.devices()[0]),
            }

    if best_estimator is None:
        raise RuntimeError(f"No candidate estimator was fit for nn{depth}.")

    test_prediction = pd.Series(
        np.mean([predict_numpy(params, x_test) for params in best_estimator], axis=0),
        index=test.index,
        name="forecast",
    )
    return ModelResult(
        model_name=f"nn{depth}",
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": best_score,
            **{f"best_{key}": value for key, value in best_params.items()},
        },
    )
