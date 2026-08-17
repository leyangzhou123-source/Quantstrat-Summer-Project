from __future__ import annotations

from itertools import pairwise, product
from typing import Any

import numpy as np
import pandas as pd

from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult


def _candidate_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("validation_grid")
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def _feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].to_numpy(dtype=np.float32, copy=False)
    return np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _weights(
    frame: pd.DataFrame,
    weight_column: str | None,
    enabled: bool,
    config: dict[str, Any],
) -> np.ndarray:
    if not enabled or not weight_column or weight_column not in frame:
        return np.ones(len(frame), dtype=np.float32)
    values = frame[weight_column].to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return np.ones(len(frame), dtype=np.float32)
    values = np.where(valid, values, 0.0)
    cap_quantile = config.get("sample_weight_cap_quantile")
    if cap_quantile is not None:
        cap = np.nanquantile(values[valid], float(cap_quantile))
        if np.isfinite(cap) and cap > 0:
            values = np.minimum(values, cap)
    power = float(config.get("sample_weight_power", 1.0))
    if power != 1.0:
        values = np.power(values, power, where=values > 0, out=np.zeros_like(values))
    return (values / values[valid].mean()).astype(np.float32, copy=False)


def _standardize(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def _parse_head_layers(value: Any) -> tuple[int, ...]:
    if value in (None, [], ""):
        return ()
    if isinstance(value, str):
        return tuple(int(part) for part in value.split("-") if part)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(part) for part in value)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The transformer_nn model now uses PyTorch. Install it with "
            "`python -m pip install torch`, or install the project optional models dependencies."
        ) from exc
    return torch


def _torch_device(torch: Any, config: dict[str, Any]) -> Any:
    requested = str(config.get("device", "auto"))
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchTabularTransformerRegressor:
    def __init__(self, torch: Any, nn: Any, config: dict[str, Any], n_features: int) -> None:
        from torch.nn import functional

        self.torch = torch
        self.nn = nn
        self.functional = functional
        self.n_features = n_features
        self.n_tokens = int(config["n_tokens"])
        self.d_model = int(config["d_model"])
        self.token_size = int(np.ceil(n_features / self.n_tokens))
        self.padded_features = self.token_size * self.n_tokens
        n_heads = int(config.get("n_heads", 1))
        if self.d_model % n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={n_heads}")
        ff_dim = int(config.get("ff_dim", self.d_model * 2))
        dropout = float(config.get("dropout", 0.0))
        head_layers = _parse_head_layers(config.get("head_layers", [16]))

        class Module(nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.token_projection = nn.Linear(self.token_size, self.d_model)
                inner_self.position = nn.Parameter(torch.zeros(1, self.n_tokens, self.d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=n_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=bool(config.get("norm_first", False)),
                )
                inner_self.encoder = nn.TransformerEncoder(
                    layer, num_layers=int(config["n_layers"])
                )
                head_dims = [self.d_model * 2, *head_layers, 1]
                modules = []
                for left, right in pairwise(head_dims):
                    linear = nn.Linear(left, right)
                    if right == 1:
                        nn.init.zeros_(linear.weight)
                        nn.init.zeros_(linear.bias)
                    modules.append(linear)
                    if right != 1:
                        modules.append(nn.ReLU())
                        if dropout:
                            modules.append(nn.Dropout(dropout))
                inner_self.head = nn.Sequential(*modules)

            def forward(inner_self, values: Any) -> Any:
                if self.padded_features > self.n_features:
                    values = self.functional.pad(
                        values, (0, self.padded_features - self.n_features)
                    )
                tokens = values.reshape(values.shape[0], self.n_tokens, self.token_size)
                hidden = inner_self.token_projection(tokens) + inner_self.position
                hidden = inner_self.encoder(hidden)
                pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=1)
                return inner_self.head(pooled).squeeze(-1)

        self.module = Module()


def _predict_batches(
    torch: Any,
    model: Any,
    x_values: np.ndarray,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_values), batch_size):
            batch = torch.as_tensor(
                x_values[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs)


def _fit_torch_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_validation: np.ndarray,
    candidate: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, str]:
    torch = _require_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _torch_device(torch, config)

    model_config = {
        "n_tokens": int(candidate.get("n_tokens", config.get("n_tokens", 12))),
        "d_model": int(candidate.get("d_model", config.get("d_model", 16))),
        "n_heads": int(candidate.get("n_heads", config.get("n_heads", 1))),
        "n_layers": int(candidate.get("n_layers", config.get("n_layers", 1))),
        "ff_dim": int(
            candidate.get("ff_dim", config.get("ff_dim", 0))
            or int(candidate.get("d_model", config.get("d_model", 16))) * 2
        ),
        "dropout": float(candidate.get("dropout", config.get("dropout", 0.0))),
        "head_layers": candidate.get("head_layers", config.get("head_layers", [16])),
    }
    wrapper = TorchTabularTransformerRegressor(torch, nn, model_config, n_features=x_train.shape[1])
    model = wrapper.module.to(device)

    learning_rate = float(
        candidate.get(
            "learning_rate",
            candidate.get(
                "head_learning_rate_init",
                config.get("learning_rate", config.get("head_learning_rate_init", 1e-3)),
            ),
        )
    )
    weight_decay = float(
        candidate.get(
            "weight_decay",
            candidate.get("head_alpha", config.get("weight_decay", config.get("head_alpha", 1e-4))),
        )
    )
    epochs = int(
        candidate.get(
            "training_epochs",
            config.get("training_epochs", min(int(config.get("head_max_iter", 20)), 20)),
        )
    )
    batch_size = int(config.get("head_batch_size", config.get("training_batch_size", 8192)))
    patience = int(config.get("head_n_iter_no_change", config.get("patience", 5)))
    prediction_batch_size = int(config.get("prediction_batch_size", 65536))

    train_dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
        torch.as_tensor(weights, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = np.inf
    stale_epochs = 0
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
        train_loss = float(loss.detach().cpu())
        if train_loss + 1e-7 < best_loss:
            best_loss = train_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    validation_prediction = _predict_batches(
        torch, model, x_validation, device, prediction_batch_size
    )
    return model, validation_prediction, str(device)


def _calibrate_scaled_forecast(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: pd.Series | np.ndarray | None,
    config: dict[str, Any],
) -> tuple[float, float]:
    if not config.get("calibrate_forecast", True):
        return 0.0, 1.0
    pred = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(y)
    if weights is None:
        w = np.ones(valid.sum(), dtype=np.float64)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        valid &= np.isfinite(weights_array) & (weights_array > 0)
        w = weights_array[valid]
    pred = pred[valid]
    y = y[valid]
    if len(y) < 2:
        return 0.0, 1.0
    if bool(config.get("calibration_intercept", False)):
        design = np.column_stack([np.ones(len(pred)), pred])
        root = np.sqrt(w / max(w.mean(), 1e-12))
        beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        intercept, slope = float(beta[0]), float(beta[1])
    else:
        denominator = float(np.sum(w * pred * pred))
        slope = 0.0 if denominator <= 0 else float(np.sum(w * pred * y) / denominator)
        intercept = 0.0
    slope_clip = config.get("calibration_slope_clip", [-1.0, 1.0])
    if slope_clip:
        slope = float(np.clip(slope, float(slope_clip[0]), float(slope_clip[1])))
    return intercept, slope


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
    x_train = _feature_matrix(train, features)
    x_validation = _feature_matrix(validation, features)
    x_test = _feature_matrix(test, features)
    if config.get("scale_features", True):
        x_train, x_validation, x_test = _standardize(x_train, x_validation, x_test)

    y_train_raw = train[target].to_numpy(dtype=np.float32, copy=False)
    train_target = y_train_raw.copy()
    target_clip = config.get("training_target_clip")
    if target_clip:
        train_target = np.clip(train_target, float(target_clip[0]), float(target_clip[1]))
    target_mean = float(np.nanmean(train_target))
    target_std = float(np.nanstd(train_target))
    if not np.isfinite(target_std) or target_std == 0:
        target_std = 1.0
    y_train = (train_target - target_mean) / target_std
    train_weights = _weights(
        train,
        weight_column,
        config.get("use_sample_weight", True),
        config,
    )
    validation_weights = (
        validation[weight_column]
        if config.get("weighted_validation", True) and weight_column and weight_column in validation
        else None
    )
    validation_target = validation[target].copy()
    validation_target_mean = target_mean
    validation_target_std = target_std

    candidates = _candidate_configs(config)
    progress_every = int(config.get("progress_every", 0) or 0)
    ensemble_seeds = config.get("ensemble_seeds", [random_seed])
    if isinstance(ensemble_seeds, int):
        ensemble_seeds = [ensemble_seeds]

    best_score = -np.inf
    best_members: list[tuple[Any, str]] | None = None
    best_params: dict[str, Any] = {}
    best_calibration: tuple[float, float] = (0.0, 1.0)
    for candidate_idx, candidate in enumerate(candidates, start=1):
        if progress_every and (candidate_idx == 1 or candidate_idx % progress_every == 0):
            print(
                f"  transformer_nn candidate {candidate_idx}/{len(candidates)}: {candidate}",
                flush=True,
            )
        validation_forecasts = []
        members = []
        for seed in ensemble_seeds:
            model, validation_scaled, device = _fit_torch_candidate(
                x_train=x_train,
                y_train=y_train,
                weights=train_weights,
                x_validation=x_validation,
                candidate=candidate,
                config=config,
                seed=int(seed),
            )
            forecast = validation_scaled * validation_target_std + validation_target_mean
            clip = config.get("forecast_clip")
            if clip:
                forecast = np.clip(forecast, float(clip[0]), float(clip[1]))
            validation_forecasts.append(forecast)
            members.append((model, device))

        validation_forecast = np.mean(validation_forecasts, axis=0)
        calibration_intercept, calibration_slope = _calibrate_scaled_forecast(
            validation_forecast,
            validation_target.to_numpy(dtype=np.float64, copy=False),
            validation_weights,
            config,
        )
        validation_forecast = calibration_intercept + calibration_slope * validation_forecast
        clip = config.get("forecast_clip")
        if clip:
            validation_forecast = np.clip(validation_forecast, float(clip[0]), float(clip[1]))
        validation_prediction = pd.Series(
            validation_forecast,
            index=validation.index,
            name="forecast",
        )
        score = out_of_sample_r2(
            validation_target, validation_prediction, weights=validation_weights
        )
        if best_members is None or (np.isfinite(score) and score > best_score):
            best_score = score
            best_members = members
            best_calibration = (calibration_intercept, calibration_slope)
            best_params = {
                **candidate,
                "backend": "pytorch_transformer_encoder",
                "ensemble_size": len(members),
                "n_features": len(features),
                "device": device,
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
            }
    if best_members is None:
        raise RuntimeError("No transformer_nn candidate was fit.")

    torch = _require_torch()
    test_forecasts = []
    for model, device_name in best_members:
        device = torch.device(device_name)
        forecast = _predict_batches(
            torch,
            model,
            x_test,
            device,
            int(config.get("prediction_batch_size", 65536)),
        )
        forecast = forecast * target_std + target_mean
        clip = config.get("forecast_clip")
        if clip:
            forecast = np.clip(forecast, float(clip[0]), float(clip[1]))
        test_forecasts.append(forecast)

    test_forecast = best_calibration[0] + best_calibration[1] * np.mean(test_forecasts, axis=0)
    clip = config.get("forecast_clip")
    if clip:
        test_forecast = np.clip(test_forecast, float(clip[0]), float(clip[1]))
    test_prediction = pd.Series(test_forecast, index=test.index, name="forecast")
    return ModelResult(
        model_name="transformer_nn",
        predictions=test_prediction,
        validation_metrics={
            "oos_r2": best_score,
            **{f"best_{k}": v for k, v in best_params.items()},
        },
    )
