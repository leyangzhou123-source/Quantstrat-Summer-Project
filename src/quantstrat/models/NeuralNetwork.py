from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.neural_network import MLPRegressor

from quantstrat.models.base import ModelResult
from quantstrat.models.jax_neural_network import train_validate_predict_jax_depth
from quantstrat.models.sklearn_training import train_validate_predict_sklearn


def train_validate_predict_depth(
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
    if config.get("backend", "sklearn") == "jax":
        return train_validate_predict_jax_depth(
            depth=depth,
            train=train,
            validation=validation,
            test=test,
            target=target,
            features=features,
            config=config,
            random_seed=random_seed,
            weight_column=weight_column,
        )

    def model(candidate: dict[str, Any]):
        configured = candidate.get("layer_widths", config.get("layer_widths"))
        if configured:
            hidden_layer_sizes = tuple(int(width) for width in configured[:depth])
        else:
            width = int(candidate.get("width", config.get("width", 32)))
            hidden_layer_sizes = tuple(max(1, width // (2**layer)) for layer in range(depth))
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=config.get("activation", "relu"),
            alpha=candidate.get("alpha", config.get("alpha", 1e-4)),
            learning_rate_init=candidate.get(
                "learning_rate_init", config.get("learning_rate_init", 1e-3)
            ),
            batch_size=config.get("batch_size", 4096),
            max_iter=config.get("max_iter", 100),
            early_stopping=config.get("early_stopping", True),
            validation_fraction=config.get("validation_fraction", 0.1),
            n_iter_no_change=config.get("n_iter_no_change", 10),
            random_state=random_seed,
        )

    return train_validate_predict_sklearn(
        model_name=f"nn{depth}",
        estimator=model,
        train=train,
        validation=validation,
        test=test,
        target=target,
        features=features,
        config=config,
        weight_column=weight_column,
    )
