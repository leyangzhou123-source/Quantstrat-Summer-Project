from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class ModelAdapter(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, target: str, features: list[str]) -> "ModelAdapter":
        ...

    def predict(self, panel: pd.DataFrame, features: list[str]) -> pd.Series:
        ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    supports_huber: bool = False
    captures_interactions: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "ols": ModelSpec("ols", "linear"),
    "ols_3": ModelSpec("ols_3", "linear"),
    "elastic_net_huber": ModelSpec("elastic_net_huber", "penalized_linear", supports_huber=True),
    "pcr": ModelSpec("pcr", "dimension_reduction"),
    "pls": ModelSpec("pls", "dimension_reduction"),
    "random_forest": ModelSpec("random_forest", "tree", captures_interactions=True),
    "gbrt_huber": ModelSpec(
        "gbrt_huber",
        "boosted_tree",
        supports_huber=True,
        captures_interactions=True,
    ),
    "neural_net": ModelSpec("neural_net", "neural_network", captures_interactions=True),
}

