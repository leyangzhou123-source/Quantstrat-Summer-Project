from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ArrayLike = np.ndarray | sparse.spmatrix


@dataclass
class SklearnModelAdapter:
    name: str
    estimator: RegressorMixin
    dense_required: bool = False

    def fit_array(self, x: ArrayLike, y: np.ndarray) -> "SklearnModelAdapter":
        self.estimator.fit(self._as_model_input(x), y)
        return self

    def predict_array(self, x: ArrayLike) -> np.ndarray:
        pred = self.estimator.predict(self._as_model_input(x))
        return np.asarray(pred).reshape(-1)

    def _as_model_input(self, x: ArrayLike) -> ArrayLike:
        if self.dense_required and sparse.issparse(x):
            return x.toarray()
        return x


def make_paper_model(name: str, random_state: int = 42) -> SklearnModelAdapter:
    """Create one model family from Gu, Kelly, and Xiu's comparison set."""
    if name == "ols":
        return SklearnModelAdapter(name, LinearRegression())
    if name == "ols_3":
        return SklearnModelAdapter(name, LinearRegression())
    if name == "elastic_net_huber":
        return SklearnModelAdapter(
            name,
            make_pipeline(
                StandardScaler(with_mean=False),
                SGDRegressor(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=1e-4,
                    l1_ratio=0.5,
                    max_iter=2000,
                    tol=1e-4,
                    random_state=random_state,
                ),
            ),
        )
    if name == "pcr":
        return SklearnModelAdapter(
            name,
            make_pipeline(
                StandardScaler(with_mean=False),
                TruncatedSVD(n_components=50, random_state=random_state),
                Ridge(alpha=1.0),
            ),
        )
    if name == "pls":
        return SklearnModelAdapter(
            name,
            make_pipeline(StandardScaler(), PLSRegression(n_components=15)),
            dense_required=True,
        )
    if name == "random_forest":
        return SklearnModelAdapter(
            name,
            RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=100,
                n_jobs=-1,
                random_state=random_state,
            ),
            dense_required=True,
        )
    if name == "gbrt_huber":
        return SklearnModelAdapter(
            name,
            GradientBoostingRegressor(
                loss="huber",
                n_estimators=500,
                learning_rate=0.01,
                max_depth=2,
                min_samples_leaf=100,
                random_state=random_state,
            ),
            dense_required=True,
        )
    if name.startswith("nn") and name[2:].isdigit():
        depth = int(name[2:])
        hidden = tuple([32] * depth)
        return SklearnModelAdapter(
            name,
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    batch_size=4096,
                    max_iter=100,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
            dense_required=True,
        )
    if name == "transformer_nn":
        raise NotImplementedError(
            "Transformer NN is an extension architecture. Add a PyTorch adapter before running it."
        )
    raise KeyError(f"Unknown paper model: {name}")


PAPER_MODEL_NAMES = [
    "ols",
    "ols_3",
    "elastic_net_huber",
    "pcr",
    "pls",
    "random_forest",
    "gbrt_huber",
    "nn1",
    "nn2",
    "nn3",
    "nn4",
    "nn5",
]
