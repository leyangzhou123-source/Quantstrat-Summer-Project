from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


def add_macro_interactions(
    panel: pd.DataFrame,
    characteristic_columns: list[str],
    macro_columns: list[str],
) -> pd.DataFrame:
    expanded = panel.copy()
    interaction_blocks = {}
    for char_col in characteristic_columns:
        for macro_col in macro_columns:
            interaction_blocks[f"{char_col}__x__{macro_col}"] = expanded[char_col] * expanded[macro_col]
    if interaction_blocks:
        expanded = pd.concat([expanded, pd.DataFrame(interaction_blocks, index=expanded.index)], axis=1)
    return expanded


@dataclass(frozen=True)
class SparseInteractionMatrix:
    matrix: sparse.csr_matrix
    feature_names: list[str]


def industry_characteristic_feature_names(
    characteristic_columns: list[str],
    industry_dummy_columns: list[str],
) -> list[str]:
    return [
        f"{char_col}__x__{industry_col}"
        for industry_col in industry_dummy_columns
        for char_col in characteristic_columns
    ]


def build_sparse_industry_characteristic_interactions(
    panel: pd.DataFrame,
    characteristic_columns: list[str],
    industry_dummy_columns: list[str],
    dtype: np.dtype = np.float32,
) -> SparseInteractionMatrix:
    """Build characteristic-by-industry interactions without dense expansion.

    Each industry dummy is zero for most rows, so the interaction block is a
    natural sparse matrix. This is the right representation for the 4M-row
    paper panel because a dense DataFrame would be unnecessarily huge.
    """
    if not characteristic_columns:
        raise ValueError("characteristic_columns cannot be empty")
    if not industry_dummy_columns:
        raise ValueError("industry_dummy_columns cannot be empty")

    char_matrix = sparse.csr_matrix(
        panel[characteristic_columns].fillna(0.0).to_numpy(dtype=dtype, copy=False)
    )
    blocks = []
    for industry_col in industry_dummy_columns:
        industry_vector = sparse.csr_matrix(
            panel[industry_col].fillna(0.0).to_numpy(dtype=dtype, copy=False)
        ).T
        blocks.append(char_matrix.multiply(industry_vector))
    matrix = sparse.hstack(blocks, format="csr", dtype=dtype)
    return SparseInteractionMatrix(
        matrix=matrix,
        feature_names=industry_characteristic_feature_names(
            characteristic_columns=characteristic_columns,
            industry_dummy_columns=industry_dummy_columns,
        ),
    )


def combine_dense_and_sparse_features(
    panel: pd.DataFrame,
    dense_columns: list[str],
    sparse_block: sparse.spmatrix | None = None,
    dtype: np.dtype = np.float32,
) -> sparse.csr_matrix:
    dense = sparse.csr_matrix(panel[dense_columns].fillna(0.0).to_numpy(dtype=dtype, copy=False))
    if sparse_block is None:
        return dense
    return sparse.hstack([dense, sparse_block], format="csr", dtype=dtype)
