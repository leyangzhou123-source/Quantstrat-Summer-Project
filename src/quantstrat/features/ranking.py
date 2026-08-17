from __future__ import annotations

import pandas as pd


def rank_characteristics(
    panel: pd.DataFrame,
    date_column: str,
    characteristic_columns: list[str],
    low: float = -1.0,
    high: float = 1.0,
) -> pd.DataFrame:
    ranked = panel.copy()
    grouped = ranked.groupby(date_column, group_keys=False)
    ranks = grouped[characteristic_columns].rank(method="average", na_option="keep")
    counts = grouped[characteristic_columns].transform("count")
    scaled = low + (high - low) * (ranks - 1.0) / (counts - 1.0)
    ranked[characteristic_columns] = scaled.where(counts > 1)
    return ranked
