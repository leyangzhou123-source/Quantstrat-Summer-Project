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
    percentile = grouped[characteristic_columns].rank(pct=True)
    ranked[characteristic_columns] = low + (high - low) * percentile
    return ranked

