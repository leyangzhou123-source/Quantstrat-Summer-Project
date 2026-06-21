from __future__ import annotations

import pandas as pd


def add_macro_interactions(
    panel: pd.DataFrame,
    characteristic_columns: list[str],
    macro_columns: list[str],
) -> pd.DataFrame:
    expanded = panel.copy()
    for char_col in characteristic_columns:
        for macro_col in macro_columns:
            expanded[f"{char_col}__x__{macro_col}"] = expanded[char_col] * expanded[macro_col]
    return expanded

