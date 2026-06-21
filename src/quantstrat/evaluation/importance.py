from __future__ import annotations

import pandas as pd


def prediction_drop_importance(
    panel: pd.DataFrame,
    feature: str,
    baseline_r2: float,
    recomputed_r2: float,
) -> dict[str, float | str]:
    return {
        "feature": feature,
        "baseline_r2": baseline_r2,
        "ablated_r2": recomputed_r2,
        "importance": baseline_r2 - recomputed_r2,
    }

