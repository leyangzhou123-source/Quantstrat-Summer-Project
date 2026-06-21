from __future__ import annotations

import pandas as pd


def assign_forecast_deciles(
    panel: pd.DataFrame,
    date_column: str,
    forecast_column: str = "forecast",
    deciles: int = 10,
) -> pd.Series:
    def _bucket(group: pd.Series) -> pd.Series:
        return pd.qcut(group.rank(method="first"), q=deciles, labels=False) + 1

    return panel.groupby(date_column, group_keys=False)[forecast_column].apply(_bucket)


def value_weighted_returns(
    panel: pd.DataFrame,
    date_column: str,
    bucket_column: str,
    return_column: str,
    weight_column: str,
) -> pd.DataFrame:
    weighted = panel.copy()
    weighted["_weighted_return"] = weighted[return_column] * weighted[weight_column]
    grouped = weighted.groupby([date_column, bucket_column])
    result = grouped["_weighted_return"].sum() / grouped[weight_column].sum()
    return result.rename("return").reset_index()

