from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_year_splits(
    dates: pd.Series,
    train_years: int,
    validation_years: int,
    test_years: int = 1,
) -> list[TimeSplit]:
    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values().reset_index(drop=True)
    splits: list[TimeSplit] = []
    first_year = unique_dates.dt.year.min()
    last_year = unique_dates.dt.year.max()

    for test_year in range(first_year + train_years + validation_years, last_year + 1, test_years):
        train_start = pd.Timestamp(year=test_year - validation_years - train_years, month=1, day=1)
        train_end = pd.Timestamp(year=test_year - validation_years, month=1, day=1) - pd.offsets.Day(1)
        validation_start = train_end + pd.offsets.Day(1)
        validation_end = pd.Timestamp(year=test_year, month=1, day=1) - pd.offsets.Day(1)
        test_start = validation_end + pd.offsets.Day(1)
        test_end = pd.Timestamp(year=test_year + test_years, month=1, day=1) - pd.offsets.Day(1)
        if test_start <= unique_dates.max():
            splits.append(
                TimeSplit(
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
    return splits

