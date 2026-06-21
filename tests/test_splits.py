from __future__ import annotations

import pandas as pd

from quantstrat.models.splits import rolling_year_splits


def test_rolling_year_splits_preserve_temporal_order() -> None:
    dates = pd.date_range("1957-03-31", "1990-12-31", freq="ME").to_series()

    split = rolling_year_splits(dates, train_years=18, validation_years=12, test_years=1)[0]

    assert split.train_end < split.validation_start
    assert split.validation_end < split.test_start

