from __future__ import annotations

import pandas as pd

from quantstrat.evaluation.metrics import out_of_sample_r2


def test_out_of_sample_r2_improves_on_zero_benchmark() -> None:
    actual = pd.Series([1.0, -1.0, 2.0])
    forecast = pd.Series([0.5, -0.5, 1.5])

    assert out_of_sample_r2(actual, forecast) > 0

