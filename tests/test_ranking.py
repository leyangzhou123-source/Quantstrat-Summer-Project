from __future__ import annotations

import pandas as pd

from quantstrat.features.ranking import rank_characteristics


def test_rank_characteristics_maps_cross_section_to_full_range() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.to_datetime(["2000-01-31"] * 3 + ["2000-02-29"] * 3),
            "permno": [1, 2, 3, 1, 2, 3],
            "size": [10.0, 20.0, 30.0, 5.0, 5.0, 15.0],
        }
    )

    ranked = rank_characteristics(panel, "month", ["size"])

    assert ranked.loc[:2, "size"].tolist() == [-1.0, 0.0, 1.0]
    assert ranked.loc[3:4, "size"].tolist() == [-0.5, -0.5]
    assert ranked.loc[5, "size"] == 1.0
