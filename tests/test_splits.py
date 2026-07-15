from __future__ import annotations

import pandas as pd

from quantstrat.engine import ResearchEngine


def test_engine_split_preserves_temporal_order() -> None:
    panel = pd.DataFrame({"month": pd.date_range("1957-03-31", "1990-12-31", freq="ME")})
    config = {
        "project": {"random_seed": 42},
        "data": {
            "date_column": "month",
            "asset_id_column": "permno",
            "target_column": "ret_excess_lead1",
            "weight_column": "market_equity",
            "industry_column": "sic2",
            "processed_panel_path": "unused.parquet",
        },
        "splits": {
            "scheme": "fixed",
            "train_start": "1957-03-31",
            "train_end": "1974-12-31",
            "validation_start": "1975-01-01",
            "validation_end": "1986-12-31",
            "test_start": "1987-01-01",
            "test_end": "1990-12-31",
        },
        "models": {"enabled": []},
    }
    split = ResearchEngine(config).make_split(panel)

    assert split.train_end < split.validation_start
    assert split.validation_end < split.test_start
