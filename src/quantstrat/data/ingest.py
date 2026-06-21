from __future__ import annotations

import pandas as pd

from quantstrat.data.schema import PanelSchema


def validate_panel(panel: pd.DataFrame, schema: PanelSchema) -> None:
    missing = schema.required_columns.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")


def load_processed_panel(path: str, schema: PanelSchema | None = None) -> pd.DataFrame:
    schema = schema or PanelSchema()
    panel = pd.read_parquet(path)
    validate_panel(panel, schema)
    return panel.sort_values([schema.date, schema.asset_id]).reset_index(drop=True)

