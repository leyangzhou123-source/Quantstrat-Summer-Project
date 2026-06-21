from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelSchema:
    date: str = "date"
    asset_id: str = "permno"
    target: str = "ret_excess_lead1"
    weight: str = "market_equity"
    industry: str = "sic2"

    @property
    def required_columns(self) -> set[str]:
        return {self.date, self.asset_id, self.target}

