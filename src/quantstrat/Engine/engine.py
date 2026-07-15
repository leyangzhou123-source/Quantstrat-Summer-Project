from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantstrat.data.ingest import load_processed_panel
from quantstrat.data.schema import PanelSchema
from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.models.base import ModelResult
from quantstrat.utils.config import load_config


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class EngineResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


class ResearchEngine:
    """Load data, delegate model work, and analyze returned forecasts."""

    def __init__(self, config: dict[str, Any], project_root: str | Path = ".") -> None:
        self.config = config
        self.project_root = Path(project_root)
        data_config = config["data"]
        self.schema = PanelSchema(
            date=data_config["date_column"],
            asset_id=data_config["asset_id_column"],
            target=data_config["target_column"],
            weight=data_config["weight_column"],
            industry=data_config["industry_column"],
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/default.yaml",
        project_root: str | Path = ".",
    ) -> "ResearchEngine":
        return cls(load_config(config_path), project_root=project_root)

    def run(self) -> EngineResult:
        panel = self.load_data()
        features = self.feature_columns(panel)
        split = self.make_split(panel)
        train, validation, test = self.apply_split(panel, split)

        model_results = [
            self.run_model(model_name, train, validation, test, features)
            for model_name in self.config["models"]["enabled"]
        ]
        predictions = pd.concat(
            [self.prediction_frame(result, test) for result in model_results],
            ignore_index=True,
        )
        metrics = pd.DataFrame([self.analyze(result, test) for result in model_results])
        return EngineResult(predictions=predictions, metrics=metrics)

    def load_data(self) -> pd.DataFrame:
        path = self.project_root / self.config["data"]["processed_panel_path"]
        return load_processed_panel(str(path), self.schema)

    def feature_columns(self, panel: pd.DataFrame) -> list[str]:
        configured = self.config.get("features", {}).get("columns")
        if configured:
            missing = set(configured).difference(panel.columns)
            if missing:
                raise ValueError(f"Configured feature columns are missing: {sorted(missing)}")
            return list(configured)

        manifest_path = self.project_root / self.config["data"].get(
            "manifest_path", "data/processed/model_panel_manifest.json"
        )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            macro_cols = [
                f"macro_{name}"
                for name in self.config.get("features", {}).get("macro_predictors", [])
            ]
            manifest_features = (
                manifest.get("characteristics", [])
                + macro_cols
                + manifest.get("industry_dummies", [])
            )
            return [column for column in manifest_features if column in panel.columns]

        excluded = {
            self.schema.date,
            self.schema.asset_id,
            self.schema.target,
            self.schema.weight,
            self.schema.industry,
            "model",
            "forecast",
        }
        return [
            column
            for column in panel.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(panel[column])
        ]

    def make_split(self, panel: pd.DataFrame) -> TimeSplit:
        split_config = self.config["splits"]
        if split_config["scheme"] != "fixed":
            raise ValueError("ResearchEngine currently expects splits.scheme: fixed")
        return TimeSplit(
            train_start=pd.Timestamp(split_config["train_start"]),
            train_end=pd.Timestamp(split_config["train_end"]),
            validation_start=pd.Timestamp(split_config["validation_start"]),
            validation_end=pd.Timestamp(split_config["validation_end"]),
            test_start=pd.Timestamp(split_config["test_start"]),
            test_end=pd.Timestamp(split_config["test_end"]),
        )

    def apply_split(
        self,
        panel: pd.DataFrame,
        split: TimeSplit,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = pd.to_datetime(panel[self.schema.date])
        train = panel[(dates >= split.train_start) & (dates <= split.train_end)].copy()
        validation = panel[
            (dates >= split.validation_start) & (dates <= split.validation_end)
        ].copy()
        test = panel[(dates >= split.test_start) & (dates <= split.test_end)].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(
                "Configured split produced an empty train, validation, or test frame. "
                "Check configs/default.yaml against the panel date range."
            )
        return train, validation, test

    def run_model(
        self,
        model_name: str,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        features: list[str],
    ) -> ModelResult:
        module = import_module(f"quantstrat.models.{self.model_module_name(model_name)}")
        model_config = self.config.get("model_params", {}).get(model_name, {})
        return module.train_validate_predict(
            train=train,
            validation=validation,
            test=test,
            target=self.schema.target,
            features=features,
            config=model_config,
            random_seed=self.config["project"]["random_seed"],
        )

    def prediction_frame(self, result: ModelResult, test: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.schema.date: test[self.schema.date].to_numpy(),
                self.schema.asset_id: test[self.schema.asset_id].to_numpy(),
                self.schema.target: test[self.schema.target].to_numpy(),
                "forecast": result.predictions.reindex(test.index).to_numpy(),
                "model": result.model_name,
            }
        )

    def analyze(self, result: ModelResult, test: pd.DataFrame) -> dict[str, Any]:
        forecast = result.predictions.reindex(test.index)
        metrics = {
            "model": result.model_name,
            "test_rows": int(len(test)),
            "test_oos_r2": out_of_sample_r2(test[self.schema.target], forecast),
        }
        metrics.update({f"validation_{k}": v for k, v in result.validation_metrics.items()})
        return metrics

    @staticmethod
    def model_module_name(model_name: str) -> str:
        module_names = {
            "ols": "OLS",
            "ridge": "Ridge",
        }
        try:
            return module_names[model_name]
        except KeyError as exc:
            raise ValueError(
                f"No model module is registered for {model_name!r}. "
                "Add a file under src/quantstrat/models and register it in ResearchEngine."
            ) from exc
