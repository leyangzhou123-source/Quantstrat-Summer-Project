from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
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


@dataclass(frozen=True)
class PaperRollingSplit:
    test_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


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
    ) -> ResearchEngine:
        return cls(load_config(config_path), project_root=project_root)

    def run(self) -> EngineResult:
        features = self.configured_feature_columns()
        panel = self.load_data(features)
        features = self.feature_columns(panel)
        split = self.make_split(panel)
        train, validation, test = self.apply_split(panel, split)
        train, validation, test = self.limit_split_rows(train, validation, test)

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

    def run_paper_rolling(self) -> EngineResult:
        features = self.configured_feature_columns()
        panel = self.load_data(features)
        features = self.feature_columns(panel)

        all_predictions = []
        all_metrics = []
        for split in self.make_paper_rolling_splits():
            train, validation, test = self.apply_split(panel, split)
            train, validation, test = self.limit_split_rows(train, validation, test)
            for model_name in self.config["models"]["enabled"]:
                result = self.run_model(
                    model_name,
                    train.copy(deep=False),
                    validation.copy(deep=False),
                    test.copy(deep=False),
                    features,
                )
                prediction = self.prediction_frame(result, test)
                prediction["test_year"] = split.test_year
                all_predictions.append(prediction)
                metric = self.analyze(result, test)
                metric["test_year"] = split.test_year
                metric["train_end"] = split.train_end
                metric["validation_start"] = split.validation_start
                metric["validation_end"] = split.validation_end
                all_metrics.append(metric)

        predictions = pd.concat(all_predictions, ignore_index=True)
        annual_metrics = pd.DataFrame(all_metrics)
        pooled_metrics = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "test_year": "pooled",
                    "test_rows": len(group),
                    "test_oos_r2": out_of_sample_r2(
                        group[self.schema.target],
                        group["forecast"],
                        weights=group[self.schema.weight]
                        if self.config.get("evaluation", {}).get("weighted_oos_r2", True)
                        and self.schema.weight in group
                        else None,
                    ),
                    "validation_oos_r2": annual_metrics.loc[
                        annual_metrics["model"] == model_name, "validation_oos_r2"
                    ].mean(),
                }
                for model_name, group in predictions.groupby("model", sort=False)
            ]
        )
        metrics = pd.concat([annual_metrics, pooled_metrics], ignore_index=True)
        return EngineResult(predictions=predictions, metrics=metrics)

    def load_data(self, features: list[str] | None = None) -> pd.DataFrame:
        path = self.project_root / self.config["data"]["processed_panel_path"]
        columns = None
        if features is not None:
            required = [
                self.schema.date,
                self.schema.asset_id,
                self.schema.target,
                self.schema.weight,
                self.schema.industry,
            ]
            columns = list(dict.fromkeys(required + features))
        return load_processed_panel(str(path), self.schema, columns=columns)

    def feature_columns(self, panel: pd.DataFrame) -> list[str]:
        configured_features = self.configured_feature_columns()
        if configured_features is not None:
            return configured_features

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
            manifest_features = self.features_from_manifest(manifest, macro_cols)
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

    def configured_feature_columns(self) -> list[str] | None:
        feature_config = self.config.get("features", {})
        configured = feature_config.get("columns")
        if configured:
            return list(configured)

        feature_set = feature_config.get("feature_set", "baseline")
        manifest_path = self.project_root / self.config["data"].get(
            "manifest_path", "data/processed/model_panel_manifest.json"
        )
        if not manifest_path.exists():
            return self.features_from_panel_schema(feature_set)
        manifest = json.loads(manifest_path.read_text())
        macro_cols = [f"macro_{name}" for name in feature_config.get("macro_predictors", [])]
        if not macro_cols:
            macro_cols = manifest.get("macro_predictors", [])
        return self.features_from_manifest(manifest, macro_cols, feature_set=feature_set)

    def features_from_panel_schema(self, feature_set: str = "baseline") -> list[str] | None:
        """Infer model features when the panel parquet is present but the manifest is not."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None

        panel_path = self.project_root / self.config["data"]["processed_panel_path"]
        if not panel_path.exists():
            return None

        schema = pq.read_schema(panel_path)
        excluded = {
            self.schema.date,
            self.schema.asset_id,
            self.schema.target,
            self.schema.weight,
            self.schema.industry,
            "permco",
            "date",
            "yyyymm",
            "ret",
            "retx",
            "prc",
            "shrout",
            "vol",
            "ticker",
            "comnam",
            "shrcd",
            "exchcd",
            "siccd",
            "rf_welch_goyal",
            "ret_excess",
            "mktrf",
            "smb",
            "hml",
            "rmw",
            "cma",
            "umd",
            "rf_fama_french",
            "model",
            "forecast",
        }
        numeric_columns = [
            field.name
            for field in schema
            if (
                pa.types.is_integer(field.type)
                or pa.types.is_floating(field.type)
                or pa.types.is_boolean(field.type)
            )
            and field.name not in excluded
        ]
        if feature_set == "ols_3":
            return [column for column in ["mvel1", "bm", "mom12m"] if column in numeric_columns]
        if feature_set == "characteristics":
            return [
                column
                for column in numeric_columns
                if not column.startswith("macro_")
                and not column.startswith("sic2_")
                and "__x__" not in column
            ]
        if feature_set == "no_interactions":
            return [column for column in numeric_columns if "__x__" not in column]
        if feature_set == "baseline":
            return numeric_columns
        raise ValueError(f"Unknown features.feature_set: {feature_set!r}")

    @staticmethod
    def features_from_manifest(
        manifest: dict[str, Any],
        macro_cols: list[str],
        feature_set: str = "baseline",
    ) -> list[str]:
        characteristics = manifest.get("stock_characteristics", manifest.get("characteristics", []))
        industry = manifest.get("industry_dummies", [])
        interactions = manifest.get("macro_interactions", [])
        if feature_set == "ols_3":
            selected = [c for c in ["mvel1", "bm", "mom12m"] if c in characteristics]
            return selected
        if feature_set == "characteristics":
            return list(characteristics)
        if feature_set == "baseline":
            return list(characteristics) + list(macro_cols) + list(interactions) + list(industry)
        if feature_set == "no_interactions":
            return list(characteristics) + list(macro_cols) + list(industry)
        raise ValueError(f"Unknown features.feature_set: {feature_set!r}")

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

    def make_paper_rolling_splits(self) -> list[PaperRollingSplit]:
        split_config = self.config["splits"]
        if split_config.get("scheme") != "paper_rolling":
            raise ValueError("run_paper_rolling expects splits.scheme: paper_rolling")
        train_start = pd.Timestamp(split_config.get("train_start", "1957-03-31"))
        first_test_year = int(split_config.get("first_test_year", 1987))
        last_test_year = int(split_config.get("last_test_year", 2016))
        validation_years = int(split_config.get("validation_years", 12))
        test_years = int(split_config.get("test_years", 1))
        step_years = int(split_config.get("step_years", test_years))
        if test_years < 1:
            raise ValueError("splits.test_years must be at least 1")
        if step_years < 1:
            raise ValueError("splits.step_years must be at least 1")
        splits = []
        for test_year in range(first_test_year, last_test_year + 1, step_years):
            test_end_year = min(test_year + test_years - 1, last_test_year)
            train_end_year = test_year - validation_years - 1
            validation_start_year = test_year - validation_years
            train_end = pd.Timestamp(f"{train_end_year}-12-31")
            validation_start = pd.Timestamp(f"{validation_start_year}-01-01")
            validation_end = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_end_year}-12-31")
            splits.append(
                PaperRollingSplit(
                    test_year=test_year,
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        return splits

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

    def limit_split_rows(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        limits = self.config.get("data", {}).get("max_rows_per_split", {})
        if not limits:
            return train, validation, test
        seed = self.config["project"]["random_seed"]

        def limit(frame: pd.DataFrame, key: str) -> pd.DataFrame:
            max_rows = limits.get(key)
            if max_rows is None or len(frame) <= int(max_rows):
                return frame
            return (
                frame.sample(n=int(max_rows), random_state=seed)
                .sort_values([self.schema.date, self.schema.asset_id])
                .copy()
            )

        return limit(train, "train"), limit(validation, "validation"), limit(test, "test")

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
        model_features = self.model_feature_columns(model_config, features)
        return module.train_validate_predict(
            train=train,
            validation=validation,
            test=test,
            target=self.schema.target,
            features=model_features,
            config=model_config,
            random_seed=self.config["project"]["random_seed"],
            weight_column=self.schema.weight,
        )

    def model_feature_columns(
        self, model_config: dict[str, Any], default_features: list[str]
    ) -> list[str]:
        if model_config.get("features"):
            return list(model_config["features"])
        feature_set = model_config.get("feature_set")
        if not feature_set:
            return default_features
        manifest_path = self.project_root / self.config["data"].get(
            "manifest_path", "data/processed/model_panel_manifest.json"
        )
        if not manifest_path.exists():
            raise ValueError(f"Cannot use model feature_set without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        macro_cols = [
            f"macro_{name}" for name in self.config.get("features", {}).get("macro_predictors", [])
        ]
        if not macro_cols:
            macro_cols = manifest.get("macro_predictors", [])
        return self.features_from_manifest(manifest, macro_cols, feature_set=feature_set)

    def prediction_frame(self, result: ModelResult, test: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.schema.date: test[self.schema.date].to_numpy(),
                self.schema.asset_id: test[self.schema.asset_id].to_numpy(),
                self.schema.target: test[self.schema.target].to_numpy(),
                self.schema.weight: test[self.schema.weight].to_numpy(),
                "forecast": result.predictions.reindex(test.index).to_numpy(),
                "model": result.model_name,
            }
        )

    def analyze(self, result: ModelResult, test: pd.DataFrame) -> dict[str, Any]:
        forecast = result.predictions.reindex(test.index)
        metrics = {
            "model": result.model_name,
            "test_rows": len(test),
            "test_oos_r2": out_of_sample_r2(
                test[self.schema.target],
                forecast,
                weights=test[self.schema.weight]
                if self.config.get("evaluation", {}).get("weighted_oos_r2", True)
                else None,
            ),
        }
        metrics.update({f"validation_{k}": v for k, v in result.validation_metrics.items()})
        return metrics

    @staticmethod
    def model_module_name(model_name: str) -> str:
        module_names = {
            "ols": "OLS",
            "ols_huber": "OLSHuber",
            "ols_3": "OLS",
            "ridge": "Ridge",
            "elastic_net": "ElasticNet",
            "elastic_net_huber": "ElasticNetHuber",
            "pcr": "PCR",
            "pls": "PLS",
            "glm_huber": "GLMHuber",
            "random_forest": "RandomForest",
            "gbrt_huber": "GBRTHuber",
            "nn1": "NN1",
            "nn2": "NN2",
            "nn3": "NN3",
            "nn4": "NN4",
            "nn5": "NN5",
            "transformer_nn": "TransformerNN",
        }
        try:
            return module_names[model_name]
        except KeyError as exc:
            raise ValueError(
                f"No model module is registered for {model_name!r}. "
                "Add a file under src/quantstrat/models and register it in ResearchEngine."
            ) from exc
