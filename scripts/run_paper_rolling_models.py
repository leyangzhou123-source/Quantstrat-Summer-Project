from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.engine import ResearchEngine
from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.utils.config import load_config


def load_split_frame(
    path: Path,
    columns: list[str],
    date_column: str,
    asset_id_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feature_columns: list[str],
    asset_sample_fraction: float | None = None,
    asset_sample_seed: int = 42,
) -> pd.DataFrame:
    expressions = []
    feature_set = set(feature_columns)
    for column in columns:
        expression = pl.col(column)
        if column in feature_set:
            expression = expression.cast(pl.Float32)
        expressions.append(expression)
    frame = (
        pl.scan_parquet(path)
        .filter((pl.col(date_column) >= start.to_pydatetime()) & (pl.col(date_column) <= end.to_pydatetime()))
        .select(expressions)
        .sort(date_column)
        .collect()
        .to_pandas()
        .reset_index(drop=True)
    )
    if asset_sample_fraction is not None and asset_sample_fraction < 1.0:
        hashed = pd.util.hash_pandas_object(
            frame[asset_id_column].astype("int64") + int(asset_sample_seed),
            index=False,
        )
        keep = (hashed / float(2**64 - 1)) < float(asset_sample_fraction)
        frame = frame.loc[keep].reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-style rolling models with incremental output.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out-prefix", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    config["models"]["enabled"] = args.models
    engine = ResearchEngine(config, project_root=ROOT)

    features = engine.configured_feature_columns()
    if features is None:
        panel = engine.load_data(features)
        features = engine.feature_columns(panel)
    else:
        panel = None
    out_dir = ROOT / "reports" / "model_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.out_prefix}_metrics.csv"
    predictions_path = out_dir / f"{args.out_prefix}_predictions.parquet"

    all_predictions = []
    all_metrics = []
    data_path = ROOT / config["data"]["processed_panel_path"]
    asset_sample_fraction = config["data"].get("asset_sample_fraction")
    asset_sample_seed = int(config["data"].get("asset_sample_seed", config["project"]["random_seed"]))
    required_columns = list(
        dict.fromkeys(
            [
                engine.schema.date,
                engine.schema.asset_id,
                engine.schema.target,
                engine.schema.weight,
                engine.schema.industry,
            ]
            + features
        )
    )
    for split in engine.make_paper_rolling_splits():
        if panel is None:
            train = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.train_start,
                split.train_end,
                features,
                asset_sample_fraction=asset_sample_fraction,
                asset_sample_seed=asset_sample_seed,
            )
            validation = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.validation_start,
                split.validation_end,
                features,
                asset_sample_fraction=asset_sample_fraction,
                asset_sample_seed=asset_sample_seed,
            )
            test = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.test_start,
                split.test_end,
                features,
                asset_sample_fraction=asset_sample_fraction,
                asset_sample_seed=asset_sample_seed,
            )
        else:
            train, validation, test = engine.apply_split(panel, split)
        train, validation, test = engine.limit_split_rows(train, validation, test)
        for model_name in args.models:
            print(f"Fitting {model_name} for test year {split.test_year}", flush=True)
            model_train = train.copy(deep=False)
            model_validation = validation.copy(deep=False)
            model_test = test.copy(deep=False)
            result = engine.run_model(
                model_name,
                model_train,
                model_validation,
                model_test,
                features,
            )
            prediction = engine.prediction_frame(result, test)
            prediction["test_year"] = split.test_year
            metric = engine.analyze(result, test)
            metric["test_year"] = split.test_year
            metric["train_end"] = split.train_end
            metric["validation_start"] = split.validation_start
            metric["validation_end"] = split.validation_end
            all_predictions.append(prediction)
            all_metrics.append(metric)
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

    predictions = pd.concat(all_predictions, ignore_index=True)
    annual_metrics = pd.DataFrame(all_metrics)
    pooled = pd.DataFrame(
        [
            {
                "model": model_name,
                "test_rows": int(len(group)),
                "test_oos_r2": out_of_sample_r2(
                    group[engine.schema.target],
                    group["forecast"],
                    weights=group[engine.schema.weight]
                    if config.get("evaluation", {}).get("weighted_oos_r2", True)
                    and engine.schema.weight in group
                    else None,
                ),
                "validation_oos_r2": annual_metrics.loc[
                    annual_metrics["model"] == model_name, "validation_oos_r2"
                ].mean(),
                "test_year": "pooled",
                "train_end": pd.NaT,
                "validation_start": pd.NaT,
                "validation_end": pd.NaT,
            }
            for model_name, group in predictions.groupby("model", sort=False)
        ]
    )
    metrics = pd.concat([annual_metrics, pooled], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    print(metrics.tail(len(args.models)).to_string(index=False))


if __name__ == "__main__":
    main()
