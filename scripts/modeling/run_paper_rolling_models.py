from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.Engine.engine import ResearchEngine
from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.utils.config import load_config


def annualized_sharpe(monthly_returns: pd.Series, annualization_factor: int = 12) -> float:
    values = monthly_returns.dropna().astype(float)
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((annualization_factor**0.5) * values.mean() / std)


def monthly_spearman_ic(
    predictions: pd.DataFrame,
    date_column: str,
    target_column: str,
) -> float:
    monthly_ic = []
    for _, group in predictions.groupby(date_column, sort=False):
        valid = group[["forecast", target_column]].dropna()
        if valid["forecast"].nunique() < 2 or valid[target_column].nunique() < 2:
            continue
        monthly_ic.append(valid["forecast"].corr(valid[target_column], method="spearman"))
    if not monthly_ic:
        return float("nan")
    return float(pd.Series(monthly_ic, dtype="float64").mean())


def decile_spread_sharpe(
    predictions: pd.DataFrame,
    date_column: str,
    target_column: str,
    weight_column: str,
    annualization_factor: int = 12,
    deciles: int = 10,
) -> float:
    returns = []
    for _, group in predictions.groupby(date_column, sort=False):
        valid = group[["forecast", target_column, weight_column]].dropna()
        if len(valid) < deciles or valid["forecast"].nunique() < deciles:
            continue
        ranks = valid["forecast"].rank(method="first")
        valid = valid.assign(decile=pd.qcut(ranks, deciles, labels=False, duplicates="drop") + 1)
        if valid["decile"].nunique() < deciles:
            continue
        top = valid[valid["decile"] == deciles]
        bottom = valid[valid["decile"] == 1]
        top_return = weighted_average(top[target_column], top[weight_column])
        bottom_return = weighted_average(bottom[target_column], bottom[weight_column])
        returns.append(top_return - bottom_return)
    return annualized_sharpe(pd.Series(returns, dtype="float64"), annualization_factor)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    clean = clean[clean["weight"] > 0]
    if clean.empty:
        return float("nan")
    return float((clean["value"] * clean["weight"]).sum() / clean["weight"].sum())


def final_summary(
    predictions: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    engine: ResearchEngine,
    config: dict,
) -> pd.DataFrame:
    rows = []
    annualization_factor = int(config.get("evaluation", {}).get("annualization_factor", 12))
    weighted_oos = bool(config.get("evaluation", {}).get("weighted_oos_r2", True))
    for model_name, group in predictions.groupby("model", sort=False):
        model_annual = annual_metrics[annual_metrics["model"] == model_name]
        rows.append(
            {
                "model": model_name,
                "pooled_oos_r2": out_of_sample_r2(
                    group[engine.schema.target],
                    group["forecast"],
                    weights=group[engine.schema.weight]
                    if weighted_oos and engine.schema.weight in group
                    else None,
                ),
                "mean_oos_r2": float(model_annual["test_oos_r2"].mean()),
                "monthly_spearman_ic": monthly_spearman_ic(
                    group,
                    engine.schema.date,
                    engine.schema.target,
                ),
                "decile_10_minus_1_sharpe": decile_spread_sharpe(
                    group,
                    engine.schema.date,
                    engine.schema.target,
                    engine.schema.weight,
                    annualization_factor=annualization_factor,
                    deciles=int(config.get("evaluation", {}).get("portfolio_deciles", 10)),
                ),
            }
        )
    return pd.DataFrame(rows)


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
        .filter(
            (pl.col(date_column) >= start.to_pydatetime())
            & (pl.col(date_column) <= end.to_pydatetime())
        )
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
    parser = argparse.ArgumentParser(
        description="Run paper-style rolling models with incremental output."
    )
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
    predictions_path = out_dir / f"{args.out_prefix}_predictions.parquet"

    all_predictions = []
    all_metrics = []
    data_path = ROOT / config["data"]["processed_panel_path"]
    asset_sample_fraction = config["data"].get("asset_sample_fraction")
    asset_sample_seed = int(
        config["data"].get("asset_sample_seed", config["project"]["random_seed"])
    )
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
            if split.test_start.year == split.test_end.year:
                test_label = str(split.test_start.year)
            else:
                test_label = f"{split.test_start.year}-{split.test_end.year}"
            print(f"Fitting {model_name} for test year {test_label}", flush=True)
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

    predictions = pd.concat(all_predictions, ignore_index=True)
    annual_metrics = pd.DataFrame(all_metrics)
    predictions.to_parquet(predictions_path, index=False)
    summary = final_summary(predictions, annual_metrics, engine, config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
