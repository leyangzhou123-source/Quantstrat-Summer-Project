from __future__ import annotations

import argparse
import sys
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "feature_engineering"))

from run_feature_engineering_nn5_experiment import (
    decile_spread_sharpe,
    feature_ic_importance,
    load_split_frame,
    monthly_spearman_ic,
)

from quantstrat.Engine.engine import ResearchEngine
from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.utils.config import load_config

OUT_DIR = ROOT / "reports" / "feature_engineering"


def _sign(value: float) -> int:
    return -1 if np.isfinite(value) and value < 0 else 1


def _split_label(split: Any) -> str:
    if split.test_start.year == split.test_end.year:
        return str(split.test_start.year)
    return f"{split.test_start.year}-{split.test_end.year}"


def _read_selection_ranks(path: Path, min_months: int) -> pd.DataFrame:
    ranks = pd.read_csv(path)
    ranks = ranks[ranks["selection_months"] >= min_months].copy()
    ranks = ranks[np.isfinite(ranks["selection_abs_ic"])].copy()
    ranks["sign"] = ranks["selection_mean_ic"].map(_sign)
    return ranks.sort_values("selection_abs_ic", ascending=False)


def _build_rank_feature_sets(ranks: pd.DataFrame) -> dict[str, list[str]]:
    by_category = {
        category: group.sort_values("selection_abs_ic", ascending=False)
        for category, group in ranks.groupby("category")
    }

    def take(
        category: str, n: int, low: float | None = None, high: float | None = None
    ) -> list[str]:
        group = by_category.get(category, pd.DataFrame())
        if low is not None:
            group = group[group["selection_abs_ic"] >= low]
        if high is not None:
            group = group[group["selection_abs_ic"] <= high]
        return group.head(n)["feature"].tolist()

    top = ranks.head(60)["feature"].tolist()
    high_core_32 = top[:32]
    high_core_46 = top[:46]
    no_macro_42 = [feature for feature in top if not feature.startswith("macro_")][:42]
    no_industry_42 = [feature for feature in top if not feature.startswith("sic2_")][:42]

    balanced_46 = (
        take("risk_liquidity", 7)
        + take("value", 6)
        + take("profitability_quality", 6)
        + take("momentum", 5)
        + take("accounting_balance", 6)
        + take("intangibles_attention", 5)
        + take("growth_operations", 4)
        + take("macro", 4)
        + take("industry", 3)
    )
    small_ic_blend_46 = (
        take("risk_liquidity", 5)
        + take("value", 5)
        + take("profitability_quality", 5)
        + take("momentum", 4)
        + take("accounting_balance", 5)
        + take("intangibles_attention", 5)
        + take("growth_operations", 5, low=0.005, high=0.025)
        + take("industry", 8, low=0.005, high=0.02)
        + take("macro", 4)
    )
    anti_crowded_38 = (
        take("risk_liquidity", 4)
        + take("value", 6)
        + take("profitability_quality", 6)
        + take("momentum", 5)
        + take("accounting_balance", 6)
        + take("intangibles_attention", 5)
        + take("growth_operations", 3, low=0.005)
        + take("macro", 3)
    )

    raw_sets = {
        "rank_signed_top32": high_core_32,
        "rank_signed_top46": high_core_46,
        "rank_signed_balanced46": balanced_46,
        "rank_signed_small_ic_blend46": small_ic_blend_46,
        "rank_signed_no_macro42": no_macro_42,
        "rank_signed_no_industry42": no_industry_42,
        "rank_signed_anti_crowded38": anti_crowded_38,
    }
    out: dict[str, list[str]] = {}
    for name, features in raw_sets.items():
        deduped = list(dict.fromkeys(features))
        out[name] = deduped[:49]
    return out


def _feature_categories(ranks: pd.DataFrame, features: list[str]) -> dict[str, str]:
    mapping = ranks.drop_duplicates("feature").set_index("feature")["category"].to_dict()
    return {feature: mapping.get(feature, "unknown") for feature in features}


def _load_oos_frame(
    data_path: Path,
    columns: list[str],
    date_column: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    return (
        pl.scan_parquet(data_path)
        .select(columns)
        .filter(
            (pl.col(date_column) >= pd.Timestamp(f"{start_year}-01-01"))
            & (pl.col(date_column) <= pd.Timestamp(f"{end_year}-12-31"))
        )
        .collect()
        .to_pandas()
    )


def _apply_signs(frame: pd.DataFrame, features: list[str], signs: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    for feature in features:
        if signs.get(feature, 1) < 0:
            out[feature] = -out[feature]
    return out


def _rank_weighted_composite(
    frame: pd.DataFrame,
    features: list[str],
    signs: dict[str, int],
    weights: dict[str, float],
) -> pd.Series:
    score = np.zeros(len(frame), dtype=np.float64)
    total_weight = 0.0
    for feature in features:
        values = frame[feature].to_numpy(dtype=np.float64, copy=False)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        weight = float(weights.get(feature, 0.0))
        score += signs.get(feature, 1) * weight * values
        total_weight += abs(weight)
    if total_weight > 0:
        score /= total_weight
    return pd.Series(score, index=frame.index, name="forecast")


def _summarize_predictions(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    engine: ResearchEngine,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for name, group in predictions.groupby("model", sort=False):
        model_metrics = metrics[metrics["model"] == name]
        rows.append(
            {
                "model": name,
                "n_features": int(group["n_features"].iloc[0]),
                "pooled_oos_r2": out_of_sample_r2(group[engine.schema.target], group["forecast"]),
                "mean_oos_r2": float(model_metrics["test_oos_r2"].mean())
                if not model_metrics.empty
                else np.nan,
                "monthly_spearman_ic": monthly_spearman_ic(
                    group, engine.schema.date, engine.schema.target
                ),
                "decile_10_minus_1_sharpe": decile_spread_sharpe(
                    group,
                    engine.schema.date,
                    engine.schema.target,
                    engine.schema.weight,
                    annualization_factor=int(config["evaluation"]["annualization_factor"]),
                    deciles=int(config["evaluation"]["portfolio_deciles"]),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("monthly_spearman_ic", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-optimized signed-feature NN5 research.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_like_nn5_backprop_no_interactions_rankfix.yaml"),
    )
    parser.add_argument(
        "--selection-ranks",
        type=Path,
        default=Path(
            "reports/feature_engineering/nn5_feature_selected_1987_1996_to_1997_2016_selection_ranks.csv"
        ),
    )
    parser.add_argument("--out-prefix", default="nn5_rank_optimized_feature_research")
    parser.add_argument("--oos-start-year", type=int, default=1997)
    parser.add_argument("--oos-end-year", type=int, default=2016)
    parser.add_argument("--min-selection-months", type=int, default=60)
    parser.add_argument("--run-nn5", action="store_true")
    parser.add_argument("--groups", nargs="*", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    config["models"]["enabled"] = ["nn5"]
    config.setdefault("splits", {})
    config["splits"]["first_test_year"] = int(args.oos_start_year)
    config["splits"]["last_test_year"] = int(args.oos_end_year)
    config.setdefault("model_params", {}).setdefault("nn5", {})
    config["model_params"]["nn5"]["validation_selection_metric"] = "spearman_ic"
    config["model_params"]["nn5"]["validation_date_column"] = config["data"]["date_column"]
    config["model_params"]["nn5"]["zero_output_init"] = False
    config["model_params"]["nn5"].setdefault("output_init_scale", 0.01)

    selection_path = (
        args.selection_ranks if args.selection_ranks.is_absolute() else ROOT / args.selection_ranks
    )
    ranks = _read_selection_ranks(selection_path, args.min_selection_months)
    feature_sets = _build_rank_feature_sets(ranks)
    if args.groups:
        missing = sorted(set(args.groups) - set(feature_sets))
        if missing:
            raise ValueError(f"Unknown groups: {missing}")
        feature_sets = {name: feature_sets[name] for name in args.groups}

    signs = ranks.drop_duplicates("feature").set_index("feature")["sign"].to_dict()
    weights = ranks.drop_duplicates("feature").set_index("feature")["selection_abs_ic"].to_dict()
    engine = ResearchEngine(config, project_root=ROOT)
    data_path = ROOT / config["data"]["processed_panel_path"]

    all_features = sorted(set(chain.from_iterable(feature_sets.values())))
    columns = list(
        dict.fromkeys(
            [
                engine.schema.date,
                engine.schema.asset_id,
                engine.schema.target,
                engine.schema.weight,
                engine.schema.industry,
            ]
            + all_features
        )
    )
    oos = _load_oos_frame(
        data_path,
        columns,
        engine.schema.date,
        args.oos_start_year,
        args.oos_end_year,
    )

    design_rows = []
    composite_predictions = []
    composite_metrics = []
    importance_frames = []
    for name, features in feature_sets.items():
        category_map = _feature_categories(ranks, features)
        for feature in features:
            row = ranks[ranks["feature"] == feature].head(1)
            design_rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "category": category_map[feature],
                    "selection_mean_ic": float(row["selection_mean_ic"].iloc[0]),
                    "selection_abs_ic": float(row["selection_abs_ic"].iloc[0]),
                    "sign_multiplier": int(signs[feature]),
                }
            )
        prediction = oos[
            [
                engine.schema.date,
                engine.schema.asset_id,
                engine.schema.target,
                engine.schema.weight,
            ]
            + features
        ].copy()
        prediction["forecast"] = _rank_weighted_composite(prediction, features, signs, weights)
        prediction["model"] = f"{name}_ic_weighted_composite"
        prediction["n_features"] = len(features)
        composite_predictions.append(
            prediction[
                [
                    engine.schema.date,
                    engine.schema.asset_id,
                    engine.schema.target,
                    engine.schema.weight,
                    "forecast",
                    "model",
                    "n_features",
                ]
            ]
        )
        composite_metrics.append(
            {
                "model": f"{name}_ic_weighted_composite",
                "test_oos_r2": out_of_sample_r2(
                    prediction[engine.schema.target], prediction["forecast"]
                ),
            }
        )
        signed_oos = _apply_signs(
            prediction[[engine.schema.date, engine.schema.target] + features], features, signs
        )
        importance_frames.append(
            feature_ic_importance(
                signed_oos,
                features,
                category_map,
                name,
                engine.schema.date,
                engine.schema.target,
            )
        )

    design = pd.DataFrame(design_rows).sort_values(
        ["model", "selection_abs_ic"], ascending=[True, False]
    )
    design.to_csv(OUT_DIR / f"{args.out_prefix}_design.csv", index=False)
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(OUT_DIR / f"{args.out_prefix}_signed_feature_oos_ic.csv", index=False)
    composite_predictions_df = pd.concat(composite_predictions, ignore_index=True)
    composite_metrics_df = pd.DataFrame(composite_metrics)
    composite_summary = _summarize_predictions(
        composite_predictions_df, composite_metrics_df, engine, config
    )
    composite_summary.to_csv(OUT_DIR / f"{args.out_prefix}_composite_summary.csv", index=False)
    composite_predictions_df.to_parquet(
        OUT_DIR / f"{args.out_prefix}_composite_predictions.parquet", index=False
    )

    print("\nRank-composite summary", flush=True)
    print(composite_summary.to_string(index=False), flush=True)
    print("\nTop signed OOS feature IC rows", flush=True)
    print(importance.nsmallest(30, "importance_rank").to_string(index=False), flush=True)

    if not args.run_nn5:
        return

    nn_predictions: list[pd.DataFrame] = []
    nn_metrics: list[dict[str, Any]] = []
    selected_groups = feature_sets
    for feature_set_name, features in selected_groups.items():
        print(f"\n=== {feature_set_name}: {len(features)} signed features ===", flush=True)
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
            print(
                f"Fitting rank-optimized NN5 feature set {feature_set_name} for test years {_split_label(split)}",
                flush=True,
            )
            train = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.train_start,
                split.train_end,
                features,
            )
            validation = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.validation_start,
                split.validation_end,
                features,
            )
            test = load_split_frame(
                data_path,
                required_columns,
                engine.schema.date,
                engine.schema.asset_id,
                split.test_start,
                split.test_end,
                features,
            )
            train, validation, test = engine.limit_split_rows(train, validation, test)
            train = _apply_signs(train, features, signs)
            validation = _apply_signs(validation, features, signs)
            test = _apply_signs(test, features, signs)
            result = engine.run_model(
                "nn5",
                train.copy(deep=False),
                validation.copy(deep=False),
                test.copy(deep=False),
                features,
            )
            prediction = engine.prediction_frame(result, test)
            prediction["model"] = feature_set_name
            prediction["test_year"] = split.test_year
            prediction["test_window"] = _split_label(split)
            prediction["n_features"] = len(features)
            nn_predictions.append(prediction)
            metric = engine.analyze(result, test)
            metric["model"] = feature_set_name
            metric["test_year"] = split.test_year
            metric["test_window"] = _split_label(split)
            metric["n_features"] = len(features)
            nn_metrics.append(metric)

            nn_predictions_df = pd.concat(nn_predictions, ignore_index=True)
            nn_metrics_df = pd.DataFrame(nn_metrics)
            nn_summary = _summarize_predictions(nn_predictions_df, nn_metrics_df, engine, config)
            nn_metrics_df.to_csv(
                OUT_DIR / f"{args.out_prefix}_nn5_rolling_metrics.csv", index=False
            )
            nn_summary.to_csv(OUT_DIR / f"{args.out_prefix}_nn5_summary.csv", index=False)
            nn_predictions_df.to_parquet(
                OUT_DIR / f"{args.out_prefix}_nn5_predictions.parquet", index=False
            )

    print("\nRank-optimized NN5 summary", flush=True)
    print(nn_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
