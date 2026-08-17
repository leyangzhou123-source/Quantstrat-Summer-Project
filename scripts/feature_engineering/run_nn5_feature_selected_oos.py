from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "feature_engineering"))

import run_feature_engineering_nn5_experiment as base
from run_feature_engineering_nn5_expanded_search import (
    ACCOUNTING_BALANCE,
    GROWTH_OPERATIONS,
    INTANGIBLE_ATTENTION,
    MACRO_ALL,
)

OUT_DIR = ROOT / "reports" / "feature_engineering"


STATIC_CANDIDATES = {
    "risk_liquidity": [
        "baspread",
        "idiovol",
        "retvol",
        "maxret",
        "ill",
        "beta",
        "betasq",
        "turn",
        "dolvol",
        "zerotrade",
        "std_dolvol",
        "std_turn",
    ],
    "value": ["mvel1", "bm", "bm_ia", "ep", "cfp", "cfp_ia", "ps", "dy"],
    "profitability_quality": [
        "operprof",
        "gma",
        "roic",
        "roaq",
        "roeq",
        "cashdebt",
        "cash",
        "lev",
    ],
    "momentum": ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom"],
    "accounting_balance": ACCOUNTING_BALANCE,
    "intangibles_attention": INTANGIBLE_ATTENTION,
    "growth_operations": GROWTH_OPERATIONS,
}


DEFAULT_QUOTAS = {
    "risk_liquidity": 8,
    "value": 6,
    "profitability_quality": 6,
    "momentum": 4,
    "accounting_balance": 5,
    "intangibles_attention": 4,
    "growth_operations": 3,
    "macro": 4,
    "industry": 6,
}


def monthly_feature_ic(
    frame: pd.DataFrame, feature: str, date_col: str, target_col: str
) -> tuple[float, int]:
    values = []
    for _, month_frame in frame[[date_col, target_col, feature]].groupby(date_col, sort=False):
        valid = month_frame[[target_col, feature]].dropna()
        if valid[feature].nunique() < 2 or valid[target_col].nunique() < 2:
            continue
        corr = valid[feature].corr(valid[target_col], method="spearman")
        if np.isfinite(corr):
            values.append(float(corr))
    mean_ic = float(np.mean(values)) if values else np.nan
    return mean_ic, len(values)


def load_selection_frame(
    data_path: Path,
    columns: list[str],
    date_col: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    return (
        pl.scan_parquet(data_path)
        .select(columns)
        .filter((pl.col(date_col) >= start) & (pl.col(date_col) <= end))
        .collect()
        .to_pandas()
    )


def rank_cross_sectional_features(
    frame: pd.DataFrame,
    candidates: dict[str, list[str]],
    available: set[str],
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for category, features in candidates.items():
        for feature in features:
            if feature not in available:
                continue
            mean_ic, months = monthly_feature_ic(frame, feature, date_col, target_col)
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "selection_mean_ic": mean_ic,
                    "selection_abs_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
                    "selection_months": months,
                }
            )
    return pd.DataFrame(rows).sort_values(["category", "selection_abs_ic"], ascending=[True, False])


def rank_macro_features(
    frame: pd.DataFrame,
    macros: list[str],
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    monthly = frame.groupby(date_col, as_index=False).agg(
        {target_col: "mean", **{macro: "first" for macro in macros if macro in frame.columns}}
    )
    rows = []
    for macro in macros:
        if macro not in monthly.columns:
            continue
        valid = monthly[[macro, target_col]].dropna()
        corr = (
            valid[macro].corr(valid[target_col], method="spearman")
            if valid[macro].nunique() > 1 and valid[target_col].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "feature": macro,
                "category": "macro",
                "selection_mean_ic": corr,
                "selection_abs_ic": abs(corr) if np.isfinite(corr) else np.nan,
                "selection_months": len(valid),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_abs_ic", ascending=False)


def select_by_quota(ranks: pd.DataFrame, quotas: dict[str, int]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for category, quota in quotas.items():
        category_ranks = ranks[ranks["category"] == category].sort_values(
            "selection_abs_ic", ascending=False
        )
        selected[category] = category_ranks.head(quota)["feature"].tolist()
    return {category: features for category, features in selected.items() if features}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select <50 NN5 features from an early OOS window and test later OOS years."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_like_nn5_backprop_no_interactions_rankfix.yaml"),
    )
    parser.add_argument("--out-prefix", default="nn5_feature_selected_1987_1996_to_1997_2016")
    parser.add_argument("--selection-start-year", type=int, default=1987)
    parser.add_argument("--selection-end-year", type=int, default=1996)
    parser.add_argument("--oos-start-year", type=int, default=1997)
    parser.add_argument("--oos-end-year", type=int, default=2016)
    parser.add_argument("--max-features", type=int, default=49)
    parser.add_argument("--min-selection-months", type=int, default=60)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--checkpoint-each-split", action="store_true")
    parser.add_argument("--validation-forecast-calibration", action="store_true")
    parser.add_argument("--validation-shrinkage-only", action="store_true")
    parser.add_argument("--validation-shrinkage-grid", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = base.build_base_config(config_path)
    data_path = ROOT / config["data"]["processed_panel_path"]
    manifest = json.loads((ROOT / config["data"]["manifest_path"]).read_text())

    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]
    available = set(
        manifest.get("stock_characteristics", manifest.get("characteristics", []))
        + manifest.get("macro_predictors", [])
        + manifest.get("industry_dummies", [])
    )
    industries = manifest.get("industry_dummies", [])
    candidate_columns = sorted(
        (set(chain.from_iterable(STATIC_CANDIDATES.values())) | set(MACRO_ALL) | set(industries))
        & available
    )
    selection_frame = load_selection_frame(
        data_path,
        [date_col, target_col] + candidate_columns,
        date_col,
        args.selection_start_year,
        args.selection_end_year,
    )

    cross_ranks = rank_cross_sectional_features(
        selection_frame,
        {**STATIC_CANDIDATES, "industry": industries},
        available,
        date_col,
        target_col,
    )
    macro_ranks = rank_macro_features(selection_frame, MACRO_ALL, date_col, target_col)
    ranks = pd.concat([cross_ranks, macro_ranks], ignore_index=True)

    eligible_ranks = ranks[ranks["selection_months"] >= args.min_selection_months].copy()
    selected = select_by_quota(eligible_ranks, DEFAULT_QUOTAS)
    selected_flat = []
    for category, features in selected.items():
        for feature in features:
            if feature not in selected_flat:
                selected_flat.append(feature)
    if len(selected_flat) > args.max_features:
        keep = set(
            ranks[ranks["feature"].isin(selected_flat)]
            .sort_values("selection_abs_ic", ascending=False)
            .head(args.max_features)["feature"]
        )
        selected = {
            category: [feature for feature in features if feature in keep]
            for category, features in selected.items()
        }
        selected_flat = [feature for features in selected.values() for feature in features]

    design_rows = []
    for category, features in selected.items():
        for feature in features:
            row = eligible_ranks[
                (eligible_ranks["category"] == category) & (eligible_ranks["feature"] == feature)
            ].head(1)
            design_rows.append(
                {
                    "model": "selected_lt50_1987_1996",
                    "feature": feature,
                    "category": category,
                    "selection_mean_ic": float(row["selection_mean_ic"].iloc[0])
                    if not row.empty
                    else np.nan,
                    "selection_abs_ic": float(row["selection_abs_ic"].iloc[0])
                    if not row.empty
                    else np.nan,
                    "selection_months": int(row["selection_months"].iloc[0])
                    if not row.empty
                    else 0,
                }
            )

    design = pd.DataFrame(design_rows).sort_values("selection_abs_ic", ascending=False)
    ranks.to_csv(OUT_DIR / f"{args.out_prefix}_selection_ranks.csv", index=False)
    design.to_csv(OUT_DIR / f"{args.out_prefix}_design.csv", index=False)
    print(
        f"Selected {len(selected_flat)} features from {args.selection_start_year}-{args.selection_end_year}",
        flush=True,
    )
    print(design.to_string(index=False), flush=True)

    if not args.run_model:
        return

    original = deepcopy(base.MANUAL_GROUPS)
    try:
        base.MANUAL_GROUPS = {"selected_lt50_1987_1996": selected}
        sys.argv = [
            "scripts/feature_engineering/run_feature_engineering_nn5_experiment.py",
            "--config",
            str(args.config),
            "--out-prefix",
            args.out_prefix,
            "--first-test-year",
            str(args.oos_start_year),
            "--last-test-year",
            str(args.oos_end_year),
        ]
        if args.skip_predictions:
            sys.argv.append("--skip-predictions")
        if args.checkpoint_each_split:
            sys.argv.append("--checkpoint-each-split")
        if args.validation_forecast_calibration:
            sys.argv.append("--validation-forecast-calibration")
        if args.validation_shrinkage_only:
            sys.argv.append("--validation-shrinkage-only")
        if args.validation_shrinkage_grid:
            sys.argv.extend(["--validation-shrinkage-grid", args.validation_shrinkage_grid])
        base.main()
    finally:
        base.MANUAL_GROUPS = original


if __name__ == "__main__":
    main()
