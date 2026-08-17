from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "modeling"))

from run_paper_rolling_models import (
    decile_spread_sharpe,
    load_split_frame,
    monthly_spearman_ic,
)

from quantstrat.Engine.engine import ResearchEngine
from quantstrat.evaluation.metrics import out_of_sample_r2
from quantstrat.utils.config import load_config

OUT_DIR = ROOT / "reports" / "feature_engineering"
PANEL_PATH = "data/processed/model_penal_gkx_clean_rankfix.parquet"
MANIFEST_PATH = "data/processed/model_penal_gkx_clean_rankfix_manifest.json"


MANUAL_GROUPS: dict[str, dict[str, list[str]]] = {
    "value_momentum_quality": {
        "size_value": ["mvel1", "bm", "ep", "cfp"],
        "momentum": ["mom1m", "mom6m", "mom12m", "indmom"],
        "profitability_quality": ["operprof", "gma", "roaq", "lev"],
        "liquidity": ["turn", "dolvol"],
        "macro": ["macro_dp", "macro_bm", "macro_tbl", "macro_tms"],
        "industry": ["sic2_20", "sic2_28", "sic2_35", "sic2_60", "sic2_73"],
    },
    "profitability_investment_accounting": {
        "profitability": ["operprof", "gma", "roic", "roaq", "roeq"],
        "investment_growth": ["invest", "agr", "egr", "sgr", "pchcapx_ia", "grcapx"],
        "accounting_quality": ["acc", "absacc", "pctacc", "stdacc", "stdcf"],
        "balance_sheet": ["cash", "cashdebt"],
        "macro": ["macro_ep", "macro_dfy", "macro_ntis"],
        "industry": ["sic2_28", "sic2_35", "sic2_36", "sic2_49", "sic2_73"],
    },
    "momentum_liquidity_risk": {
        "momentum": ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom"],
        "risk": ["maxret", "retvol", "idiovol", "beta", "betasq"],
        "liquidity": ["baspread", "ill", "std_dolvol", "std_turn", "zerotrade", "turn", "dolvol"],
        "macro": ["macro_svar", "macro_tbl", "macro_tms"],
        "industry": ["sic2_37", "sic2_49", "sic2_60", "sic2_73"],
    },
    "balance_sheet_payout_credit": {
        "value": ["bm", "bm_ia", "cfp", "ps"],
        "leverage_liquidity": ["lev", "cash", "cashpr", "currat", "quick", "depr"],
        "credit_security": ["secured", "securedind", "tang", "tb", "realestate"],
        "payout": ["divi", "divo", "dy"],
        "macro": ["macro_dfy", "macro_tbl", "macro_tms"],
        "industry": ["sic2_20", "sic2_49", "sic2_60", "sic2_65"],
    },
    "growth_efficiency_operations": {
        "growth": ["agr", "egr", "sgr", "lgr", "grltnoa"],
        "operating_efficiency": ["chatoia", "chempia", "chpmia", "chinv", "hire", "herf"],
        "sales_working_capital": [
            "saleinv",
            "salecash",
            "salerec",
            "pchsaleinv",
            "pchsale_pchinvt",
            "pchgm_pchsale",
        ],
        "intangibles": ["orgcap"],
        "macro": ["macro_ntis", "macro_dp", "macro_ep"],
        "industry": ["sic2_35", "sic2_36", "sic2_38", "sic2_73"],
    },
    "intangible_macro_industry": {
        "intangibles": ["rd", "rd_mve", "rd_sale", "orgcap"],
        "firm_lifecycle": ["age", "convind", "sin"],
        "earnings_attention": ["ms", "nincr", "ear", "chtx", "cinvest"],
        "balance_size": ["cashdebt", "mve_ia", "sp"],
        "macro": [
            "macro_dp",
            "macro_ep",
            "macro_bm",
            "macro_ntis",
            "macro_tbl",
            "macro_tms",
            "macro_dfy",
            "macro_svar",
        ],
        "industry": ["sic2_28", "sic2_35", "sic2_36", "sic2_38", "sic2_73", "sic2_87"],
    },
    "compact_diversified_core": {
        "size_value": ["mvel1", "bm", "dy"],
        "momentum": ["mom1m", "mom12m"],
        "profitability_investment": ["operprof", "invest", "roaq", "sgr"],
        "accounting_balance": ["acc", "stdacc", "lev", "cash"],
        "liquidity_risk": ["turn", "dolvol", "retvol", "idiovol", "maxret", "ill"],
        "intangibles": ["rd_mve"],
        "macro": ["macro_bm", "macro_tbl", "macro_dfy", "macro_svar"],
        "industry": ["sic2_20", "sic2_35", "sic2_49", "sic2_60", "sic2_73"],
    },
}


def build_base_config(
    config_path: Path | None = None,
    test_years: int | None = None,
    step_years: int | None = None,
    first_test_year: int | None = None,
    last_test_year: int | None = None,
) -> dict[str, Any]:
    if config_path is not None:
        config = load_config(config_path)
        config["models"]["enabled"] = ["nn5"]
        if test_years is not None:
            config.setdefault("splits", {})
            config["splits"]["test_years"] = int(test_years)
        if step_years is not None:
            config.setdefault("splits", {})
            config["splits"]["step_years"] = int(step_years)
        if first_test_year is not None:
            config.setdefault("splits", {})
            config["splits"]["first_test_year"] = int(first_test_year)
        if last_test_year is not None:
            config.setdefault("splits", {})
            config["splits"]["last_test_year"] = int(last_test_year)
        return config

    return {
        "project": {"name": "quantstrat_feature_engineering_nn5", "random_seed": 42},
        "data": {
            "frequency": "monthly",
            "date_column": "month",
            "asset_id_column": "permno",
            "target_column": "ret_excess_lead1",
            "weight_column": "me",
            "industry_column": "sic2",
            "processed_panel_path": PANEL_PATH,
            "manifest_path": MANIFEST_PATH,
            "max_rows_per_split": {
                "train": 250000,
                "validation": 120000,
            },
        },
        "features": {
            "feature_set": "no_interactions",
            "macro_predictors": ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"],
        },
        "splits": {
            "scheme": "paper_rolling",
            "train_start": "1957-03-31",
            "validation_years": 12,
            "first_test_year": 1987,
            "last_test_year": 2016,
            "test_years": 5,
            "step_years": 5,
        },
        "models": {"enabled": ["nn5"]},
        "model_params": {
            "nn5": {
                "backend": "jax",
                "output_ridge": False,
                "layer_widths": [32, 16, 8, 4, 2],
                "ensemble_seeds": [42],
                "activation": "relu",
                "batch_size": 16384,
                "max_iter": 12,
                "validate_every": 4,
                "early_stopping": True,
                "n_iter_no_change": 2,
                "tol": 0.000001,
                "learning_rate_decay": 0.99,
                "zero_output_init": False,
                "output_init_scale": 0.01,
                "standardize_target": True,
                "clip_grad_norm": 5.0,
                "batch_normalization": True,
                "batch_norm_epsilon": 0.00001,
                "forecast_clip": None,
                "use_sample_weight": False,
                "weighted_validation": False,
                "prediction_batch_size": 65536,
                "scale_features": True,
                "drop_features_after_matrix": True,
                "validation_grid": {
                    "alpha": [0.00001, 0.0001],
                    "l1_alpha": [0.000001],
                    "learning_rate_init": [0.0001],
                },
            }
        },
        "evaluation": {
            "benchmark_forecast": "zero",
            "weighted_oos_r2": False,
            "portfolio_deciles": 10,
            "annualization_factor": 12,
            "newey_west_lags": 12,
        },
    }


def flatten_group(
    group_definition: dict[str, list[str]], available: set[str]
) -> tuple[list[str], list[dict[str, str]]]:
    features: list[str] = []
    rows: list[dict[str, str]] = []
    for category, columns in group_definition.items():
        for column in columns:
            if column not in available:
                continue
            if column not in features:
                features.append(column)
                rows.append({"feature": column, "category": category})
    return features, rows


def split_label(split: Any) -> str:
    if split.test_start.year == split.test_end.year:
        return str(split.test_start.year)
    return f"{split.test_start.year}-{split.test_end.year}"


def summarize_predictions(
    predictions: pd.DataFrame, metrics: pd.DataFrame, engine: ResearchEngine, config: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for name, group in predictions.groupby("model", sort=False):
        model_metrics = metrics[metrics["model"] == name]
        rows.append(
            {
                "model": name,
                "n_features": int(group["n_features"].iloc[0]),
                "pooled_oos_r2": out_of_sample_r2(group[engine.schema.target], group["forecast"]),
                "mean_oos_r2": float(model_metrics["test_oos_r2"].mean()),
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
    return pd.DataFrame(rows).sort_values("pooled_oos_r2", ascending=False)


def feature_ic_importance(
    frame: pd.DataFrame,
    features: list[str],
    feature_categories: dict[str, str],
    feature_set: str,
    date_column: str,
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = []
        for _, month_frame in frame[[date_column, target_column, feature]].groupby(
            date_column, sort=False
        ):
            valid = month_frame[[target_column, feature]].dropna()
            if valid[feature].nunique() < 2 or valid[target_column].nunique() < 2:
                continue
            corr = valid[feature].corr(valid[target_column], method="spearman")
            if np.isfinite(corr):
                values.append(float(corr))
        mean_ic = float(np.mean(values)) if values else np.nan
        rows.append(
            {
                "model": feature_set,
                "feature": feature,
                "category": feature_categories.get(feature, "unknown"),
                "mean_monthly_spearman_ic": mean_ic,
                "abs_mean_monthly_spearman_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
                "months": len(values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["importance_rank"] = (
        out["abs_mean_monthly_spearman_ic"].rank(method="first", ascending=False).astype("Int64")
    )
    return out.sort_values(["model", "importance_rank"])


def macro_time_series_importance(
    data_path: Path,
    macros: list[str],
    date_column: str,
    target_column: str,
) -> pd.DataFrame:
    columns = [date_column, target_column] + macros
    frame = pl.scan_parquet(data_path).select(columns).collect().to_pandas()
    monthly = frame.groupby(date_column, as_index=False).agg(
        {target_column: "mean", **{macro: "first" for macro in macros}}
    )
    rows = []
    for macro in macros:
        valid = monthly[[macro, target_column]].dropna()
        pearson = (
            valid[macro].corr(valid[target_column], method="pearson")
            if valid[macro].nunique() > 1
            else np.nan
        )
        spearman = (
            valid[macro].corr(valid[target_column], method="spearman")
            if valid[macro].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "feature": macro,
                "months": len(valid),
                "time_series_pearson_with_ew_market_return": pearson,
                "time_series_spearman_with_ew_market_return": spearman,
                "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual reduced-feature NN5 experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base YAML config to use for data, split, and NN5 model settings.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="nn5_manual_feature_groups",
        help="Prefix for output files in reports/feature_engineering.",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=None,
        help="Optional override for rolling OOS test block size.",
    )
    parser.add_argument(
        "--step-years",
        type=int,
        default=None,
        help="Optional override for rolling step size.",
    )
    parser.add_argument(
        "--first-test-year",
        type=int,
        default=None,
        help="Optional override for first rolling OOS test year.",
    )
    parser.add_argument(
        "--last-test-year",
        type=int,
        default=None,
        help="Optional override for last rolling OOS test year.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional list of feature-set names to run.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not write the large OOS prediction parquet; keep summary and diagnostics only.",
    )
    parser.add_argument(
        "--checkpoint-each-split",
        action="store_true",
        help="Write partial summary and rolling metrics after each completed rolling test window.",
    )
    parser.add_argument(
        "--validation-forecast-calibration",
        action="store_true",
        help="Use validation-period forecast shrinkage/calibration before test scoring.",
    )
    parser.add_argument(
        "--validation-shrinkage-only",
        action="store_true",
        help="When calibrating forecasts, allow only shrinkage toward zero.",
    )
    parser.add_argument(
        "--validation-shrinkage-grid",
        type=str,
        default=None,
        help="Comma-separated shrinkage values for validation forecast calibration.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = build_base_config(
        config_path,
        test_years=args.test_years,
        step_years=args.step_years,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
    )
    if args.validation_forecast_calibration:
        config.setdefault("model_params", {}).setdefault("nn5", {})
        config["model_params"]["nn5"]["validation_forecast_calibration"] = True
    if args.validation_shrinkage_only:
        config.setdefault("model_params", {}).setdefault("nn5", {})
        config["model_params"]["nn5"]["validation_forecast_calibration"] = True
        config["model_params"]["nn5"]["validation_calibration_methods"] = ["zero_shrinkage"]
    if args.validation_shrinkage_grid:
        config.setdefault("model_params", {}).setdefault("nn5", {})
        config["model_params"]["nn5"]["validation_forecast_calibration"] = True
        config["model_params"]["nn5"]["validation_shrinkage_grid"] = [
            float(value) for value in args.validation_shrinkage_grid.split(",") if value.strip()
        ]
    manifest_path = ROOT / config["data"].get("manifest_path", MANIFEST_PATH)
    manifest = json.loads(manifest_path.read_text())
    available_features = set(
        manifest.get("stock_characteristics", manifest.get("characteristics", []))
        + manifest.get("macro_predictors", [])
        + manifest.get("industry_dummies", [])
    )
    # Macro columns in this manifest are already named macro_*, but keep this defensive.
    available_features |= set(manifest.get("macro_predictors", []))

    engine = ResearchEngine(config, project_root=ROOT)
    data_path = ROOT / config["data"]["processed_panel_path"]

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_importance: list[pd.DataFrame] = []
    design_rows: list[dict[str, Any]] = []

    selected_groups = MANUAL_GROUPS
    if args.groups:
        missing = sorted(set(args.groups) - set(MANUAL_GROUPS))
        if missing:
            raise ValueError(f"Unknown feature groups: {missing}")
        selected_groups = {name: MANUAL_GROUPS[name] for name in args.groups}

    def write_outputs() -> None:
        if not all_predictions:
            return
        predictions = pd.concat(all_predictions, ignore_index=True)
        metrics = pd.DataFrame(all_metrics)
        summary = summarize_predictions(predictions, metrics, engine, config)
        design = pd.DataFrame(design_rows)
        importance = (
            pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
        )
        if importance.empty:
            category_importance = pd.DataFrame()
        else:
            category_importance = (
                importance.groupby(["model", "category"], as_index=False)
                .agg(
                    features=("feature", "count"),
                    mean_abs_ic=("abs_mean_monthly_spearman_ic", "mean"),
                    max_abs_ic=("abs_mean_monthly_spearman_ic", "max"),
                )
                .sort_values(["model", "mean_abs_ic"], ascending=[True, False])
            )
        macro_importance = macro_time_series_importance(
            data_path,
            [
                macro
                for macro in manifest.get("macro_predictors", [])
                if macro in available_features
            ],
            engine.schema.date,
            engine.schema.target,
        )

        metrics.to_csv(OUT_DIR / f"{args.out_prefix}_rolling_metrics.csv", index=False)
        summary.to_csv(OUT_DIR / f"{args.out_prefix}_summary.csv", index=False)
        design.to_csv(OUT_DIR / f"{args.out_prefix}_design.csv", index=False)
        importance.to_csv(OUT_DIR / f"{args.out_prefix}_feature_importance.csv", index=False)
        category_importance.to_csv(
            OUT_DIR / f"{args.out_prefix}_category_importance.csv", index=False
        )
        macro_importance.to_csv(
            OUT_DIR / f"{args.out_prefix}_macro_time_series_importance.csv", index=False
        )
        if not args.skip_predictions:
            predictions.to_parquet(OUT_DIR / f"{args.out_prefix}_predictions.parquet", index=False)

    for feature_set_name, group_definition in selected_groups.items():
        features, category_rows = flatten_group(group_definition, available_features)
        category_map = {row["feature"]: row["category"] for row in category_rows}
        if len(features) < 5:
            raise ValueError(
                f"Feature set {feature_set_name} has too few available features: {features}"
            )

        design_rows.extend(
            {
                "model": feature_set_name,
                "feature": row["feature"],
                "category": row["category"],
            }
            for row in category_rows
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
        print(f"\n=== {feature_set_name}: {len(features)} features ===", flush=True)
        feature_set_oos_frames: list[pd.DataFrame] = []

        for split in engine.make_paper_rolling_splits():
            print(
                f"Fitting NN5 feature set {feature_set_name} for test years {split_label(split)}",
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
            prediction["test_window"] = split_label(split)
            prediction["n_features"] = len(features)
            all_predictions.append(prediction)

            metric = engine.analyze(result, test)
            metric["model"] = feature_set_name
            metric["test_year"] = split.test_year
            metric["test_window"] = split_label(split)
            metric["n_features"] = len(features)
            all_metrics.append(metric)

            feature_set_oos_frames.append(
                test[[engine.schema.date, engine.schema.target] + features].copy()
            )
            if args.checkpoint_each_split:
                write_outputs()

        importance_frame = feature_ic_importance(
            pd.concat(feature_set_oos_frames, ignore_index=True),
            features,
            category_map,
            feature_set_name,
            engine.schema.date,
            engine.schema.target,
        )
        all_importance.append(importance_frame)
        write_outputs()

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    summary = summarize_predictions(predictions, metrics, engine, config)
    importance = pd.concat(all_importance, ignore_index=True)

    print("\nFinal summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nTop 20 feature-IC importance rows", flush=True)
    print(importance.nsmallest(20, "importance_rank").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
