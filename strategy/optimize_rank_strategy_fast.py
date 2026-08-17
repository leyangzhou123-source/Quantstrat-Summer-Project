from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREDICTIONS = [
    Path(
        "reports/feature_engineering/nn5_rank_optimized_feature_research_2002_2016_fixed_nn5_predictions.parquet"
    ),
    Path(
        "reports/feature_engineering/nn5_rank_optimized_feature_research_2002_2016_fixed_composite_predictions.parquet"
    ),
]
DEFAULT_OUT_DIR = Path("reports/strategies/rank_optimized_existing_predictions_fast")


def source_name(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_predictions")


def annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna().astype(float)
    std = values.std(ddof=1)
    if len(values) < 2 or not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(12.0) * values.mean() / std)


def max_drawdown(returns: pd.Series) -> float:
    values = returns.fillna(0.0).astype(float)
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else float("nan")


def summarize(returns: pd.Series) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return {
            "months": 0,
            "mean_monthly_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "annualized_sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": float((1.0 + values).prod() ** (12.0 / len(values)) - 1.0),
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(12.0)),
        "annualized_sharpe": annualized_sharpe(values),
        "max_drawdown": max_drawdown(values),
    }


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["month", "permno", "ret_excess_lead1", "me", "forecast", "model"],
        )
        frame["source"] = source_name(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).rename(
        columns={"ret_excess_lead1": "realized_return"}
    )
    data["month"] = pd.to_datetime(data["month"])
    for column in ["realized_return", "me", "forecast"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = (
        data["month"].notna()
        & data["source"].notna()
        & data["model"].notna()
        & np.isfinite(data["realized_return"])
        & np.isfinite(data["forecast"])
        & np.isfinite(data["me"])
        & (data["me"] > 0)
    )
    return (
        data.loc[valid].sort_values(["source", "model", "month", "permno"]).reset_index(drop=True)
    )


def prepare_rank_panel(data: pd.DataFrame) -> pd.DataFrame:
    panel = data.copy()
    group_keys = ["source", "model", "month"]
    panel["rank_pct"] = panel.groupby(group_keys)["forecast"].rank(method="first", pct=True)
    panel["median_forecast"] = panel.groupby(group_keys)["forecast"].transform("median")
    panel["signal_strength"] = (panel["forecast"] - panel["median_forecast"]).abs().clip(lower=1e-8)
    return panel


def raw_weights(panel: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "equal":
        return pd.Series(1.0, index=panel.index)
    if scheme == "value":
        return panel["me"]
    if scheme == "sqrt_me":
        return np.sqrt(panel["me"])
    if scheme == "signal":
        return panel["signal_strength"]
    if scheme == "signal_sqrt_me":
        return panel["signal_strength"] * np.sqrt(panel["me"])
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def build_long_short_returns(panel: pd.DataFrame) -> pd.DataFrame:
    returns = []
    for top_fraction in [0.05, 0.10, 0.15, 0.20, 0.30]:
        selected = panel[
            (panel["rank_pct"] <= top_fraction) | (panel["rank_pct"] > 1.0 - top_fraction)
        ].copy()
        selected["side"] = np.where(selected["rank_pct"] > 1.0 - top_fraction, "top", "bottom")
        for weighting in ["equal", "value", "sqrt_me", "signal", "signal_sqrt_me"]:
            temp = selected[
                ["source", "model", "month", "side", "realized_return", "me", "signal_strength"]
            ].copy()
            temp["raw_weight"] = raw_weights(temp, weighting)
            denom = temp.groupby(["source", "model", "month", "side"])["raw_weight"].transform(
                "sum"
            )
            temp = temp[denom > 0].copy()
            temp["weight"] = temp["raw_weight"] / denom[denom > 0]
            temp["weighted_return"] = temp["weight"] * temp["realized_return"]
            legs = (
                temp.groupby(["source", "model", "month", "side"], as_index=False)[
                    "weighted_return"
                ]
                .sum()
                .pivot_table(
                    index=["source", "model", "month"],
                    columns="side",
                    values="weighted_return",
                )
                .reset_index()
            )
            if "top" not in legs or "bottom" not in legs:
                continue
            legs["return"] = legs["top"] - legs["bottom"]
            legs["strategy"] = f"top{int(top_fraction * 100)}_{weighting}_long_short"
            returns.append(legs[["source", "model", "month", "strategy", "return"]])
    return pd.concat(returns, ignore_index=True)


def apply_vol_target(
    returns: pd.DataFrame,
    target_vol: float | None,
    lookback_months: int = 12,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    out = returns.sort_values(["source", "model", "strategy", "month"]).copy()
    if target_vol is None:
        out["leverage"] = 1.0
        out["target_vol"] = np.nan
        out["strategy_full"] = out["strategy"] + "_volnone"
        return out
    pieces = []
    for _, group in out.groupby(["source", "model", "strategy"], sort=False):
        group = group.copy()
        realized_vol = group["return"].rolling(lookback_months, min_periods=12).std(ddof=1).shift(
            1
        ) * np.sqrt(12.0)
        leverage = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
        group["leverage"] = leverage.fillna(1.0).clip(lower=0.0, upper=max_leverage)
        group["return"] = group["return"] * group["leverage"]
        group["target_vol"] = target_vol
        group["strategy_full"] = group["strategy"] + f"_vol{target_vol}"
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def evaluate(returns: pd.DataFrame, tune_end: str) -> pd.DataFrame:
    tune_end_date = pd.Timestamp(tune_end)
    rows = []
    for (source, model, strategy), group in returns.groupby(
        ["source", "model", "strategy_full"], sort=False
    ):
        tune = group[group["month"] <= tune_end_date]
        test = group[group["month"] > tune_end_date]
        rows.append(
            {
                "source": source,
                "model": model,
                "strategy": strategy,
                **{f"tune_{key}": value for key, value in summarize(tune["return"]).items()},
                **{f"test_{key}": value for key, value in summarize(test["return"]).items()},
                **{f"full_{key}": value for key, value in summarize(group["return"]).items()},
            }
        )
    result = pd.DataFrame(rows)
    result["selection_score"] = (
        result["tune_annualized_sharpe"]
        - 0.35 * result["tune_max_drawdown"].abs()
        - 0.10 * result["tune_annualized_volatility"]
    )
    return result


def select_best(grid: pd.DataFrame) -> pd.DataFrame:
    eligible = grid[
        (grid["tune_months"] >= 48)
        & np.isfinite(grid["tune_annualized_sharpe"])
        & np.isfinite(grid["test_annualized_sharpe"])
    ].copy()
    return (
        eligible.sort_values(["source", "model", "selection_score"], ascending=[True, True, False])
        .groupby(["source", "model"], as_index=False)
        .head(1)
        .sort_values("test_annualized_sharpe", ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast rank-strategy grid over saved prediction files."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tune-end", default="2009-12-31")
    args = parser.parse_args()

    prediction_paths = [
        path if path.is_absolute() else Path.cwd() / path for path in args.predictions
    ]
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_rank_panel(load_predictions(prediction_paths))
    base_returns = build_long_short_returns(panel)
    returns = pd.concat(
        [
            apply_vol_target(base_returns, None),
            apply_vol_target(base_returns, 0.10),
            apply_vol_target(base_returns, 0.15),
            apply_vol_target(base_returns, 0.20),
        ],
        ignore_index=True,
    )
    grid = evaluate(returns, args.tune_end).sort_values("test_annualized_sharpe", ascending=False)
    best = select_best(grid)
    best_returns = returns.merge(
        best[["source", "model", "strategy"]],
        left_on=["source", "model", "strategy_full"],
        right_on=["source", "model", "strategy"],
        how="inner",
        suffixes=("", "_selected"),
    )

    grid.to_csv(out_dir / "all_strategy_grid_results.csv", index=False)
    best.to_csv(out_dir / "best_strategy_by_model.csv", index=False)
    returns.to_parquet(out_dir / "all_strategy_returns.parquet", index=False)
    best_returns.to_csv(out_dir / "best_strategy_returns.csv", index=False)
    best_returns.to_parquet(out_dir / "best_strategy_returns.parquet", index=False)

    show_columns = [
        "source",
        "model",
        "strategy",
        "tune_annualized_sharpe",
        "tune_max_drawdown",
        "test_annualized_sharpe",
        "test_max_drawdown",
        "full_annualized_sharpe",
        "full_max_drawdown",
    ]
    print(best[show_columns].to_string(index=False))
    print("\nTop 10 by post-tune test Sharpe")
    print(grid[show_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
