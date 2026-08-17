from __future__ import annotations

import argparse
import json
from itertools import product
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
DEFAULT_OUT_DIR = Path("reports/strategies/rank_optimized_existing_predictions")


def annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna().astype(float)
    std = values.std(ddof=1)
    if len(values) < 2 or not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(12.0) * values.mean() / std)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def summarize(returns: pd.Series) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return {
            "months": 0,
            "mean_monthly_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "annualized_sharpe": np.nan,
            "max_drawdown": np.nan,
            "min_monthly_return": np.nan,
        }
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": float((1.0 + values).prod() ** (12.0 / len(values)) - 1.0),
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(12.0)),
        "annualized_sharpe": annualized_sharpe(values),
        "max_drawdown": max_drawdown(values),
        "min_monthly_return": float(values.min()),
    }


def source_name(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_predictions")


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        columns = ["month", "permno", "ret_excess_lead1", "me", "forecast", "model"]
        frame = pd.read_parquet(path, columns=columns)
        frame["source"] = source_name(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={"ret_excess_lead1": "realized_return"})
    data["month"] = pd.to_datetime(data["month"])
    for column in ["realized_return", "me", "forecast"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = (
        data["month"].notna()
        & data["source"].notna()
        & data["model"].notna()
        & np.isfinite(data["realized_return"])
        & np.isfinite(data["me"])
        & (data["me"] > 0)
        & np.isfinite(data["forecast"])
    )
    return (
        data.loc[valid].sort_values(["source", "model", "permno", "month"]).reset_index(drop=True)
    )


def add_smoothed_forecast(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    data = frame.copy()
    if window <= 1:
        data["strategy_forecast"] = data["forecast"]
        return data
    data["strategy_forecast"] = data.groupby(["source", "model", "permno"], sort=False)[
        "forecast"
    ].transform(lambda values: values.rolling(window, min_periods=1).mean())
    return data


def normalize(values: pd.Series) -> pd.Series:
    total = values.sum()
    if len(values) == 0:
        return values
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def leg_weights(leg: pd.DataFrame, scheme: str) -> pd.Series:
    signal = (leg["strategy_forecast"] - leg["month_median_forecast"]).abs().clip(lower=1e-8)
    if scheme == "equal":
        return pd.Series(1.0 / len(leg), index=leg.index)
    if scheme == "value":
        return normalize(leg["me"])
    if scheme == "sqrt_me":
        return normalize(np.sqrt(leg["me"]))
    if scheme == "signal":
        return normalize(signal)
    if scheme == "signal_sqrt_me":
        return normalize(signal * np.sqrt(leg["me"]))
    if scheme == "signal_value":
        return normalize(signal * leg["me"])
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def strategy_returns(
    frame: pd.DataFrame,
    top_fraction: float,
    weighting: str,
    min_me_percentile: float,
    side: str,
) -> pd.DataFrame:
    rows = []
    for (source, model, month), group in frame.groupby(["source", "model", "month"], sort=True):
        group = group[np.isfinite(group["strategy_forecast"])].copy()
        if len(group) < 100:
            continue
        if min_me_percentile > 0:
            cutoff = group["me"].quantile(min_me_percentile)
            group = group[group["me"] >= cutoff].copy()
        if len(group) < 100:
            continue
        group["forecast_rank_pct"] = group["strategy_forecast"].rank(method="first", pct=True)
        group["month_median_forecast"] = group["strategy_forecast"].median()
        bottom = group[group["forecast_rank_pct"] <= top_fraction]
        top = group[group["forecast_rank_pct"] > 1.0 - top_fraction]
        if bottom.empty or top.empty:
            continue
        top_weights = leg_weights(top, weighting)
        bottom_weights = leg_weights(bottom, weighting)
        top_return = float((top_weights * top["realized_return"]).sum())
        bottom_return = float((bottom_weights * bottom["realized_return"]).sum())
        if side == "long_short":
            ret = top_return - bottom_return
            gross = 2.0
        elif side == "long_only":
            ret = top_return
            gross = 1.0
        elif side == "short_only":
            ret = -bottom_return
            gross = 1.0
        else:
            raise ValueError(f"Unknown side: {side}")
        rows.append(
            {
                "source": source,
                "model": model,
                "month": month,
                "return": ret,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "top_count": len(top),
                "bottom_count": len(bottom),
                "gross_leverage": gross,
            }
        )
    return pd.DataFrame(rows)


def apply_vol_target(
    returns: pd.DataFrame,
    target_vol: float | None,
    lookback: int,
    max_leverage: float,
) -> pd.DataFrame:
    if target_vol is None:
        out = returns.copy()
        out["leverage"] = 1.0
        out["target_vol"] = np.nan
        out["vol_lookback"] = lookback
        return out
    out = returns.sort_values(["source", "model", "month"]).copy()
    pieces = []
    for _, group in out.groupby(["source", "model"], sort=False):
        group = group.copy()
        trailing_vol = group["return"].rolling(lookback, min_periods=12).std(ddof=1).shift(
            1
        ) * np.sqrt(12.0)
        leverage = (target_vol / trailing_vol).replace([np.inf, -np.inf], np.nan)
        group["leverage"] = leverage.fillna(1.0).clip(lower=0.0, upper=max_leverage)
        group["return"] = group["return"] * group["leverage"]
        group["gross_leverage"] = group["gross_leverage"] * group["leverage"]
        group["target_vol"] = target_vol
        group["vol_lookback"] = lookback
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def monthly_ic(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, model, month), group in frame.groupby(["source", "model", "month"], sort=True):
        valid = group[["strategy_forecast", "realized_return"]].dropna()
        if valid["strategy_forecast"].nunique() < 2 or valid["realized_return"].nunique() < 2:
            continue
        rows.append(
            {
                "source": source,
                "model": model,
                "month": month,
                "monthly_spearman_ic": valid["strategy_forecast"].corr(
                    valid["realized_return"], method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_grid(frame: pd.DataFrame, tune_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    return_frames = []
    tune_end_date = pd.Timestamp(tune_end)
    for (
        smooth,
        top_fraction,
        weighting,
        min_me_percentile,
        side,
        target_vol,
        vol_lookback,
        max_leverage,
    ) in product(
        [1, 3],
        [0.05, 0.10, 0.15],
        ["equal", "sqrt_me", "signal_sqrt_me"],
        [0.0, 0.2],
        ["long_short"],
        [None, 0.15],
        [12],
        [1.5],
    ):
        if target_vol is None and (vol_lookback != 12 or max_leverage != 1.5):
            continue
        scored = add_smoothed_forecast(frame, smooth)
        returns = strategy_returns(scored, top_fraction, weighting, min_me_percentile, side)
        if returns.empty:
            continue
        returns = apply_vol_target(returns, target_vol, vol_lookback, max_leverage)
        returns["smooth_window"] = smooth
        returns["top_fraction"] = top_fraction
        returns["weighting"] = weighting
        returns["min_me_percentile"] = min_me_percentile
        returns["side"] = side
        returns["max_leverage"] = max_leverage
        returns["strategy_id"] = (
            f"smooth{smooth}_top{int(top_fraction * 100)}_{weighting}_"
            f"me{int(min_me_percentile * 100)}_{side}_"
            f"vol{target_vol if target_vol is not None else 'none'}_lb{vol_lookback}_max{max_leverage}"
        )
        return_frames.append(returns)
        for (source, model), group in returns.groupby(["source", "model"], sort=False):
            tune = group[group["month"] <= tune_end_date]
            test = group[group["month"] > tune_end_date]
            grid_rows.append(
                {
                    "source": source,
                    "model": model,
                    "strategy_id": returns["strategy_id"].iloc[0],
                    "smooth_window": smooth,
                    "top_fraction": top_fraction,
                    "weighting": weighting,
                    "min_me_percentile": min_me_percentile,
                    "side": side,
                    "target_vol": target_vol,
                    "vol_lookback": vol_lookback,
                    "max_leverage": max_leverage,
                    **{f"tune_{key}": value for key, value in summarize(tune["return"]).items()},
                    **{f"test_{key}": value for key, value in summarize(test["return"]).items()},
                    **{f"full_{key}": value for key, value in summarize(group["return"]).items()},
                }
            )
    return pd.DataFrame(grid_rows), pd.concat(return_frames, ignore_index=True)


def select_best(grid: pd.DataFrame) -> pd.DataFrame:
    eligible = grid[
        (grid["tune_months"] >= 48)
        & np.isfinite(grid["tune_annualized_sharpe"])
        & np.isfinite(grid["test_annualized_sharpe"])
    ].copy()
    eligible["score"] = (
        eligible["tune_annualized_sharpe"]
        - 0.35 * eligible["tune_max_drawdown"].abs()
        - 0.10 * eligible["tune_annualized_volatility"]
    )
    return (
        eligible.sort_values(["source", "model", "score"], ascending=[True, True, False])
        .groupby(["source", "model"], as_index=False)
        .head(1)
        .sort_values("test_annualized_sharpe", ascending=False)
    )


def build_best_positions(frame: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for row in best.to_dict(orient="records"):
        source = row["source"]
        model = row["model"]
        sub = frame[(frame["source"] == source) & (frame["model"] == model)].copy()
        sub = add_smoothed_forecast(sub, int(row["smooth_window"]))
        for month, group in sub.groupby("month", sort=True):
            group = group[np.isfinite(group["strategy_forecast"])].copy()
            if float(row["min_me_percentile"]) > 0:
                group = group[
                    group["me"] >= group["me"].quantile(float(row["min_me_percentile"]))
                ].copy()
            if len(group) < 100:
                continue
            group["forecast_rank_pct"] = group["strategy_forecast"].rank(method="first", pct=True)
            group["month_median_forecast"] = group["strategy_forecast"].median()
            top_fraction = float(row["top_fraction"])
            top = group[group["forecast_rank_pct"] > 1.0 - top_fraction]
            bottom = group[group["forecast_rank_pct"] <= top_fraction]
            if top.empty or bottom.empty:
                continue
            top_weight = leg_weights(top, str(row["weighting"]))
            bottom_weight = leg_weights(bottom, str(row["weighting"]))
            strategy_id = row["strategy_id"]
            if row["side"] == "long_short":
                pieces.append(top.assign(strategy_id=strategy_id, side="long", weight=top_weight))
                pieces.append(
                    bottom.assign(strategy_id=strategy_id, side="short", weight=-bottom_weight)
                )
            elif row["side"] == "long_only":
                pieces.append(top.assign(strategy_id=strategy_id, side="long", weight=top_weight))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize rank-based strategies from saved forecasts."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tune-end", default="2009-12-31")
    args = parser.parse_args()

    paths = [path if path.is_absolute() else Path.cwd() / path for path in args.predictions]
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(paths)
    grid, returns = evaluate_grid(predictions, args.tune_end)
    best = select_best(grid)
    ic = monthly_ic(add_smoothed_forecast(predictions, 1))
    best_returns = returns.merge(
        best[["source", "model", "strategy_id"]],
        on=["source", "model", "strategy_id"],
        how="inner",
    )
    best_positions = build_best_positions(predictions, best)

    grid.to_csv(out_dir / "strategy_grid_results.csv", index=False)
    returns.to_parquet(out_dir / "strategy_grid_returns.parquet", index=False)
    best.to_csv(out_dir / "best_strategy_by_model.csv", index=False)
    best_returns.to_csv(out_dir / "best_strategy_returns.csv", index=False)
    best_returns.to_parquet(out_dir / "best_strategy_returns.parquet", index=False)
    best_positions.to_parquet(out_dir / "best_strategy_positions.parquet", index=False)
    ic.to_csv(out_dir / "monthly_ic.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "predictions": [str(path) for path in paths],
                "tune_end": args.tune_end,
                "selection_rule": "maximize tune Sharpe penalized by drawdown and volatility; report post-tune test separately",
            },
            indent=2,
        )
        + "\n"
    )

    show_cols = [
        "source",
        "model",
        "strategy_id",
        "tune_annualized_sharpe",
        "tune_max_drawdown",
        "test_annualized_sharpe",
        "test_max_drawdown",
        "full_annualized_sharpe",
        "full_max_drawdown",
    ]
    print(best[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
