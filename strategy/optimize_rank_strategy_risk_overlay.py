from __future__ import annotations

import argparse
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
DEFAULT_OUT_DIR = Path("reports/strategies/rank_optimized_risk_overlay")


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
            "calmar": float("nan"),
            "positive_month_rate": float("nan"),
        }
    ann_return = float((1.0 + values).prod() ** (12.0 / len(values)) - 1.0)
    dd = max_drawdown(values)
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": ann_return,
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(12.0)),
        "annualized_sharpe": annualized_sharpe(values),
        "max_drawdown": dd,
        "calmar": float(ann_return / abs(dd)) if np.isfinite(dd) and dd < 0 else float("nan"),
        "positive_month_rate": float((values > 0).mean()),
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
        data.loc[valid].sort_values(["source", "model", "permno", "month"]).reset_index(drop=True)
    )


def prepare_panel(data: pd.DataFrame, smoothing: int) -> pd.DataFrame:
    panel = data.copy()
    if smoothing > 1:
        panel["formation_forecast"] = (
            panel.groupby(["source", "model", "permno"], sort=False)["forecast"]
            .rolling(smoothing, min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )
    else:
        panel["formation_forecast"] = panel["forecast"]
    keys = ["source", "model", "month"]
    panel["rank_pct"] = panel.groupby(keys)["formation_forecast"].rank(method="first", pct=True)
    panel["rank_centered"] = panel["rank_pct"] - 0.5
    panel["median_forecast"] = panel.groupby(keys)["formation_forecast"].transform("median")
    panel["signal_strength"] = (
        (panel["formation_forecast"] - panel["median_forecast"]).abs().clip(lower=1e-8)
    )
    return panel.sort_values(["source", "model", "month", "permno"]).reset_index(drop=True)


def raw_weights(panel: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "equal":
        return pd.Series(1.0, index=panel.index)
    if scheme == "sqrt_me":
        return np.sqrt(panel["me"])
    if scheme == "value":
        return panel["me"]
    if scheme == "signal":
        return panel["signal_strength"]
    if scheme == "rank_taper":
        return panel["rank_centered"].abs().clip(lower=1e-6) ** 1.5
    if scheme == "signal_sqrt_me":
        return panel["signal_strength"] * np.sqrt(panel["me"])
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def build_long_short_returns(panel: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    top_fractions = [0.05, 0.10, 0.15, 0.20]
    weightings = ["equal", "sqrt_me", "value", "signal"]
    for top_fraction, weighting in product(top_fractions, weightings):
        selected = panel[
            (panel["rank_pct"] <= top_fraction) | (panel["rank_pct"] > 1.0 - top_fraction)
        ].copy()
        selected["side"] = np.where(selected["rank_pct"] > 1.0 - top_fraction, "top", "bottom")
        selected["raw_weight"] = raw_weights(selected, weighting)
        denom = selected.groupby(["source", "model", "month", "side"])["raw_weight"].transform(
            "sum"
        )
        selected = selected[denom > 0].copy()
        selected["weight"] = selected["raw_weight"] / denom[denom > 0]
        selected["weighted_return"] = selected["weight"] * selected["realized_return"]
        legs = (
            selected.groupby(["source", "model", "month", "side"], as_index=False)[
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
        legs["base_strategy"] = f"top{int(top_fraction * 1000):03d}_{weighting}"
        outputs.append(legs[["source", "model", "month", "base_strategy", "return"]])
    return pd.concat(outputs, ignore_index=True)


def trailing_drawdown(returns: pd.Series, lookback: int) -> pd.Series:
    values = returns.fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - lookback)
        history = values[start:idx]
        if len(history):
            wealth = np.cumprod(1.0 + history)
            out[idx] = np.min(wealth / np.maximum.accumulate(wealth) - 1.0)
    return pd.Series(out, index=returns.index)


def apply_risk_overlay(
    returns: pd.DataFrame,
    target_vol: float | None,
    vol_lookback: int,
    max_leverage: float,
    dd_lookback: int,
    dd_trigger: float | None,
    dd_multiplier: float,
    momentum_lookback: int | None,
    momentum_cut: float,
) -> pd.DataFrame:
    pieces = []
    keys = ["source", "model", "base_strategy"]
    for _, group in returns.sort_values(keys + ["month"]).groupby(keys, sort=False):
        group = group.copy()
        exposure = pd.Series(1.0, index=group.index)
        if target_vol is not None:
            realized_vol = group["return"].rolling(
                vol_lookback, min_periods=max(3, vol_lookback // 2)
            ).std(ddof=1).shift(1) * np.sqrt(12.0)
            vol_scale = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
            exposure *= vol_scale.fillna(1.0).clip(lower=0.0, upper=max_leverage)
        if dd_trigger is not None:
            dd = trailing_drawdown(group["return"], dd_lookback)
            exposure *= np.where(dd <= -abs(dd_trigger), dd_multiplier, 1.0)
        if momentum_lookback is not None:
            trailing_return = (1.0 + group["return"]).rolling(
                momentum_lookback, min_periods=momentum_lookback
            ).apply(np.prod, raw=True).shift(1) - 1.0
            exposure *= np.where(trailing_return < 0.0, momentum_cut, 1.0)
        group["exposure"] = exposure.clip(lower=0.0, upper=max_leverage)
        group["return"] = group["return"] * group["exposure"]
        group["target_vol"] = np.nan if target_vol is None else target_vol
        group["vol_lookback"] = vol_lookback
        group["dd_lookback"] = dd_lookback
        group["dd_trigger"] = np.nan if dd_trigger is None else dd_trigger
        group["dd_multiplier"] = dd_multiplier
        group["momentum_lookback"] = np.nan if momentum_lookback is None else momentum_lookback
        group["momentum_cut"] = momentum_cut
        vol_label = "none" if target_vol is None else f"{target_vol:.2f}"
        dd_label = "none" if dd_trigger is None else f"{dd_trigger:.2f}x{dd_multiplier:.2f}"
        mom_label = (
            "none" if momentum_lookback is None else f"{momentum_lookback}x{momentum_cut:.2f}"
        )
        group["strategy"] = (
            group["base_strategy"]
            + f"_vol{vol_label}_vlb{vol_lookback}_lev{max_leverage:.2f}"
            + f"_dd{dd_label}_dlb{dd_lookback}_mom{mom_label}"
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def make_strategy_grid(base_returns: pd.DataFrame) -> pd.DataFrame:
    overlays = []
    for target_vol, vol_lookback, max_leverage in product(
        [None, 0.08, 0.10, 0.12, 0.15],
        [12],
        [1.0, 1.25],
    ):
        if target_vol is None and (vol_lookback != 12 or max_leverage != 1.0):
            continue
        for dd_trigger, dd_multiplier in [(None, 1.0), (0.12, 0.50), (0.16, 0.65)]:
            for momentum_lookback, momentum_cut in [(None, 1.0), (6, 0.50)]:
                overlays.append(
                    apply_risk_overlay(
                        base_returns,
                        target_vol=target_vol,
                        vol_lookback=vol_lookback,
                        max_leverage=max_leverage,
                        dd_lookback=12,
                        dd_trigger=dd_trigger,
                        dd_multiplier=dd_multiplier,
                        momentum_lookback=momentum_lookback,
                        momentum_cut=momentum_cut,
                    )
                )
    return pd.concat(overlays, ignore_index=True)


def evaluate(returns: pd.DataFrame, tune_end: str) -> pd.DataFrame:
    tune_end_date = pd.Timestamp(tune_end)
    rows = []
    for (source, model, strategy), group in returns.groupby(
        ["source", "model", "strategy"], sort=False
    ):
        tune = group[group["month"] <= tune_end_date]
        test = group[group["month"] > tune_end_date]
        row = {
            "source": source,
            "model": model,
            "strategy": strategy,
            "base_strategy": group["base_strategy"].iloc[0],
            "target_vol": group["target_vol"].iloc[0],
            "vol_lookback": group["vol_lookback"].iloc[0],
            "dd_trigger": group["dd_trigger"].iloc[0],
            "dd_multiplier": group["dd_multiplier"].iloc[0],
            "momentum_lookback": group["momentum_lookback"].iloc[0],
            "momentum_cut": group["momentum_cut"].iloc[0],
            "mean_exposure": float(group["exposure"].mean()),
        }
        row.update({f"tune_{key}": value for key, value in summarize(tune["return"]).items()})
        row.update({f"test_{key}": value for key, value in summarize(test["return"]).items()})
        row.update({f"full_{key}": value for key, value in summarize(group["return"]).items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["selection_score"] = (
        result["tune_annualized_sharpe"].fillna(-999.0)
        + 0.30 * result["tune_calmar"].fillna(-999.0)
        - 1.10 * result["tune_max_drawdown"].abs().fillna(999.0)
        - 0.20 * result["tune_annualized_volatility"].fillna(999.0)
    )
    return result


def select_best(grid: pd.DataFrame) -> pd.DataFrame:
    eligible = grid[
        (grid["tune_months"] >= 48)
        & np.isfinite(grid["tune_annualized_sharpe"])
        & np.isfinite(grid["test_annualized_sharpe"])
        & (grid["tune_max_drawdown"] >= -0.35)
    ].copy()
    if eligible.empty:
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
        description="Drawdown-aware strategy search over fixed forecast files."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tune-end", default="2009-12-31")
    parser.add_argument("--smoothing-grid", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()

    prediction_paths = [
        path if path.is_absolute() else Path.cwd() / path for path in args.predictions
    ]
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_predictions(prediction_paths)
    all_returns = []
    for smoothing in args.smoothing_grid:
        panel = prepare_panel(data, smoothing=smoothing)
        base_returns = build_long_short_returns(panel)
        base_returns["forecast_smoothing"] = smoothing
        returns = make_strategy_grid(base_returns)
        returns["forecast_smoothing"] = smoothing
        returns["strategy"] = returns["strategy"] + f"_smooth{smoothing}"
        all_returns.append(returns)
    returns = pd.concat(all_returns, ignore_index=True)

    grid = evaluate(returns, args.tune_end).sort_values(
        ["test_annualized_sharpe", "test_max_drawdown"], ascending=[False, False]
    )
    best = select_best(grid)
    best_returns = returns.merge(
        best[["source", "model", "strategy"]],
        on=["source", "model", "strategy"],
        how="inner",
    )

    grid.to_csv(out_dir / "all_strategy_grid_results.csv", index=False)
    best.to_csv(out_dir / "best_strategy_by_model.csv", index=False)
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
        "full_annualized_volatility",
        "mean_exposure",
    ]
    print("Best selected by tune-period risk-adjusted score")
    print(best[show_columns].to_string(index=False))
    print("\nTop 20 by post-tune test Sharpe")
    print(grid[show_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
