from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREDICTIONS = Path(
    "reports/model_runs/gkx_clean_rankfix_nn1_to_nn5_nonconstant_full_rolling_rerun_predictions.parquet"
)
DEFAULT_OUT_DIR = Path("reports/strategies/adaptive_ensemble_gkx_clean_rankfix_nn1_to_nn5")


def summarize_returns(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return {
            "months": 0,
            "mean_monthly_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "annualized_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "min_monthly_return": float("nan"),
            "max_monthly_return": float("nan"),
        }
    std = values.std(ddof=1)
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": float((1.0 + values).prod() ** (periods_per_year / len(values)) - 1.0),
        "annualized_volatility": float(std * np.sqrt(periods_per_year)),
        "annualized_sharpe": float(np.sqrt(periods_per_year) * values.mean() / std)
        if np.isfinite(std) and std > 0
        else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "min_monthly_return": float(values.min()),
        "max_monthly_return": float(values.max()),
    }


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["month", "permno", "ret_excess_lead1", "me", "forecast", "model"],
    ).rename(columns={"ret_excess_lead1": "realized_return"})
    frame["month"] = pd.to_datetime(frame["month"])
    for column in ["realized_return", "me", "forecast"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["month"].notna()
        & frame["model"].notna()
        & np.isfinite(frame["realized_return"])
        & np.isfinite(frame["forecast"])
        & np.isfinite(frame["me"])
        & (frame["me"] > 0)
    )
    return frame.loc[valid].sort_values(["month", "permno", "model"]).reset_index(drop=True)


def normalized_weights(values: pd.Series) -> pd.Series:
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def assign_deciles(score: pd.Series, deciles: int = 10) -> pd.Series:
    if score.nunique(dropna=True) < deciles:
        return pd.Series(np.nan, index=score.index)
    return pd.qcut(score.rank(method="first"), q=deciles, labels=False) + 1


def monthly_model_diagnostics(frame: pd.DataFrame, deciles: int) -> pd.DataFrame:
    rows = []
    for (month, model), group in frame.groupby(["month", "model"], sort=True):
        if group["forecast"].nunique() <= 1 or group["realized_return"].nunique() <= 1:
            ic = np.nan
        else:
            ic = group["forecast"].corr(group["realized_return"], method="spearman")
        decile = assign_deciles(group["forecast"], deciles)
        bucketed = group.assign(decile=decile).dropna(subset=["decile"])
        if bucketed.empty:
            spread = np.nan
        else:
            bottom = bucketed[bucketed["decile"] == 1]
            top = bucketed[bucketed["decile"] == deciles]
            if bottom.empty or top.empty:
                spread = np.nan
            else:
                top_return = float((normalized_weights(top["me"]) * top["realized_return"]).sum())
                bottom_return = float(
                    (normalized_weights(bottom["me"]) * bottom["realized_return"]).sum()
                )
                spread = top_return - bottom_return
        rows.append({"month": month, "model": model, "ic": ic, "vw_10_minus_1": spread})
    return pd.DataFrame(rows)


def trailing_controls(
    diagnostics: pd.DataFrame,
    month: pd.Timestamp,
    models: list[str],
    lookback_months: int,
    min_history: int,
    mode: str,
) -> tuple[dict[str, float], dict[str, float], bool]:
    history = diagnostics[diagnostics["month"] < month].sort_values("month")
    if history["month"].nunique() < min_history:
        return (
            {model: 1.0 for model in models},
            {model: 1.0 / len(models) for model in models},
            False,
        )
    months = sorted(history["month"].unique())[-lookback_months:]
    history = history[history["month"].isin(months)]

    signs = {}
    raw_scores = {}
    signed_scores = {}
    for model in models:
        model_history = history[history["model"] == model]
        if mode in {"ic", "best_abs_ic", "top2_abs_ic"}:
            values = model_history["ic"].dropna()
            score = float(values.mean()) if not values.empty else 0.0
            signed_scores[model] = score
            raw_scores[model] = abs(score)
            signs[model] = 1.0 if score >= 0 else -1.0
        elif mode == "spread_sharpe":
            values = model_history["vw_10_minus_1"].dropna()
            if len(values) < 2 or values.std(ddof=1) == 0:
                score = 0.0
            else:
                score = float(values.mean() / values.std(ddof=1))
            raw_scores[model] = abs(score)
            signs[model] = 1.0 if score >= 0 else -1.0
        else:
            raise ValueError(f"Unknown adaptive mode: {mode}")

    if mode == "best_abs_ic":
        best_model = max(raw_scores, key=raw_scores.get)
        raw_scores = {model: 1.0 if model == best_model else 0.0 for model in models}
        signs = {model: 1.0 if signed_scores.get(model, 0.0) >= 0 else -1.0 for model in models}
    elif mode == "top2_abs_ic":
        top_models = {
            model
            for model, _ in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)[:2]
        }
        raw_scores = {model: raw_scores[model] if model in top_models else 0.0 for model in models}

    total = sum(raw_scores.values())
    if not np.isfinite(total) or total <= 0:
        weights = {model: 1.0 / len(models) for model in models}
    else:
        weights = {model: raw_scores[model] / total for model in models}
    return signs, weights, True


def build_month_scores(
    month_frame: pd.DataFrame,
    signs: dict[str, float],
    model_weights: dict[str, float],
) -> pd.DataFrame:
    pieces = []
    for model, group in month_frame.groupby("model", sort=False):
        signal = group[["month", "permno", "realized_return", "me", "forecast"]].copy()
        std = signal["forecast"].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            signal["model_score"] = 0.0
        else:
            signal["model_score"] = (signal["forecast"] - signal["forecast"].mean()) / std
        signal["model_score"] *= signs.get(model, 1.0) * model_weights.get(model, 0.0)
        pieces.append(signal[["month", "permno", "realized_return", "me", "model_score"]])
    stacked = pd.concat(pieces, ignore_index=True)
    scored = (
        stacked.groupby(["month", "permno"], as_index=False)
        .agg(
            realized_return=("realized_return", "first"),
            me=("me", "first"),
            score=("model_score", "sum"),
        )
        .sort_values(["month", "permno"])
    )
    return scored


def portfolio_return(
    month_frame: pd.DataFrame, deciles: int, weighting: str
) -> tuple[float, int, int]:
    decile = assign_deciles(month_frame["score"], deciles)
    bucketed = month_frame.assign(decile=decile).dropna(subset=["decile"])
    bottom = bucketed[bucketed["decile"] == 1]
    top = bucketed[bucketed["decile"] == deciles]
    if bottom.empty or top.empty:
        return np.nan, 0, 0
    if weighting == "value":
        long_weight = normalized_weights(top["me"])
        short_weight = normalized_weights(bottom["me"])
    elif weighting == "equal":
        long_weight = pd.Series(1.0 / len(top), index=top.index)
        short_weight = pd.Series(1.0 / len(bottom), index=bottom.index)
    elif weighting == "signal_sqrt_me":
        top_strength = (top["score"] - top["score"].mean()).abs().clip(lower=1e-8)
        bottom_strength = (bottom["score"] - bottom["score"].mean()).abs().clip(lower=1e-8)
        long_weight = normalized_weights(top_strength * np.sqrt(top["me"]))
        short_weight = normalized_weights(bottom_strength * np.sqrt(bottom["me"]))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    long_return = float((long_weight * top["realized_return"]).sum())
    short_return = float((short_weight * bottom["realized_return"]).sum())
    return long_return - short_return, len(top), len(bottom)


def apply_vol_target(
    returns: pd.DataFrame,
    target_annual_volatility: float,
    trailing_months: int,
    max_leverage: float,
) -> pd.DataFrame:
    out = returns.sort_values("month").copy()
    trailing_vol = out["raw_return"].rolling(trailing_months, min_periods=12).std(ddof=1).shift(1)
    trailing_vol *= np.sqrt(12.0)
    leverage = (target_annual_volatility / trailing_vol).replace([np.inf, -np.inf], np.nan)
    out["leverage"] = leverage.fillna(1.0).clip(lower=0.0, upper=max_leverage)
    out["return"] = out["raw_return"] * out["leverage"]
    return out


def apply_momentum_gate(
    returns: pd.DataFrame,
    lookback_months: int,
    threshold: float = 0.0,
) -> pd.DataFrame:
    out = returns.sort_values("month").copy()
    trailing_return = (1.0 + out["return"]).rolling(
        lookback_months, min_periods=max(3, min(lookback_months, 12))
    ).apply(np.prod, raw=True).shift(1) - 1.0
    out["active"] = (trailing_return > threshold).astype(float)
    out["return_before_momentum_gate"] = out["return"]
    out["return"] = out["return"] * out["active"]
    out["leverage"] = out["leverage"] * out["active"]
    return out


def build_adaptive_strategies(
    predictions_path: Path,
    out_dir: Path,
    deciles: int = 10,
    lookback_months: int = 36,
    min_history: int = 24,
    target_annual_volatility: float = 0.10,
    max_leverage: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = load_predictions(predictions_path)
    models = sorted(predictions["model"].unique())
    diagnostics = monthly_model_diagnostics(predictions, deciles)
    vol_label = f"vol{round(target_annual_volatility * 100)}"

    rows = []
    control_rows = []
    strategy_specs = [
        ("adaptive_ic_value", "ic", "value", False),
        ("adaptive_ic_equal", "ic", "equal", False),
        (f"adaptive_ic_signal_sqrt_me_{vol_label}", "ic", "signal_sqrt_me", True),
        (f"adaptive_top2_ic_signal_sqrt_me_{vol_label}", "top2_abs_ic", "signal_sqrt_me", True),
        (f"adaptive_best_ic_signal_sqrt_me_{vol_label}", "best_abs_ic", "signal_sqrt_me", True),
        ("adaptive_spread_value", "spread_sharpe", "value", False),
        (f"adaptive_spread_signal_sqrt_me_{vol_label}", "spread_sharpe", "signal_sqrt_me", True),
    ]
    for month, month_frame in predictions.groupby("month", sort=True):
        for strategy, mode, weighting, vol_target in strategy_specs:
            signs, weights, active = trailing_controls(
                diagnostics,
                month,
                models,
                lookback_months=lookback_months,
                min_history=min_history,
                mode=mode,
            )
            scored = build_month_scores(month_frame, signs, weights)
            raw_return, long_count, short_count = portfolio_return(scored, deciles, weighting)
            rows.append(
                {
                    "month": month,
                    "strategy": strategy,
                    "raw_return": raw_return,
                    "return": raw_return,
                    "long_count": long_count,
                    "short_count": short_count,
                    "active_history": active,
                    "vol_targeted": vol_target,
                }
            )
            control_rows.append(
                {
                    "month": month,
                    "strategy": strategy,
                    **{f"{model}_sign": signs[model] for model in models},
                    **{f"{model}_weight": weights[model] for model in models},
                }
            )

    returns = pd.DataFrame(rows).dropna(subset=["raw_return"]).sort_values(["strategy", "month"])
    adjusted = []
    for strategy, group in returns.groupby("strategy", sort=True):
        if bool(group["vol_targeted"].iloc[0]):
            adjusted.append(
                apply_vol_target(
                    group,
                    target_annual_volatility=target_annual_volatility,
                    trailing_months=lookback_months,
                    max_leverage=max_leverage,
                )
            )
        else:
            group = group.copy()
            group["leverage"] = 1.0
            adjusted.append(group)
    returns = pd.concat(adjusted, ignore_index=True).sort_values(["strategy", "month"])
    momentum_variants = []
    for strategy in [
        f"adaptive_ic_signal_sqrt_me_{vol_label}",
        f"adaptive_top2_ic_signal_sqrt_me_{vol_label}",
        f"adaptive_best_ic_signal_sqrt_me_{vol_label}",
    ]:
        base = returns[returns["strategy"] == strategy]
        for lookback in [6, 12]:
            gated = apply_momentum_gate(base, lookback_months=lookback)
            gated["strategy"] = f"{strategy}_mom{lookback}"
            momentum_variants.append(gated)
    if momentum_variants:
        returns = pd.concat([returns, *momentum_variants], ignore_index=True).sort_values(
            ["strategy", "month"]
        )
    controls = pd.DataFrame(control_rows).sort_values(["strategy", "month"])
    metrics = pd.DataFrame(
        [
            {"strategy": strategy, **summarize_returns(group.sort_values("month")["return"])}
            for strategy, group in returns.groupby("strategy", sort=True)
        ]
    ).sort_values("annualized_sharpe", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out_dir / "adaptive_strategy_returns.csv", index=False)
    returns.to_parquet(out_dir / "adaptive_strategy_returns.parquet", index=False)
    controls.to_csv(out_dir / "adaptive_strategy_controls.csv", index=False)
    metrics.to_csv(out_dir / "adaptive_strategy_metrics.csv", index=False)
    (out_dir / "adaptive_strategy_metrics.json").write_text(
        json.dumps(metrics.to_dict(orient="records"), indent=2) + "\n"
    )
    return returns, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build adaptive NN ensemble strategies.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--lookback-months", type=int, default=36)
    parser.add_argument("--min-history", type=int, default=24)
    parser.add_argument("--target-annual-volatility", type=float, default=0.10)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    args = parser.parse_args()

    _, metrics = build_adaptive_strategies(
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        deciles=args.deciles,
        lookback_months=args.lookback_months,
        min_history=args.min_history,
        target_annual_volatility=args.target_annual_volatility,
        max_leverage=args.max_leverage,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
