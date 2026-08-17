from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREDICTIONS = Path(
    "reports/model_runs/gkx_clean_nn5_no_shrinkage_full_rolling_predictions.parquet"
)
DEFAULT_OUT_DIR = Path("reports/strategies/nn5_no_shrinkage")


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    values = returns.dropna().astype(float)
    std = values.std(ddof=1)
    if len(values) < 2 or not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * values.mean() / std)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def summarize_returns(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return {
            "months": 0,
            "mean_monthly_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "annualized_sharpe": float("nan"),
            "min_monthly_return": float("nan"),
            "max_monthly_return": float("nan"),
            "max_drawdown": float("nan"),
        }
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": float((1.0 + values).prod() ** (periods_per_year / len(values)) - 1.0),
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(periods_per_year)),
        "annualized_sharpe": annualized_sharpe(values, periods_per_year),
        "min_monthly_return": float(values.min()),
        "max_monthly_return": float(values.max()),
        "max_drawdown": max_drawdown(values),
    }


def load_predictions(path: Path) -> pd.DataFrame:
    required = ["month", "permno", "ret_excess_lead1", "me", "forecast"]
    frame = pd.read_parquet(path, columns=required)
    frame = frame.rename(columns={"ret_excess_lead1": "realized_return"})
    frame["month"] = pd.to_datetime(frame["month"])
    for column in ["realized_return", "me", "forecast"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["month"].notna()
        & np.isfinite(frame["realized_return"])
        & np.isfinite(frame["forecast"])
        & np.isfinite(frame["me"])
        & (frame["me"] > 0)
    )
    return frame.loc[valid].sort_values(["month", "permno"]).reset_index(drop=True)


def assign_deciles(frame: pd.DataFrame, deciles: int = 10) -> pd.Series:
    def bucket(group: pd.Series) -> pd.Series:
        if group.nunique(dropna=True) < deciles:
            return pd.Series(np.nan, index=group.index)
        ranks = group.rank(method="first")
        return pd.qcut(ranks, q=deciles, labels=False) + 1

    return frame.groupby("month", group_keys=False)["forecast"].apply(bucket)


def normalized_leg_weights(values: pd.Series) -> pd.Series:
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def top_bottom_decile(
    frame: pd.DataFrame,
    deciles: int = 10,
    inverse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["decile"] = assign_deciles(data, deciles=deciles)
    data = data.dropna(subset=["decile"]).copy()
    data["decile"] = data["decile"].astype(int)
    strategy_name = "inverse_top_bottom_decile" if inverse else "top_bottom_decile"

    positions = []
    returns = []
    for month, group in data.groupby("month", sort=True):
        bottom = group[group["decile"] == 1]
        top = group[group["decile"] == deciles]
        if bottom.empty or top.empty:
            continue
        long_leg = bottom if inverse else top
        short_leg = top if inverse else bottom
        long_weight = normalized_leg_weights(long_leg["me"])
        short_weight = normalized_leg_weights(short_leg["me"])
        long_return = float((long_weight * long_leg["realized_return"]).sum())
        short_return = float((short_weight * short_leg["realized_return"]).sum())
        returns.append(
            {
                "month": month,
                "strategy": strategy_name,
                "long_return": long_return,
                "short_leg_return": short_return,
                "return": long_return - short_return,
                "long_count": len(long_leg),
                "short_count": len(short_leg),
                "gross_leverage": 2.0,
                "net_leverage": 0.0,
            }
        )
        positions.append(long_leg.assign(strategy=strategy_name, side="long", weight=long_weight))
        positions.append(
            short_leg.assign(strategy=strategy_name, side="short", weight=-short_weight)
        )

    position_frame = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()
    return pd.DataFrame(returns), position_frame


def blended_contrarian_decile(
    frame: pd.DataFrame,
    deciles: int = 10,
    value_weight: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["decile"] = assign_deciles(data, deciles=deciles)
    data = data.dropna(subset=["decile"]).copy()
    data["decile"] = data["decile"].astype(int)
    data["month_signal_mean"] = data.groupby("month")["forecast"].transform("mean")
    data["signal_strength"] = (data["forecast"] - data["month_signal_mean"]).abs().clip(lower=1e-8)
    strategy_name = "blended_contrarian_decile"

    positions = []
    returns = []
    for month, group in data.groupby("month", sort=True):
        long_leg = group[group["decile"] == 1]
        short_leg = group[group["decile"] == deciles]
        if long_leg.empty or short_leg.empty:
            continue
        long_value_weight = normalized_leg_weights(long_leg["me"])
        short_value_weight = normalized_leg_weights(short_leg["me"])
        long_signal_weight = normalized_leg_weights(
            long_leg["signal_strength"] * np.sqrt(long_leg["me"])
        )
        short_signal_weight = normalized_leg_weights(
            short_leg["signal_strength"] * np.sqrt(short_leg["me"])
        )
        long_weight = value_weight * long_value_weight + (1.0 - value_weight) * long_signal_weight
        short_weight = (
            value_weight * short_value_weight + (1.0 - value_weight) * short_signal_weight
        )
        long_return = float((long_weight * long_leg["realized_return"]).sum())
        short_return = float((short_weight * short_leg["realized_return"]).sum())
        returns.append(
            {
                "month": month,
                "strategy": strategy_name,
                "long_return": long_return,
                "short_leg_return": short_return,
                "return": long_return - short_return,
                "long_count": len(long_leg),
                "short_count": len(short_leg),
                "gross_leverage": 2.0,
                "net_leverage": 0.0,
                "value_weight": value_weight,
            }
        )
        positions.append(long_leg.assign(strategy=strategy_name, side="long", weight=long_weight))
        positions.append(
            short_leg.assign(strategy=strategy_name, side="short", weight=-short_weight)
        )

    position_frame = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()
    return pd.DataFrame(returns), position_frame


def signal_weighted_vol_target(
    frame: pd.DataFrame,
    deciles: int = 10,
    target_annual_volatility: float = 0.10,
    trailing_months: int = 36,
    max_leverage: float = 3.0,
    inverse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["decile"] = assign_deciles(data, deciles=deciles)
    data = data.dropna(subset=["decile"]).copy()
    data["decile"] = data["decile"].astype(int)
    data["month_signal_mean"] = data.groupby("month")["forecast"].transform("mean")
    data["signal_strength"] = (data["forecast"] - data["month_signal_mean"]).abs()
    data["signal_strength"] = data["signal_strength"].clip(lower=1e-8)

    strategy_name = (
        "inverse_signal_weighted_vol_target" if inverse else "signal_weighted_vol_target"
    )

    raw_rows = []
    positions_by_month: list[pd.DataFrame] = []
    for month, group in data.groupby("month", sort=True):
        bottom = group[group["decile"] == 1]
        top = group[group["decile"] == deciles]
        if bottom.empty or top.empty:
            continue
        long_leg = bottom if inverse else top
        short_leg = top if inverse else bottom
        long_weight = normalized_leg_weights(long_leg["signal_strength"] * np.sqrt(long_leg["me"]))
        short_weight = normalized_leg_weights(
            short_leg["signal_strength"] * np.sqrt(short_leg["me"])
        )
        raw_return = float(
            (long_weight * long_leg["realized_return"]).sum()
            - (short_weight * short_leg["realized_return"]).sum()
        )
        raw_rows.append(
            {
                "month": month,
                "raw_return": raw_return,
                "long_count": len(long_leg),
                "short_count": len(short_leg),
            }
        )
        positions_by_month.append(
            pd.concat(
                [
                    long_leg.assign(strategy=strategy_name, side="long", raw_weight=long_weight),
                    short_leg.assign(
                        strategy=strategy_name,
                        side="short",
                        raw_weight=-short_weight,
                    ),
                ]
            )
        )

    returns = pd.DataFrame(raw_rows).sort_values("month").reset_index(drop=True)
    if returns.empty:
        return returns, pd.DataFrame()
    trailing_vol = returns["raw_return"].rolling(trailing_months, min_periods=12).std(ddof=1).shift(
        1
    ) * np.sqrt(12)
    leverage = (target_annual_volatility / trailing_vol).replace([np.inf, -np.inf], np.nan)
    returns["leverage"] = leverage.fillna(1.0).clip(lower=0.0, upper=max_leverage)
    returns["return"] = returns["raw_return"] * returns["leverage"]
    returns["strategy"] = strategy_name
    returns["gross_leverage"] = 2.0 * returns["leverage"]
    returns["net_leverage"] = 0.0
    returns["target_annual_volatility"] = target_annual_volatility
    returns["trailing_months"] = trailing_months

    positions = pd.concat(positions_by_month, ignore_index=True)
    positions = positions.merge(returns[["month", "leverage"]], on="month", how="left")
    positions["weight"] = positions["raw_weight"] * positions["leverage"]
    return returns.drop(columns=["raw_return"]), positions.drop(columns=["raw_weight"])


def apply_strategy_momentum_filter(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    strategy_name: str,
    lookback_months: int,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_returns = returns.sort_values("month").copy()
    trailing_return = (1.0 + filtered_returns["return"]).rolling(
        lookback_months, min_periods=max(3, min(lookback_months, 12))
    ).apply(np.prod, raw=True).shift(1) - 1.0
    active = (trailing_return > threshold).astype(float)
    filtered_returns["base_return"] = filtered_returns["return"]
    filtered_returns["active"] = active
    filtered_returns["return"] = filtered_returns["base_return"] * filtered_returns["active"]
    filtered_returns["gross_leverage"] = (
        filtered_returns["gross_leverage"] * filtered_returns["active"]
    )
    filtered_returns["strategy"] = strategy_name
    filtered_returns["momentum_lookback_months"] = lookback_months
    filtered_returns["momentum_threshold"] = threshold

    filtered_positions = positions.merge(
        filtered_returns[["month", "active"]],
        on="month",
        how="left",
    )
    filtered_positions["weight"] = filtered_positions["weight"] * filtered_positions["active"]
    filtered_positions["strategy"] = strategy_name
    filtered_positions = filtered_positions[filtered_positions["active"] > 0].drop(
        columns=["active"]
    )
    return filtered_returns, filtered_positions


def write_outputs(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out_dir / "strategy_returns.csv", index=False)
    returns.to_parquet(out_dir / "strategy_returns.parquet", index=False)
    positions.to_parquet(out_dir / "strategy_positions.parquet", index=False)
    metrics.to_csv(out_dir / "strategy_metrics.csv", index=False)
    (out_dir / "strategy_metrics.json").write_text(
        json.dumps(metrics.to_dict(orient="records"), indent=2) + "\n"
    )


def build_strategies(
    predictions_path: Path,
    out_dir: Path,
    deciles: int = 10,
    target_annual_volatility: float = 0.10,
    trailing_months: int = 36,
    max_leverage: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = load_predictions(predictions_path)
    simple_returns, simple_positions = top_bottom_decile(predictions, deciles=deciles)
    inverse_returns, inverse_positions = top_bottom_decile(
        predictions, deciles=deciles, inverse=True
    )
    blended_returns, blended_positions = blended_contrarian_decile(predictions, deciles=deciles)
    inverse_mom12_returns, inverse_mom12_positions = apply_strategy_momentum_filter(
        inverse_returns,
        inverse_positions,
        strategy_name="inverse_top_bottom_decile_mom12",
        lookback_months=12,
    )
    inverse_mom6_returns, inverse_mom6_positions = apply_strategy_momentum_filter(
        inverse_returns,
        inverse_positions,
        strategy_name="inverse_top_bottom_decile_mom6",
        lookback_months=6,
    )
    blended_mom12_returns, blended_mom12_positions = apply_strategy_momentum_filter(
        blended_returns,
        blended_positions,
        strategy_name="blended_contrarian_decile_mom12",
        lookback_months=12,
    )
    advanced_returns, advanced_positions = signal_weighted_vol_target(
        predictions,
        deciles=deciles,
        target_annual_volatility=target_annual_volatility,
        trailing_months=trailing_months,
        max_leverage=max_leverage,
    )
    inverse_advanced_returns, inverse_advanced_positions = signal_weighted_vol_target(
        predictions,
        deciles=deciles,
        target_annual_volatility=target_annual_volatility,
        trailing_months=trailing_months,
        max_leverage=max_leverage,
        inverse=True,
    )
    returns = pd.concat(
        [
            simple_returns,
            inverse_returns,
            blended_returns,
            inverse_mom12_returns,
            inverse_mom6_returns,
            blended_mom12_returns,
            advanced_returns,
            inverse_advanced_returns,
        ],
        ignore_index=True,
    )
    positions = pd.concat(
        [
            simple_positions,
            inverse_positions,
            blended_positions,
            inverse_mom12_positions,
            inverse_mom6_positions,
            blended_mom12_positions,
            advanced_positions,
            inverse_advanced_positions,
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        [
            {"strategy": strategy, **summarize_returns(group.sort_values("month")["return"])}
            for strategy, group in returns.groupby("strategy", sort=True)
        ]
    )
    write_outputs(returns, positions, metrics, out_dir)
    return returns, positions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strategy returns from model forecasts.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--target-annual-volatility", type=float, default=0.10)
    parser.add_argument("--trailing-months", type=int, default=36)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    args = parser.parse_args()

    _, _, metrics = build_strategies(
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        deciles=args.deciles,
        target_annual_volatility=args.target_annual_volatility,
        trailing_months=args.trailing_months,
        max_leverage=args.max_leverage,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
