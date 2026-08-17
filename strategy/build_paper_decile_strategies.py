from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NN_PREDICTIONS = Path(
    "reports/model_runs/gkx_clean_nn1_to_nn5_nonconstant_full_rolling_predictions.parquet"
)
DEFAULT_TRANSFORMER_PREDICTIONS = Path(
    "reports/model_runs/transformer_nn_model_penal_pytorch_r2_small_full_grid_1987_2016_predictions.parquet"
)
DEFAULT_OUT_DIR = Path("reports/strategies/paper_decile_nn1_to_nn5_transformer")


def _strategy_label(path: Path) -> str:
    stem = path.stem
    stem = stem.removesuffix("_predictions")
    return stem


def _load_predictions(path: Path) -> pd.DataFrame:
    required = ["month", "permno", "ret_excess_lead1", "me", "forecast", "model"]
    frame = pd.read_parquet(path, columns=required)
    frame = frame.rename(columns={"ret_excess_lead1": "realized_return"})
    frame["source"] = _strategy_label(path)
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
    return frame.loc[valid].sort_values(["source", "model", "month", "permno"])


def _assign_deciles(group: pd.DataFrame, deciles: int) -> pd.Series:
    if group["forecast"].nunique(dropna=True) < deciles:
        return pd.Series(np.nan, index=group.index)
    ranks = group["forecast"].rank(method="first")
    return pd.qcut(ranks, q=deciles, labels=False) + 1


def _weighted_return(frame: pd.DataFrame, scheme: str) -> float:
    if scheme == "equal":
        return float(frame["realized_return"].mean())
    weights = frame["me"] / frame["me"].sum()
    return float((weights * frame["realized_return"]).sum())


def _summarize(returns: pd.Series) -> dict[str, float]:
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
    std = values.std(ddof=1)
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "months": len(values),
        "mean_monthly_return": float(values.mean()),
        "annualized_return": float((1.0 + values).prod() ** (12.0 / len(values)) - 1.0),
        "annualized_volatility": float(std * np.sqrt(12.0)),
        "annualized_sharpe": float(np.sqrt(12.0) * values.mean() / std)
        if np.isfinite(std) and std > 0
        else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }


def build_paper_decile_strategies(
    prediction_paths: list[Path],
    out_dir: Path,
    deciles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.concat([_load_predictions(path) for path in prediction_paths], ignore_index=True)
    panel["decile"] = (
        panel.groupby(["source", "model", "month"], group_keys=False)
        .apply(_assign_deciles, deciles=deciles)
        .astype("float")
    )
    panel = panel.dropna(subset=["decile"]).copy()
    panel["decile"] = panel["decile"].astype(int)

    return_rows = []
    position_rows = []
    decile_rows = []
    for (source, model, month), group in panel.groupby(["source", "model", "month"], sort=True):
        for decile, decile_group in group.groupby("decile", sort=True):
            value_weighted_return = _weighted_return(decile_group, "value")
            equal_weighted_return = _weighted_return(decile_group, "equal")
            value_weights = decile_group["me"] / decile_group["me"].sum()
            decile_rows.append(
                {
                    "source": source,
                    "model": model,
                    "month": month,
                    "decile": int(decile),
                    "predicted_return": float((value_weights * decile_group["forecast"]).sum()),
                    "value_weighted_return": value_weighted_return,
                    "equal_weighted_return": equal_weighted_return,
                    "stock_count": len(decile_group),
                }
            )
        bottom = group[group["decile"] == 1]
        top = group[group["decile"] == deciles]
        if bottom.empty or top.empty:
            continue
        for weighting in ["value", "equal"]:
            long_return = _weighted_return(top, weighting)
            short_return = _weighted_return(bottom, weighting)
            return_rows.append(
                {
                    "source": source,
                    "model": model,
                    "month": month,
                    "strategy": f"{weighting}_weighted_10_minus_1",
                    "long_return": long_return,
                    "short_return": short_return,
                    "return": long_return - short_return,
                    "long_count": len(top),
                    "short_count": len(bottom),
                }
            )
            return_rows.append(
                {
                    "source": source,
                    "model": model,
                    "month": month,
                    "strategy": f"inverse_{weighting}_weighted_1_minus_10",
                    "long_return": short_return,
                    "short_return": long_return,
                    "return": short_return - long_return,
                    "long_count": len(bottom),
                    "short_count": len(top),
                }
            )
        value_long_weight = top["me"] / top["me"].sum()
        value_short_weight = bottom["me"] / bottom["me"].sum()
        position_rows.append(
            top[["source", "model", "month", "permno", "forecast", "realized_return", "me"]].assign(
                strategy="value_weighted_10_minus_1", side="long", weight=value_long_weight
            )
        )
        position_rows.append(
            bottom[
                ["source", "model", "month", "permno", "forecast", "realized_return", "me"]
            ].assign(strategy="value_weighted_10_minus_1", side="short", weight=-value_short_weight)
        )

    returns = pd.DataFrame(return_rows).sort_values(["source", "model", "strategy", "month"])
    decile_returns = pd.DataFrame(decile_rows).sort_values(["source", "model", "decile", "month"])
    positions = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    metrics = pd.DataFrame(
        [
            {
                "source": source,
                "model": model,
                "strategy": strategy,
                **_summarize(group.sort_values("month")["return"]),
            }
            for (source, model, strategy), group in returns.groupby(
                ["source", "model", "strategy"], sort=True
            )
        ]
    ).sort_values("annualized_sharpe", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out_dir / "paper_decile_returns.csv", index=False)
    returns.to_parquet(out_dir / "paper_decile_returns.parquet", index=False)
    decile_returns.to_csv(out_dir / "paper_decile_by_decile_returns.csv", index=False)
    decile_returns.to_parquet(out_dir / "paper_decile_by_decile_returns.parquet", index=False)
    decile_summary = pd.DataFrame(
        [
            {
                "source": source,
                "model": model,
                "decile": decile,
                "predicted_monthly_return": float(group["predicted_return"].mean()),
                "average_monthly_return": float(group["value_weighted_return"].mean()),
                "monthly_sd": float(group["value_weighted_return"].std(ddof=1)),
                "annualized_sharpe": _summarize(group["value_weighted_return"])[
                    "annualized_sharpe"
                ],
                "average_stock_count": float(group["stock_count"].mean()),
            }
            for (source, model, decile), group in decile_returns.groupby(
                ["source", "model", "decile"], sort=True
            )
        ]
    )
    decile_summary.to_csv(out_dir / "paper_decile_table_value_weighted.csv", index=False)
    positions.to_parquet(out_dir / "paper_decile_value_weighted_positions.parquet", index=False)
    metrics.to_csv(out_dir / "paper_decile_metrics.csv", index=False)
    return returns, positions, metrics, decile_returns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-style forecast-sorted 10-minus-1 decile strategies."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[DEFAULT_NN_PREDICTIONS, DEFAULT_TRANSFORMER_PREDICTIONS],
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--deciles", type=int, default=10)
    args = parser.parse_args()

    _, _, metrics, _ = build_paper_decile_strategies(
        prediction_paths=args.predictions,
        out_dir=args.out_dir,
        deciles=args.deciles,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
