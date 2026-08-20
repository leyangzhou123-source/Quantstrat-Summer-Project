# Strategy

This folder contains the model-agnostic strategy grid generator.

The main script is:

```bash
python -B -m strategy.optimize_rank_strategy_risk_overlay \
  --predictions reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet \
  --out-dir reports/strategies/model_strategy_grid \
  --smoothing-grid 1 3 \
  --turnover-cost-bps 5
```

The prediction parquet must contain `month`, `permno`, `ret_excess_lead1`,
`me`, `forecast`, and `model`. Multiple models can be stored in the same file;
the strategy grid is evaluated separately for each source/model pair.

The selected strategy is chosen by tune-period annualized Sharpe only. Other
metrics such as drawdown, volatility, Calmar, turnover, and test Sharpe are
reported for evaluation but do not enter the selection score.
