# Strategy

This folder contains standalone code for turning model prediction parquet files into
portfolio strategy returns.

Current strategies:

- `top_bottom_decile`: value-weighted top forecast decile minus bottom forecast decile.
- `signal_weighted_vol_target`: market-neutral long-short strategy using top and bottom
  forecast deciles, signal-strength weights within each side, and trailing volatility
  targeting.

Example:

```bash
PYTHONPATH=. python -m strategy.build_strategies \
  --predictions reports/model_runs/gkx_clean_nn5_no_shrinkage_full_rolling_predictions.parquet \
  --out-dir reports/strategies/nn5_no_shrinkage
```

