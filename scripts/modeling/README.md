# Modeling Scripts

`run_rank38_nn5_predictions.sh` regenerates the fixed 38-feature NN5 prediction
source.

It uses:

- rank-fixed panel: `data/processed/model_penal_gkx_clean_rankfix.parquet`
- config: `configs/best_rank38_nn5.yaml`
- fixed feature design: `configs/feature_designs/rank_signed_anti_crowded38.csv`
- OOS years: `2002-2016`
- feature set: `rank_signed_anti_crowded38`

`run_all_ml_models.sh` runs the broader machine-learning benchmark on the
rank-fixed no-interaction panel. It includes:

- `ols`, `ols_huber`, and `ols_3`
- `ridge`, `elastic_net`, and `elastic_net_huber`
- `pcr` and `pls`
- `random_forest` and `gbrt_huber`
- `nn1`, `nn2`, `nn3`, `nn4`, and `nn5`
- `transformer_nn`

`run_strategy_grid.sh` applies the strategy generator to any compatible
prediction parquet. The selected strategy for each source/model pair is chosen
by tune-period annualized Sharpe only.
