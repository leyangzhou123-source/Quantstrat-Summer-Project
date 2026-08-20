# Quantstrat ML Asset-Pricing Models And Strategy Grid

This is the clean GitHub version of the project. It keeps the code needed to:

1. reproduce the rank-optimized NN5 prediction research path, and
2. run the broader machine-learning model comparison on the same rank-fixed
   no-interaction panel,
3. generate strategy grids from any compatible ML prediction parquet.

The retained NN5 research result is:

- prediction model: `rank_signed_anti_crowded38`
- feature count: `38`
- neural network: NN5 with hidden layers `[32, 16, 8, 4, 2]`
- out-of-sample years: `2002-2016`
- strategy grid: top/bottom rank portfolios with multiple weighting schemes,
  volatility targeting, momentum gating, drawdown gates, smoothing, and `5 bps`
  turnover cost

Generated datasets, model outputs, strategy returns, reports, and local cache
files are intentionally excluded from Git.

## Structure

```text
.
|-- configs/
|   |-- best_rank38_nn5.yaml
|   |-- all_models_rankfix_no_interactions.yaml
|   |-- all_models_light_rankfix_no_interactions.yaml
|   `-- feature_designs/rank_signed_anti_crowded38.csv
|-- data/
|   `-- README.md
|-- docs/
|   `-- methodology.md
|-- reports/
|   `-- README.md
|-- scripts/
|   |-- run_full_research_workflow.sh
|   |-- run_rank38_nn5_workflow.sh
|   |-- run_paper_rolling_models.py
|   |-- feature_engineering/run_rank_optimized_feature_research.py
|   |-- modeling/run_all_ml_models.sh
|   |-- modeling/run_rank38_nn5_predictions.sh
|   |-- modeling/run_strategy_grid.sh
|   `-- modeling/run_paper_rolling_models.py
|-- src/quantstrat/
|   |-- Engine/
|   |-- data/
|   |-- evaluation/
|   |-- features/
|   |-- models/
|   `-- utils/
|-- strategy/
|   `-- optimize_rank_strategy_risk_overlay.py
`-- tests/
```

## Install

```bash
python -m pip install -e ".[dev]"
```

The PyTorch transformer model is optional. Install it when running
`transformer_nn`:

```bash
python -m pip install -e ".[dev,models]"
```

## Required Data

Place the rank-fixed monthly model panel at:

```text
data/processed/model_penal_gkx_clean_rankfix.parquet
data/processed/model_penal_gkx_clean_rankfix_manifest.json
```

The parquet should contain monthly stock observations with:

- `month`
- `permno`
- `ret_excess_lead1`
- `me`
- `sic2`
- the 38 features listed in
  `configs/feature_designs/rank_signed_anti_crowded38.csv`

Large parquet data files are ignored by Git.

## Run The Rank-Optimized NN5 Path

To run the full project in the correct order:

```bash
bash scripts/run_full_research_workflow.sh
```

This runs:

1. load the rank-fixed monthly panel,
2. rebuild the fixed 38-feature research outputs and NN5 predictions,
3. fit the all-model rolling benchmark with validation grid search,
4. run the Sharpe-selected strategy grid on the NN5 prediction file,
5. run the Sharpe-selected strategy grid on the all-model prediction file.

Run the rank-optimized 38-feature NN5 prediction workflow:

```bash
bash scripts/run_rank38_nn5_workflow.sh
```

This command:

1. Trains the rank-optimized NN5 model from the fixed 38-feature design.
2. Saves predictions under `reports/feature_engineering/`.

To run only the NN5 prediction source:

```bash
bash scripts/modeling/run_rank38_nn5_predictions.sh
```

## Run All Machine-Learning Models

The full benchmark config includes:

- linear models: `ols`, `ols_huber`, `ols_3`
- penalized linear models: `ridge`, `elastic_net`, `elastic_net_huber`
- dimension-reduction models: `pcr`, `pls`
- tree models: `random_forest`, `gbrt_huber`
- neural networks: `nn1`, `nn2`, `nn3`, `nn4`, `nn5`
- transformer extension: `transformer_nn`

Run the all-model rolling benchmark:

```bash
bash scripts/modeling/run_all_ml_models.sh
```

This writes:

- `reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet`
- `reports/model_runs/all_models_rankfix_no_interactions_rolling_metrics.csv`
- `reports/model_runs/all_models_rankfix_no_interactions_summary.csv`

For a lighter local check of non-NN models, use:

```bash
python -u scripts/modeling/run_paper_rolling_models.py \
  --config configs/all_models_light_rankfix_no_interactions.yaml \
  --models ols_huber ols_3 pls pcr elastic_net_huber glm_huber random_forest gbrt_huber \
  --out-prefix all_models_light_rankfix_no_interactions
```

## Run Strategy Grids

The strategy generator accepts any prediction parquet with:

- `month`
- `permno`
- `ret_excess_lead1`
- `me`
- `forecast`
- `model`

Run it on all-model predictions:

```bash
bash scripts/modeling/run_strategy_grid.sh \
  reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet \
  reports/strategies/model_strategy_grid
```

For each source/model pair, strategy selection is based only on tune-period
annualized Sharpe. Other metrics are still reported for evaluation, but they do
not enter the selection score.

This writes:

- `all_strategy_grid_results.csv`
- `selected_strategy_by_model.csv`
- `selected_strategy_returns.csv`
- `selected_strategy_returns.parquet`

The strategy ranks stocks by each model's forecast every month, buys the top
tail, shorts the bottom tail, and can use signal-strength weights:

```text
signal_strength_i,t = |forecast_i,t - median_forecast_t|
```

Weights are normalized separately inside the long and short legs. The raw
long-short return is:

```text
R_t = sum(long_weight_i,t * realized_return_i,t)
    - sum(short_weight_i,t * realized_return_i,t)
```

The strategy grid can vary:

- top/bottom breadth
- equal, value, square-root market equity, or signal weights
- annual volatility target
- `12` month volatility lookback
- leverage cap
- drawdown and momentum exposure gates
- forecast smoothing
- `5 bps` turnover cost

## Checks

```bash
python -m ruff check src scripts strategy tests
python -m pytest
```
