# Scripts

This folder contains reproducible entry points for model runs and strategy
grid generation.

Full ordered workflow:

```bash
bash scripts/run_full_research_workflow.sh
```

It expects:

```text
data/processed/model_penal_gkx_clean_rankfix.parquet
data/processed/model_penal_gkx_clean_rankfix_manifest.json
```

Rank-optimized 38-feature NN5 prediction workflow:

```bash
bash scripts/run_rank38_nn5_workflow.sh
```

The runner trains the fixed 38-feature NN5 model and stores predictions.

For all machine-learning models:

```bash
bash scripts/modeling/run_all_ml_models.sh
```

For strategy grids over any compatible prediction parquet:

```bash
bash scripts/modeling/run_strategy_grid.sh \
  reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet \
  reports/strategies/model_strategy_grid
```
