#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"

PANEL="data/processed/model_penal_gkx_clean_rankfix.parquet"
MANIFEST="data/processed/model_penal_gkx_clean_rankfix_manifest.json"

if [[ ! -f "${PANEL}" ]]; then
  echo "Missing ${PANEL}. Place the rank-fixed monthly model panel there before running."
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing ${MANIFEST}. Place the matching panel manifest there before running."
  exit 1
fi

echo "Step 1/4: feature engineering and fixed 38-feature NN5 predictions"
bash scripts/run_rank38_nn5_workflow.sh rank38_nn5

echo "Step 2/4: all machine-learning rolling models"
bash scripts/modeling/run_all_ml_models.sh

echo "Step 3/4: strategy grid for fixed 38-feature NN5 predictions"
bash scripts/modeling/run_strategy_grid.sh \
  reports/feature_engineering/rank38_nn5_nn5_predictions.parquet \
  reports/strategies/rank38_nn5_strategy_grid

echo "Step 4/4: strategy grid for all machine-learning predictions"
bash scripts/modeling/run_strategy_grid.sh \
  reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet \
  reports/strategies/all_models_strategy_grid

echo "Workflow complete."
echo "Prediction summaries:"
echo "  reports/feature_engineering/rank38_nn5_nn5_summary.csv"
echo "  reports/model_runs/all_models_rankfix_no_interactions_summary.csv"
echo "Strategy selections:"
echo "  reports/strategies/rank38_nn5_strategy_grid/selected_strategy_by_model.csv"
echo "  reports/strategies/all_models_strategy_grid/selected_strategy_by_model.csv"
