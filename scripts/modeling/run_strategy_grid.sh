#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"

PREDICTIONS="${1:-reports/model_runs/all_models_rankfix_no_interactions_predictions.parquet}"
OUT_DIR="${2:-reports/strategies/model_strategy_grid}"

python -B -m strategy.optimize_rank_strategy_risk_overlay \
  --predictions "${PREDICTIONS}" \
  --out-dir "${OUT_DIR}" \
  --smoothing-grid 1 3 \
  --turnover-cost-bps 5
