#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"

PREFIX="${1:-transformer_nn_model_penal_pytorch_local_grid_1987_1988}"
CONFIG="${2:-configs/transformer_nn_model_penal_pytorch_local_grid.yaml}"
STRATEGY_DIR="${3:-reports/strategies/transformer_nn_model_penal_pytorch_local_grid_1987_1988_advanced}"

python -B -u scripts/run_paper_rolling_models.py \
  --config "${CONFIG}" \
  --models transformer_nn \
  --out-prefix "${PREFIX}"

python -B -m strategy.build_strategies \
  --predictions "reports/model_runs/${PREFIX}_predictions.parquet" \
  --out-dir "${STRATEGY_DIR}"
