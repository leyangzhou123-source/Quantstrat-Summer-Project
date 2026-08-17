#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"

PREFIX="${1:-transformer_nn_model_penal_mlp_full_grid_full_rolling}"
CONFIG="${2:-configs/transformer_nn_model_penal_mlp_full_grid.yaml}"
STRATEGY_DIR="${3:-reports/strategies/transformer_nn_model_penal_mlp_full_grid_advanced}"

python -B -u scripts/run_paper_rolling_models.py \
  --config "${CONFIG}" \
  --models transformer_nn \
  --out-prefix "${PREFIX}"

python -B -m strategy.build_strategies \
  --predictions "reports/model_runs/${PREFIX}_predictions.parquet" \
  --out-dir "${STRATEGY_DIR}"
