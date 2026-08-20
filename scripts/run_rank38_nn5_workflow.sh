#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-}:src"

PRED_PREFIX="${1:-rank38_nn5}"

python -B -u scripts/feature_engineering/run_rank_optimized_feature_research.py \
  --config configs/best_rank38_nn5.yaml \
  --fixed-design configs/feature_designs/rank_signed_anti_crowded38.csv \
  --out-prefix "${PRED_PREFIX}" \
  --oos-start-year 2002 \
  --oos-end-year 2016 \
  --groups rank_signed_anti_crowded38 \
  --run-nn5
