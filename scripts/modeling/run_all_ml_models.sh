#!/usr/bin/env bash
set -euo pipefail

python -u scripts/modeling/run_paper_rolling_models.py \
  --config configs/all_models_rankfix_no_interactions.yaml \
  --models \
    ols ridge elastic_net ols_huber ols_3 pcr pls glm_huber \
    random_forest gbrt_huber nn1 nn2 nn3 nn4 nn5 transformer_nn \
  --out-prefix all_models_rankfix_no_interactions
