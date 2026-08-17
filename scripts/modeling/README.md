# Modeling Scripts

Rolling model training and prediction entry points.

Primary command:

```bash
python -u scripts/run_paper_rolling_models.py \
  --config configs/paper_all_models_no_interactions_gkx_clean_rankfix_nonconstant.yaml \
  --models ols ridge elastic_net pcr random_forest nn1 nn2 nn3 nn4 nn5 \
  --out-prefix gkx_clean_rankfix_no_interactions
```

The root-level `scripts/run_paper_rolling_models.py` file is a compatibility
wrapper. The implementation lives here.
