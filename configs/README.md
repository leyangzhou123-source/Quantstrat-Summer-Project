# Configs

This clean repo keeps three modeling configs:

- `best_rank38_nn5.yaml`: rank-fixed, no-interaction NN5 setup used to
  regenerate the fixed 38-feature NN5 prediction source.
- `all_models_rankfix_no_interactions.yaml`: full rolling benchmark config for
  OLS, Ridge, Elastic Net, Huber models, PCR, PLS, random forest, GBRT, NN1-NN5,
  and the transformer extension.
- `all_models_light_rankfix_no_interactions.yaml`: lighter non-NN benchmark for
  local smoke testing and quick comparison.

The fixed feature design lives in:

- `feature_designs/rank_signed_anti_crowded38.csv`
