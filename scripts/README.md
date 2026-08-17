# Scripts

This folder keeps command-line entry points for the Gu-Kelly-Xiu replication.
The stable wrapper commands remain at the top level, while implementation
scripts are grouped by purpose.

```text
scripts/
|-- run_paper_rolling_models.py   # stable wrapper for VM/GitHub commands
|-- run_pipeline.py               # small orchestration helper
|-- data_processing/              # panel building and data merges
|-- modeling/                     # rolling model-training implementations
|-- feature_engineering/          # ranks and feature-set experiments
|-- reporting/                    # report builders
`-- experiments/                  # shell launchers and one-off experiments
```

## Build the model panel

```bash
python scripts/data_processing/build_model_panel.py
```

This creates:

- `data/processed/model_panel.parquet`
- `data/processed/model_panel_manifest.json`
- `data/processed/industry_characteristic_interactions.npz`
- `data/processed/industry_characteristic_interaction_names.json`

For a quick smoke test:

```bash
python scripts/data_processing/build_model_panel.py --sample-rows 5000
```

## Run paper model families

```bash
python -u scripts/run_paper_rolling_models.py \
  --config configs/paper_all_models_no_interactions_gkx_clean_rankfix_nonconstant.yaml \
  --models ols ridge elastic_net pcr random_forest nn1 nn2 nn3 nn4 nn5 \
  --out-prefix gkx_clean_rankfix_no_interactions
```

Implemented paper model names:

- `ols`
- `ols_3`
- `elastic_net_huber`
- `pcr`
- `pls`
- `random_forest`
- `gbrt_huber`
- `nn1`
- `nn2`
- `nn3`
- `nn4`
- `nn5`
- `transformer_nn`

Use configs ending in `no_interactions` for the smaller GitHub/VM workflow.

## Feature engineering

```bash
python scripts/feature_engineering/rank_stock_characteristics.py --help
python scripts/feature_engineering/run_rank_optimized_feature_research.py --help
```

The root-level `scripts/rank_stock_characteristics.py` wrapper is kept for old
commands.

## Transformer experiments

```bash
bash scripts/experiments/transformer/run_transformer_mlp_light_grid.sh
```
