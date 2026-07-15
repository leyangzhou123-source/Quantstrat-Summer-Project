# Scripts

This folder keeps the original pipeline entry point plus data/model scripts for
the Gu-Kelly-Xiu replication.

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
python scripts/run_paper_models.py
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

Add `--use-industry-characteristic-interactions` to include the sparse
characteristic-by-industry interaction block.
