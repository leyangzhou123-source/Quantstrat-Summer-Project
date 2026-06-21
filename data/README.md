# Data Directory

Keep raw data immutable. Place source files in:

- `data/raw/` for CRSP-style returns and stock characteristics.
- `data/external/` for macro predictors, factor files, and metadata.
- `data/interim/` for cleaned intermediate files.
- `data/processed/` for final modeling panels.

The pipeline expects a monthly panel with one row per stock-month and a
lead-one-month excess return target.

