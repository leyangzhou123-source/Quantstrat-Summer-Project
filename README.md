# Quantstrat ML Asset Pricing

This repository is a research scaffold for reproducing the empirical workflow
in Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*
(`eHtkga-hhaa009.pdf`, DOI `10.1093/rfs/hhaa009`).

The project treats expected stock excess return as a supervised prediction
problem. The intended pipeline builds a monthly stock panel, ranks firm
characteristics cross-sectionally, adds macro predictors and
characteristic-by-macro interactions, trains models with time-ordered
train/validation/test splits, and evaluates both forecast accuracy and
portfolio performance.

The paper studies roughly 30,000 stocks from 1957 to 2016, using 94 firm
characteristics, eight macro predictors, 74 industry dummies, and machine
learning models that range from linear baselines to trees and neural networks.
This codebase mirrors that structure, but many files are still scaffolding
waiting for vendor-specific data and concrete model adapters.

## What Goes Where

```text
.
|-- README.md
|-- eHtkga-hhaa009.pdf
|-- pyproject.toml
|-- configs/
|   |-- default.toml
|   `-- default.yaml
|-- data/
|   `-- README.md
|-- docs/
|   `-- methodology.md
|-- reports/
|   `-- README.md
|-- scripts/
|   `-- run_pipeline.py
|-- src/
|   `-- quantstrat/
|       |-- data/
|       |-- features/
|       |-- models/
|       |-- evaluation/
|       |-- portfolio/
|       `-- utils/
`-- tests/
```

### Root Files

- `README.md`: the project map. Keep this file updated when adding new
  directories, scripts, data expectations, or major workflow steps.
- `eHtkga-hhaa009.pdf`: the reference paper. Use it as the methodological
  anchor for the model families, predictors, train/validation/test design,
  out-of-sample evaluation, variable importance, and portfolio sorts.
- `pyproject.toml`: Python package metadata and dependencies. Add required
  runtime packages under `dependencies`, development tools under `dev`, and
  optional heavy modeling libraries such as `xgboost` or `torch` under
  `models`.

### Configuration

- `configs/default.toml`: the main configuration file used by the code through
  `quantstrat.utils.config.load_config`. Update paths, enabled model families,
  split windows, macro predictors, and evaluation settings here first.
- `configs/default.yaml`: a YAML mirror of the default settings. Keep it in
  sync only if you need YAML for notebooks, external tools, or documentation.
  The current Python loader reads the TOML file.

### Data

- `data/README.md`: data storage rules and expected subdirectories.
- `data/raw/`: immutable source data, such as CRSP-style monthly returns and
  stock-level characteristics.
- `data/external/`: outside reference data, such as macro predictors, factor
  files, risk-free rates, and metadata.
- `data/interim/`: cleaned intermediate outputs that are not yet modeling-ready.
- `data/processed/`: final monthly panel files. The expected modeling panel has
  one row per stock-month and includes `date`, `permno`,
  `ret_excess_lead1`, optional `market_equity`, optional `sic2`, ranked firm
  characteristics, macro predictors, interaction terms, and industry dummies.

### Documentation

- `docs/methodology.md`: a concise translation from the paper to this codebase.
  Expand it when implementation choices need explanation, such as data filters,
  sample periods, hyperparameter grids, or departures from the paper.

### Reports

- `reports/README.md`: report output rules.
- `reports/tables/`: generated tables for out-of-sample R2, model comparisons,
  Diebold-Mariano-style tests, decile returns, long-short spreads, Sharpe
  ratios, and variable-importance rankings.
- `reports/figures/`: generated plots for model performance, variable
  importance, marginal relationships, and portfolio results.

### Scripts

- `scripts/run_pipeline.py`: the command-line entry point. It currently loads
  the TOML config and prints enabled models. As the project matures, this file
  should orchestrate ingestion, feature assembly, rolling splits, model tuning,
  out-of-sample prediction, evaluation, and report generation.

### Source Package

- `src/quantstrat/__init__.py`: package marker and future package-level version
  or public imports.

#### `src/quantstrat/data/`

- `schema.py`: defines `PanelSchema`, the canonical column names for the
  stock-month panel. Update this when the target, asset identifier, weights, or
  industry coding changes.
- `ingest.py`: loading and validation helpers. Add vendor-specific ingestion
  here for returns, characteristics, macro data, risk-free rates, and final
  panel assembly. Raw files should be transformed into a sorted, validated
  monthly panel.
- `__init__.py`: package marker for data utilities.

#### `src/quantstrat/features/`

- `ranking.py`: cross-sectional ranking of firm characteristics by month into
  the paper-style `[-1, 1]` range. Extend this with missing-value treatment,
  winsorization, or characteristic-specific preprocessing rules.
- `interactions.py`: characteristic-by-macro interaction construction. This is
  where the paper's state-dependent predictor expansion belongs.
- `__init__.py`: package marker for feature utilities.

#### `src/quantstrat/models/`

- `registry.py`: model family metadata and the `ModelAdapter` protocol. Add
  concrete adapters or specs for OLS, OLS-3, elastic net with Huber loss, PCR,
  PLS, random forest, boosted trees, and neural networks.
- `splits.py`: time-ordered rolling train/validation/test split construction.
  This protects the paper's out-of-sample design by avoiding random shuffles
  and future leakage.
- `train.py`: generic fit-and-predict wrapper for model adapters. Expand this
  with validation tuning, prediction storage, and split-by-split execution.
- `__init__.py`: package marker for model utilities.

#### `src/quantstrat/evaluation/`

- `metrics.py`: forecast and portfolio metrics. It currently includes panel
  out-of-sample R2 and annualized Sharpe ratio.
- `model_comparison.py`: pairwise forecast comparison helpers. This is the home
  for Diebold-Mariano-style error-difference tests and Newey-West adjustments.
- `importance.py`: variable-importance helpers. Extend this with the paper's
  drop-in-R2 importance calculations and model-specific importance summaries.
- `__init__.py`: package marker for evaluation utilities.

#### `src/quantstrat/portfolio/`

- `sorts.py`: forecast-sorted portfolio utilities. This should contain decile
  assignment, value-weighted returns, long-short spreads, turnover, and related
  economic evaluation logic.
- `__init__.py`: package marker for portfolio utilities.

#### `src/quantstrat/utils/`

- `config.py`: TOML configuration loading. Add path normalization, schema
  checks, or environment overrides here if the pipeline needs them.
- `__init__.py`: package marker for shared utilities.

### Tests

- `tests/test_metrics.py`: focused tests for evaluation metrics, beginning with
  out-of-sample R2.
- `tests/test_splits.py`: focused tests for rolling split behavior and temporal
  ordering.

Add tests whenever a file starts carrying real project logic, especially for
date handling, panel validation, feature construction, split boundaries,
benchmark forecasts, and portfolio sorting edge cases.

## Methodology Map

1. Collect monthly stock returns, risk-free rates, firm characteristics,
   industry codes, macro predictors, and market equity weights.
2. Build the lead-one-month excess return target `ret_excess_lead1`.
3. Rank firm characteristics cross-sectionally each month into `[-1, 1]`.
4. Add macro predictors, industry dummies, and characteristic-by-macro
   interactions.
5. Preserve time order with rolling train, validation, and test windows.
6. Tune hyperparameters on validation samples only.
7. Generate true out-of-sample forecasts for each model family.
8. Evaluate predictive performance against a zero forecast with panel
   out-of-sample R2.
9. Compare model errors with Diebold-Mariano-style tests.
10. Convert forecasts into value-weighted decile portfolios and long-short
    spreads.
11. Report Sharpe ratios, portfolio returns, variable importance, and marginal
    relationships.

## First Implementation Targets

- Replace placeholder data paths in `configs/default.toml`.
- Implement vendor-specific ingestion in `src/quantstrat/data/ingest.py`.
- Build the final monthly modeling panel under `data/processed/`.
- Add concrete model adapters for the paper's model families.
- Extend `scripts/run_pipeline.py` into a full orchestration entry point.
- Generate tables and figures under `reports/`.
