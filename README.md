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
This codebase mirrors that structure with a local, no-WRDS workflow built
around processed parquet panels and rolling out-of-sample model runners.

## Quick Start

Create an environment and install the project dependencies:

```bash
python -m pip install -e ".[dev,models]"
```

Run the lightweight test suite:

```bash
python -m ruff check src scripts strategy tests
python -m pytest
```

Run the paper-style rolling model engine with the rank-fixed model panel:

```bash
python -u scripts/run_paper_rolling_models.py \
  --config configs/paper_all_models_no_interactions_gkx_clean_rankfix_nonconstant.yaml \
  --models ols ridge elastic_net pcr random_forest nn1 nn2 nn3 nn4 nn5 \
  --out-prefix gkx_clean_rankfix_no_interactions
```

## What Goes Where

```text
.
|-- README.md
|-- eHtkga-hhaa009.pdf
|-- pyproject.toml
|-- configs/
|   |-- README.md
|   |-- default.toml
|   `-- *.yaml
|-- data/
|   `-- README.md
|-- docs/
|   `-- methodology.md
|-- reports/
|   `-- README.md
|-- scripts/
|   |-- README.md
|   |-- run_paper_rolling_models.py
|   |-- run_pipeline.py
|   |-- data_processing/
|   |-- modeling/
|   |-- feature_engineering/
|   |-- reporting/
|   `-- experiments/
|-- strategy/
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
- `configs/README.md`: naming convention for paper, smoke/local, rank-fixed,
  no-interaction, and transformer configs.

### Data

- `data/README.md`: data storage rules and expected subdirectories.
- `data/raw/`: immutable source data, such as CRSP-style monthly returns and
  stock-level characteristics.
- `data/external/`: outside reference data, such as macro predictors, factor
  files, risk-free rates, and metadata.
- `data/interim/`: cleaned intermediate outputs that are not yet modeling-ready.
- `data/processed/`: final monthly panel files. The expected modeling panel has
  one row per stock-month and includes `month`, `permno`,
  `ret_excess_lead1`, market-equity weights, industry codes or dummies, ranked
  firm characteristics, and macro predictors. Interaction-expanded panels can
  be generated separately, but the main GitHub workflow uses the smaller
  no-interaction panel.

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

- `scripts/run_paper_rolling_models.py`: the main rolling train, validation,
  prediction, and summary entry point. This is a stable wrapper used by VM
  commands; the implementation lives in `scripts/modeling/`.
- `scripts/run_pipeline.py`: a smaller orchestration helper for configured
  workflow checks.
- `scripts/data_processing/`: model-panel construction and merge scripts.
- `scripts/modeling/`: rolling model-training implementations.
- `scripts/feature_engineering/`: characteristic ranking, NN5 feature
  experiments, and rank-optimized feature research.
- `scripts/reporting/`: report generation utilities.
- `scripts/experiments/`: shell launchers and one-off experiment runs.

### Strategy

- `strategy/`: portfolio construction and strategy-optimization scripts that
  consume saved prediction parquet files.

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

- `registry.py`: model family metadata and adapter lookup for OLS, OLS-3,
  elastic net, PCR, PLS, random forest, boosted trees, neural networks, and the
  transformer-style tabular neural model.
- Rolling split construction lives in `src/quantstrat/Engine/engine.py`. This
  protects the paper's out-of-sample design by avoiding random shuffles and
  future leakage.
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
- `tests/test_ranking.py`: focused tests for cross-sectional rank scaling.

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

## GitHub Hygiene

Large local parquet panels, model predictions, strategy outputs, bytecode, and
OS metadata are ignored by `.gitignore`. Commit source code, configs, tests, and
small summary CSV/report files only.
