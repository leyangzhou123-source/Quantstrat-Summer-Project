# Methodology Translation

The paper frames expected excess return as a prediction problem:

```text
E_t[r_{i,t+1}] = g(z_{i,t})
```

where `z_{i,t}` is the stock-month predictor vector. This project mirrors that
logic with a panel pipeline rather than isolated notebooks.

## Data

- Monthly individual equity returns.
- Risk-free rate to construct next-month excess returns.
- Firm characteristics, ranked cross-sectionally each month into `[-1, 1]`.
- Industry dummies based on two-digit SIC.
- Macro predictors following the Welch-Goyal style set: `dp`, `ep`, `bm`,
  `ntis`, `tbl`, `tms`, `dfy`, and `svar`.
- Characteristic-by-macro interactions to allow state-dependent predictors.

## Sample Splitting

All splits preserve temporal ordering:

- Train: estimate model parameters for a fixed hyperparameter choice.
- Validation: choose hyperparameters by validation forecast loss.
- Test: evaluate forecasts never used for estimation or tuning.

The default config uses an 18-year train window, 12-year validation window, and
1-year test window. Adjust `configs/default.toml` if you choose an expanding
or recursive variant.

## Models

The model registry is organized around the paper's comparative families:

- Linear baselines: full OLS and sparse `OLS-3` using size, value, momentum.
- Penalized linear model: elastic net, with a Huber-loss variant.
- Dimension reduction: PCR and PLS.
- Nonlinear/interactions: random forest and gradient boosted regression trees.
- Neural networks: one or more feed-forward architectures.

## Evaluation

The evaluation layer separates:

- Predictive performance: panel out-of-sample R2 against a zero forecast.
- Statistical comparison: Diebold-Mariano style error-difference tests.
- Economic value: forecast-sorted decile portfolios and long-short spreads.
- Interpretation: variable importance and marginal relationships.
