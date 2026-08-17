# Current Modeling Workflow

## Data Input

The current paper-like run uses the rank-fixed processed panel:

```text
data/processed/model_penal_gkx_clean_rankfix.parquet
```

with manifest:

```text
data/processed/model_penal_gkx_clean_rankfix_manifest.json
```

The active NN5 config is:

```text
configs/paper_like_nn5_backprop_bn_l1_no_interactions_rankfix.yaml
```

The model uses the no-interaction feature set:

```text
stock characteristics + macro variables + industry dummies
```

It does not use characteristic-by-macro interaction terms.

## Rolling Window Structure

The current rolling setup keeps the paper-style 12-year validation period, but predicts two out-of-sample years at once.

```text
Train:      1957-1974
Validate:  1975-1986
Predict:   1987-1988

Train:      1957-1976
Validate:  1977-1988
Predict:   1989-1990

Train:      1957-1978
Validate:  1979-1990
Predict:   1991-1992
```

This reduces the number of rolling model fits from roughly 30 one-year fits to roughly 15 two-year fits.

## End-To-End Workflow

```text
1. Load YAML config.
2. Read feature list from the rank-fixed manifest.
3. Build rolling train, validation, and test windows.
4. Load only required columns from parquet.
5. Convert pandas frames to NumPy float32 arrays.
6. Fill NaN and infinite feature values with 0.
7. Standardize features using train-set mean and standard deviation.
8. Standardize target returns using train-set mean and standard deviation.
9. Train model candidates on the train set.
10. Select hyperparameters using validation OOS R2.
11. Predict the two-year OOS test block.
12. Store OOS predictions as parquet.
13. Print pooled OOS R2, mean OOS R2, IC, and Sharpe.
```

The prediction output is stored as:

```text
reports/model_runs/<out-prefix>_predictions.parquet
```

## NN5 Algorithm

NN5 is a feedforward neural network with five hidden layers.

```text
Input features
 -> Dense layer 32
 -> Batch normalization
 -> ReLU
 -> Dense layer 16
 -> Batch normalization
 -> ReLU
 -> Dense layer 8
 -> Batch normalization
 -> ReLU
 -> Dense layer 4
 -> Batch normalization
 -> ReLU
 -> Dense layer 2
 -> Batch normalization
 -> ReLU
 -> Dense output layer
 -> Forecast next-month excess return
```

The training objective is:

```text
MSE loss + L2 penalty + L1 penalty
```

Current key settings:

```yaml
backend: jax
output_ridge: false
layer_widths: [32, 16, 8, 4, 2]
ensemble_seeds: [42, 43]
batch_size: 8192
max_iter: 100
validate_every: 5
batch_normalization: true
l1_alpha: 0.000001
```

The validation grid has five alpha candidates:

```yaml
alpha: [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]
l1_alpha: [0.000001]
learning_rate_init: [0.0001]
```

For each rolling block, NN5 trains:

```text
5 alpha candidates x 2 ensemble seeds = 10 neural networks
```

The two seed forecasts are averaged for each candidate. The candidate with the best validation OOS R2 is selected, then used to predict the two-year OOS block.

## Epoch Definition

One epoch is one approximate pass over the training data in mini-batches.

The code defines:

```python
steps_per_epoch = max(1, len(x_train) // batch_size)
```

With `batch_size = 8192`, if a rolling block has 1,086,953 training rows:

```text
steps_per_epoch = 1,086,953 // 8,192 = 132
```

So one epoch means 132 mini-batch gradient updates. The small remainder after integer division is not used in that epoch.

## Transformer Algorithm

The transformer model is an experimental extension, not part of the original paper. It is implemented in:

```text
src/quantstrat/models/TransformerNN.py
```

The transformer turns the tabular feature vector into tokens:

```python
token_size = ceil(n_features / n_tokens)
padded_features = token_size * n_tokens
tokens = values.reshape(batch_size, n_tokens, token_size)
```

Then each token is projected into a learned embedding:

```text
token_size raw features -> d_model-dimensional token embedding
```

The full flow is:

```text
Raw features
 -> split into contiguous feature groups
 -> linear projection into token embeddings
 -> transformer encoder with self-attention
 -> mean pooling and max pooling across tokens
 -> MLP prediction head
 -> forecast next-month excess return
```

## How Features Are Divided Into Tokens

Feature grouping is deterministic, not random. However, it is also not manually designed by economic category.

The model takes the feature list in its existing order:

```text
stock characteristics first
macro variables next
industry dummies last
```

Then it splits that list into contiguous chunks.

In the current rank-fixed data there are 180 no-interaction features:

```text
94 stock characteristics
8 macro variables
78 industry dummies
```

If `n_tokens = 6`, then:

```text
token_size = 180 / 6 = 30
```

The token groups are:

```text
Token 1: stock characteristics 1-30
Token 2: stock characteristics 31-60
Token 3: stock characteristics 61-90
Token 4: last stock characteristics + macro variables + first industry dummies
Token 5: industry dummies
Token 6: industry dummies
```

If `n_tokens = 12`, then each token has 15 features:

```text
Token 1-7: mostly stock characteristics
Token 7: also macro variables and first industry dummies
Token 8-12: industry dummies
```

This means the current grouping is convenient but not economically clean. A better future version could group features by financial category:

```text
size/value
momentum/reversal
liquidity/trading frictions
profitability/investment
accounting quality
macro predictors
industry dummies
```

## How Token Information Is Gathered

Within each token, raw features are compressed by a learned linear projection.

Example with 180 features, 6 tokens, and `d_model = 8`:

```text
30 raw features in one token
 -> Linear(30, 8)
 -> one 8-dimensional token embedding
```

This learned projection gathers within-token information.

Then self-attention gathers cross-token information:

```text
Token 1 attends to Token 2, Token 3, ..., Token 6
```

Finally, the model pools the token outputs:

```text
mean pooling across tokens
max pooling across tokens
concatenate mean + max
MLP head
forecast
```

## Current Reported Result

The reported NN5 result is:

```text
model  pooled_oos_r2  mean_oos_r2  monthly_spearman_ic  decile_10_minus_1_sharpe
nn5       0.002051     0.002611             0.020585                  0.349799
```

Interpretation:

```text
pooled_oos_r2 = 0.2051%
mean_oos_r2 = 0.2611%
monthly_spearman_ic = 0.0206
decile_10_minus_1_sharpe = 0.35
```

The model has a positive but weak ranking signal. The positive IC means forecasts tend to rank stocks in the correct direction on average, but the Sharpe of 0.35 suggests the portfolio signal is still modest.

## Main Cost Drivers

The expensive part is repeated NN training:

```text
15 rolling blocks
x 5 validation candidates
x 2 ensemble seeds
= about 150 NN trainings
```

Each training can run up to 100 epochs, and each epoch contains many mini-batch gradient updates over hundreds of thousands to millions of stock-month observations.

The transformer can also be expensive because each candidate trains a PyTorch transformer encoder, and self-attention operates across tokens for every training batch.
