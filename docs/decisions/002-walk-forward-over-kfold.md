# ADR-002: Walk-Forward CV Over K-Fold for Model Evaluation

## Status

Accepted

## Context

Financial time-series data contains strong temporal dependencies: autocorrelation in volatility, momentum regimes, and structural breaks. Standard k-fold cross-validation randomly shuffles observations, allowing the model to train on future data and evaluate on past data. This introduces look-ahead bias and produces inflated performance metrics that do not reflect real-world trading conditions.

## Decision

Use walk-forward cross-validation with expanding training windows for all model evaluation. Each fold trains on data from the start of the dataset up to time T, then evaluates on the subsequent out-of-sample window (T to T+21 days). The training window expands forward with each fold while the test window always lies strictly in the future relative to training data.

## Consequences

**Positive:**
- Zero future data leakage: the model never sees data from its evaluation period during training.
- Metrics reflect realistic deployment conditions where the model predicts unseen future returns.
- Captures regime changes: evaluation windows span different market conditions over time.
- Aligns with how the model will actually be used (trained on history, deployed forward).

**Negative:**
- Fewer effective training samples per fold compared to k-fold, especially in early windows.
- Higher computational cost: each fold requires a full model retrain on an expanding dataset.
- Early folds have small training sets, potentially producing noisy performance estimates.
- Cannot be parallelized as easily as independent k-fold splits.

## Alternatives Considered

- **K-fold CV:** Fast and well-understood but fundamentally inappropriate for time-series due to look-ahead bias.
- **Time-series split (fixed window):** Avoids leakage but discards older data, reducing training set size.
- **Purged/embargo k-fold:** Adds buffer zones around folds to reduce leakage, but does not fully eliminate temporal contamination for multi-day prediction horizons.
