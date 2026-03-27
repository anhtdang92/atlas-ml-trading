# ADR-001: Huber Loss Over MSE for Stock Return Prediction

## Status

Accepted

## Context

Stock return distributions exhibit fat tails and occasional extreme outliers caused by earnings surprises, macroeconomic shocks, and flash crashes. Mean Squared Error (MSE) squares the residual, giving disproportionate weight to these outliers. A single earnings gap of +15% can dominate an entire training batch and destabilize gradient updates, causing the model to overfit to rare extreme events rather than learning the general return distribution.

## Decision

Use Huber loss with delta=0.1 as the primary training objective for the LSTM model.

Huber loss behaves as MSE for residuals smaller than delta (preserving sensitivity to normal-range returns) and transitions to MAE (linear penalty) for residuals beyond delta. With delta=0.1, any predicted return error exceeding 10% is penalized linearly rather than quadratically.

## Consequences

**Positive:**
- Robust to fat-tailed outliers from earnings surprises and market dislocations.
- More stable gradient updates during training, reducing the need for gradient clipping.
- Better generalization on walk-forward validation windows that include volatile periods.

**Negative:**
- Slightly less sensitivity to small return differences (sub-1% moves) compared to pure MSE.
- The delta hyperparameter (0.1) requires tuning; an inappropriate value could under- or over-smooth.
- Less common in literature, making direct comparison with published MSE-based results harder.

## Alternatives Considered

- **MSE:** Standard but vulnerable to outlier-driven gradient explosions.
- **MAE:** Fully robust but provides weak gradients near zero, slowing convergence on the majority of normal returns.
- **Quantile loss:** Useful for interval prediction but adds complexity without clear benefit for point forecasts.
