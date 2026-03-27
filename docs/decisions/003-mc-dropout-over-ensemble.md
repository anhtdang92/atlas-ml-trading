# ADR-003: MC Dropout Over Model Ensembles for Uncertainty Estimation

## Status

Accepted

## Context

Position sizing in the trading system requires confidence estimates alongside point predictions. A prediction of +3% return with high uncertainty should receive a smaller allocation than the same prediction with low uncertainty. The system needs calibrated uncertainty quantification that distinguishes between reliable and unreliable forecasts without adding excessive infrastructure or training cost.

## Decision

Use Monte Carlo Dropout for uncertainty estimation: perform 50 stochastic forward passes at inference time with dropout layers active (`training=True`), then compute the mean as the point prediction and standard deviation as the uncertainty estimate. Predictions with confidence below 0.6 are excluded from trade signals.

## Consequences

**Positive:**
- Single model to train, store, and serve -- no need to maintain N separate model checkpoints.
- Theoretically grounded: MC Dropout approximates variational inference over model weights (Gal & Ghahramani, 2016).
- Simple implementation: requires only toggling `training=True` during inference passes.
- Well-calibrated uncertainty: high dropout variance correlates with genuinely harder-to-predict stocks and regimes.

**Negative:**
- Approximately 50x slower inference compared to a single deterministic forward pass.
- Uncertainty quality depends on dropout rate (0.2) being appropriate; too low underestimates uncertainty.
- Does not capture epistemic uncertainty from architecture choice (all passes use the same LSTM structure).
- Less expressive than deep ensembles, which capture multi-modal posterior distributions.

## Alternatives Considered

- **Deep ensembles (5-10 models):** Better calibration in literature but requires 5-10x training cost and model storage. Operationally complex for a system that retrains on 33 stocks.
- **Bayesian LSTM (variational layers):** Principled but significantly harder to implement and tune, with slower convergence.
- **Quantile regression:** Produces prediction intervals directly but does not provide a full uncertainty distribution for downstream position sizing.
- **Conformal prediction:** Distribution-free coverage guarantees but requires a calibration set and does not naturally integrate with gradient-based training.
