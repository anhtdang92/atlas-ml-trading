# Model Results & Experiment Tracking

This directory contains model training results, experiment logs, and performance metrics.

## Directory Structure

```
results/
├── experiments/          # Per-run experiment logs (JSON + CSV)
│   ├── experiment_log.csv    # Aggregated experiment log
│   └── run_*.json            # Individual run details
└── README.md
```

## Experiment Tracking

All training runs are tracked via `ml/experiment_tracker.py`. Each run logs:

- **Hyperparameters**: lookback, prediction_horizon, lstm_units, dropout, epochs, batch_size
- **Metrics**: MSE, MAE, RMSE, directional accuracy, R²
- **Metadata**: symbol, timestamp, duration, model path

### Viewing Results

```python
from ml.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
tracker.print_summary()

# Get best model for a symbol
best = tracker.get_best_run("AAPL", metric="directional_accuracy")
print(f"Best AAPL model: {best.metrics}")
```

## Baseline Comparison

Run `python -m ml.baseline_models` to compare LSTM against baselines:

| Model | Description | When It Wins |
|-------|-------------|--------------|
| Buy & Hold | Predicts historical mean return | Trending markets |
| Mean Reversion | Predicts return toward mean | Range-bound markets |
| Momentum | Predicts trend continuation | Strong trends |
| Ridge Regression | Linear model on flattened features | When features are linearly predictive |
| XGBoost | Gradient boosted trees | Complex non-linear but non-temporal patterns |
| **LSTM** | 2-layer recurrent neural network | Temporal patterns in feature sequences |

The LSTM should beat all baselines to justify its complexity. If it doesn't, the simpler model should be preferred (Occam's razor).
