# Model Card: ATLAS Stock Return Prediction LSTM

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ATLAS LSTM Stock Return Predictor |
| **Version** | 1.0 |
| **Architecture** | 2-layer LSTM (64 units each, 0.2 dropout per layer) |
| **Loss Function** | Huber loss (delta=0.1) |
| **Optimizer** | Adam (default learning rate) |
| **Regularization** | L2 weight regularization + MC Dropout |
| **Output** | Predicted 21-day return percentage per stock symbol |
| **Author** | Anh Dang |
| **License** | MIT |
| **Framework** | TensorFlow / Keras |

## Intended Use

- **Primary use:** Educational and research tool for stock return forecasting.
- **Secondary use:** Paper trading experimentation with Alpaca integration.
- **Out of scope:** Production trading with real capital without independent validation. This model has NOT been audited for live deployment and should not be used as the sole basis for investment decisions.

## Training Data

- **Source:** Yahoo Finance via yfinance (free, no API key required).
- **Period:** 2 years of daily OHLCV (Open, High, Low, Close, Volume) data.
- **Universe:** 33 US equities and ETFs across 5 categories (Tech, Sector Leaders, Defensive, ETFs, Growth).
- **Features:** 34 engineered technical indicators across 9 categories:
  - Moving averages and price ratios (MA10/20/50/200, golden cross)
  - Momentum oscillators (RSI, MACD signal + histogram)
  - Volatility measures (Bollinger Bands, ATR, 14/30-day volatility)
  - Volume indicators (log volume, moving average, rate of change)
  - Momentum (daily return, 14/30-day momentum, ROC-10)
  - Market regime (volatility regime, distance to 52-week high, z-score, trend strength)
  - Calendar features (month and day-of-week, sin/cos encoded)
  - Relative strength
- **Input shape:** 30-day lookback windows of normalized features.
- **Preprocessing:** Per-feature StandardScaler normalization fitted on training windows only.

## Evaluation

- **Validation strategy:** Walk-forward cross-validation with expanding training windows. No random splits -- temporal ordering is strictly preserved to prevent future data leakage.
- **Directional accuracy:** 55-62% (percentage of correct up/down predictions).
- **RMSE:** 0.035-0.045 on held-out forward windows.
- **Sharpe ratio:** 0.8-1.2 (annualized, based on backtest tearsheet).
- **Statistical rigor:**
  - Bootstrap confidence intervals (10,000 resamples) for all reported metrics.
  - Diebold-Mariano tests for pairwise statistical significance vs. baseline models.

## Ablation Study

Seven architectures were compared under identical walk-forward evaluation:

| Architecture | Description |
|-------------|-------------|
| LSTM (base) | 2-layer, 64 units, Huber loss |
| BiLSTM + Attention | Bidirectional LSTM with attention mechanism |
| Transformer | Multi-head self-attention encoder |
| CNN-LSTM | 1D convolution front-end with LSTM backend |
| Multi-Horizon | Simultaneous 7/14/21-day prediction heads |
| XGBoost | Gradient-boosted trees baseline |
| Ridge Regression | Linear baseline with L2 regularization |

Results and statistical comparisons are logged via the experiment tracker (CSV + optional MLflow/W&B).

## Uncertainty Quantification

- **Method:** Monte Carlo Dropout (50 stochastic forward passes at inference time with `training=True`).
- **Output:** Mean prediction and standard deviation across passes, used as a confidence score.
- **Usage:** Predictions with confidence below 0.6 are excluded from trade signals. Position sizing is scaled by inverse uncertainty.

## Limitations

- **Single data source:** Relies exclusively on Yahoo Finance price/volume data. No alternative data (sentiment, satellite, web traffic).
- **No fundamental signals:** Does not incorporate earnings announcements, SEC filings, or macroeconomic indicators.
- **Fixed horizon:** Predicts 21-day returns only. Not suitable for intraday or multi-year horizons.
- **US equities only:** Trained and evaluated on US-listed stocks and ETFs.
- **Survivorship bias:** The fixed 33-stock universe does not account for delisted securities.
- **Regime sensitivity:** Performance may degrade during market regimes not represented in the 2-year training window (e.g., sustained bear markets, black swan events).
- **Stationarity assumption:** Technical indicators assume locally stationary price dynamics, which may not hold during structural breaks.

## Ethical Considerations

- **Not financial advice.** This model is an educational tool. Users assume all risk.
- **Potential for financial loss.** Even with positive backtested metrics, live trading involves slippage, fees, and regime changes not captured in historical evaluation.
- **Paper trading recommended.** The system defaults to Alpaca paper trading mode. Live trading requires explicit configuration.
- **Transparency:** MC Dropout provides honest uncertainty estimates. The system fails loudly (no silent mock fallback) when no trained model is available, preventing false confidence.
- **Risk controls:** Hard-coded position limits (10% max), sector caps, stop-losses (8%), and a 12% portfolio drawdown circuit breaker are enforced regardless of model output.

## How to Cite

```
ATLAS Stock ML Intelligence System - LSTM Return Predictor
Author: Anh Dang
Repository: kraken-ml-trading-strategy
License: MIT
```

## Additional Resources

- [Walk-forward backtest](tests/integration/walk_forward_backtest.py)
- [Ablation study](ml/ablation_study.py)
- [Statistical tests](ml/statistical_tests.py)
- [Architecture Decision Records](docs/decisions/)
