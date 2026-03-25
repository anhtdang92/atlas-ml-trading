# ATLAS - Stock ML Intelligence System

## Project Overview

AI-powered stock trading dashboard with ML price predictions (LSTM neural networks), real-time Yahoo Finance data (yfinance), and Google Cloud ML infrastructure. Built with Python 3.9+ and Streamlit.

**Author:** Anh Dang | **License:** MIT | **GCP Project:** `stock-ml-trading-487`

## Architecture

```
app.py (Streamlit Dashboard)
├── data/          → Stock market data via yfinance (free, no API key)
├── ml/            → ML pipeline (LSTM, feature engineering, predictions, API)
├── gcp/           → Google Cloud Platform (Vertex AI, BigQuery, Storage)
├── ui/            → Cyberpunk-themed UI components (glass morphism, neon)
├── config/        → YAML/JSON configuration files
├── bin/           → Shell scripts (training, setup, status checks)
├── tests/         → Unit, integration, and E2E tests
├── docs/          → Documentation files
├── notebooks/     → Model evaluation & EDA notebooks
└── models/        → Trained model artifacts (gitignored)
```

## Key Entry Points

- **Dashboard:** `streamlit run app.py` (localhost:8501)
- **REST API:** `uvicorn ml.api:app --reload` (localhost:8000/api/v1/docs)
- **Local Training:** `./bin/train_local.sh` (train on local GPU/CPU — no cloud needed)
- **Cloud Training:** `./bin/train_now.sh` (train on Google Cloud Vertex AI)
- **Setup:** `./bin/quick-start.sh` (first-time environment setup)
- **Tests:** `python -m pytest tests/unit/ -v`

## Dashboard Pages

1. **Portfolio** - Stock portfolio with real-time prices, P&L tracking, sector allocation (persistent via JSON)
2. **Live Prices** - Real-time prices by category (Tech, Sector Leaders, Defensive, ETFs, Growth) with candlestick charts
3. **ML Predictions** - LSTM 21-day forecasts with confidence scores for position trading
4. **Rebalancing** - ML-enhanced portfolio allocation with paper/live trading via Alpaca
5. **Cloud Progress** - Training job monitoring, cost tracking, endpoint status

## Stock Universe (33 stocks)

- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **Defensive:** JNJ, KO, PEP, MCD (consumer staples / healthcare for crash protection)
- **ETFs:** IWM, IJR, XLK, XLF, XLE, XLV, TLT (bonds), GLD (gold)
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN (speculative — capped at 2% each)

## ML Pipeline

- **Model:** 2-layer LSTM (64 units each, 0.2 dropout), Adam optimizer, Huber loss
- **Features:** 34 technical indicators across 9 categories:
  - Moving averages (MA10/20/50/200, price ratios, golden cross)
  - RSI, MACD (signal + histogram), Bollinger Bands (width + position)
  - Volume indicators (log volume, MA, ROC, ratio)
  - Momentum (daily return, momentum 14/30, ROC 10)
  - Volatility (14/30-day, ATR 14)
  - Market regime (vol regime, distance to 52w high, price z-score, trend strength)
  - Calendar (month + day-of-week, sin/cos encoded)
  - Relative strength
- **Input:** 30-day lookback windows of OHLCV data + 34 indicators
- **Output:** Predicted 21-day return percentage per symbol (position trading)
- **Uncertainty:** MC Dropout (50 forward passes with training=True)
- **Hybrid system:** Vertex AI → Local ML → Explicit failure (no silent mock fallback)
- **Mock predictions:** Disabled by default (`allow_mock=False`). Only enabled for dashboard display, NEVER for trade decisions.

### Key ML Files

| File | Purpose |
|------|---------|
| `ml/lstm_model.py` | Base LSTM architecture (Huber loss, MC Dropout, L2 reg) |
| `ml/lstm_model_gpu.py` | GPU models (BiLSTM+Attention, Transformer, multi-horizon) |
| `ml/feature_engineering.py` | 34 technical indicators, sequence creation, normalization |
| `ml/prediction_service.py` | Main ML orchestrator (train, predict, walk-forward CV) |
| `ml/hybrid_prediction_service.py` | Fallback chain: Vertex AI → Local → Fail loudly |
| `ml/ablation_study.py` | 7-architecture comparison (LSTM, BiLSTM, Transformer, CNN-LSTM, XGBoost, Ridge) |
| `ml/statistical_tests.py` | Bootstrap CIs (10K resamples), Diebold-Mariano tests |
| `ml/hyperparameter_tuning.py` | Optuna Bayesian optimization with walk-forward CV |
| `ml/baseline_models.py` | 5 baselines (Buy&Hold, MeanReversion, Momentum, Ridge, XGBoost) |
| `ml/backtest_tearsheet.py` | Sharpe, Sortino, Calmar, drawdown, monthly heatmap |
| `ml/feature_importance.py` | SHAP, permutation, mutual information, correlation |
| `ml/api.py` | FastAPI REST endpoint (/predict, /health, /models, /features) |
| `ml/alpaca_trading.py` | Alpaca paper/live trading integration |
| `ml/portfolio_rebalancer.py` | ML-weighted allocation with risk controls |
| `ml/experiment_tracker.py` | CSV + MLflow + W&B experiment logging |

## Data Layer

- `data/stock_api.py` - Stock data client via yfinance (free, no API key needed)
- Supports: current prices, quotes, batch quotes, historical OHLCV, fundamentals
- Smart caching: 60s during market hours, 5min pre/post market, 1hr when closed
- Stock universe defined in `data/stock_api.py` (`STOCK_UNIVERSE` dict)

## GCP Integration

- **Vertex AI** - Cloud ML training (n1-standard-4, optional Tesla T4 GPU) and prediction endpoints
- **BigQuery** - Dataset `stock_data` with tables for prices, predictions, trades, metrics
- **Cloud Storage** - Model artifacts and training data
- **Cost:** ~$18-32/month, $3-8 per training run (preemptible instances, scale-to-zero)

## Configuration

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main app config (tickers, ML params, scheduling, risk limits) |
| `config/gcp_config.yaml` | GCP settings (Vertex AI, BigQuery, Storage, IAM) |
| `config/secrets.yaml` | Brokerage API keys (gitignored, see `secrets.yaml.example`) |
| `config/rebalancing_config.json` | Portfolio rules (10% max position, $100 min trade) |
| `config/portfolio.json` | Persistent portfolio holdings (demo data by default) |
| `.env` | Environment variables for GCP project/region/buckets |

## Risk Controls

- Max position: 10% (reduced from 15%), Min position: 2%
- Speculative caps: COIN/PLTR/SNOW at 2%, TSLA at 5%
- Sector limits: Tech max 25%, Financials/Healthcare/Consumer Staples max 15%, etc.
- Cash reserve: 5% always held uninvested
- Position stop-loss: 8% (auto-sell if individual position drops 8%)
- Portfolio drawdown circuit breaker: 12% (halt all trading if portfolio drops 12% from peak)
- Minimum trade size: $100
- ML weight factor: 0.05 (reduced from 0.3 — until LSTM model is validated via walk-forward)
- Mock predictions disabled by default — system fails loudly if no trained model exists
- Confidence threshold: 0.6
- Paper trading mode enabled by default
- Alpaca paper trading integration (requires API keys in config/secrets.yaml)

## Testing

```bash
# All unit tests (~116 tests)
python -m pytest tests/unit/ -v

# Individual test suites
python -m pytest tests/unit/test_ml_pipeline.py -v          # Feature engineering, data validation
python -m pytest tests/unit/test_statistical_analysis.py -v  # Bootstrap CIs, SHAP, tearsheet
python -m pytest tests/unit/test_model_comparison.py -v      # Ablation, tuning, baselines
python -m pytest tests/unit/test_api.py -v                   # FastAPI endpoint tests

# Integration tests
python -m pytest tests/integration/test_e2e_pipeline.py -v   # Full pipeline E2E
python tests/integration/walk_forward_backtest.py             # Walk-forward backtesting

# GCP connection
python tests/unit/test_gcp.py
```

## Common Commands

```bash
# Development
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# REST API
uvicorn ml.api:app --reload

# ML Training
./bin/train_local.sh          # Local GPU/CPU
./bin/train_now.sh            # Google Cloud Vertex AI
./bin/check_training.sh       # Check training status

# Dev setup
./bin/dev-setup.sh
```
