# ATLAS - Stock ML Intelligence System

## Project Overview

AI-powered stock trading dashboard with ML price predictions (LSTM neural networks), real-time Yahoo Finance data (yfinance), and Google Cloud ML infrastructure. Built with Python 3.9+ and Streamlit.

**Author:** Anh Dang | **License:** MIT | **GCP Project:** `stock-ml-trading-487`

## Architecture

```
app.py (Streamlit Dashboard)
├── data/          → Stock market data via yfinance (free, no API key)
├── ml/            → ML pipeline (LSTM, feature engineering, predictions)
├── gcp/           → Google Cloud Platform (Vertex AI, BigQuery, Storage)
├── ui/            → Cyberpunk-themed UI components (glass morphism, neon)
├── config/        → YAML/JSON configuration files
├── bin/           → Shell scripts (training, setup, status checks)
├── tests/         → Unit, integration, and manual tests
├── docs/          → Documentation files
└── models/        → Trained model artifacts (gitignored)
```

## Key Entry Points

- **Dashboard:** `streamlit run app.py` (localhost:8501)
- **Local Training:** `./bin/train_local.sh` (train on local GPU/CPU — no cloud needed)
- **Cloud Training:** `./bin/train_now.sh` (train on Google Cloud Vertex AI)
- **Setup:** `./bin/quick-start.sh` (first-time environment setup)
- **Tests:** `python -m pytest tests/` or `python tests/unit/test_stock_api.py`

## Dashboard Pages

1. **Portfolio** - Demo stock portfolio with real-time prices, P&L tracking, allocation
2. **Live Prices** - Real-time stock prices by category (Tech, Sector Leaders, ETFs, Growth) with charts
3. **ML Predictions** - LSTM 21-day forecasts with confidence scores for position trading
4. **Rebalancing** - ML-enhanced portfolio allocation with paper/live trading
5. **Cloud Progress** - Training job monitoring, cost tracking, endpoint status

## Stock Universe (~33 stocks)

- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **Defensive:** JNJ, KO, PEP, MCD (consumer staples / healthcare for crash protection)
- **ETFs:** IWM, IJR, XLK, XLF, XLE, XLV, TLT (bonds), GLD (gold)
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN (speculative — capped at 2% each)

*Removed SPY/QQQ/DIA (redundant with individual holdings) and ARKK (overlaps growth picks).*

## ML Pipeline

- **Model:** 2-layer LSTM (64 units each, 0.2 dropout), Adam optimizer, MSE loss
- **Features:** 25 technical indicators (MAs, RSI, MACD, Bollinger Bands, volume, momentum, volatility, ATR)
- **Input:** 30-day lookback windows of OHLCV data + indicators
- **Output:** Predicted 21-day return percentage per symbol (position trading)
- **Hybrid system:** Vertex AI → Local ML → Explicit failure (no silent mock fallback)
- **Mock predictions:** Disabled by default (`allow_mock=False`). Only enabled for dashboard display, NEVER for trade decisions.
- **Key files:** `ml/lstm_model.py`, `ml/feature_engineering.py`, `ml/prediction_service.py`, `ml/hybrid_prediction_service.py`

## Data Layer

- `data/stock_api.py` - Stock data client via yfinance (free, no API key needed)
- Supports: current prices, quotes, batch quotes, historical OHLCV, fundamentals
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
| `config/rebalancing_config.json` | Portfolio rules (15% max position, $100 min trade) |
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
- Zero-commission trading (most modern brokers)

## Testing

```bash
python tests/unit/test_stock_api.py      # Stock API connectivity (yfinance)
python tests/unit/test_gcp.py            # GCP services verification
python tests/integration/run_backtest.py  # Backtesting
python -m pytest tests/                   # All tests
```

## Common Commands

```bash
# Development
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# ML Training (GCP)
./bin/train_now.sh
./bin/check_training.sh

# Dev setup
./bin/dev-setup.sh
```
