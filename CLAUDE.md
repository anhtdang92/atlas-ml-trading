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
- **Training:** `./bin/train_now.sh` (interactive ML training launcher)
- **Setup:** `./bin/quick-start.sh` (first-time environment setup)
- **Tests:** `python -m pytest tests/` or `python tests/unit/test_stock_api.py`

## Dashboard Pages

1. **Portfolio** - Demo stock portfolio with real-time prices, P&L tracking, allocation
2. **Live Prices** - Real-time stock prices by category (Tech, Sector Leaders, ETFs, Growth) with charts
3. **ML Predictions** - LSTM 21-day forecasts with confidence scores for position trading
4. **Rebalancing** - ML-enhanced portfolio allocation with paper/live trading
5. **Cloud Progress** - Training job monitoring, cost tracking, endpoint status

## Stock Universe (~30 stocks)

- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **ETFs:** SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, ARKK
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN

## ML Pipeline

- **Model:** 2-layer LSTM (64 units each, 0.2 dropout), Adam optimizer, MSE loss
- **Features:** 25 technical indicators (MAs, RSI, MACD, Bollinger Bands, volume, momentum, volatility, ATR)
- **Input:** 30-day lookback windows of OHLCV data + indicators
- **Output:** Predicted 21-day return percentage per symbol (position trading)
- **Hybrid system:** Vertex AI → Local ML → Enhanced Mock → Basic Mock (fallback chain)
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

- Max position: 15%, Min position: 2%
- Minimum trade size: $100
- ML weight factor: 0.3 (how much predictions affect allocation)
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
