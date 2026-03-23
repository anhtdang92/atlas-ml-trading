# NOVA - Crypto Intelligence System

## Project Overview

AI-powered cryptocurrency trading dashboard with ML price predictions (LSTM neural networks), real-time Kraken API integration, and Google Cloud ML infrastructure. Built with Python 3.9+ and Streamlit.

**Author:** Anh Dang | **License:** MIT | **GCP Project:** `crypto-ml-trading-487`

## Architecture

```
app.py (Streamlit Dashboard - 2,699 lines)
├── data/          → Kraken API integration (public + authenticated)
├── ml/            → ML pipeline (LSTM, feature engineering, predictions)
├── gcp/           → Google Cloud Platform (Vertex AI, BigQuery, Storage)
├── ui/            → Cyberpunk-themed UI components (glass morphism, neon)
├── config/        → YAML/JSON configuration files
├── bin/           → Shell scripts (training, setup, status checks)
├── tests/         → Unit, integration, and manual tests
├── docs/          → 18 documentation files
└── models/        → Trained model artifacts (gitignored)
```

## Key Entry Points

- **Dashboard:** `streamlit run app.py` (localhost:8501)
- **Training:** `./bin/train_now.sh` (interactive ML training launcher)
- **Setup:** `./bin/quick-start.sh` (first-time environment setup)
- **Tests:** `python -m pytest tests/` or individual test files

## Dashboard Pages

1. **Portfolio** - Real-time Kraken portfolio sync, P&L tracking, asset allocation
2. **Live Prices** - Real-time prices for BTC, ETH, SOL, ADA, DOT, XRP with charts
3. **ML Predictions** - LSTM 7-day forecasts with confidence scores
4. **Rebalancing** - ML-enhanced portfolio allocation with paper/live trading
5. **Cloud Progress** - Training job monitoring, cost tracking, endpoint status

## ML Pipeline

- **Model:** 2-layer LSTM (50 units each, 0.2 dropout), Adam optimizer, MSE loss
- **Features:** 11 technical indicators (MAs, RSI, volume, momentum, volatility)
- **Input:** 7-day lookback windows of OHLCV data + indicators
- **Output:** Predicted 7-day return percentage per symbol
- **Hybrid system:** Vertex AI → Local ML → Enhanced Mock → Basic Mock (fallback chain)
- **Key files:** `ml/lstm_model.py`, `ml/feature_engineering.py`, `ml/prediction_service.py`, `ml/hybrid_prediction_service.py`

## Data Layer

- `data/kraken_api.py` - Public API client (rate-limited 15 req/min, retry with backoff)
- `data/kraken_auth.py` - Authenticated API client (HMAC-SHA512 signed requests)
- Credentials in `config/secrets.yaml` (gitignored)

## GCP Integration

- **Vertex AI** - Cloud ML training (n1-standard-4, optional Tesla T4 GPU) and prediction endpoints
- **BigQuery** - Dataset `crypto_data` with tables for prices, predictions, trades, metrics
- **Cloud Storage** - Model artifacts and training data
- **Cost:** ~$18-32/month, $3-8 per training run (preemptible instances, scale-to-zero)
- **Key files:** `gcp/deployment/vertex_prediction_service.py`, `gcp/scripts/`

## Configuration

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main app config (pairs, ML params, scheduling, risk limits) |
| `config/gcp_config.yaml` | GCP settings (Vertex AI, BigQuery, Storage, IAM) |
| `config/secrets.yaml` | Kraken API keys (gitignored, see `secrets.yaml.example`) |
| `config/rebalancing_config.json` | Portfolio rules (40% max position, $50 min trade, 0.16% fee) |
| `.env` | Environment variables for GCP project/region/buckets |

## Risk Controls

- Max position: 40%, Min position: 10%
- Minimum trade size: $50
- ML weight factor: 0.3 (how much predictions affect allocation)
- Confidence threshold: 0.6
- Paper trading mode enabled by default

## Testing

```bash
python tests/unit/kraken_test.py        # Kraken API connectivity
python tests/unit/test_gcp.py           # GCP services verification
python tests/integration/run_backtest.py # Backtesting
python -m pytest tests/                  # All tests
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
