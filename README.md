# ⚡ ATLAS - Stock ML Intelligence System

> **AI-Powered Stock Trading Dashboard with Machine Learning Price Predictions**

[![CI Pipeline](https://github.com/anhtdang92/kraken-ml-trading-strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/anhtdang92/kraken-ml-trading-strategy/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A cyberpunk-themed stock trading dashboard powered by **LSTM neural networks**, real-time Yahoo Finance data, and Google Cloud ML infrastructure. Features 25 technical indicators, walk-forward backtesting, and production risk controls.

**Author:** Anh Dang | **License:** MIT | **GCP Project:** `stock-ml-trading-487`

---

## Demo

<table>
<tr>
<td width="50%">

**Portfolio Dashboard**
![Portfolio View](docs/screenshots/portfolio_dashboard.png)
*Real-time portfolio tracking with P&L, sector allocation, and cyberpunk glass UI*

</td>
<td width="50%">

**ML Predictions**
![ML Predictions](docs/screenshots/ml_predictions.png)
*LSTM 21-day price forecasts with confidence scores for ~30 stocks*

</td>
</tr>
<tr>
<td width="50%">

**Model Evaluation**
![Training Curves](docs/screenshots/training_curves.png)
*Training/validation loss convergence with early stopping*

</td>
<td width="50%">

**Backtest Tearsheet**
![Tearsheet](docs/screenshots/tearsheet.png)
*Professional quant-style performance report with Sharpe, drawdown, monthly heatmap*

</td>
</tr>
</table>

> **Screenshots:** Run `streamlit run app.py` and see `notebooks/model_evaluation.ipynb` for interactive visualizations.

---

## Quick Start

```bash
# Option 1: Docker (one command)
docker compose up

# Option 2: Local
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** — no API keys needed!

---

## ML Model Performance

> Metrics from walk-forward backtesting on out-of-sample data. See [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb) for full analysis.

| Metric | LSTM | Ridge Regression | Momentum | Buy & Hold |
|--------|------|-----------------|----------|------------|
| **Directional Accuracy** | 55-62% | 50-54% | 48-52% | 50% |
| **RMSE** | 0.035-0.045 | 0.045-0.060 | 0.055-0.070 | N/A |
| **Sharpe Ratio** | 0.8-1.2 | 0.4-0.7 | 0.3-0.5 | 0.6-0.9 |
| **Max Drawdown** | 8-15% | 12-20% | 15-25% | 15-25% |
| **Information Coefficient** | 0.15-0.30 | 0.05-0.15 | -0.05-0.10 | 0.0 |

*Ranges reflect variation across symbols and time periods.*

---

## ✨ Features

### 🎨 Cyberpunk UI
- **Dark mode only** with neon aesthetics
- **Liquid glass** morphism effects
- **Font Awesome** icons with neon glows
- **Orbitron & Rajdhani** futuristic fonts

### 📊 Portfolio Management
- **Demo stock portfolio** with real-time prices
- **Live price** updates via Yahoo Finance
- **P&L calculations** with visual indicators
- **Sector allocation** pie charts

### 🧠 ML Price Predictions
- **LSTM neural networks** (2-layer, 64 units each)
- **25 technical indicators** (MAs, RSI, MACD, Bollinger, volume, momentum, volatility, ATR)
- **Hybrid system**: Vertex AI → Local ML → Enhanced Mock → Basic Mock (fallback chain)
- **Confidence scoring** for each prediction
- **21-day forecasts** for ~30 stocks (position trading)

### ⚖️ Portfolio Rebalancing
- **ML-enhanced allocation** strategy
- **Risk controls** (15% max position, 2% min, $100 min trade)
- **Paper trading** mode for testing
- **Zero-commission** trading (most modern brokers)

### ☁️ Cloud ML Training
- **Google Cloud** Vertex AI integration
- **Budget-optimized** ($3-8 per training run)
- **Real-time** progress tracking
- **Auto-scaling** prediction endpoints

---

## System Architecture

```mermaid
graph TB
    subgraph Data Layer
        YF[Yahoo Finance<br/>yfinance API] -->|OHLCV data| SA[StockAPI<br/>data/stock_api.py]
        SA -->|30 stocks| HDF[HistoricalDataFetcher<br/>ml/historical_data_fetcher.py]
    end

    subgraph Feature Engineering
        HDF -->|2yr daily data| FE[FeatureEngineer<br/>ml/feature_engineering.py]
        FE -->|25 technical indicators| SEQ[Sequence Builder<br/>30-day lookback windows]
    end

    subgraph ML Pipeline
        SEQ -->|X: n×30×25| LSTM[StockLSTM<br/>2-layer, 64 units<br/>ml/lstm_model.py]
        SEQ -->|X_flat| BL[Baseline Models<br/>Ridge, XGBoost<br/>ml/baseline_models.py]
        LSTM -->|predicted return| PS[PredictionService<br/>ml/prediction_service.py]
        BL -.->|comparison| PS
    end

    subgraph Prediction & Portfolio
        PS -->|per-symbol forecasts| HPS[HybridPredictionService<br/>Vertex AI → Local ML → TA]
        HPS -->|21-day predictions| PR[PortfolioRebalancer<br/>ml/portfolio_rebalancer.py]
        PR -->|buy/sell orders| PT[Paper Trading<br/>Risk Controls]
    end

    subgraph Infrastructure
        PS -.->|optional| VAI[Google Vertex AI<br/>Cloud Training]
        HDF -.->|optional| BQ[BigQuery<br/>Data Storage]
        ET[ExperimentTracker<br/>ml/experiment_tracker.py] -->|logs| RES[results/experiments/]
        DV[DataSchemas<br/>ml/validation/] -->|validates| FE
    end

    subgraph Dashboard
        APP[Streamlit Dashboard<br/>app.py] --> P1[Portfolio]
        APP --> P2[Live Prices]
        APP --> P3[ML Predictions]
        APP --> P4[Rebalancing]
        APP --> P5[Cloud Progress]
    end

    HPS --> APP
```

## 🚀 Detailed Setup

### Local Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Docker (One Command)
```bash
docker compose up
```

### With MLflow Tracking
```bash
docker compose --profile mlflow up
# MLflow UI: http://localhost:5000
# Dashboard: http://localhost:8501
```

### (Optional) Configure GCP
```bash
cp config/secrets.yaml.example config/secrets.yaml
./bin/quick-start.sh
```

---

## 🎯 Stock Universe (~30 stocks)

| Category | Symbols |
|----------|---------|
| **Tech (FAANG+)** | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA |
| **Sector Leaders** | JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN |
| **ETFs** | SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, ARKK |
| **Growth** | PLTR, CRWD, SNOW, SQ, COIN |

---

## 🧠 Train ML Models

### Local Training
```bash
# Interactive training launcher
./bin/train_now.sh
```

### Google Cloud (Vertex AI)
```bash
# Budget training ($3-8, 30-60 min)
./bin/train_now.sh  # Choose option 1

# Check training status
./bin/check_training.sh
```

### Track Progress
- **Dashboard:** http://localhost:8501 → ☁️ Cloud Progress
- **Terminal:** `./bin/check_training.sh`
- **Web:** [Vertex AI Console](https://console.cloud.google.com/vertex-ai)

---

## 📁 Project Structure

```
kraken-ml-trading-strategy/
│
├── 📱 app.py                 # Main Streamlit dashboard (ATLAS)
├── 📋 requirements.txt       # Pinned dependencies
│
├── 🚀 bin/                   # User scripts
│   ├── train_now.sh         # Train ML models
│   ├── check_training.sh    # Check training status
│   ├── quick-start.sh       # First-time setup
│   └── dev-setup.sh         # Development setup
│
├── ⚙️  config/               # Configuration
│   ├── config.yaml          # App settings (stocks, ML params, risk)
│   ├── gcp_config.yaml      # Google Cloud config
│   ├── secrets.yaml.example # API credentials template
│   └── rebalancing_config.json  # Portfolio rules
│
├── 📊 data/                  # Stock data layer
│   └── stock_api.py         # Yahoo Finance client (yfinance)
│
├── 🧠 ml/                    # Machine Learning
│   ├── prediction_service.py           # Main ML service
│   ├── hybrid_prediction_service.py    # Hybrid predictions (Vertex AI → Local → TA)
│   ├── lstm_model.py                   # LSTM architecture
│   ├── feature_engineering.py          # 25 technical indicators
│   ├── feature_importance.py          # SHAP & permutation importance analysis
│   ├── baseline_models.py             # Baseline comparison (Ridge, XGBoost)
│   ├── backtest_tearsheet.py          # Quant-style performance tearsheet
│   ├── experiment_tracker.py          # Experiment tracking (CSV + MLflow + W&B)
│   ├── historical_data_fetcher.py      # Data collection
│   ├── portfolio_rebalancer.py         # Rebalancing logic
│   └── validation/                    # Data validation schemas (pandera)
│
├── 🎨 ui/                    # UI components
│   ├── styles.py            # Cyberpunk theme
│   └── components.py        # Reusable UI elements
│
├── ☁️  gcp/                  # Google Cloud Platform
│   ├── training/            # ML training containers
│   └── deployment/          # Prediction services
│
├── 📓 notebooks/             # EDA & analysis notebooks
│   ├── eda_model_analysis.ipynb   # Exploratory data analysis
│   └── model_evaluation.ipynb     # LSTM evaluation, SHAP, tearsheet
├── 📦 models/                # Trained models (gitignored)
├── 📊 results/               # Experiment logs & metrics
├── 🧪 tests/                 # Unit & integration tests
│   ├── unit/test_ml_pipeline.py   # ML pipeline tests (pytest)
│   └── integration/walk_forward_backtest.py  # Walk-forward backtesting
└── 📚 docs/                  # Documentation
```

---

## 💰 Costs (Google Cloud)

| Service | Monthly Cost | Optimization |
|---------|--------------|--------------|
| Vertex AI Training | $5-10 | Preemptible instances (60% off) |
| Prediction Endpoint | $10-15 | Auto-scaling, scale-to-zero |
| BigQuery | $2-5 | Partitioned tables |
| Cloud Storage | $1-2 | Lifecycle policies |
| **Total** | **$18-32** | **~3 months on $50 credit** |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, Font Awesome |
| **ML/AI** | TensorFlow/Keras (LSTM), scikit-learn, SHAP, Google Vertex AI |
| **Experiment Tracking** | CSV + MLflow + Weights & Biases (optional) |
| **Model Evaluation** | SHAP, permutation importance, backtest tearsheets |
| **Data Validation** | Pandera schemas for OHLCV, features, and predictions |
| **Data** | Yahoo Finance (yfinance - free, no API key) |
| **Storage** | Google BigQuery, Cloud Storage |
| **Infrastructure** | Google Cloud Platform, Docker Compose |
| **CI/CD** | GitHub Actions (lint, test, coverage, Docker build) |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **⚡ Portfolio** | Demo stock portfolio with real-time prices, P&L, allocation |
| **↗ Live Prices** | Real-time stock prices by category with candlestick/line charts |
| **◉ ML Predictions** | LSTM 21-day forecasts with confidence scores |
| **◉ Rebalancing** | ML-enhanced portfolio allocation with paper/live trading |
| **☁️ Cloud Progress** | Training job monitoring, cost tracking, endpoint status |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# ML pipeline tests (feature engineering, data validation, portfolio logic)
python -m pytest tests/unit/test_ml_pipeline.py -v

# Stock API connectivity
python -m pytest tests/unit/test_stock_api.py -v

# Walk-forward backtesting with financial metrics
python tests/integration/walk_forward_backtest.py

# Baseline model comparison
python -m ml.baseline_models

# GCP connection
python tests/unit/test_gcp.py
```

---

## 🔐 Security

- ✅ No API keys needed for stock data (yfinance is free)
- ✅ GCP service account keys stored in `config/keys/` (gitignored)
- ✅ Service accounts with minimal permissions
- ✅ Paper trading mode by default
- ✅ No private keys in code

---

## 📈 Performance

- **Prediction speed:** <100ms per stock
- **Data refresh:** 60-second cache
- **Dashboard load:** <2 seconds
- **Training time:** 30-60 minutes (7 tech stocks on Vertex AI)
- **Inference:** <50ms via Vertex AI endpoint

---

## 🎯 Roadmap

- [x] Real-time stock portfolio tracking
- [x] LSTM price predictions (25 features)
- [x] Google Cloud ML training
- [x] Portfolio rebalancing with risk controls
- [x] Cyberpunk UI theme
- [x] Walk-forward backtesting with financial metrics
- [x] Baseline model comparison (Ridge, XGBoost)
- [x] Experiment tracking pipeline (CSV + MLflow + W&B)
- [x] Data validation schemas
- [x] CI/CD with GitHub Actions (lint, test, coverage, Docker)
- [x] EDA & model analysis notebook
- [x] SHAP & permutation feature importance analysis
- [x] Backtest tearsheet (Sharpe, Sortino, Calmar, drawdown)
- [x] Docker Compose one-command setup
- [x] Model evaluation notebook with training curves & confusion matrix
- [ ] Automated trading via Alpaca API
- [ ] Multi-timeframe analysis
- [ ] Earnings calendar integration
- [ ] Options strategy overlay
- [ ] Mobile-responsive dashboard

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Stock trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before making investment decisions.

---

**Built with 💙 by Anh Dang**

🚀 **ATLAS** - Stock ML Intelligence System
