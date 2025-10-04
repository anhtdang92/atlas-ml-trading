# Crypto ML Trading Dashboard

A machine learning-powered cryptocurrency trading application that uses LSTM neural networks to predict price movements and automatically rebalance a crypto portfolio using the Kraken API. Built with Streamlit, TensorFlow, and Google Cloud Platform.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![GCP](https://img.shields.io/badge/Google%20Cloud-Platform-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎉 Current Status: Phase 1 Complete + Portfolio Integration!

**Last Updated:** October 4, 2025

### ✅ What's Working Now:
- **Streamlit Dashboard** - Fully functional web application with professional UI
- **Live Price Tracking** - Real-time data from Kraken API (6+ cryptocurrencies)
- **Interactive Charts** - Candlestick charts with multiple time intervals
- **Real Portfolio Integration** - Connected to actual Kraken account via API
- **Authenticated API** - Secure connection to view account balances and holdings
- **Staking Support** - Separate tracking for staked/bonded assets
- **Advanced KPIs** - Easy-to-read cards showing liquid value, staked value, P&L, and total assets
- **Refresh Functionality** - One-click data refresh from Kraken
- **API Client** - Kraken API client with retry logic and rate limiting
- **HTML Dashboard** - Alternative standalone HTML interface

### 🆕 New Features:
- **🔐 Authenticated Portfolio** - View your real Kraken holdings
- **🔒 Staking Dashboard** - Track bonded/staked assets earning rewards
- **💰 Liquid vs Staked** - Clear separation of tradeable vs locked assets
- **🎨 Improved UI** - Large, colorful KPI cards with better readability
- **🔄 Live Refresh** - Manual refresh button to update data on demand

### 🚧 In Progress:
- Phase 2: ML Model Development (LSTM for price predictions)

### 📍 Current Phase: Phase 1 ✅ COMPLETE | Portfolio Integration ✅ COMPLETE

## 🎯 Project Overview

This application combines quantitative finance, machine learning, and automated trading to:
- **Predict** 7-day cryptocurrency price movements using LSTM models
- **Analyze** 4-12 major cryptocurrencies (BTC, ETH, SOL, ADA, etc.)
- **Rebalance** portfolio weekly based on ML predictions and risk parameters
- **Execute** trades automatically via Kraken API
- **Display** real-time portfolio analytics and trading signals in an interactive dashboard
- **Backtest** strategies on historical data using QuantConnect

## ✨ Key Features

### 📊 ML Prediction Engine
- LSTM neural network trained on historical OHLCV data
- 7-day prediction windows with feature engineering (MA, RSI, volume indicators)
- Model versioning and automated weekly retraining
- Confidence intervals and performance metrics

### 💼 Intelligent Portfolio Management
- ML-enhanced rebalancing strategy
- Risk management: Max 40% position size, minimum $50 trades
- Real-time portfolio tracking and valuation
- Paper trading mode for strategy validation

### 📈 Interactive Streamlit Dashboard
- **Portfolio View**: Current holdings, values, and performance
- **Predictions View**: ML forecasts with confidence scores and visualizations
- **Rebalancing View**: Clear action items and trade recommendations
- One-click trade execution with confirmations

### 🔄 Cloud Automation
- Automated weekly rebalancing via Cloud Scheduler
- Serverless architecture on Google Cloud Run
- Comprehensive logging to BigQuery
- Email/Slack notifications for trade summaries

### 🧪 QuantConnect Backtesting
- Historical strategy validation (2-3 years of data)
- Kraken-specific brokerage modeling
- Performance metrics: Sharpe ratio, max drawdown, total return
- ML strategy vs. equal-weight baseline comparison

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **ML Framework** | TensorFlow/Keras (LSTM) |
| **Cloud Platform** | Google Cloud Platform |
| **Hosting** | Cloud Run |
| **ML Training** | Vertex AI |
| **Database** | BigQuery |
| **Storage** | Cloud Storage |
| **Automation** | Cloud Scheduler, Cloud Functions |
| **Secrets** | Secret Manager |
| **Trading API** | Kraken API (python-kraken-sdk) |
| **Backtesting** | QuantConnect |
| **Data Sources** | Kraken API, CoinAPI |

## 📁 Project Structure

```
crypto-trading-app/
├── .cursorrules              # Cursor AI project rules
├── project-context.md        # Detailed project documentation
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit dashboard main file
│
├── data/
│   ├── data_fetcher.py      # Kraken API data collection
│   └── bigquery_handler.py   # BigQuery read/write operations
│
├── ml/
│   ├── lstm_model.py        # LSTM model architecture
│   ├── train.py             # Model training script (Vertex AI)
│   ├── predict.py           # Generate weekly predictions
│   └── feature_engineering.py
│
├── trading/
│   ├── kraken_client.py     # Kraken API wrapper
│   ├── portfolio.py         # Portfolio tracking logic
│   └── rebalancer.py        # Rebalancing calculation and execution
│
├── cloud_functions/
│   └── weekly_rebalance/    # Cloud Function for automation
│       ├── main.py
│       └── requirements.txt
│
├── quantconnect/
│   └── backtest_strategy.py # QuantConnect algorithm
│
├── config/
│   ├── config.yaml          # App configuration
│   └── secrets.yaml.example # API key template
│
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_lstm.py
│   └── test_rebalancer.py
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Internet connection (for Kraken API)
- *Optional:* Google Cloud Platform account (for cloud deployment - Phase 7)
- *Optional:* Kraken API keys (for live trading - not needed yet!)
- *Optional:* QuantConnect account (for backtesting - Phase 6)

### Quick Start (5 Minutes)

1. **Clone the repository**
   ```bash
   git clone https://github.com/anhtdang92/Kraken_Cloud_ML_Strat.git
   cd Kraken_Cloud_ML_Strat
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - Dashboard will automatically open at `http://localhost:8501`
   - Or open `index.html` for the HTML version

That's it! No API keys needed to start exploring live crypto prices.

### Testing Kraken API

```bash
# Test API connectivity (no keys required)
python kraken_test.py

# Test with authentication (requires API keys)
export KRAKEN_API_KEY="your_key"
export KRAKEN_API_SECRET="your_secret"
python kraken_test.py --with-auth
```

### Adding API Keys (View Your Real Portfolio!)

1. **Get API keys from Kraken:**
   - Go to Kraken → Settings → API
   - Create key named "ML Dashboard Read Only"
   - Select permissions:
     - ✅ Query (Funds)
     - ✅ Query open orders & trades
     - ✅ Query closed orders & trades
     - ✅ Query ledger entries (optional)
   - ⚠️ **DO NOT** enable: Deposit, Withdraw, Create & Modify Orders (for now)

2. **Configure secrets:**
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   # Edit secrets.yaml and paste your API key and Private key
   ```

3. **Test authentication:**
   ```bash
   python test_auth.py
   ```

4. **View your portfolio:**
   - Restart the dashboard: `streamlit run app.py`
   - You'll see your real holdings, including staked assets!
   - Click "🔄 Refresh Data" to update from Kraken

### Optional: Configure GCP (Phase 7 - Cloud Deployment)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Upload secrets to Secret Manager
gcloud secrets create kraken-api-key --data-file=config/secrets.yaml
```

### Configuration

Edit `config/config.yaml` to customize:
- Cryptocurrency symbols to trade
- Rebalancing schedule (default: Sunday 10 PM CDT)
- Risk parameters (max position size, minimum trade amount)
- ML model hyperparameters
- Initial portfolio value

## 📖 Usage

### Running the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the dashboard.

### Training ML Models

```bash
# Local training
python ml/train.py --symbols BTC ETH SOL ADA --epochs 100

# Vertex AI training
python ml/train.py --use-vertex-ai --machine-type n1-standard-4
```

### Manual Rebalancing

```bash
# Paper trading (dry run)
python trading/rebalancer.py --paper-trade

# Live trading (requires confirmation)
python trading/rebalancer.py --live
```

### Running Backtests

Upload `quantconnect/backtest_strategy.py` to QuantConnect and configure:
- Start date: 2-3 years ago
- End date: Today
- Initial capital: $5,000
- Brokerage: Kraken

## 🏗️ Development Roadmap

### Phase 1: Data Pipeline ✅ COMPLETE
- [x] Kraken API data fetcher
- [x] Kraken API client with retry logic
- [x] Rate limiting and error handling
- [x] Public endpoint integration (no API keys required)
- [x] Private endpoint authentication (API keys)
- [x] Streamlit dashboard foundation
- [x] Live price tracking for 6+ cryptocurrencies
- [x] Interactive candlestick charts
- [x] Portfolio view with real-time updates
- [x] Real Kraken account integration
- [x] Staking/bonded assets tracking
- [x] Professional UI with large KPI cards
- [x] Manual refresh functionality
- [x] HTML alternative dashboard
- [ ] BigQuery schema and storage (Phase 7)
- [ ] Automated daily data updates (Phase 7)

### Phase 2: ML Foundation 🚧
- [ ] LSTM model architecture
- [ ] Feature engineering pipeline
- [ ] Local model training
- [ ] Vertex AI integration

### Phase 3: Dashboard Core 📋
- [ ] Portfolio view with test data
- [ ] Price charts and visualizations
- [ ] Basic UI/UX design

### Phase 4: Predictions Integration 📋
- [ ] Display ML predictions
- [ ] Confidence intervals
- [ ] Model performance metrics

### Phase 5: Rebalancing Logic 📋
- [ ] Portfolio calculation engine
- [ ] Trade recommendation generator
- [ ] Paper trading validation

### Phase 6: Backtesting 📋
- [ ] QuantConnect strategy implementation
- [ ] Performance analysis
- [ ] Strategy comparison

### Phase 7: Cloud Deployment 📋
- [ ] Cloud Run deployment
- [ ] Cloud Scheduler setup
- [ ] Cloud Function automation

### Phase 8: Live Trading 📋
- [ ] Kraken API integration
- [ ] Order execution
- [ ] Trade verification
- [ ] Production monitoring

## 🧠 ML Model Architecture

### LSTM Configuration
- **Layers**: 2 LSTM layers with 50 units each
- **Dropout**: 0.2 to prevent overfitting
- **Input**: 7-day historical windows (OHLCV + features)
- **Output**: Predicted 7-day return percentage
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam

### Feature Engineering
- Moving Averages: 7-day, 14-day, 30-day
- Relative Strength Index (RSI)
- Volume indicators
- Price momentum
- Volatility measures

### Training Strategy
- Rolling window validation
- Weekly model retraining
- Model versioning in Cloud Storage
- Performance monitoring and alerting

## 🔒 Security & Risk Management

### API Security
- **Never commit API keys** to version control
- All secrets stored in Google Secret Manager
- API key rotation every 90 days
- Rate limit handling and retry logic

### Trading Risk Controls
- Maximum position size: 40% per cryptocurrency
- Minimum trade amount: $50 to avoid excessive fees
- Stop-loss mechanisms (planned)
- Paper trading validation before live deployment

### Data Security
- Parameterized SQL queries (SQL injection prevention)
- Encrypted data in transit and at rest
- Audit logging for all trades
- Regular security reviews

## 💰 Cost Optimization

**Target**: Run on $50 Google Cloud credit for 3-4 months

- Use Cloud Run's free tier (2M requests/month)
- Optimize BigQuery queries (avoid full table scans)
- Schedule model training during off-peak hours
- Use preemptible instances for Vertex AI
- Monitor billing alerts

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_rebalancer.py -v

# Run with coverage
pytest --cov=. tests/
```

## 📊 Performance Metrics

Backtest results will be compared against:
- **Benchmark**: Equal-weight rebalancing strategy
- **Metrics**:
  - Total Return
  - Sharpe Ratio (target: > 1.5)
  - Maximum Drawdown (target: < 30%)
  - Win Rate
  - Average Trade P&L

## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guide
- Add comprehensive docstrings
- Include type hints
- Pass flake8 linting
- Write unit tests for new features

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [Kraken API](https://www.kraken.com/features/api) for cryptocurrency trading
- [QuantConnect](https://www.quantconnect.com/) for backtesting framework
- [Streamlit](https://streamlit.io/) for rapid dashboard development
- [TensorFlow](https://www.tensorflow.org/) for ML framework

## 📧 Contact

Project Link: [https://github.com/yourusername/crypto-trading-app](https://github.com/yourusername/crypto-trading-app)

---

## ⚠️ Disclaimer

**This software is for educational purposes only. Cryptocurrency trading carries significant risk of loss. Past performance does not guarantee future results. Always conduct your own research and never invest more than you can afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.**

---

## 📚 Additional Resources

- [Kraken API Documentation](https://docs.kraken.com/rest/)
- [QuantConnect Documentation](https://www.quantconnect.com/docs/v2)
- [LSTM for Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built with ❤️ as a learning project | Spring, TX | 2025**
