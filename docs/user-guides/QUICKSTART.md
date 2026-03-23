# Quick Start Guide

## Running the Dashboard

### First Time Setup
```bash
# 1. Navigate to project directory
cd yfinance-ml-trading-strategy

# 2. Create virtual environment (if not already created)
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit Dashboard
```bash
# Activate virtual environment
source venv/bin/activate

# Launch dashboard
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## What You'll See

### Portfolio View
- Current holdings and values
- Profit/Loss calculations
- Portfolio allocation pie chart
- Real-time updates from Yahoo Finance (yfinance)

### Live Prices
- Real-time stock prices by category (Tech, Sector Leaders, ETFs, Growth)
- Daily statistics (high, low, volume, change %)
- Interactive candlestick charts
- Volume charts

### ML Predictions
- LSTM model 21-day price forecasts
- Confidence scores
- Position trading recommendations

### Rebalancing
- ML-enhanced portfolio allocation
- Trading recommendations
- Risk analysis

## Other Tools

### Test Stock API
```bash
source venv/bin/activate

# Test stock data connectivity (no API keys needed)
python tests/unit/test_stock_api.py
```

## Data Source

ATLAS uses Yahoo Finance (yfinance) for all market data. This is a free data source that requires no API keys or authentication. Stock data is available immediately after setup.

### Stock Universe (~30 stocks):
- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **ETFs:** SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, ARKK
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN

## Stopping the Dashboard

Press `Ctrl+C` in the terminal where Streamlit is running.

## Next Steps

1. Dashboard is running with live stock prices
2. Build ML models (Phase 2)
3. Backtest strategies
4. Deploy to Google Cloud
5. Enable paper/live trading (with caution!)

---

**Need help?** Check `README.md` for full documentation.
