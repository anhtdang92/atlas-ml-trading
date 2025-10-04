# 🚀 Quick Start Guide

## Running the Dashboard

### First Time Setup
```bash
# 1. Navigate to project directory
cd "/Users/anhdang/Documents/GitHub/Kraken Crypto ML/Kraken_Cloud_ML_Strat"

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

### 📊 Portfolio View
- Current holdings and values
- Profit/Loss calculations
- Portfolio allocation pie chart
- Real-time updates from Kraken API

### 📈 Live Prices
- Real-time cryptocurrency prices
- 24-hour statistics (high, low, volume, change %)
- Interactive candlestick charts
- Volume charts

### 🧠 ML Predictions
- Coming soon in Phase 2
- LSTM model predictions
- Confidence scores

### ⚖️ Rebalancing
- Coming soon in Phase 3
- Trading recommendations
- Risk analysis

## Other Tools

### Test Kraken API
```bash
source venv/bin/activate

# Test public endpoints (no API keys needed)
python kraken_test.py

# Test private endpoints (requires API keys)
export KRAKEN_API_KEY="your_key"
export KRAKEN_API_SECRET="your_secret"
python kraken_test.py --with-auth
```

### View HTML Dashboard
```bash
# Simple HTML version with live prices
open index.html
```

## Adding API Keys (Optional)

When you're ready for live trading:

1. Get API keys from Kraken:
   - Go to Kraken → Settings → API
   - Create key with: "Query Funds" + "Create & Modify Orders"
   - Copy API Key and API Secret

2. Create secrets file:
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   ```

3. Edit `config/secrets.yaml` with your keys

4. Test authentication:
   ```bash
   python kraken_test.py --with-auth
   ```

## Stopping the Dashboard

Press `Ctrl+C` in the terminal where Streamlit is running.

## Next Steps

1. ✅ Dashboard is running with live prices
2. 📚 Build ML models (Phase 2)
3. 🧪 Backtest on QuantConnect
4. ☁️ Deploy to Google Cloud
5. 💰 Enable live trading (with caution!)

---

**Need help?** Check `README.md` for full documentation.

