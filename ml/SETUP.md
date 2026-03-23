# ML Model Setup Notes

## Python Version Issue

**TensorFlow Requirement:** Python 3.9 - 3.11

If your local Python is 3.12+, TensorFlow won't install locally. Use one of the solutions below.

---

## Solutions

### Option 1: Use Vertex AI for Training (Recommended)
- Google provides correct Python environment
- Uses your $50 GCP credit
- Scalable (can use GPUs)

```bash
# Submit training job to Vertex AI
./bin/train_now.sh
```

### Option 2: Create Separate Python 3.11 Environment

```bash
# Install Python 3.11
brew install python@3.11

# Create new environment
python3.11 -m venv venv_ml
source venv_ml/bin/activate

# Install TensorFlow
pip install tensorflow==2.15.0
pip install -r requirements.txt
```

### Option 3: Use Google Colab (Free GPUs)

1. Go to: https://colab.research.google.com/
2. Upload training notebook
3. Runtime → Change runtime type → GPU
4. Train for free
5. Download trained model to `models/`

---

## Current Status

**Completed:**
- Historical data fetcher (2 years of stock data via yfinance)
- Feature engineering pipeline (25 technical indicators)
- LSTM model architecture (2-layer, 64 units)
- Prediction service with hybrid fallback chain
- Portfolio rebalancer with risk controls

**Pending:**
- TensorFlow installation (use Colab/Vertex AI)
- Model training for all 30 stocks
- Prediction endpoint deployment

---

## Stock Universe

The system tracks ~30 stocks across 4 categories:
- **Tech (FAANG+):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Sector Leaders:** JPM, UNH, XOM, CAT, PG, HD, NEE, AMT, LIN
- **ETFs:** SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, ARKK
- **Growth:** PLTR, CRWD, SNOW, SQ, COIN

Stock universe is defined in `data/stock_api.py` (`STOCK_UNIVERSE` dict) as the single source of truth.
