# Train Stock Price Prediction Models on Google Cloud

## Quick Start - Train Models Now!

You have everything set up! Here's how to train ML models for stock price prediction using your Google Cloud credits:

### Option 1: Quick Training (Recommended for Testing)
```bash
# Train models for major stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
bash scripts/deployment/deploy_budget_training.sh
```

**Cost:** $3-8 per training run
**Time:** 30-60 minutes
**What it trains:** LSTM models for ~30 stocks

### Option 2: Full Training (All Supported Stocks)
```bash
# Train models for all ~30 stocks including sector leaders, ETFs, and growth stocks
bash scripts/deployment/deploy_vertex_training.sh
```

**Cost:** $8-15 per training run
**Time:** 1-2 hours
**What it trains:** LSTM models for ~30 stocks across all categories

---

## What Gets Trained

### Model Architecture
- **Type:** LSTM (Long Short-Term Memory) Neural Network
- **Layers:** 2-layer LSTM with 64 units each
- **Dropout:** 0.2 for regularization
- **Features:** 25 technical indicators
  - Moving averages (7, 14, 30-day)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume indicators
  - Price momentum
  - Volatility measures
  - ATR (Average True Range)

### Training Data
- **Source:** Yahoo Finance via yfinance (free, no API key needed)
- **Duration:** 365 days of OHLCV data
- **Prediction Target:** 21-day future price returns (position trading)
- **Validation:** 80/20 train/test split with early stopping

### Stock Universe (~30 Stocks)
**Tech (FAANG+):**
1. **AAPL** (Apple)
2. **MSFT** (Microsoft)
3. **GOOGL** (Alphabet)
4. **AMZN** (Amazon)
5. **NVDA** (NVIDIA)
6. **META** (Meta Platforms)
7. **TSLA** (Tesla)

**Sector Leaders:**
8. **JPM** (JPMorgan Chase)
9. **UNH** (UnitedHealth)
10. **XOM** (ExxonMobil)
11. **CAT** (Caterpillar)
12. **PG** (Procter & Gamble)
13. **HD** (Home Depot)
14. **NEE** (NextEra Energy)
15. **AMT** (American Tower)
16. **LIN** (Linde)

**ETFs:**
17. **SPY** (S&P 500)
18. **QQQ** (Nasdaq 100)
19. **DIA** (Dow Jones)
20. **IWM** (Russell 2000)
21. **XLK** (Tech Sector)
22. **XLF** (Financial Sector)
23. **XLE** (Energy Sector)
24. **XLV** (Healthcare Sector)
25. **ARKK** (ARK Innovation)

**Growth:**
26. **PLTR** (Palantir)
27. **CRWD** (CrowdStrike)
28. **SNOW** (Snowflake)
29. **SQ** (Block)
30. **COIN** (Coinbase)

---

## Cost Breakdown

### Budget-Optimized Training (Recommended)
```
Machine Type: e2-standard-4
GPU: 1x NVIDIA Tesla T4
Preemptible: Yes (60-80% savings)
Cost per run: $3-8
Credits used: ~10-16% of $50 budget
```

### Regular Training
```
Machine Type: n1-standard-4
GPU: 1x NVIDIA Tesla T4
Preemptible: No
Cost per run: $15-25
Credits used: ~30-50% of $50 budget
```

**Recommendation:** Use budget-optimized training! You can train 6-15 times with your $50 credit.

---

## Step-by-Step Training Guide

### Step 1: Verify Setup
```bash
# Check your GCP project
gcloud config get-value project

# Should show: stock-ml-trading-487

# Check if you're authenticated
gcloud auth list
```

### Step 2: Enable Required APIs (One-time setup)
```bash
# Run this only once
bash scripts/deployment/enable_apis.sh
```

This enables:
- Vertex AI API
- Container Registry API
- Cloud Storage API
- BigQuery API

### Step 3: Set Up Storage (One-time setup)
```bash
# Create storage buckets for models
bash scripts/deployment/setup_storage.sh
```

### Step 4: Train Models!
```bash
# Budget-optimized training (recommended)
bash scripts/deployment/deploy_budget_training.sh
```

### Step 5: Monitor Training
```bash
# Option 1: Stream logs in terminal
gcloud ai custom-jobs list --region=us-central1

# Get the job name from above, then:
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1

# Option 2: Check in web console
# Visit: https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

### Step 6: Deploy Trained Models
```bash
# After training completes, deploy to endpoint
bash scripts/deployment/deploy_budget_endpoint.sh
```

### Step 7: Test Predictions
```bash
# Test the deployed model
python gcp/deployment/test_endpoint.py
```

---

## Training Progress Tracking

### Check Training Status
```bash
# List all training jobs
gcloud ai custom-jobs list --region=us-central1 --sort-by=~createTime --limit=5

# Get detailed status
gcloud ai custom-jobs describe JOB_NAME --region=us-central1
```

### View Training Metrics
Your Streamlit dashboard shows training progress in real-time!
- Go to **Cloud Progress** tab
- See live training status
- View cost tracking
- Monitor job completion

---

## Advanced: Custom Training Configuration

### Train Specific Stocks
Create a custom training script:

```bash
# Edit the symbols you want to train
export SYMBOLS="AAPL,MSFT,GOOGL"  # Only train these 3

# Run custom training
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="custom-stock-training" \
  --worker-pool-spec=machine-type=e2-standard-4,replica-count=1,\
accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,\
container-image-uri=gcr.io/stock-ml-trading-487/stock-lstm-training:latest
```

### Adjust Training Parameters

Edit `scripts/deployment/deploy_budget_training.sh`:

```bash
# Line 93-96: Adjust these parameters
--lookback_days=30      # Days of history to look at (default: 30)
--prediction_horizon=21  # Days ahead to predict (default: 21)
--epochs=50            # Training epochs (more = better but slower)
--batch_size=32        # Batch size (32 is optimal for T4 GPU)
```

**Recommendations:**
- `lookback_days=30`: Good for position trading predictions
- `lookback_days=14`: Better for shorter-term swing trades
- `epochs=50-100`: Balance between quality and cost
- `batch_size=32`: Optimal for T4 GPU

---

## What Happens During Training

### Phase 1: Data Collection (5-10 min)
- Fetches 365 days of OHLCV data from Yahoo Finance via yfinance
- Downloads for all specified stocks
- Validates data quality

### Phase 2: Feature Engineering (5-10 min)
- Calculates 25 technical indicators
- Normalizes data
- Creates time series sequences
- Splits into train/validation sets

### Phase 3: Model Training (20-40 min)
- Trains LSTM neural network
- Uses GPU acceleration
- Early stopping on validation loss
- Saves best model weights

### Phase 4: Model Saving (2-5 min)
- Uploads trained model to Cloud Storage
- Registers in Vertex AI Model Registry
- Creates model metadata

**Total Time:** 30-60 minutes per training run

---

## After Training - Deploy & Use

### 1. Deploy to Endpoint
```bash
bash scripts/deployment/deploy_budget_endpoint.sh
```

This creates a prediction endpoint with:
- Auto-scaling (scales to zero when not used)
- Fast inference (<100ms per prediction)
- Cost: Only pay when making predictions

### 2. Use in Your Dashboard
Your Streamlit app automatically uses deployed models!
- Go to **ML Predictions** tab
- Select "Vertex AI" mode
- Get real ML predictions

### 3. Test via API
```python
from google.cloud import aiplatform

# Initialize client
aiplatform.init(project="stock-ml-trading-487", location="us-central1")

# Get endpoint
endpoint = aiplatform.Endpoint.list()[0]

# Make prediction
prediction = endpoint.predict(instances=[{
    "symbol": "AAPL",
    "lookback_days": 30
}])

print(f"AAPL 21-day prediction: {prediction.predictions[0]}")
```

---

## Best Practices

### Cost Optimization
1. **Use preemptible instances** (60-80% savings)
2. **Train during off-peak hours** (slightly cheaper)
3. **Batch train multiple models** in one job
4. **Use budget alerts** to avoid overspending
5. **Delete old endpoints** when not needed

### Training Schedule
```bash
# Weekly training (recommended)
0 0 * * 0 cd /path/to/project && bash scripts/deployment/deploy_budget_training.sh

# This keeps models fresh with latest market data
```

### Model Quality
1. **Monitor validation loss** - should decrease consistently
2. **Check prediction accuracy** on test data
3. **Retrain weekly** to adapt to market changes
4. **Compare with baseline** (simple moving average)

---

## Troubleshooting

### Error: "Quota exceeded"
```bash
# Check quotas
gcloud compute project-info describe --project=stock-ml-trading-487

# Request quota increase:
# https://console.cloud.google.com/iam-admin/quotas
```

### Error: "GPU not available"
Try different regions:
```bash
# Use europe-west4 instead
--region=europe-west4
```

### Training Fails After 10 Minutes
Preemptible instance was stopped. Solutions:
1. Run again (it's random, usually succeeds on retry)
2. Use non-preemptible (more expensive)
3. Train during off-peak hours

### Out of Memory
Reduce batch size:
```bash
--batch_size=16  # Instead of 32
```

---

## Monitoring Costs

### Check Current Spend
```bash
# Via gcloud
gcloud billing accounts list
gcloud billing projects describe stock-ml-trading-487

# Or visit:
# https://console.cloud.google.com/billing
```

### Set Budget Alerts
Already configured at $50 limit with alerts at:
- 50% ($25)
- 75% ($37.50)
- 90% ($45)
- 100% ($50)

---

## Next Steps

### After First Training
1. Check training succeeded in Cloud Progress tab
2. Deploy model to endpoint
3. Test predictions in ML Predictions tab
4. Compare with enhanced mock predictions
5. Set up weekly retraining

### Optimize Your Models
1. **Experiment with hyperparameters**
   - Try different lookback periods
   - Adjust LSTM units (64, 100, 128)
   - Test different learning rates

2. **Add more features**
   - Market sentiment data
   - News indicators
   - Sector correlation metrics

3. **Ensemble models**
   - Combine multiple models
   - Vote on predictions
   - Improve accuracy

---

## Resources

- **Vertex AI Docs:** https://cloud.google.com/vertex-ai/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator
- **Your Project Console:** https://console.cloud.google.com/home/dashboard?project=stock-ml-trading-487

---

## Ready to Train!

You're all set! Run this command to start training:

```bash
bash scripts/deployment/deploy_budget_training.sh
```

Then watch the progress in your **Cloud Progress** dashboard tab!

**Estimated cost:** $3-8 per run
**Time:** 30-60 minutes
**Result:** Trained LSTM models for ~30 stocks ready for 21-day predictions!
