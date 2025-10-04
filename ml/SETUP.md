# ML Model Setup Notes

## 🐍 Python Version Issue

**Current Environment:** Python 3.13.7  
**TensorFlow Requirement:** Python 3.9 - 3.11

**Issue:** TensorFlow doesn't support Python 3.13 yet.

---

## ✅ Solutions

### Option 1: Use Vertex AI for Training (Recommended)
**Pros:**
- Google provides correct Python environment
- Uses your $50 GCP credit
- Scalable (can use GPUs)
- Production-ready workflow

**How:**
```bash
# Upload code to Cloud Storage
gsutil cp ml/*.py gs://crypto-ml-models-487/code/

# Submit training job to Vertex AI
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=crypto-lstm-training \
  --python-package-uris=gs://crypto-ml-models-487/code/trainer.tar.gz \
  --python-module=trainer.train \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1
```

### Option 2: Create Separate Python 3.11 Environment
**For local training:**

```bash
# Install Python 3.11
brew install python@3.11

# Create new environment
python3.11 -m venv venv_ml

# Activate
source venv_ml/bin/activate

# Install TensorFlow
pip install tensorflow==2.15.0
```

### Option 3: Use Google Colab (Free GPUs!)
**Easiest for now:**

1. Go to: https://colab.research.google.com/
2. Upload training notebook
3. Runtime → Change runtime type → GPU
4. Train for free!
5. Download trained model
6. Upload to Cloud Storage

---

## 🎯 Recommended Approach for This Project

**Short term (Testing):**
- Use Google Colab for quick model training
- Free GPU access
- No environment issues

**Long term (Production):**
- Use Vertex AI for automated training
- Integrated with GCP
- Uses $50 credit efficiently
- Professional ML workflow

---

## 📊 Current Status

✅ **Completed:**
- Historical data fetcher (1 year of data collected)
- Feature engineering pipeline (11 indicators)
- LSTM model architecture defined
- All code written and documented

🚧 **Pending:**
- TensorFlow installation (use Colab/Vertex AI)
- Model training
- Prediction generation
- Integration with dashboard

---

## 🚀 Next Steps

1. **Create Google Colab notebook** for training
2. Upload historical data
3. Train LSTM model
4. Download trained model
5. Upload to Cloud Storage
6. Generate predictions
7. Integrate with dashboard

**Estimated time:** 1-2 hours (mostly training time)

---

## 💡 Why This Is Actually Better

Training on Vertex AI/Colab instead of locally:
- ✅ Professional workflow (production-ready)
- ✅ Free GPU access (trains faster)
- ✅ No local environment issues
- ✅ Reproducible (same environment every time)
- ✅ Scales easily (can use bigger machines)
- ✅ Uses your $50 credit wisely

---

**Bottom Line:** We'll train on the cloud, which is the professional approach anyway! 🎯

