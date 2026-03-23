# Google Cloud ML Setup Guide - $50 Budget Optimized

This guide will help you set up Google Cloud Platform ML infrastructure for your stock ML trading system within a $50 budget.

## 🎯 Budget Target
- **Total Budget**: $50 GCP credit
- **Target Duration**: 3-4 months
- **Monthly Cost**: ~$15-18/month
- **Cost Optimization**: 60-80% savings through preemptible instances and auto-scaling

## 📋 Prerequisites

### 1. Google Cloud Account
- GCP account with $50 credit
- Billing enabled on your project
- Google Cloud SDK installed and authenticated

### 2. Local Environment
- Python 3.9+ installed
- Docker installed
- Git installed

### 3. Authentication
```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project
gcloud config set project stock-ml-trading-487
```

## 🚀 Quick Setup (One Command)

Run the complete budget-optimized setup:

```bash
# Make scripts executable
chmod +x gcp/scripts/*.sh

# Run complete setup
bash gcp/scripts/setup_budget_optimized.sh
```

This will:
- ✅ Enable required APIs
- ✅ Create service accounts with minimal permissions
- ✅ Set up cost-optimized storage buckets
- ✅ Create partitioned BigQuery tables
- ✅ Configure budget alerts
- ✅ Install required Python packages

## 📊 Cost Breakdown

| Service | Monthly Cost | Optimization |
|---------|-------------|--------------|
| Vertex AI Training | $5-8 | Preemptible instances (60-80% savings) |
| Vertex AI Endpoint | $10-15 | Auto-scaling, scale-to-zero |
| BigQuery | $2-4 | Partitioned tables, optimized queries |
| Cloud Storage | $1-2 | Lifecycle policies, auto-delete |
| **Total** | **$18-29** | **3-4 months on $50** |

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Vertex AI    │    │   BigQuery     │
│   Dashboard     │◄──►│   Endpoint     │◄──►│   Data Store   │
│   (Local/Cloud) │    │   (Auto-scale) │    │   (Partitioned)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud Run     │    │   Vertex AI     │    │   Cloud Storage │
│   (Optional)    │    │   Training      │    │   (Lifecycle)   │
│                 │    │   (Preemptible) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Step-by-Step Setup

### Step 1: Infrastructure Setup
```bash
# Run the budget-optimized setup
bash gcp/scripts/setup_budget_optimized.sh
```

### Step 2: Deploy Training Job
```bash
# Deploy cost-optimized training
bash gcp/scripts/deploy_budget_training.sh
```

### Step 3: Deploy Prediction Endpoint
```bash
# Deploy auto-scaling endpoint
bash gcp/scripts/deploy_budget_endpoint.sh
```

### Step 4: Test the Setup
```bash
# Test endpoint (after deployment)
python gcp/deployment/test_endpoint.py --endpoint_id=YOUR_ENDPOINT_ID
```

## 💡 Cost Optimization Features

### 1. Preemptible Instances
- **Training**: Uses preemptible instances for 60-80% cost savings
- **Risk**: Training may be interrupted, but models are saved incrementally
- **Benefit**: $15-25 → $3-8 per training run

### 2. Auto-Scaling Endpoints
- **Min replicas**: 0 (scale to zero when not in use)
- **Max replicas**: 3 (auto-scale based on demand)
- **Benefit**: $30-50 → $5-15 per month

### 3. Partitioned BigQuery Tables
- **Partitioning**: By date for efficient queries
- **Benefit**: Reduces query costs by 70-90%

### 4. Lifecycle Policies
- **Training data**: Auto-delete after 30 days
- **Models**: Archive to Coldline after 90 days, delete after 1 year
- **Benefit**: Reduces storage costs by 50-70%

### 5. Budget Alerts
- **50%**: $25 (warning)
- **75%**: $37.50 (alert)
- **90%**: $45 (critical)
- **100%**: $50 (emergency)

## 🔍 Monitoring & Management

### Check Training Status
```bash
# List training jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1

# Check job details
gcloud ai custom-jobs describe JOB_NAME --region=us-central1
```

### Monitor Endpoints
```bash
# List endpoints
gcloud ai endpoints list --region=us-central1

# Check endpoint status
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Check Costs
```bash
# View billing
gcloud billing accounts list

# Check BigQuery usage
bq query --use_legacy_sql=false "SELECT * FROM \`stock-ml-trading-487.stock_data.__TABLES__\`"
```

## 🧪 Testing the Setup

### 1. Test Local Prediction Service
```python
from ml.prediction_service import PredictionService

# Test with mock data
service = PredictionService()
predictions = service.get_all_predictions()
print(predictions)
```

### 2. Test Vertex AI Integration
```python
# Test with Vertex AI
service = PredictionService(provider="vertex")
predictions = service.get_all_predictions()
print(predictions)
```

### 3. Test Endpoint Directly
```bash
python gcp/deployment/test_endpoint.py --endpoint_id=YOUR_ENDPOINT_ID
```

## 🔧 Integration with Your App

### Update app.py
```python
from ml.prediction_service import PredictionService

# Initialize with Vertex AI
service = PredictionService(provider="vertex")

# Get predictions
predictions = service.get_all_predictions()

# Display in Streamlit
for pred in predictions:
    st.metric(
        label=f"{pred['symbol']} Prediction",
        value=f"${pred['predicted_price']:,.2f}",
        delta=f"{pred['predicted_return']*100:+.2f}%"
    )
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Training Job Fails
```bash
# Check logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=us-central1

# Common fixes:
# - Ensure billing is enabled
# - Check service account permissions
# - Verify Docker image builds successfully
```

#### 2. Endpoint Not Responding
```bash
# Check endpoint status
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1

# Common fixes:
# - Wait for deployment to complete
# - Check model deployment status
# - Verify endpoint has deployed models
```

#### 3. High Costs
```bash
# Check current usage
gcloud billing budgets list

# Optimize:
# - Ensure preemptible instances are used
# - Check auto-scaling is enabled
# - Verify lifecycle policies are active
```

### Debug Commands
```bash
# Check project configuration
gcloud config list

# Check authentication
gcloud auth list

# Check service accounts
gcloud iam service-accounts list

# Check storage
gsutil ls gs://stock-ml-trading-487-models/
```

## 📈 Performance Optimization

### 1. Model Training
- Use smaller datasets initially (30-90 days)
- Reduce epochs for faster training
- Use batch size optimization

### 2. Prediction Serving
- Enable request batching
- Use connection pooling
- Cache predictions when possible

### 3. Data Pipeline
- Use streaming inserts for BigQuery
- Compress data in Cloud Storage
- Use regional persistent disks

## 🔒 Security Best Practices

### 1. Service Accounts
- Use minimal permissions
- Rotate keys every 90 days
- Store keys securely

### 2. Data Protection
- Enable encryption at rest
- Use VPC connectors for network isolation
- Implement audit logging

### 3. API Security
- Use Secret Manager for API keys
- Implement rate limiting
- Validate all inputs

## 📞 Support & Resources

### Documentation
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)

### Cost Management
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Billing Alerts](https://cloud.google.com/billing/docs/how-to/budgets)
- [Cost Optimization Guide](https://cloud.google.com/cost-optimization)

### Community
- [GCP Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform)
- [Reddit r/googlecloud](https://reddit.com/r/googlecloud)

## ✅ Success Checklist

- [ ] GCP project created with billing enabled
- [ ] Service accounts created with minimal permissions
- [ ] Storage buckets created with lifecycle policies
- [ ] BigQuery tables created with partitioning
- [ ] Training job deployed and completed
- [ ] Prediction endpoint deployed and tested
- [ ] Budget alerts configured
- [ ] App integrated with Vertex AI
- [ ] Cost monitoring set up
- [ ] Documentation updated

## 🎉 Next Steps

After successful setup:

1. **Monitor Performance**: Check prediction accuracy and latency
2. **Optimize Costs**: Review usage and adjust settings
3. **Scale Gradually**: Increase data and model complexity over time
4. **Add Features**: Implement additional ML models or strategies
5. **Production Ready**: Move from paper trading to live trading

---

**Remember**: This setup is optimized for learning and development. For production trading, ensure thorough testing and consider additional security measures.
