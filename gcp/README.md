# GCP ML Infrastructure Setup

This directory contains the Google Cloud Platform infrastructure setup for stock ML predictions using Vertex AI.

## Overview

The GCP setup enables:
- **Vertex AI**: Training and serving LSTM models for stock price prediction
- **BigQuery**: Storing historical stock data and predictions
- **Cloud Storage**: Model artifacts and training data
- **IAM**: Secure service accounts with minimal permissions
- **Cost Optimization**: Preemptible instances, auto-scaling, lifecycle policies

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Vertex AI     │    │   BigQuery      │
│   Dashboard     │◄──►│   Endpoint      │◄──►│   Data Store    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud Run     │    │   Vertex AI     │    │   Cloud Storage │
│   (App Host)    │    │   Training      │    │   (Models)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Prerequisites

- Google Cloud SDK installed and authenticated
- Docker installed for container builds
- Python 3.9+ with required dependencies

### 2. One-Command Setup

```bash
bash gcp/scripts/setup_gcp_ml.sh
```

This will:
- Enable all required APIs
- Create service accounts and IAM roles
- Set up BigQuery datasets and tables
- Create Cloud Storage buckets
- Deploy Vertex AI training job
- Deploy prediction endpoint

### 3. Manual Setup

```bash
bash gcp/scripts/enable_apis.sh
bash gcp/scripts/setup_iam.sh
bash gcp/scripts/setup_storage.sh
bash gcp/scripts/setup_bigquery.sh
bash gcp/scripts/deploy_vertex_training.sh
bash gcp/scripts/deploy_vertex_endpoint.sh
```

## Configuration

### Environment Variables

The setup uses `.env` with:

```bash
GOOGLE_CLOUD_PROJECT=stock-ml-trading-487
GCP_REGION=us-central1
BIGQUERY_DATASET=stock_data
STORAGE_BUCKET=stock-ml-models-487
VERTEX_ENDPOINT_ID=<endpoint_id_after_deployment>
GOOGLE_APPLICATION_CREDENTIALS=config/keys/stock-app-sa-key.json
```

### Service Accounts

Three service accounts are created with minimal permissions:

- **ml-training-sa**: Vertex AI training jobs
- **ml-prediction-sa**: Vertex AI predictions
- **stock-app-sa**: Streamlit app access

### BigQuery Schema

Tables created:
- `historical_prices`: OHLCV data from Yahoo Finance
- `predictions`: ML model predictions
- `trades`: Trading history
- `model_metrics`: Training performance
- `portfolio_snapshots`: Portfolio state
- `rebalancing_events`: Rebalancing history

## Usage

### Using Vertex AI Predictions

```python
from ml.prediction_service import PredictionService

service = PredictionService(provider="vertex")
predictions = service.get_all_predictions()
```

### Monitoring

```bash
gcloud ai custom-jobs list --region=us-central1
gcloud ai endpoints list --region=us-central1
bq ls stock-ml-trading-487:stock_data
gsutil ls gs://stock-ml-models-487/
```

## Cost Breakdown (Estimated)

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| Vertex AI Training | $5-10 | Preemptible instances |
| Vertex AI Endpoint | $10-20 | Auto-scaling, scale to zero |
| BigQuery | $2-5 | Storage + queries |
| Cloud Storage | $1-2 | Model artifacts |
| **Total** | **$18-37** | **3-4 months on $50** |

## Files Structure

```
gcp/
├── scripts/                 # Setup scripts
│   ├── enable_apis.sh
│   ├── setup_iam.sh
│   ├── setup_storage.sh
│   ├── setup_bigquery.sh
│   ├── deploy_vertex_training.sh
│   ├── deploy_vertex_endpoint.sh
│   └── setup_gcp_ml.sh
├── training/                # Training job code
│   ├── Dockerfile
│   └── vertex_training_job.py
├── deployment/              # Deployment code
│   └── vertex_prediction_service.py
└── README.md
```
