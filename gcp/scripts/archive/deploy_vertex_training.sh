#!/bin/bash

# Deploy Vertex AI training job
# This script builds and deploys the LSTM training job to Vertex AI

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"
DATASET_ID="stock_data"
SYMBOLS="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY"

echo "◈ Deploying Vertex AI training job..."

# Set the project
gcloud config set project $PROJECT_ID

# Build and push Docker image
echo "◉ Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/stock-lstm-training:latest -f gcp/training/Dockerfile .

echo "◉ Pushing Docker image to Container Registry..."
docker push gcr.io/$PROJECT_ID/stock-lstm-training:latest

# Create training job
echo "◉ Creating Vertex AI training job..."

# Create training job configuration
cat > /tmp/training_job_config.json << EOF
{
  "displayName": "stock-lstm-training-$(date +%Y%m%d-%H%M%S)",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "n1-standard-4",
          "acceleratorType": "NVIDIA_TESLA_T4",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "gcr.io/$PROJECT_ID/stock-lstm-training:latest",
          "command": ["python"],
          "args": [
            "gcp/training/vertex_training_job.py",
            "--project_id=$PROJECT_ID",
            "--region=$REGION",
            "--bucket_name=$BUCKET_NAME",
            "--dataset_id=$DATASET_ID",
            "--symbols=$SYMBOLS",
            "--lookback_days=7",
            "--prediction_horizon=7"
          ],
          "env": [
            {
              "name": "GOOGLE_CLOUD_PROJECT",
              "value": "$PROJECT_ID"
            }
          ]
        }
      }
    ]
  },
  "encryptionSpec": {
    "kmsKeyName": "projects/$PROJECT_ID/locations/$REGION/keyRings/stock-ml-keyring/cryptoKeys/stock-ml-key"
  }
}
EOF

# Submit training job
gcloud ai custom-jobs create \
    --region=$REGION \
    --config=/tmp/training_job_config.json

# Clean up
rm /tmp/training_job_config.json

echo "◈ Vertex AI training job deployed!"
echo "◊ Job submitted to region: $REGION"
echo "◊ Monitor progress in Vertex AI console"
echo "◊ Next steps:"
echo "  1. Wait for training to complete"
echo "  2. Run deploy_vertex_endpoint.sh to deploy prediction endpoint"
