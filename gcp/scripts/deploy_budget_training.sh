#!/bin/bash

# Budget-Optimized Vertex AI Training Deployment
# Uses preemptible instances and cost-effective machine types

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"
DATASET_ID="stock_data"
SYMBOLS="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY"

echo "◈ Deploying BUDGET-OPTIMIZED Vertex AI training job..."
echo "◊ Using preemptible instances for 60-80% cost savings"

# Set the project
gcloud config set project $PROJECT_ID

# Create optimized Dockerfile for smaller image size
echo "◉ Creating optimized Dockerfile..."
cat > gcp/training/Dockerfile.budget << EOF
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install core dependencies first
RUN pip install --no-cache-dir \\
    tensorflow==2.13.0 \\
    google-cloud-aiplatform \\
    google-cloud-bigquery \\
    google-cloud-storage \\
    pandas \\
    numpy \\
    scikit-learn \\
    yfinance==0.2.46 \\
    requests \\
    python-dotenv

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/outputs/models /app/outputs/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_CLOUD_PROJECT=$PROJECT_ID

# Default command
CMD ["python", "gcp/training/vertex_training_job.py"]
EOF

# Build optimized Docker image
echo "◉ Building optimized Docker image..."
docker build -t gcr.io/$PROJECT_ID/stock-lstm-training-budget:latest -f gcp/training/Dockerfile.budget .

echo "◉ Pushing Docker image to Container Registry..."
docker push gcr.io/$PROJECT_ID/stock-lstm-training-budget:latest

# Create cost-optimized training job configuration
echo "◉ Creating budget-optimized training job..."
cat > /tmp/budget_training_job_config.json << EOF
{
  "displayName": "stock-lstm-training-budget-$(date +%Y%m%d-%H%M%S)",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "e2-standard-4",
          "acceleratorType": "NVIDIA_TESLA_T4",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "preemptible": true,
        "containerSpec": {
          "imageUri": "gcr.io/$PROJECT_ID/stock-lstm-training-budget:latest",
          "command": ["python"],
          "args": [
            "gcp/training/vertex_training_job.py",
            "--project_id=$PROJECT_ID",
            "--region=$REGION",
            "--bucket_name=$BUCKET_NAME-models",
            "--dataset_id=$DATASET_ID",
            "--symbols=$SYMBOLS",
            "--lookback_days=7",
            "--prediction_horizon=7",
            "--epochs=50",
            "--batch_size=32"
          ],
          "env": [
            {
              "name": "GOOGLE_CLOUD_PROJECT",
              "value": "$PROJECT_ID"
            },
            {
              "name": "GOOGLE_APPLICATION_CREDENTIALS",
              "value": "/app/config/keys/ml-training-sa-key.json"
            }
          ]
        }
      }
    ],
    "scheduling": {
      "timeout": "3600s"
    }
  }
}
EOF

# Submit training job
echo "◉ Submitting training job..."
JOB_NAME=$(gcloud ai custom-jobs create \
    --region=$REGION \
    --config=/tmp/budget_training_job_config.json \
    --format="value(name)")

echo "◊ Training job submitted: $JOB_NAME"

# Clean up
rm /tmp/budget_training_job_config.json

echo ""
echo "◈ Budget-optimized training job deployed!"
echo ""
echo "◊ Job details:"
echo "  - Job name: $JOB_NAME"
echo "  - Machine type: e2-standard-4 (cost-effective)"
echo "  - Preemptible: true (60-80% savings)"
echo "  - Accelerator: 1x NVIDIA_TESLA_T4"
echo "  - Timeout: 1 hour"
echo "  - Estimated cost: $3-8 (vs $15-25 for non-preemptible)"
echo ""
echo "◊ Monitor progress:"
echo "  gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs $JOB_NAME --region=$REGION"
echo ""
echo "◊ Or check in Vertex AI console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "◊ Next steps:"
echo "  1. Wait for training to complete (~30-60 minutes)"
echo "  2. Run: bash gcp/scripts/deploy_budget_endpoint.sh"
echo "  3. Test predictions"
echo ""
echo "◈ Training deployment complete! 🚀"
