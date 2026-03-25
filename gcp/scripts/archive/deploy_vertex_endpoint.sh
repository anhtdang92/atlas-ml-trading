#!/bin/bash

# Deploy Vertex AI prediction endpoint
# This script creates a Vertex AI endpoint for serving LSTM predictions

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"
ENDPOINT_NAME="stock-lstm-endpoint"

echo "◈ Deploying Vertex AI prediction endpoint..."

# Set the project
gcloud config set project $PROJECT_ID

# Create endpoint
echo "◉ Creating Vertex AI endpoint..."
gcloud ai endpoints create \
    --display-name=$ENDPOINT_NAME \
    --region=$REGION \
    --project=$PROJECT_ID

# Get endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list \
    --filter="displayName=$ENDPOINT_NAME" \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(name)" \
    --limit=1)

echo "◊ Endpoint created with ID: $ENDPOINT_ID"

# Create model registry entry
echo "◉ Creating model registry entry..."
MODEL_NAME="stock-lstm-model"
MODEL_VERSION="v1"

# Upload model to Vertex AI Model Registry
gcloud ai models upload \
    --display-name=$MODEL_NAME \
    --container-image-uri="gcr.io/$PROJECT_ID/stock-lstm-prediction:latest" \
    --region=$REGION \
    --project=$PROJECT_ID

# Get model ID
MODEL_ID=$(gcloud ai models list \
    --filter="displayName=$MODEL_NAME" \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(name)" \
    --limit=1)

echo "◊ Model uploaded with ID: $MODEL_ID"

# Deploy model to endpoint
echo "◉ Deploying model to endpoint..."
gcloud ai endpoints deploy-model \
    $ENDPOINT_ID \
    --model=$MODEL_ID \
    --display-name="stock-lstm-deployment" \
    --machine-type="n1-standard-2" \
    --min-replica-count=0 \
    --max-replica-count=3 \
    --region=$REGION \
    --project=$PROJECT_ID

echo "◈ Vertex AI endpoint deployed!"
echo "◊ Endpoint ID: $ENDPOINT_ID"
echo "◊ Model ID: $MODEL_ID"
echo "◊ Endpoint URL: https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/endpoints/$ENDPOINT_ID"
echo "◊ Next steps:"
echo "  1. Test the endpoint with sample predictions"
echo "  2. Update PredictionService to use Vertex endpoint"
echo "  3. Monitor endpoint usage and costs"
