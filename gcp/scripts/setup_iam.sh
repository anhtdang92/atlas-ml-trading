#!/bin/bash

# Set up IAM roles and service accounts for ML predictions
# This script creates service accounts with minimal required permissions

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "◈ Setting up IAM for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Create service account for ML training
echo "◉ Creating ML training service account..."
gcloud iam service-accounts create ml-training-sa \
    --display-name="ML Training Service Account" \
    --description="Service account for Vertex AI training jobs"

# Create service account for predictions
echo "◉ Creating prediction service account..."
gcloud iam service-accounts create ml-prediction-sa \
    --display-name="ML Prediction Service Account" \
    --description="Service account for Vertex AI predictions"

# Create service account for app
echo "◉ Creating app service account..."
gcloud iam service-accounts create stock-app-sa \
    --display-name="Stock App Service Account" \
    --description="Service account for the Streamlit app"

# Grant roles to ML training service account
echo "◉ Granting roles to ML training service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

# Grant roles to prediction service account
echo "◉ Granting roles to prediction service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Grant roles to app service account
echo "◉ Granting roles to app service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download service account keys
echo "◉ Creating service account keys..."
mkdir -p ../config/keys

gcloud iam service-accounts keys create ../config/keys/ml-training-sa-key.json \
    --iam-account=ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts keys create ../config/keys/ml-prediction-sa-key.json \
    --iam-account=ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts keys create ../config/keys/stock-app-sa-key.json \
    --iam-account=stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com

echo "◈ IAM setup complete!"
echo "◊ Service account keys saved to config/keys/"
echo "◊ Next steps:"
echo "  1. Run setup_storage.sh to create buckets"
echo "  2. Run setup_bigquery.sh to create datasets"
echo "  3. Add keys to .gitignore if not already there"
