#!/bin/bash

# Enable required GCP APIs for ML predictions
# Run this script to set up the necessary APIs

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "◈ Enabling GCP APIs for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "◉ Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com

echo "◉ Enabling BigQuery API..."
gcloud services enable bigquery.googleapis.com

echo "◉ Enabling Cloud Storage API..."
gcloud services enable storage.googleapis.com

echo "◉ Enabling Secret Manager API..."
gcloud services enable secretmanager.googleapis.com

echo "◉ Enabling Cloud Build API..."
gcloud services enable cloudbuild.googleapis.com

echo "◉ Enabling Container Registry API..."
gcloud services enable container.googleapis.com

echo "◉ Enabling Cloud Run API..."
gcloud services enable run.googleapis.com

echo "◉ Enabling Cloud Functions API..."
gcloud services enable cloudfunctions.googleapis.com

echo "◉ Enabling Cloud Scheduler API..."
gcloud services enable cloudscheduler.googleapis.com

echo "◈ All APIs enabled successfully!"
echo "◊ Next steps:"
echo "  1. Run setup_iam.sh to configure service accounts"
echo "  2. Run setup_storage.sh to create buckets"
echo "  3. Run setup_bigquery.sh to create datasets"
