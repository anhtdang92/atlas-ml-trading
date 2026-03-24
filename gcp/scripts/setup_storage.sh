#!/bin/bash

# Set up Cloud Storage buckets for ML models and data
# This script creates the necessary buckets with proper permissions

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "◈ Setting up Cloud Storage for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Create models bucket
echo "◉ Creating models bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-models

# Create backups bucket
echo "◉ Creating backups bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-backups

# Create training data bucket
echo "◉ Creating training data bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-training-data

# Set bucket permissions
echo "◉ Setting bucket permissions..."

# Models bucket - allow ML service accounts to read/write
gsutil iam ch serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-models
gsutil iam ch serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com:objectViewer gs://$PROJECT_ID-models
gsutil iam ch serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com:objectViewer gs://$PROJECT_ID-models

# Training data bucket - allow ML training to read/write
gsutil iam ch serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-training-data

# Backups bucket - allow all service accounts to read/write
gsutil iam ch serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-backups
gsutil iam ch serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-backups
gsutil iam ch serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-backups

# Create folder structure (using touch to create placeholder files)
echo "◉ Creating folder structure..."
echo "placeholder" | gsutil cp - gs://$PROJECT_ID-models/lstm-models/.gitkeep
echo "placeholder" | gsutil cp - gs://$PROJECT_ID-models/vertex-models/.gitkeep
echo "placeholder" | gsutil cp - gs://$PROJECT_ID-training-data/features/.gitkeep
echo "placeholder" | gsutil cp - gs://$PROJECT_ID-training-data/labels/.gitkeep
echo "placeholder" | gsutil cp - gs://$PROJECT_ID-backups/daily-backups/.gitkeep

# Set lifecycle policies for cost optimization
echo "◉ Setting lifecycle policies..."

# Models bucket - keep for 1 year, then archive
cat > /tmp/models-lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 30}
    },
    {
      "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
      "condition": {"age": 90}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 365}
    }
  ]
}
EOF

gsutil lifecycle set /tmp/models-lifecycle.json gs://$PROJECT_ID-models

# Training data bucket - delete after 30 days
cat > /tmp/training-lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF

gsutil lifecycle set /tmp/training-lifecycle.json gs://$PROJECT_ID-training-data

# Clean up temp files
rm /tmp/models-lifecycle.json /tmp/training-lifecycle.json

echo "◈ Cloud Storage setup complete!"
echo "◊ Buckets created:"
echo "  - gs://$PROJECT_ID-models (models storage)"
echo "  - gs://$PROJECT_ID-training-data (training data)"
echo "  - gs://$PROJECT_ID-backups (backups)"
echo "◊ Next steps:"
echo "  1. Run setup_bigquery.sh to create datasets"
echo "  2. Run deploy_vertex_training.sh to deploy training job"
