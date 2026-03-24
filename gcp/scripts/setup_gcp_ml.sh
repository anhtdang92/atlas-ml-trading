#!/bin/bash

# Complete GCP ML Setup Script
# This script sets up the entire GCP infrastructure for ML predictions

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"
DATASET_ID="stock_data"

echo "◈ Setting up GCP ML infrastructure for project: $PROJECT_ID"
echo "◊ This will enable Vertex AI, BigQuery, Cloud Storage, and IAM"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "◊ Error: gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "◊ Error: Not authenticated with gcloud. Please run:"
    echo "   gcloud auth login"
    exit 1
fi

# Set the project
echo "◉ Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Step 1: Enable APIs
echo ""
echo "◈ Step 1: Enabling GCP APIs..."
bash gcp/scripts/enable_apis.sh

# Step 2: Set up IAM
echo ""
echo "◈ Step 2: Setting up IAM roles and service accounts..."
bash gcp/scripts/setup_iam.sh

# Step 3: Create storage buckets
echo ""
echo "◈ Step 3: Creating Cloud Storage buckets..."
bash gcp/scripts/setup_storage.sh

# Step 4: Set up BigQuery
echo ""
echo "◈ Step 4: Creating BigQuery datasets and tables..."
bash gcp/scripts/setup_bigquery.sh

# Step 5: Build and deploy training job
echo ""
echo "◈ Step 5: Building and deploying Vertex AI training job..."
bash gcp/scripts/deploy_vertex_training.sh

# Step 6: Deploy prediction endpoint
echo ""
echo "◈ Step 6: Deploying Vertex AI prediction endpoint..."
bash gcp/scripts/deploy_vertex_endpoint.sh

# Step 7: Update app configuration
echo ""
echo "◈ Step 7: Updating app configuration..."

# Create .env file for environment variables
cat > .env << EOF
# GCP Configuration
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GCP_REGION=$REGION
BIGQUERY_DATASET=$DATASET_ID
STORAGE_BUCKET=$BUCKET_NAME

# Vertex AI (will be updated after endpoint creation)
VERTEX_ENDPOINT_ID=

# Service Account Keys
GOOGLE_APPLICATION_CREDENTIALS=config/keys/stock-app-sa-key.json
EOF

echo "◊ Created .env file with GCP configuration"

# Update .gitignore to exclude service account keys
if ! grep -q "config/keys/" .gitignore; then
    echo "" >> .gitignore
    echo "# GCP Service Account Keys" >> .gitignore
    echo "config/keys/" >> .gitignore
    echo ".env" >> .gitignore
    echo "◊ Updated .gitignore to exclude service account keys"
fi

# Step 8: Test the setup
echo ""
echo "◈ Step 8: Testing the setup..."

# Test BigQuery connection
echo "◉ Testing BigQuery connection..."
bq query --use_legacy_sql=false "SELECT COUNT(*) as table_count FROM \`$PROJECT_ID.$DATASET_ID.__TABLES__\`"

# Test Cloud Storage
echo "◉ Testing Cloud Storage..."
gsutil ls gs://$BUCKET_NAME/

echo ""
echo "◈ GCP ML infrastructure setup complete!"
echo ""
echo "◊ Next steps:"
echo "  1. Wait for Vertex AI training job to complete (check in console)"
echo "  2. Update VERTEX_ENDPOINT_ID in .env file"
echo "  3. Test predictions with: python gcp/deployment/vertex_prediction_service.py"
echo "  4. Update app.py to use Vertex AI predictions"
echo ""
echo "◊ Useful commands:"
echo "  - View training jobs: gcloud ai custom-jobs list --region=$REGION"
echo "  - View endpoints: gcloud ai endpoints list --region=$REGION"
echo "  - View BigQuery: bq ls $PROJECT_ID:$DATASET_ID"
echo "  - View storage: gsutil ls gs://$BUCKET_NAME/"
echo ""
echo "◊ Cost monitoring:"
echo "  - Set up billing alerts in GCP Console"
echo "  - Monitor Vertex AI usage in Cloud Console"
echo "  - Check BigQuery usage in billing dashboard"
echo ""
echo "◈ Setup complete! 🚀"
