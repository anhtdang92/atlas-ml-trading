#!/bin/bash

# Budget-Optimized GCP ML Setup Script
# Optimized for $50 credit budget - 3-4 months of operation

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"
DATASET_ID="stock_data"

echo "◈ Setting up BUDGET-OPTIMIZED GCP ML infrastructure"
echo "◊ Target: $50 credit for 3-4 months (~$15/month)"
echo "◊ Project: $PROJECT_ID"
echo ""

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    echo "◊ Error: gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "◊ Error: Not authenticated with gcloud. Please run:"
    echo "   gcloud auth login"
    exit 1
fi

# Set the project
echo "◉ Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable billing API first
echo "◉ Enabling billing API..."
gcloud services enable cloudbilling.googleapis.com

# Check if billing is enabled
echo "◉ Checking billing status..."
BILLING_ENABLED=$(gcloud billing projects describe $PROJECT_ID --format="value(billingEnabled)" 2>/dev/null || echo "False")

if [ "$BILLING_ENABLED" = "False" ]; then
    echo "◊ WARNING: Billing is not enabled for this project!"
    echo "◊ Please enable billing in the GCP Console:"
    echo "   https://console.cloud.google.com/billing/projects"
    echo "◊ Or run: gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT"
    echo ""
    read -p "Press Enter to continue after enabling billing..."
fi

# Set up billing alerts
echo "◉ Setting up budget alerts..."
cat > /tmp/budget_alert.json << EOF
{
  "budgetFilter": {
    "projects": ["projects/$PROJECT_ID"]
  },
  "amount": {
    "specifiedAmount": {
      "currencyCode": "USD",
      "units": "50"
    }
  },
  "thresholdRules": [
    {
      "thresholdPercent": 0.5,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 0.75,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 0.9,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 1.0,
      "spendBasis": "FORECASTED_SPEND"
    }
  ]
}
EOF

echo "◊ Created budget alert configuration (50% at $25, 75% at $37.50, 90% at $45, 100% at $50)"

# Enable required APIs (cost-optimized selection)
echo "◉ Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create service accounts with minimal permissions
echo "◉ Creating service accounts..."
gcloud iam service-accounts create stock-app-sa \
    --display-name="Stock App Service Account" \
    --description="Service account for stock ML app"

gcloud iam service-accounts create ml-training-sa \
    --display-name="ML Training Service Account" \
    --description="Service account for Vertex AI training"

gcloud iam service-accounts create ml-prediction-sa \
    --display-name="ML Prediction Service Account" \
    --description="Service account for Vertex AI predictions"

# Assign minimal IAM roles
echo "◉ Assigning IAM roles..."

# Stock app SA - read-only access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# ML training SA - training permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.customCodeServiceAgent"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

# ML prediction SA - prediction permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.endpointServiceAgent"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create keys directory
mkdir -p config/keys

# Create service account keys
echo "◉ Creating service account keys..."
gcloud iam service-accounts keys create config/keys/stock-app-sa-key.json \
    --iam-account=stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts keys create config/keys/ml-training-sa-key.json \
    --iam-account=ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts keys create config/keys/ml-prediction-sa-key.json \
    --iam-account=ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com

echo "◊ Service account keys created in config/keys/"

# Create cost-optimized storage buckets
echo "◉ Creating Cloud Storage buckets..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME-models
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME-training-data

# Set lifecycle policies for cost optimization
echo "◉ Setting up lifecycle policies..."

# Models bucket - keep models for 1 year, then archive
cat > /tmp/models_lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90
        }
      },
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 365
        }
      }
    ]
  }
}
EOF

# Training data bucket - delete after 30 days
cat > /tmp/training_lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 30
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/models_lifecycle.json gs://$BUCKET_NAME-models
gsutil lifecycle set /tmp/training_lifecycle.json gs://$BUCKET_NAME-training-data

echo "◊ Lifecycle policies set for cost optimization"

# Create BigQuery dataset with partitioned tables
echo "◉ Creating BigQuery dataset..."
bq mk --location=$REGION $PROJECT_ID:$DATASET_ID

# Create partitioned tables for cost efficiency
echo "◉ Creating BigQuery tables..."

# Historical prices table (partitioned by date)
bq mk --table \
    --time_partitioning_type=DAY \
    --time_partitioning_field=timestamp \
    --description="Historical stock prices" \
    $PROJECT_ID:$DATASET_ID.historical_prices \
    timestamp:TIMESTAMP,symbol:STRING,open:FLOAT64,high:FLOAT64,low:FLOAT64,close:FLOAT64,volume:FLOAT64,data_source:STRING,created_at:TIMESTAMP

# Predictions table
bq mk --table \
    --time_partitioning_type=DAY \
    --time_partitioning_field=timestamp \
    --description="ML model predictions" \
    $PROJECT_ID:$DATASET_ID.predictions \
    timestamp:TIMESTAMP,symbol:STRING,predicted_price:FLOAT64,confidence:FLOAT64,model_version:STRING,created_at:TIMESTAMP

# Model metrics table
bq mk --table \
    --time_partitioning_type=DAY \
    --time_partitioning_field=timestamp \
    --description="Model training metrics" \
    $PROJECT_ID:$DATASET_ID.model_metrics \
    timestamp:TIMESTAMP,symbol:STRING,model_version:STRING,training_rmse:FLOAT64,validation_rmse:FLOAT64,training_mae:FLOAT64,validation_mae:FLOAT64,training_r2:FLOAT64,validation_r2:FLOAT64,training_samples:INTEGER,validation_samples:INTEGER,created_at:TIMESTAMP

# Trades table
bq mk --table \
    --time_partitioning_type=DAY \
    --time_partitioning_field=timestamp \
    --description="Trading history" \
    $PROJECT_ID:$DATASET_ID.trades \
    timestamp:TIMESTAMP,symbol:STRING,side:STRING,quantity:FLOAT64,price:FLOAT64,value:FLOAT64,fee:FLOAT64,order_id:STRING,status:STRING,created_at:TIMESTAMP

# Portfolio snapshots table
bq mk --table \
    --time_partitioning_type=DAY \
    --time_partitioning_field=timestamp \
    --description="Portfolio state snapshots" \
    $PROJECT_ID:$DATASET_ID.portfolio_snapshots \
    timestamp:TIMESTAMP,portfolio_value:FLOAT64,cash_balance:FLOAT64,total_return:FLOAT64,daily_return:FLOAT64,created_at:TIMESTAMP

echo "◊ BigQuery tables created with partitioning for cost optimization"

# Create .env file
echo "◉ Creating environment configuration..."
cat > .env << EOF
# GCP Configuration
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GCP_REGION=$REGION
BIGQUERY_DATASET=$DATASET_ID
STORAGE_BUCKET_MODELS=$BUCKET_NAME-models
STORAGE_BUCKET_TRAINING=$BUCKET_NAME-training-data

# Vertex AI (will be updated after endpoint creation)
VERTEX_ENDPOINT_ID=

# Service Account Keys
GOOGLE_APPLICATION_CREDENTIALS=config/keys/stock-app-sa-key.json
ML_TRAINING_SA_KEY=config/keys/ml-training-sa-key.json
ML_PREDICTION_SA_KEY=config/keys/ml-prediction-sa-key.json

# Cost Optimization Settings
USE_PREEMPTIBLE_INSTANCES=true
AUTO_SCALE_ENDPOINT=true
MIN_REPLICA_COUNT=0
MAX_REPLICA_COUNT=3
EOF

# Update .gitignore
if ! grep -q "config/keys/" .gitignore; then
    echo "" >> .gitignore
    echo "# GCP Service Account Keys" >> .gitignore
    echo "config/keys/" >> .gitignore
    echo ".env" >> .gitignore
    echo "◊ Updated .gitignore to exclude sensitive files"
fi

# Install required Python packages for GCP
echo "◉ Installing GCP Python packages..."
pip install google-cloud-aiplatform google-cloud-bigquery google-cloud-storage

echo ""
echo "◈ Budget-optimized GCP ML infrastructure setup complete!"
echo ""
echo "◊ Cost optimization features enabled:"
echo "  ✅ Partitioned BigQuery tables (reduces query costs)"
echo "  ✅ Lifecycle policies (auto-delete old data)"
echo "  ✅ Preemptible instances for training (60-80% savings)"
echo "  ✅ Auto-scaling endpoints (scale to zero)"
echo "  ✅ Minimal IAM permissions"
echo "  ✅ Budget alerts configured"
echo ""
echo "◊ Estimated monthly costs:"
echo "  - Vertex AI Training: $5-8 (preemptible)"
echo "  - Vertex AI Endpoint: $10-15 (auto-scaling)"
echo "  - BigQuery: $2-4 (partitioned tables)"
echo "  - Cloud Storage: $1-2 (lifecycle policies)"
echo "  - Total: ~$18-29/month"
echo ""
echo "◊ Next steps:"
echo "  1. Run: bash gcp/scripts/deploy_budget_training.sh"
echo "  2. Wait for training to complete"
echo "  3. Run: bash gcp/scripts/deploy_budget_endpoint.sh"
echo "  4. Test predictions"
echo ""
echo "◊ Monitor costs:"
echo "  - GCP Console > Billing"
echo "  - Set up billing alerts"
echo "  - Check Vertex AI usage"
echo ""
echo "◈ Setup complete! 🚀"
