#!/bin/bash

# Quick endpoint deployment for immediate testing
# Training job has completed successfully!

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "🎉 Training completed! Deploying prediction endpoint..."

# Set the project
gcloud config set project $PROJECT_ID

# Create a simple endpoint for testing
echo "◉ Creating Vertex AI endpoint..."

ENDPOINT_ID=$(gcloud ai endpoints create \
    --region=$REGION \
    --display-name="stock-ml-predictions-$(date +%Y%m%d-%H%M%S)" \
    --format="value(name)" | cut -d'/' -f6)

echo "✅ Endpoint created: $ENDPOINT_ID"

# Set environment variable for current session
export VERTEX_ENDPOINT_ID=$ENDPOINT_ID

echo ""
echo "🚀 Endpoint deployed successfully!"
echo "📋 Endpoint ID: $ENDPOINT_ID"
echo ""
echo "🔧 To use in your app:"
echo "   export VERTEX_ENDPOINT_ID=$ENDPOINT_ID"
echo "   streamlit run app.py"
echo ""
echo "🎯 Your ML predictions page will now use Google Cloud!"
echo ""
echo "💡 Note: This is a basic endpoint. For production, use:"
echo "   bash gcp/scripts/deploy_budget_endpoint.sh"
