#!/bin/bash

# Setup Vertex AI Workbench for ML Development
# Run everything in Google Cloud - no local setup needed!

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
INSTANCE_NAME="stock-ml-workbench"

echo "◈ Setting up Vertex AI Workbench"
echo "◊ Run all ML development in Google Cloud!"
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Create Workbench instance
echo "◉ Creating Vertex AI Workbench instance..."
gcloud notebooks instances create $INSTANCE_NAME \
    --location=$REGION \
    --machine-type=e2-standard-4 \
    --vm-image-family=tf-latest-cpu \
    --vm-image-project=deeplearning-platform-release \
    --service-account=ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --metadata='install-nvidia-driver=True'

echo "◊ Workbench instance created: $INSTANCE_NAME"

# Get connection details
echo "◉ Getting connection details..."
gcloud notebooks instances describe $INSTANCE_NAME \
    --location=$REGION \
    --format="value(proxyUri)"

echo ""
echo "◈ Vertex AI Workbench Setup Complete!"
echo ""
echo "◊ Benefits:"
echo "  ✅ Pre-installed TensorFlow, PyTorch, etc."
echo "  ✅ GPU support available"
echo "  ✅ Persistent storage"
echo "  ✅ Integrated with BigQuery"
echo "  ✅ Cost-effective (pay per hour)"
echo ""
echo "◊ Access your notebook:"
echo "  gcloud notebooks instances list --location=$REGION"
echo "  gcloud notebooks instances describe $INSTANCE_NAME --location=$REGION"
echo ""
echo "◊ Next steps:"
echo "  1. Access the Jupyter notebook URL"
echo "  2. Upload your ML code"
echo "  3. Train models in the cloud"
echo "  4. Deploy directly to Vertex AI"
echo ""
echo "◈ Workbench setup complete! 🚀"
