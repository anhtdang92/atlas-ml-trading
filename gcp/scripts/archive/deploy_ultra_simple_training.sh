#!/bin/bash

# Ultra Simple Vertex AI Training
# Uses Google's pre-built containers with minimal configuration

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "◈ Deploying Ultra Simple Vertex AI Training"
echo "◊ Using Google's pre-built TensorFlow containers!"
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Create the simplest possible training job
echo "◉ Creating ultra simple training job..."

JOB_NAME="stock-simple-$(date +%Y%m%d-%H%M%S)"

# Submit training job
echo "◉ Submitting training job: $JOB_NAME"

gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="$JOB_NAME" \
    --worker-pool-spec=machine-type=e2-standard-2,container-image-uri=gcr.io/deeplearning-platform-release/tf2-cpu.2-13:latest,command='["python3","-c","import tensorflow as tf; print(\"🚀 TensorFlow version:\", tf.__version__); print(\"✅ Success! TensorFlow is working in Google Cloud!\"); import numpy as np; x = np.array([1,2,3,4,5]); y = np.array([2,4,6,8,10]); model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))]); model.compile(optimizer=\"adam\", loss=\"mse\"); model.fit(x, y, epochs=5, verbose=1); print(\"🎯 Training complete! This proves ML works in Google Cloud!\")"]'

echo "◊ Training job submitted: $JOB_NAME"

echo ""
echo "◈ Ultra Simple Training Setup Complete!"
echo ""
echo "◊ What just happened:"
echo "  ✅ Created a Vertex AI training job"
echo "  ✅ Uses Google's pre-built TensorFlow container"
echo "  ✅ No Docker build needed"
echo "  ✅ No local TensorFlow installation needed"
echo "  ✅ Everything runs in Google Cloud"
echo ""
echo "◊ Monitor training:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo "  gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
echo ""
echo "◊ Check Vertex AI console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "◈ Ultra simple training deployed! 🚀"
echo ""
echo "🎯 Next Steps:"
echo "1. Check the training job status"
echo "2. Once complete, we'll create a prediction endpoint"
echo "3. Integrate with your Streamlit dashboard"
echo ""
echo "This proves everything can run on Google Cloud without local setup!"
