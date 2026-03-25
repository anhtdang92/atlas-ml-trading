#!/bin/bash

# Quick Vertex AI Training - Simplest Approach
# Uses Google's pre-built containers with correct syntax

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"

echo "◈ Deploying Quick Vertex AI Training"
echo "◊ Using Google's pre-built TensorFlow containers!"
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Create training job using correct syntax
echo "◉ Creating Vertex AI training job..."

JOB_NAME="stock-quick-training-$(date +%Y%m%d-%H%M%S)"

# Submit training job with inline Python code
echo "◉ Submitting training job: $JOB_NAME"

gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="$JOB_NAME" \
    --worker-pool-spec=machine-type=e2-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,preemptible=true,container-image-uri=gcr.io/deeplearning-platform-release/tf2-gpu.2-13:latest,command='["python3","-c","import tensorflow as tf; import numpy as np; print(\"🚀 TensorFlow version:\", tf.__version__); model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation=\"relu\", input_shape=(10,)), tf.keras.layers.Dense(32, activation=\"relu\"), tf.keras.layers.Dense(1)]); model.compile(optimizer=\"adam\", loss=\"mse\"); X = np.random.randn(1000, 10); y = np.random.randn(1000, 1); history = model.fit(X, y, epochs=5, batch_size=32, verbose=1); print(\"✅ Training complete! Final loss:\", history.history[\"loss\"][-1])"]'

echo "◊ Training job submitted: $JOB_NAME"

echo ""
echo "◈ Quick Vertex AI Training Setup Complete!"
echo ""
echo "◊ Benefits:"
echo "  ✅ No Docker build needed"
echo "  ✅ Uses Google's pre-built TensorFlow container"
echo "  ✅ GPU support included"
echo "  ✅ Preemptible instances (60-80% cost savings)"
echo "  ✅ Complete in ~5 minutes"
echo ""
echo "◊ Monitor training:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo "  gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs $JOB_NAME --region=$REGION"
echo ""
echo "◊ Check Vertex AI console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "◈ Quick training deployment complete! 🚀"
