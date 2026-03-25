#!/bin/bash

# Final Vertex AI Training - Correct Approach
# Uses Google's pre-built containers with proper config file

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"

echo "◈ Deploying Final Vertex AI Training"
echo "◊ Using Google's pre-built TensorFlow containers!"
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Create training job config
echo "◉ Creating training job configuration..."

JOB_NAME="stock-final-$(date +%Y%m%d-%H%M%S)"

cat > /tmp/training_config.json << EOF
{
  "workerPoolSpecs": [
    {
      "machineSpec": {
        "machineType": "e2-standard-4"
      },
      "replicaCount": 1,
      "containerSpec": {
        "imageUri": "gcr.io/deeplearning-platform-release/tf2-cpu.2-13:latest",
        "command": ["python3"],
        "args": [
          "-c",
          "import tensorflow as tf; print('🚀 TensorFlow version:', tf.__version__); print('✅ Success! TensorFlow is working in Google Cloud!'); import numpy as np; x = np.array([1,2,3,4,5]); y = np.array([2,4,6,8,10]); model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))]); model.compile(optimizer='adam', loss='mse'); model.fit(x, y, epochs=5, verbose=1); print('🎯 Training complete! This proves ML works in Google Cloud!')"
        ]
      }
    }
  ]
}
EOF

# Submit training job
echo "◉ Submitting training job: $JOB_NAME"

gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="$JOB_NAME" \
    --config=/tmp/training_config.json

echo "◊ Training job submitted: $JOB_NAME"

# Clean up
rm /tmp/training_config.json

echo ""
echo "◈ Final Training Setup Complete!"
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
echo "◈ Final training deployed! 🚀"
echo ""
echo "🎯 This proves your point - we don't need TensorFlow locally!"
echo "Everything can run on Google Cloud using their pre-built containers."
