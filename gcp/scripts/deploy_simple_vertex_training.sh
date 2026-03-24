#!/bin/bash

# Deploy Simple Vertex AI Training
# Use Google's pre-built containers - no custom Docker needed!

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
REGION="us-central1"
BUCKET_NAME="stock-ml-models-$PROJECT_ID"

echo "◈ Deploying Simple Vertex AI Training"
echo "◊ Using Google's pre-built containers!"
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Create training job using Google's TensorFlow container
echo "◉ Creating Vertex AI training job..."

cat > /tmp/simple_training_job.json << EOF
{
  "displayName": "stock-simple-training-$(date +%Y%m%d-%H%M%S)",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "e2-standard-4",
          "acceleratorType": "NVIDIA_TESLA_T4",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "preemptible": true,
        "containerSpec": {
          "imageUri": "gcr.io/deeplearning-platform-release/tf2-gpu.2-13:latest",
          "command": ["python3"],
          "args": [
            "-c",
            "
import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import os

print('🚀 Starting simple stock training...')

# Get project info
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
bucket_name = '$BUCKET_NAME-models'

# Initialize clients
bq_client = bigquery.Client(project=project_id)
storage_client = storage.Client(project=project_id)

# Create simple model
def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Generate synthetic training data
print('📊 Generating training data...')
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)

# Create and train model
print('🧠 Training model...')
model = create_simple_model()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Save model
print('💾 Saving model...')
model_path = '/tmp/stock_model'
model.save(model_path)

# Upload to Cloud Storage
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob('models/simple_stock_model')
blob.upload_from_filename(model_path + '/saved_model.pb')

print('✅ Training complete! Model saved to Cloud Storage.')
print(f'📈 Final loss: {history.history[\"loss\"][-1]:.4f}')
            "
          ],
          "env": [
            {
              "name": "GOOGLE_CLOUD_PROJECT",
              "value": "$PROJECT_ID"
            }
          ]
        }
      }
    ]
  }
}
EOF

# Submit training job
echo "◉ Submitting training job..."
JOB_NAME=$(gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="stock-simple-training-$(date +%Y%m%d-%H%M%S)" \
    --config=/tmp/simple_training_job.json \
    --format="value(name)")

echo "◊ Training job submitted: $JOB_NAME"

# Clean up
rm /tmp/simple_training_job.json

echo ""
echo "◈ Simple Vertex AI Training Setup Complete!"
echo ""
echo "◊ Benefits:"
echo "  ✅ No Docker build needed"
echo "  ✅ Uses Google's pre-built containers"
echo "  ✅ TensorFlow pre-installed"
echo "  ✅ GPU support included"
echo "  ✅ Preemptible instances (60-80% cost savings)"
echo ""
echo "◊ Monitor training:"
echo "  gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
echo "  gcloud ai custom-jobs stream-logs $JOB_NAME --region=$REGION"
echo ""
echo "◊ Check Vertex AI console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "◈ Simple training deployment complete! 🚀"
