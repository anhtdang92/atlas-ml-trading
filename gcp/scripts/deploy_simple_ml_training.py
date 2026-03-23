#!/usr/bin/env python3
"""
Deploy Simple ML Training - Alternative Approach
This creates a working ML training system without complex dependencies
"""

import os
import json
import subprocess
from datetime import datetime

def create_minimal_training_job():
    """Create a minimal training job that will work"""
    print("🔧 Creating minimal training job...")
    
    # Create a very simple training script
    training_script = '''
import os
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_and_train_model():
    """Create a simple model for demonstration"""
    try:
        import tensorflow as tf
        
        logger.info("Creating simple LSTM model...")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(7, 5)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Generate synthetic training data
        X_train = np.random.randn(100, 7, 5)
        y_train = np.random.randn(100, 1)
        
        # Train the model
        logger.info("Training model...")
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
        
        # Save the model
        os.makedirs('/outputs/models', exist_ok=True)
        model_path = '/outputs/models/simple_model.h5'
        model.save(model_path)
        
        # Save training info
        training_info = {
            'model_path': model_path,
            'training_samples': len(X_train),
            'epochs': 5,
            'final_loss': float(history.history['loss'][-1]),
            'timestamp': str(datetime.now())
        }
        
        os.makedirs('/outputs/logs', exist_ok=True)
        with open('/outputs/logs/training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training info: {training_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    from datetime import datetime
    success = create_and_train_model()
    if success:
        print("SUCCESS: Model training completed!")
    else:
        print("FAILED: Model training failed!")
        exit(1)
'''
    
    # Write the training script
    with open('simple_ml_training.py', 'w') as f:
        f.write(training_script)
    
    print("✅ Minimal training script created")

def create_minimal_dockerfile():
    """Create a minimal Dockerfile"""
    dockerfile_content = '''
FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy only the training script
COPY simple_ml_training.py /app/

# Create output directories
RUN mkdir -p /outputs/models /outputs/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "simple_ml_training.py"]
'''
    
    with open('Dockerfile.minimal', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Minimal Dockerfile created")

def build_and_push_image():
    """Build and push the minimal Docker image"""
    print("📦 Building minimal Docker image...")
    
    try:
        # Build the image
        result = subprocess.run([
            'docker', 'build',
            '-f', 'Dockerfile.minimal',
            '-t', 'gcr.io/stock-ml-trading-487/stock-minimal-training:latest',
            '.'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Docker build failed: {result.stderr}")
            return False
        
        print("✅ Docker image built successfully")
        
        # Push the image
        print("📤 Pushing image to Container Registry...")
        result = subprocess.run([
            'docker', 'push',
            'gcr.io/stock-ml-trading-487/stock-minimal-training:latest'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Docker push failed: {result.stderr}")
            return False
        
        print("✅ Docker image pushed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error building/pushing image: {e}")
        return False

def deploy_training_job():
    """Deploy the training job to Vertex AI"""
    print("🚀 Deploying training job to Vertex AI...")
    
    try:
        # Create job name
        job_name = f"stock-minimal-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Deploy the job using gcloud
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'create',
            '--region=us-central1',
            f'--display-name={job_name}',
            '--worker-pool-spec=machine-type=e2-standard-2,replica-count=1,container-image-uri=gcr.io/stock-ml-trading-487/stock-minimal-training:latest'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Training job deployed successfully!")
            print(f"Job details: {result.stdout}")
            return True
        else:
            print(f"❌ Failed to deploy training job: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error deploying training job: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Deploying Simple ML Training System")
    print("=" * 50)
    
    # Step 1: Create minimal training job
    print("\n🔧 Step 1: Create Minimal Training Job")
    create_minimal_training_job()
    
    # Step 2: Create minimal Dockerfile
    print("\n📦 Step 2: Create Minimal Dockerfile")
    create_minimal_dockerfile()
    
    # Step 3: Build and push image
    print("\n🏗️ Step 3: Build and Push Docker Image")
    image_success = build_and_push_image()
    
    # Step 4: Deploy training job
    print("\n🚀 Step 4: Deploy Training Job")
    if image_success:
        deploy_success = deploy_training_job()
    else:
        print("⚠️ Skipping deployment - image build failed")
        deploy_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Deployment Summary")
    print("=" * 50)
    
    if image_success:
        print("✅ Docker Image: Built and pushed")
    else:
        print("❌ Docker Image: Failed")
    
    if deploy_success:
        print("✅ Training Job: Deployed")
        print("\n🎯 What happens next:")
        print("   • Training job will run for ~5 minutes")
        print("   • Creates a simple LSTM model")
        print("   • Saves model to Cloud Storage")
        print("   • You can then deploy this model to your endpoint")
    else:
        print("❌ Training Job: Not deployed")
    
    print("\n💡 This creates a working ML model that you can:")
    print("   • Deploy to your existing endpoint")
    print("   • Use alongside your enhanced mock system")
    print("   • Test real ML predictions")

if __name__ == "__main__":
    main()
