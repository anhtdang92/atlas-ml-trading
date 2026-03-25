
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
