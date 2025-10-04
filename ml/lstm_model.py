"""
LSTM Model Architecture for Cryptocurrency Price Prediction

Model Design:
- 2-layer LSTM with 50 units each
- Dropout layers (0.2) for regularization
- Dense layer for final prediction
- Predicts 7-day future return (%)

Architecture Rationale:
- 2 layers: Enough to capture patterns, not too complex
- 50 units: Sweet spot for time series (tested range: 30-100)
- Dropout 0.2: Prevents overfitting on volatile crypto data
- Returns not prices: Easier to learn, scale-invariant

Input Shape: (batch_size, 7 timesteps, N features)
Output: Single value (predicted 7-day return as decimal, e.g., 0.05 = 5%)

Training Strategy:
- Loss: Mean Squared Error (regression task)
- Optimizer: Adam with learning rate 0.001
- Early stopping: Patience 10 epochs on validation loss
- Metrics: MSE, MAE, Directional Accuracy
"""

import logging
from typing import Tuple, Optional
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


class CryptoLSTM:
    """
    LSTM model for cryptocurrency price prediction.
    
    Model Architecture:
        Input (7 days × features) 
            ↓
        LSTM Layer 1 (50 units) → Dropout (0.2)
            ↓
        LSTM Layer 2 (50 units) → Dropout (0.2)
            ↓
        Dense Layer (25 units, ReLU)
            ↓
        Output Layer (1 unit) → Predicted Return
    """
    
    def __init__(
        self,
        lookback: int = 7,
        num_features: int = 11,
        lstm_units: int = 50,
        dropout_rate: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Number of timesteps to look back (default: 7 days)
            num_features: Number of input features (default: 11)
            lstm_units: Number of units in LSTM layers (default: 50)
            dropout_rate: Dropout rate for regularization (default: 0.2)
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")
        
        self.lookback = lookback
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        logger.info("🧠 Initializing LSTM model...")
        logger.info(f"   Lookback: {lookback} days")
        logger.info(f"   Features: {num_features}")
        logger.info(f"   LSTM Units: {lstm_units}")
        logger.info(f"   Dropout: {dropout_rate}")
    
    def build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info("🏗️  Building model architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.lookback, self.num_features)),
            
            # LSTM Layer 1
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,  # Pass sequences to next LSTM
                name='lstm_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # LSTM Layer 2
            layers.LSTM(
                self.lstm_units,
                return_sequences=False,  # Output single vector
                name='lstm_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            
            # Dense layer for processing
            layers.Dense(25, activation='relu', name='dense_1'),
            
            # Output layer (single value: predicted return)
            layers.Dense(1, name='output')
        ], name='CryptoLSTM')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        self.model = model
        
        logger.info("✅ Model built successfully!")
        logger.info(f"\nModel Summary:")
        model.summary(print_fn=lambda x: logger.info(x))
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (samples, lookback, features)
            y_train: Training targets (samples,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum epochs (default: 100)
            batch_size: Batch size (default: 32)
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history object
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"🚀 Starting training...")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Validation samples: {len(X_val)}")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        self.history = history
        
        # Log training results
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Final Training Loss: {final_train_loss:.6f}")
        logger.info(f"   Final Validation Loss: {final_val_loss:.6f}")
        logger.info(f"   Epochs Trained: {len(history.history['loss'])}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences (samples, lookback, features)
            
        Returns:
            Predicted returns (samples,)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("📊 Evaluating model...")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Directional accuracy (did we predict up/down correctly?)
        pred_direction = predictions > 0
        true_direction = y_test > 0
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"✅ Evaluation complete:")
        logger.info(f"   MSE: {mse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   RMSE: {rmse:.6f}")
        logger.info(f"   Directional Accuracy: {directional_accuracy*100:.2f}%")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model (e.g., 'models/btc_model.h5')
        """
        if self.model is None:
            logger.error("No model to save!")
            return
        
        self.model.save(filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"📂 Model loaded from {filepath}")


def main():
    """Test LSTM model with dummy data."""
    if not HAS_TENSORFLOW:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        return
    
    print("="*60)
    print("🧪 Testing LSTM Model")
    print("="*60)
    
    # Create dummy data
    samples = 100
    lookback = 7
    features = 11
    
    X_train = np.random.rand(samples, lookback, features)
    y_train = np.random.rand(samples) * 0.1 - 0.05  # Returns between -5% and +5%
    
    X_val = np.random.rand(20, lookback, features)
    y_val = np.random.rand(20) * 0.1 - 0.05
    
    # Build and train model
    model = CryptoLSTM(lookback=lookback, num_features=features)
    model.build_model()
    
    print("\n🚀 Training on dummy data (10 epochs)...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=0)
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    test_input = np.random.rand(1, lookback, features)
    prediction = model.predict(test_input)
    
    print(f"✅ Predicted return: {prediction[0]*100:+.2f}%")
    
    print("\n✅ LSTM model test complete!")
    print("   Model is ready for training on real data")


if __name__ == "__main__":
    main()

