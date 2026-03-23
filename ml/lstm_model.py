"""
LSTM Model Architecture for Stock Price Prediction

Model Design - Position Trading (Weeks-Months):
- 2-layer LSTM with 64 units each
- Dropout layers (0.2) for regularization
- Dense layers for final prediction
- Predicts 21-day (~1 month) future return (%)

Architecture Rationale:
- 2 layers: Enough to capture patterns, not too complex
- 64 units: More capacity for richer stock features (25 indicators)
- Dropout 0.2: Prevents overfitting
- Returns not prices: Scale-invariant, easier to learn

Input Shape: (batch_size, 30 timesteps, 25 features)
Output: Single value (predicted 21-day return as decimal, e.g., 0.05 = 5%)

Training Strategy:
- Loss: Mean Squared Error (regression task)
- Optimizer: Adam with learning rate 0.001
- Early stopping: Patience 15 epochs on validation loss
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
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TENSORFLOW = False

    class keras:
        class Model:
            pass
        class Sequential:
            def __init__(self, *args, **kwargs):
                pass
        class optimizers:
            class Adam:
                def __init__(self, *args, **kwargs):
                    pass
        class callbacks:
            class EarlyStopping:
                def __init__(self, *args, **kwargs):
                    pass
            class ReduceLROnPlateau:
                def __init__(self, *args, **kwargs):
                    pass
        class models:
            @staticmethod
            def load_model(*args, **kwargs):
                return None

    class layers:
        class Input:
            def __init__(self, *args, **kwargs):
                pass
        class LSTM:
            def __init__(self, *args, **kwargs):
                pass
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        class Dense:
            def __init__(self, *args, **kwargs):
                pass

    class callbacks:
        class EarlyStopping:
            def __init__(self, *args, **kwargs):
                pass
        class ReduceLROnPlateau:
            def __init__(self, *args, **kwargs):
                pass
        class History:
            def __init__(self, *args, **kwargs):
                self.history = {'loss': [0.1], 'val_loss': [0.1]}

logger = logging.getLogger(__name__)


class StockLSTM:
    """LSTM model for stock price prediction (position trading)."""

    def __init__(
        self,
        lookback: int = 30,
        num_features: int = 25,
        lstm_units: int = 64,
        dropout_rate: float = 0.2
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")

        self.lookback = lookback
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None

        logger.info("Initializing Stock LSTM model...")
        logger.info(f"   Lookback: {lookback} days")
        logger.info(f"   Features: {num_features}")
        logger.info(f"   LSTM Units: {lstm_units}")
        logger.info(f"   Dropout: {dropout_rate}")

    def build_model(self) -> keras.Model:
        """Build the LSTM model architecture."""
        logger.info("Building model architecture...")

        model = keras.Sequential([
            layers.Input(shape=(self.lookback, self.num_features)),

            # LSTM Layer 1
            layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_1'),
            layers.Dropout(self.dropout_rate, name='dropout_1'),

            # LSTM Layer 2
            layers.LSTM(self.lstm_units, return_sequences=False, name='lstm_2'),
            layers.Dropout(self.dropout_rate, name='dropout_2'),

            # Dense layers
            layers.Dense(32, activation='relu', name='dense_1'),
            layers.Dense(1, name='output')
        ], name='StockLSTM')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        logger.info("Model built successfully!")
        model.summary(print_fn=lambda x: logger.info(x))
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        verbose: int = 1
    ) -> callbacks.History:
        """Train the LSTM model."""
        if self.model is None:
            self.build_model()

        logger.info(f"Starting training...")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Validation samples: {len(X_val)}")

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=0.00001, verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        self.history = history

        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        logger.info(f"Training complete!")
        logger.info(f"   Final Training Loss: {final_train_loss:.6f}")
        logger.info(f"   Final Validation Loss: {final_val_loss:.6f}")
        logger.info(f"   Epochs Trained: {len(history.history['loss'])}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        predictions = self.predict(X_test)

        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)

        pred_direction = predictions > 0
        true_direction = y_test > 0
        directional_accuracy = np.mean(pred_direction == true_direction)

        metrics = {
            'mse': mse, 'mae': mae, 'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }

        logger.info(f"Evaluation complete:")
        logger.info(f"   MSE: {mse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   RMSE: {rmse:.6f}")
        logger.info(f"   Directional Accuracy: {directional_accuracy*100:.2f}%")

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            logger.error("No model to save!")
            return
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


# Keep backward compatibility alias
CryptoLSTM = StockLSTM


def main():
    """Test LSTM model with dummy data."""
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return

    print("=" * 60)
    print("Testing Stock LSTM Model")
    print("=" * 60)

    samples = 200
    lookback = 30
    features = 25

    X_train = np.random.rand(samples, lookback, features)
    y_train = np.random.rand(samples) * 0.2 - 0.1

    X_val = np.random.rand(40, lookback, features)
    y_val = np.random.rand(40) * 0.2 - 0.1

    model = StockLSTM(lookback=lookback, num_features=features)
    model.build_model()

    print("\nTraining on dummy data (10 epochs)...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=0)

    print("\nTesting prediction...")
    test_input = np.random.rand(1, lookback, features)
    prediction = model.predict(test_input)
    print(f"Predicted return: {prediction[0]*100:+.2f}%")

    print("\nStock LSTM model test complete!")


if __name__ == "__main__":
    main()
