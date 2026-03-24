"""
LSTM Model Architecture for Stock Price Prediction

Model Design - Position Trading (Weeks-Months):
- 2-layer LSTM with 64 units each + L2 regularization
- Dropout layers (0.2) for regularization and MC Dropout inference
- Dense layers for final prediction
- Predicts 21-day (~1 month) future return (%)

Improvements over v1:
- Huber loss instead of MSE (robust to outlier earnings-day moves)
- MC Dropout: dropout stays active at inference for uncertainty estimation
- L2 regularization on LSTM recurrent weights
- Ensemble support: train N models, average predictions, use std as confidence

Input Shape: (batch_size, 30 timesteps, 29 features)
Output: Single value (predicted 21-day return as decimal, e.g., 0.05 = 5%)

Training Strategy:
- Loss: Huber (delta=0.1) - robust to outliers
- Optimizer: Adam with learning rate 0.001
- Early stopping: Patience 15 epochs on validation loss
- Metrics: MAE, Directional Accuracy
"""

import logging
from typing import Tuple, Optional, List
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
        class losses:
            class Huber:
                def __init__(self, *args, **kwargs):
                    pass

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
        num_features: int = 29,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")

        self.lookback = lookback
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.history = None

        logger.info("Initializing Stock LSTM model...")
        logger.info(f"   Lookback: {lookback} days")
        logger.info(f"   Features: {num_features}")
        logger.info(f"   LSTM Units: {lstm_units}")
        logger.info(f"   Dropout: {dropout_rate}")
        logger.info(f"   L2 Regularization: {l2_reg}")

    def build_model(self) -> keras.Model:
        """Build the LSTM model architecture with Huber loss and L2 regularization."""
        logger.info("Building model architecture...")

        regularizer = keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        model = keras.Sequential([
            layers.Input(shape=(self.lookback, self.num_features)),

            # LSTM Layer 1
            layers.LSTM(self.lstm_units, return_sequences=True,
                        recurrent_regularizer=regularizer, name='lstm_1'),
            layers.Dropout(self.dropout_rate, name='dropout_1'),

            # LSTM Layer 2
            layers.LSTM(self.lstm_units, return_sequences=False,
                        recurrent_regularizer=regularizer, name='lstm_2'),
            layers.Dropout(self.dropout_rate, name='dropout_2'),

            # Dense layers
            layers.Dense(32, activation='relu',
                         kernel_regularizer=regularizer, name='dense_1'),
            layers.Dense(1, name='output')
        ], name='StockLSTM')

        # Huber loss: robust to outlier returns (earnings, macro events)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.Huber(delta=0.1),
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
        """Make predictions (standard, dropout disabled)."""
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_forward_passes: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout: run N forward passes with dropout enabled for uncertainty.

        Returns:
            mean_predictions: Average prediction across passes
            std_predictions: Standard deviation (uncertainty estimate)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")

        # Run multiple forward passes with dropout active (training=True)
        all_predictions = []
        for _ in range(n_forward_passes):
            pred = self.model(X, training=True)  # training=True keeps dropout active
            all_predictions.append(pred.numpy().flatten())

        all_predictions = np.array(all_predictions)
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)

        logger.info(f"MC Dropout: {n_forward_passes} passes, "
                     f"mean pred: {mean_predictions.mean():.4f}, "
                     f"mean uncertainty: {std_predictions.mean():.4f}")

        return mean_predictions, std_predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance with comprehensive metrics."""
        logger.info("Evaluating model...")
        predictions = self.predict(X_test)

        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)

        pred_direction = predictions > 0
        true_direction = y_test > 0
        directional_accuracy = np.mean(pred_direction == true_direction)

        # Information Coefficient (rank correlation)
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(predictions, y_test)
        except ImportError:
            ic = 0.0

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'information_coefficient': float(ic)
        }

        logger.info(f"Evaluation complete:")
        logger.info(f"   MSE: {mse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   RMSE: {rmse:.6f}")
        logger.info(f"   Directional Accuracy: {directional_accuracy*100:.2f}%")
        logger.info(f"   Information Coefficient: {ic:.4f}")

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


class EnsembleStockLSTM:
    """Ensemble of LSTM models for improved predictions and natural uncertainty.

    Trains N models with different random seeds. Prediction = mean of all models.
    Confidence = inverse of prediction standard deviation across models.
    """

    def __init__(
        self,
        n_models: int = 3,
        lookback: int = 30,
        num_features: int = 29,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")

        self.n_models = n_models
        self.models: List[StockLSTM] = []
        self.lookback = lookback
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        logger.info(f"Initializing Ensemble with {n_models} LSTM models")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 150,
        batch_size: int = 32,
        verbose: int = 0
    ) -> List[dict]:
        """Train all ensemble members with different random seeds."""
        all_metrics = []

        for i in range(self.n_models):
            logger.info(f"Training ensemble member {i+1}/{self.n_models}...")
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)

            model = StockLSTM(
                lookback=self.lookback,
                num_features=self.num_features,
                lstm_units=self.lstm_units,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg
            )
            model.build_model()
            model.train(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)
            metrics = model.evaluate(X_val, y_val)

            self.models.append(model)
            all_metrics.append(metrics)

        logger.info(f"Ensemble training complete: {len(self.models)} models")
        return all_metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction with uncertainty from model disagreement.

        Returns:
            mean_predictions: Average across all models
            std_predictions: Standard deviation (natural uncertainty)
        """
        if not self.models:
            raise ValueError("No models trained yet!")

        all_preds = np.array([m.predict(X) for m in self.models])
        return all_preds.mean(axis=0), all_preds.std(axis=0)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_mc_passes: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combined ensemble + MC Dropout uncertainty.

        Runs MC Dropout on each ensemble member, combines all predictions.
        """
        if not self.models:
            raise ValueError("No models trained yet!")

        all_preds = []
        for model in self.models:
            mc_mean, _ = model.predict_with_uncertainty(X, n_mc_passes)
            all_preds.append(mc_mean)

        all_preds = np.array(all_preds)
        return all_preds.mean(axis=0), all_preds.std(axis=0)

    def save_models(self, base_dir: str, symbol: str) -> None:
        """Save all ensemble models."""
        import os
        for i, model in enumerate(self.models):
            path = os.path.join(base_dir, f"{symbol}_ensemble_{i}.h5")
            model.save_model(path)

    def load_models(self, base_dir: str, symbol: str) -> None:
        """Load all ensemble models."""
        import os
        self.models = []
        for i in range(self.n_models):
            path = os.path.join(base_dir, f"{symbol}_ensemble_{i}.h5")
            if os.path.exists(path):
                model = StockLSTM(
                    lookback=self.lookback,
                    num_features=self.num_features,
                    lstm_units=self.lstm_units,
                    dropout_rate=self.dropout_rate,
                    l2_reg=self.l2_reg
                )
                model.load_model(path)
                self.models.append(model)
        logger.info(f"Loaded {len(self.models)} ensemble models for {symbol}")


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
    features = 29

    X_train = np.random.rand(samples, lookback, features)
    y_train = np.random.rand(samples) * 0.2 - 0.1

    X_val = np.random.rand(40, lookback, features)
    y_val = np.random.rand(40) * 0.2 - 0.1

    # Test single model
    model = StockLSTM(lookback=lookback, num_features=features)
    model.build_model()

    print("\nTraining on dummy data (10 epochs)...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=0)

    print("\nTesting prediction...")
    test_input = np.random.rand(1, lookback, features)
    prediction = model.predict(test_input)
    print(f"Predicted return: {prediction[0]*100:+.2f}%")

    print("\nTesting MC Dropout uncertainty...")
    mean_pred, std_pred = model.predict_with_uncertainty(test_input, n_forward_passes=20)
    print(f"MC Dropout: {mean_pred[0]*100:+.2f}% +/- {std_pred[0]*100:.2f}%")

    print("\nStock LSTM model test complete!")


if __name__ == "__main__":
    main()
