"""
Prediction Service for Crypto ML Trading Dashboard

This service integrates all ML components to provide real-time predictions:
- Fetches historical data from Kraken API
- Applies feature engineering
- Loads trained LSTM models
- Generates predictions with confidence scores
- Handles model training and evaluation

Architecture:
    Data Flow: Kraken API → Feature Engineering → LSTM Model → Predictions
    Caching: Predictions cached for 1 hour to reduce API calls
    Error Handling: Graceful fallbacks for missing models/data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path

# Local imports
from .historical_data_fetcher import HistoricalDataFetcher
from .feature_engineering import FeatureEngineer

# Try to import LSTM model, fallback to mock if not available
try:
    from .lstm_model import CryptoLSTM
    HAS_LSTM = True
except ImportError:
    print("⚠️  LSTM model not available (TensorFlow required)")
    HAS_LSTM = False
    CryptoLSTM = None

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Main service for cryptocurrency price predictions.
    
    This service orchestrates the entire ML pipeline:
    1. Data Collection: Fetch recent data from Kraken
    2. Feature Engineering: Create technical indicators
    3. Model Prediction: Generate forecasts using LSTM
    4. Confidence Scoring: Assess prediction reliability
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize prediction service.
        
        Args:
            models_dir: Directory to store/load trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_fetcher = HistoricalDataFetcher()
        self.feature_engineer = FeatureEngineer()
        
        # Model cache
        self.models = {}
        self.scalers = {}
        
        # Supported symbols
        self.symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP']
        
        logger.info("🧠 Prediction Service initialized")
        logger.info(f"   Models directory: {self.models_dir}")
        logger.info(f"   Supported symbols: {', '.join(self.symbols)}")
    
    def get_prediction(
        self,
        symbol: str,
        days_ahead: int = 7,
        use_cache: bool = True
    ) -> Dict:
        """
        Get price prediction for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days_ahead: Days to predict ahead (default: 7)
            use_cache: Use cached prediction if available
            
        Returns:
            Dictionary with prediction data:
            {
                'symbol': 'BTC',
                'current_price': 122001.0,
                'predicted_price': 128500.0,
                'predicted_return': 0.053,
                'confidence': 0.78,
                'prediction_date': '2025-10-04',
                'model_version': 'v1.0',
                'features_used': ['close', 'MA_7', 'RSI', ...],
                'status': 'success'
            }
        """
        logger.info(f"🔮 Generating prediction for {symbol} ({days_ahead} days ahead)")
        
        # For now, always use mock predictions since TensorFlow is not available
        if not HAS_LSTM:
            logger.info("Using mock predictions (TensorFlow not available)")
            return self._create_mock_prediction(symbol, days_ahead)
        
        try:
            # Check if we have a trained model
            if not self._has_model(symbol):
                return self._create_mock_prediction(symbol, days_ahead)
            
            # Fetch recent data
            df = self.data_fetcher.fetch_historical_data(symbol, days=90)
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}, using mock prediction")
                return self._create_mock_prediction(symbol, days_ahead)
            
            # Apply feature engineering
            df_features = self.feature_engineer.calculate_features(df)
            
            # Normalize features
            df_normalized = self.feature_engineer.normalize_features(df_features)
            
            # Load model
            model = self._load_model(symbol)
            if model is None:
                return self._create_mock_prediction(symbol, days_ahead)
            
            # Prepare input sequence (last 7 days)
            lookback = 7
            feature_data = df_normalized[self.feature_engineer.features].values
            last_sequence = feature_data[-lookback:].reshape(1, lookback, -1)
            
            # Make prediction
            predicted_return = model.predict(last_sequence)[0]
            
            # Calculate predicted price
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence (simplified - based on model certainty)
            confidence = min(0.95, max(0.1, 1.0 - abs(predicted_return) * 2))
            
            # Create result
            result = {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_return': float(predicted_return),
                'confidence': float(confidence),
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'model_version': 'v1.0',
                'features_used': self.feature_engineer.features,
                'status': 'success',
                'data_points': len(df)
            }
            
            logger.info(f"✅ Prediction generated for {symbol}: {predicted_return*100:+.2f}% return")
            return result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return self._create_mock_prediction(symbol, days_ahead)
    
    def get_all_predictions(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get predictions for all supported cryptocurrencies.
        
        Args:
            days_ahead: Days to predict ahead
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"🔮 Generating predictions for all {len(self.symbols)} symbols")
        
        predictions = []
        for symbol in self.symbols:
            pred = self.get_prediction(symbol, days_ahead)
            predictions.append(pred)
        
        # Sort by predicted return (highest first)
        predictions.sort(key=lambda x: x['predicted_return'], reverse=True)
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        return predictions
    
    def train_model(
        self,
        symbol: str,
        days: int = 365,
        epochs: int = 50,
        save_model: bool = True
    ) -> Dict:
        """
        Train LSTM model for a specific cryptocurrency.
        
        Args:
            symbol: Crypto symbol to train
            days: Days of historical data to use
            epochs: Number of training epochs
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        logger.info(f"🚀 Training LSTM model for {symbol}...")
        
        # Check if TensorFlow is available
        if not HAS_LSTM:
            return {
                'status': 'error',
                'message': 'TensorFlow not available. Please install TensorFlow to train models.',
                'symbol': symbol,
                'solution': 'Run: pip install tensorflow (requires Python 3.9-3.11)'
            }
        
        try:
            # Fetch historical data
            df = self.data_fetcher.fetch_historical_data(symbol, days)
            if df is None or len(df) < 100:
                return {
                    'status': 'error',
                    'message': f'Insufficient data for {symbol}',
                    'symbol': symbol
                }
            
            # Apply feature engineering
            df_features = self.feature_engineer.calculate_features(df)
            df_normalized = self.feature_engineer.normalize_features(df_features)
            
            # Create sequences
            X, y = self.feature_engineer.create_sequences(
                df_normalized, 
                lookback=7, 
                prediction_horizon=7
            )
            
            if len(X) < 50:
                return {
                    'status': 'error',
                    'message': f'Insufficient sequences for {symbol}',
                    'symbol': symbol
                }
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build and train model
            model = CryptoLSTM(
                lookback=7,
                num_features=len(self.feature_engineer.features),
                lstm_units=50,
                dropout_rate=0.2
            )
            
            model.build_model()
            history = model.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=32, verbose=0
            )
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            # Save model if requested
            if save_model:
                model_path = self.models_dir / f"{symbol}_model.h5"
                model.save_model(str(model_path))
                
                # Save scaler info
                scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
                scaler_info = {
                    'features': self.feature_engineer.features,
                    'min_values': df_normalized[self.feature_engineer.features].min().to_dict(),
                    'max_values': df_normalized[self.feature_engineer.features].max().to_dict()
                }
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler_info, f)
            
            # Create result
            result = {
                'status': 'success',
                'symbol': symbol,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'metrics': metrics,
                'model_saved': save_model
            }
            
            logger.info(f"✅ Model training complete for {symbol}")
            logger.info(f"   Final validation loss: {result['final_val_loss']:.6f}")
            logger.info(f"   Directional accuracy: {metrics['directional_accuracy']*100:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'symbol': symbol
            }
    
    def train_all_models(self, days: int = 365, epochs: int = 50) -> Dict:
        """
        Train models for all supported cryptocurrencies.
        
        Args:
            days: Days of historical data to use
            epochs: Number of training epochs per model
            
        Returns:
            Training results for all symbols
        """
        logger.info(f"🚀 Training models for all {len(self.symbols)} symbols...")
        
        results = {}
        for i, symbol in enumerate(self.symbols):
            logger.info(f"\n[{i+1}/{len(self.symbols)}] Training {symbol}...")
            result = self.train_model(symbol, days, epochs)
            results[symbol] = result
        
        # Summary
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        logger.info(f"\n✅ Training complete: {successful}/{len(self.symbols)} models trained successfully")
        
        return results
    
    def _has_model(self, symbol: str) -> bool:
        """Check if trained model exists for symbol."""
        model_path = self.models_dir / f"{symbol}_model.h5"
        return model_path.exists()
    
    def _load_model(self, symbol: str) -> Optional[CryptoLSTM]:
        """Load trained model for symbol."""
        try:
            model_path = self.models_dir / f"{symbol}_model.h5"
            if not model_path.exists():
                return None
            
            # Create model instance
            model = CryptoLSTM(
                lookback=7,
                num_features=len(self.feature_engineer.features),
                lstm_units=50,
                dropout_rate=0.2
            )
            
            # Load weights
            model.load_model(str(model_path))
            return model
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return None
    
    def _create_mock_prediction(self, symbol: str, days_ahead: int) -> Dict:
        """Create mock prediction when model is not available."""
        # Get current price from Kraken
        try:
            from data.kraken_api import KrakenAPI
            kraken = KrakenAPI()
            
            pair_map = {
                'BTC': 'XXBTZUSD',
                'ETH': 'XETHZUSD', 
                'SOL': 'SOLUSD',
                'ADA': 'ADAUSD',
                'DOT': 'DOTUSD',
                'XRP': 'XXRPZUSD'
            }
            
            pair = pair_map.get(symbol)
            if pair:
                ticker_data = kraken.get_ticker([pair])
                if ticker_data:
                    matching_key = [k for k in ticker_data.keys() if pair in k or k in pair]
                    if matching_key:
                        current_price = float(ticker_data[matching_key[0]]['c'][0])
                    else:
                        current_price = 1000.0  # Fallback
                else:
                    current_price = 1000.0
            else:
                current_price = 1000.0
                
        except:
            current_price = 1000.0
        
        # Generate mock prediction
        np.random.seed(hash(symbol) % 2**32)  # Consistent mock data
        predicted_return = np.random.normal(0.02, 0.05)  # 2% mean, 5% std
        predicted_price = current_price * (1 + predicted_return)
        confidence = np.random.uniform(0.6, 0.8)
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_return': float(predicted_return),
            'confidence': float(confidence),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'model_version': 'mock',
            'features_used': [],
            'status': 'mock',
            'data_points': 0
        }


def main():
    """Test the prediction service."""
    print("="*60)
    print("🧪 Testing Prediction Service")
    print("="*60)
    
    # Initialize service
    service = PredictionService()
    
    # Test mock predictions
    print("\n🔮 Testing mock predictions...")
    for symbol in ['BTC', 'ETH', 'SOL']:
        pred = service.get_prediction(symbol)
        print(f"\n{symbol}:")
        print(f"   Current: ${pred['current_price']:,.2f}")
        print(f"   Predicted: ${pred['predicted_price']:,.2f}")
        print(f"   Return: {pred['predicted_return']*100:+.2f}%")
        print(f"   Confidence: {pred['confidence']*100:.1f}%")
        print(f"   Status: {pred['status']}")
    
    print("\n✅ Prediction service test complete!")


if __name__ == "__main__":
    main()
