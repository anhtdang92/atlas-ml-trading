#!/usr/bin/env python3
"""
Hybrid Prediction Service for Stock ML Trading Dashboard

Combines enhanced mock predictions with real ML models for the best of both worlds.
Supports all stock categories: tech, sector leaders, ETFs, growth.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ml.prediction_service import PredictionService, DEFAULT_SYMBOLS
try:
    from gcp.deployment.vertex_prediction_service import VertexPredictionService
except ImportError:
    VertexPredictionService = None

logger = logging.getLogger(__name__)


class HybridPredictionService:
    """Hybrid prediction service combining real ML models with enhanced mocks."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models_status = {}

        self.enhanced_service = PredictionService(provider="local", models_dir=models_dir)
        self.vertex_service = None
        self.local_ml_models = {}

        self._init_vertex_ai()
        self._init_local_ml_models()

        logger.info("Hybrid Prediction Service initialized")
        logger.info(f"   Enhanced Mock: Available")
        logger.info(f"   Vertex AI: {'Available' if self.vertex_service else 'Not available'}")
        logger.info(f"   Local ML Models: {'Available' if self.local_ml_models else 'Not available'}")

    def _init_vertex_ai(self):
        try:
            if VertexPredictionService is None:
                return

            project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'stock-ml-trading-487')
            region = os.getenv('GCP_REGION', 'us-central1')
            endpoint_id = os.getenv('VERTEX_ENDPOINT_ID', '')

            if endpoint_id:
                self.vertex_service = VertexPredictionService(
                    project_id=project_id, region=region, endpoint_id=endpoint_id
                )
        except Exception as e:
            logger.warning(f"Vertex AI initialization failed: {e}")

    def _init_local_ml_models(self):
        try:
            if os.path.exists(self.models_dir):
                model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
                for model_file in model_files:
                    symbol = model_file.replace('_model.h5', '').upper()
                    model_path = os.path.join(self.models_dir, model_file)
                    self.local_ml_models[symbol] = {
                        'path': model_path, 'type': 'lstm', 'loaded': False
                    }
                if self.local_ml_models:
                    logger.info(f"Found {len(self.local_ml_models)} local ML models")
        except Exception as e:
            logger.warning(f"Local ML model initialization failed: {e}")

    def get_prediction(self, symbol: str, days_ahead: int = 21) -> Dict:
        """Get prediction using hybrid approach.

        Priority: Vertex AI → Local ML → Enhanced Mock → Basic Mock
        """
        logger.info(f"Getting hybrid prediction for {symbol} ({days_ahead} days ahead)")

        # Try Vertex AI
        if self.vertex_service:
            try:
                ml_prediction = self.vertex_service.get_prediction(symbol, days_ahead)
                if ml_prediction and ml_prediction.get('status') == 'success':
                    ml_prediction['prediction_source'] = 'vertex_ai_ml'
                    ml_prediction['prediction_type'] = 'real_ml'
                    return ml_prediction
            except Exception as e:
                logger.warning(f"Vertex AI error for {symbol}: {e}")

        # Try local ML
        if symbol in self.local_ml_models:
            try:
                local_prediction = self._predict_with_local_model(symbol, days_ahead)
                if local_prediction:
                    local_prediction['prediction_source'] = 'local_ml'
                    local_prediction['prediction_type'] = 'real_ml'
                    return local_prediction
            except Exception as e:
                logger.warning(f"Local ML error for {symbol}: {e}")

        # Enhanced mock prediction
        try:
            enhanced_prediction = self.enhanced_service.get_prediction(symbol, days_ahead)
            enhanced_prediction['prediction_source'] = 'enhanced_mock'
            enhanced_prediction['prediction_type'] = 'enhanced_mock'
            return enhanced_prediction
        except Exception as e:
            logger.warning(f"Enhanced mock error for {symbol}: {e}")
            basic = self._get_basic_mock_prediction(symbol, days_ahead)
            basic['prediction_source'] = 'basic_mock'
            basic['prediction_type'] = 'basic_mock'
            return basic

    def _predict_with_local_model(self, symbol: str, days_ahead: int) -> Optional[Dict]:
        """Load a trained local LSTM model and generate a real prediction."""
        try:
            from ml.lstm_model import StockLSTM, HAS_TENSORFLOW
            if not HAS_TENSORFLOW:
                return None

            from ml.feature_engineering import FeatureEngineer
            from ml.historical_data_fetcher import HistoricalDataFetcher

            model_info = self.local_ml_models.get(symbol)
            if not model_info or not os.path.exists(model_info['path']):
                return None

            # Load model
            fe = FeatureEngineer()
            fetcher = HistoricalDataFetcher()
            df = fetcher.fetch_historical_data(symbol, days=365)
            if df is None or len(df) < 60:
                return None

            df_features = fe.calculate_features(df)
            df_normalized = fe.normalize_features(df_features)

            model = StockLSTM(lookback=30, num_features=len(fe.features),
                              lstm_units=64, dropout_rate=0.2)
            model.load_model(model_info['path'])

            feature_data = df_normalized[fe.features].values
            last_sequence = feature_data[-30:].reshape(1, 30, -1)
            predicted_return = float(model.predict(last_sequence)[0])

            current_price = float(df['close'].iloc[-1])
            predicted_price = current_price * (1 + predicted_return)
            confidence = min(0.95, max(0.1, 1.0 - abs(predicted_return) * 2))

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': predicted_return,
                'confidence': confidence,
                'days_ahead': days_ahead,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Local ML prediction failed for {symbol}: {e}")
            return None

    def _get_basic_mock_prediction(self, symbol: str, days_ahead: int) -> Dict:
        import numpy as np

        fallback_prices = {
            'AAPL': 185.0, 'MSFT': 420.0, 'GOOGL': 175.0, 'AMZN': 185.0,
            'NVDA': 880.0, 'META': 510.0, 'TSLA': 175.0,
            'SPY': 520.0, 'QQQ': 450.0,
            'JPM': 200.0, 'UNH': 530.0, 'XOM': 110.0
        }
        current_price = fallback_prices.get(symbol, 100.0)

        np.random.seed(hash(symbol + str(days_ahead)) % 2**32)
        return_change = np.random.normal(0.03, 0.08)
        predicted_price = current_price * (1 + return_change)

        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': return_change,
            'confidence': 0.45,
            'days_ahead': days_ahead,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
        }

    def get_all_predictions(self, symbols: List[str] = None, days_ahead: int = 21) -> Dict[str, Dict]:
        """Get predictions for all symbols."""
        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        logger.info(f"Getting hybrid predictions for {len(symbols)} symbols")

        predictions = {}
        source_counts = {'real_ml': 0, 'enhanced_mock': 0, 'basic_mock': 0}

        for symbol in symbols:
            prediction = self.get_prediction(symbol, days_ahead)
            predictions[symbol] = prediction
            prediction_type = prediction.get('prediction_type', 'basic_mock')
            source_counts[prediction_type] = source_counts.get(prediction_type, 0) + 1

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def _has_model(self, symbol: str) -> bool:
        if symbol in self.local_ml_models:
            return True
        model_path = os.path.join(self.models_dir, f"{symbol}_model.h5")
        return os.path.exists(model_path)

    def train_model(self, symbol: str, days: int = 730, epochs: int = 150, save_model: bool = True) -> Dict:
        try:
            return self.enhanced_service.train_model(symbol, days, epochs, save_model)
        except Exception as e:
            return {'status': 'error', 'message': f'Training failed: {e}', 'symbol': symbol}

    def train_all_models(self, days: int = 730, epochs: int = 150) -> Dict:
        try:
            return self.enhanced_service.train_all_models(days, epochs)
        except Exception as e:
            return {'status': 'error', 'message': f'Training failed: {e}', 'results': {}}

    def get_prediction_summary(self) -> Dict:
        return {
            'enhanced_mock_available': True,
            'vertex_ai_available': self.vertex_service is not None,
            'local_ml_models_available': len(self.local_ml_models) > 0,
            'local_ml_models_count': len(self.local_ml_models),
            'supported_symbols': DEFAULT_SYMBOLS,
            'system_status': 'hybrid_operational'
        }
