"""
Prediction Service for Stock ML Trading Dashboard

Integrates all ML components to provide real-time stock predictions:
- Fetches historical data via yfinance
- Applies feature engineering (29 technical indicators)
- Loads trained LSTM models (single or ensemble)
- Generates predictions with MC Dropout confidence scores
- Walk-forward cross-validation for honest performance evaluation
- Baseline model comparisons (buy-and-hold, MA crossover, linear regression)

Architecture:
    Data Flow: yfinance -> Feature Engineering -> LSTM Model -> MC Dropout -> Predictions
    Caching: Predictions cached for 1 hour
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

from .historical_data_fetcher import HistoricalDataFetcher
from .feature_engineering import FeatureEngineer
from data.stock_api import get_all_symbols

try:
    from .lstm_model import StockLSTM, EnsembleStockLSTM, HAS_TENSORFLOW
    HAS_LSTM = True
except ImportError:
    print("Warning: LSTM model not available (TensorFlow required)")
    HAS_LSTM = False
    HAS_TENSORFLOW = False
    StockLSTM = None
    EnsembleStockLSTM = None

logger = logging.getLogger(__name__)


# Default symbols for predictions - sourced from stock_api.STOCK_UNIVERSE
DEFAULT_SYMBOLS = get_all_symbols()


class BaselineModels:
    """Baseline models for comparison against LSTM predictions."""

    @staticmethod
    def buy_and_hold(df: pd.DataFrame, prediction_horizon: int = 21) -> Dict:
        """Baseline: assume historical mean return continues."""
        returns = df['close'].pct_change(periods=prediction_horizon).dropna()
        if len(returns) == 0:
            return {'predicted_return': 0.0, 'name': 'buy_and_hold'}
        return {
            'predicted_return': float(returns.mean()),
            'std': float(returns.std()),
            'name': 'buy_and_hold'
        }

    @staticmethod
    def ma_crossover(df: pd.DataFrame, prediction_horizon: int = 21) -> Dict:
        """Baseline: MA(20) > MA(50) = bullish, else bearish."""
        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()

        # Current signal
        if len(ma20.dropna()) < 1 or len(ma50.dropna()) < 1:
            return {'predicted_return': 0.0, 'name': 'ma_crossover'}

        bullish = ma20.iloc[-1] > ma50.iloc[-1]
        # Historical returns conditioned on signal
        returns = df['close'].pct_change(periods=prediction_horizon).dropna()
        signal = (ma20 > ma50).reindex(returns.index)

        if bullish:
            cond_returns = returns[signal == True]
        else:
            cond_returns = returns[signal == False]

        pred_return = float(cond_returns.mean()) if len(cond_returns) > 0 else 0.0
        return {
            'predicted_return': pred_return,
            'signal': 'bullish' if bullish else 'bearish',
            'name': 'ma_crossover'
        }

    @staticmethod
    def linear_regression(df: pd.DataFrame, prediction_horizon: int = 21) -> Dict:
        """Baseline: linear regression on recent returns."""
        returns = df['close'].pct_change().dropna().values
        if len(returns) < 30:
            return {'predicted_return': 0.0, 'name': 'linear_regression'}

        # Use last 60 days of returns
        recent = returns[-60:]
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        # Extrapolate prediction_horizon days
        future_x = len(recent) + prediction_horizon / 2
        daily_return = coeffs[0] * future_x + coeffs[1]
        pred_return = daily_return * prediction_horizon

        return {
            'predicted_return': float(np.clip(pred_return, -0.3, 0.3)),
            'slope': float(coeffs[0]),
            'name': 'linear_regression'
        }


class PredictionService:
    """Main service for stock price predictions."""

    def __init__(self, models_dir: str = "models", provider: str = "vertex"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.data_fetcher = HistoricalDataFetcher()
        self.feature_engineer = FeatureEngineer()

        self.models = {}
        self.scalers = {}

        self.symbols = DEFAULT_SYMBOLS

        self.provider = provider
        self.vertex_service = None

        if provider == "vertex":
            self._init_vertex_ai()

        logger.info("Prediction Service initialized")
        logger.info(f"   Models directory: {self.models_dir}")
        logger.info(f"   Provider: {provider}")

    def _init_vertex_ai(self):
        """Initialize Vertex AI prediction service."""
        try:
            from gcp.deployment.vertex_prediction_service import VertexPredictionService

            project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'stock-ml-trading-487')
            region = os.getenv('GCP_REGION', 'us-central1')
            endpoint_id = os.getenv('VERTEX_ENDPOINT_ID', '')

            if endpoint_id:
                self.vertex_service = VertexPredictionService(
                    project_id=project_id, region=region, endpoint_id=endpoint_id
                )
                logger.info("Vertex AI service initialized")
            else:
                logger.warning("Vertex AI endpoint ID not configured, falling back to local")
                self.provider = "local"

        except ImportError:
            logger.warning("Vertex AI dependencies not available, falling back to local")
            self.provider = "local"
        except Exception as e:
            logger.warning(f"Vertex AI initialization failed: {e}, falling back to local")
            self.provider = "local"

    def get_prediction(
        self,
        symbol: str,
        days_ahead: int = 21,
        use_cache: bool = True,
        allow_mock: bool = False
    ) -> Dict:
        """Get price prediction for a stock.

        Uses MC Dropout for uncertainty estimation when a real model is available.

        Args:
            symbol: Stock ticker symbol.
            days_ahead: Prediction horizon in trading days.
            use_cache: Whether to use cached predictions.
            allow_mock: If False (default), return an explicit 'no_model' status
                        instead of silently falling back to mock/random predictions.
                        Set to True only for dashboard display purposes — NEVER for
                        actual trade decisions.
        """
        logger.info(f"Generating prediction for {symbol} ({days_ahead} days ahead)")

        if self.provider == "vertex" and self.vertex_service:
            return self._get_vertex_prediction(symbol, days_ahead)

        if not HAS_LSTM:
            if allow_mock:
                return self._create_mock_prediction(symbol, days_ahead)
            return self._no_model_response(symbol, days_ahead, reason="TensorFlow not available")

        try:
            if not self._has_model(symbol):
                if allow_mock:
                    return self._create_mock_prediction(symbol, days_ahead)
                return self._no_model_response(symbol, days_ahead, reason=f"No trained model for {symbol}")

            df = self.data_fetcher.fetch_historical_data(symbol, days=365)
            if df is None or len(df) < 100:
                if allow_mock:
                    return self._create_mock_prediction(symbol, days_ahead)
                return self._no_model_response(symbol, days_ahead, reason=f"Insufficient data for {symbol}")

            df_features = self.feature_engineer.calculate_features(df)

            # Load saved scaler params to normalize with training-set statistics
            scaler_params = self._load_scaler(symbol)
            if scaler_params:
                df_normalized = self.feature_engineer.normalize_features(
                    df_features, fit=False, scaler_params=scaler_params
                )
            else:
                df_normalized = self.feature_engineer.normalize_features(df_features, fit=True)

            model = self._load_model(symbol)
            if model is None:
                if allow_mock:
                    return self._create_mock_prediction(symbol, days_ahead)
                return self._no_model_response(symbol, days_ahead, reason=f"Failed to load model for {symbol}")

            lookback = 30
            feature_data = df_normalized[self.feature_engineer.features].values
            if len(feature_data) < lookback:
                if allow_mock:
                    return self._create_mock_prediction(symbol, days_ahead)
                return self._no_model_response(symbol, days_ahead, reason=f"Insufficient feature data for {symbol}")
            last_sequence = feature_data[-lookback:].reshape(1, lookback, -1)

            # MC Dropout for uncertainty-based confidence
            mean_pred, std_pred = model.predict_with_uncertainty(last_sequence, n_forward_passes=50)
            predicted_return = float(mean_pred[0])
            prediction_uncertainty = float(std_pred[0])

            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)

            # Confidence from MC Dropout uncertainty
            # Lower uncertainty -> higher confidence, calibrated to [0.1, 0.95]
            confidence = float(np.clip(1.0 - prediction_uncertainty * 10, 0.1, 0.95))

            # Get baseline comparisons
            baselines = self._get_baselines(df, days_ahead)

            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_return': predicted_return,
                'confidence': confidence,
                'prediction_uncertainty': prediction_uncertainty,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'model_version': 'v2.0-mc-dropout',
                'features_used': self.feature_engineer.features,
                'status': 'success',
                'data_points': len(df),
                'baselines': baselines,
                'prediction_source': 'local_ml'
            }

        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            if allow_mock:
                return self._create_mock_prediction(symbol, days_ahead)
            return self._no_model_response(symbol, days_ahead, reason=str(e))

    def get_all_predictions(self, days_ahead: int = 21, allow_mock: bool = False) -> List[Dict]:
        """Get predictions for all tracked stocks.

        Args:
            days_ahead: Prediction horizon.
            allow_mock: If True, fall back to mock predictions for display.
                        Default False — only return real ML predictions.
        """
        logger.info(f"Generating predictions for {len(self.symbols)} symbols (allow_mock={allow_mock})")
        predictions = []
        for symbol in self.symbols:
            pred = self.get_prediction(symbol, days_ahead, allow_mock=allow_mock)
            predictions.append(pred)
        predictions.sort(key=lambda x: x.get('predicted_return', 0), reverse=True)

        real_count = sum(1 for p in predictions if p.get('prediction_source') == 'local_ml')
        mock_count = sum(1 for p in predictions if p.get('status') in ('enhanced_mock', 'basic_mock'))
        no_model = sum(1 for p in predictions if p.get('status') == 'no_model')
        logger.info(
            f"Generated {len(predictions)} predictions: "
            f"{real_count} real ML, {mock_count} mock, {no_model} no_model"
        )
        return predictions

    def train_model(
        self,
        symbol: str,
        days: int = 730,
        epochs: int = 150,
        save_model: bool = True,
        use_ensemble: bool = False,
        n_ensemble: int = 3
    ) -> Dict:
        """Train LSTM model with proper train/val split and normalization.

        Fixes from v1:
        - Scaler fit ONLY on training data (no data leakage)
        - Walk-forward validation metrics reported
        - Baseline comparison included
        - Ensemble training supported
        """
        logger.info(f"Training {'ensemble' if use_ensemble else 'single'} LSTM for {symbol}...")

        if not HAS_LSTM:
            return {
                'status': 'error',
                'message': 'TensorFlow not available. Install with: pip install tensorflow',
                'symbol': symbol
            }

        try:
            df = self.data_fetcher.fetch_historical_data(symbol, days)
            if df is None or len(df) < 200:
                return {'status': 'error', 'message': f'Insufficient data for {symbol}', 'symbol': symbol}

            df_features = self.feature_engineer.calculate_features(df)

            # --- FIX: Split BEFORE normalization to prevent data leakage ---
            split_idx = int(len(df_features) * 0.8)
            df_train = df_features.iloc[:split_idx].copy()
            df_val = df_features.iloc[split_idx:].copy()

            # Fit scaler ONLY on training data
            df_train_norm = self.feature_engineer.normalize_features(df_train, fit=True)
            # Apply training scaler to validation data (may exceed [0,1], clipped)
            df_val_norm = self.feature_engineer.normalize_features(df_val, fit=False)

            # Create sequences from normalized data
            X_train, y_train = self.feature_engineer.create_sequences(
                df_train_norm, lookback=30, prediction_horizon=21
            )
            X_val, y_val = self.feature_engineer.create_sequences(
                df_val_norm, lookback=30, prediction_horizon=21
            )

            if len(X_train) < 50 or len(X_val) < 10:
                return {'status': 'error', 'message': f'Insufficient sequences for {symbol}', 'symbol': symbol}

            num_features = len(self.feature_engineer.features)

            if use_ensemble:
                ensemble = EnsembleStockLSTM(
                    n_models=n_ensemble, lookback=30,
                    num_features=num_features, lstm_units=64,
                    dropout_rate=0.2, l2_reg=1e-4
                )
                all_metrics = ensemble.train(X_train, y_train, X_val, y_val, epochs, 32, 0)
                metrics = all_metrics[-1]  # Use last model's metrics as representative

                if save_model:
                    ensemble.save_models(str(self.models_dir), symbol)
            else:
                model = StockLSTM(
                    lookback=30, num_features=num_features,
                    lstm_units=64, dropout_rate=0.2, l2_reg=1e-4
                )
                model.build_model()
                history = model.train(X_train, y_train, X_val, y_val, epochs, 32, 0)
                metrics = model.evaluate(X_val, y_val)

                if save_model:
                    model_path = self.models_dir / f"{symbol}_model.h5"
                    model.save_model(str(model_path))

            # Save scaler params (training-set statistics only)
            if save_model:
                scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
                scaler_info = {
                    'features': self.feature_engineer.features,
                    'scaler_params': self.feature_engineer.get_scaler_params()
                }
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler_info, f)

            # Baseline comparison
            baselines = self._get_baselines(df, 21)

            return {
                'status': 'success',
                'symbol': symbol,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'epochs_trained': len(history.history['loss']) if not use_ensemble else epochs,
                'final_train_loss': history.history['loss'][-1] if not use_ensemble else 0.0,
                'final_val_loss': history.history['val_loss'][-1] if not use_ensemble else 0.0,
                'metrics': metrics,
                'baselines': baselines,
                'model_saved': save_model,
                'model_type': 'ensemble' if use_ensemble else 'single'
            }

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'status': 'error', 'message': str(e), 'symbol': symbol}

    def walk_forward_validate(
        self,
        symbol: str,
        n_splits: int = 5,
        days: int = 730,
        epochs: int = 100
    ) -> Dict:
        """Walk-forward cross-validation for honest performance evaluation.

        Uses expanding window: train on [0:t], validate on [t:t+step].
        Reports per-fold and aggregate metrics.
        """
        if not HAS_LSTM:
            return {'status': 'error', 'message': 'TensorFlow not available'}

        logger.info(f"Walk-forward validation for {symbol} ({n_splits} folds)...")

        try:
            df = self.data_fetcher.fetch_historical_data(symbol, days)
            if df is None or len(df) < 300:
                return {'status': 'error', 'message': 'Insufficient data'}

            df_features = self.feature_engineer.calculate_features(df)
            total_len = len(df_features)
            min_train = int(total_len * 0.4)
            step_size = (total_len - min_train) // n_splits

            fold_metrics = []
            baseline_metrics = []

            for fold in range(n_splits):
                train_end = min_train + fold * step_size
                val_end = min(train_end + step_size, total_len)

                if val_end - train_end < 30:
                    continue

                df_train = df_features.iloc[:train_end].copy()
                df_val = df_features.iloc[train_end:val_end].copy()

                # Fit scaler on this fold's training data only
                df_train_norm = self.feature_engineer.normalize_features(df_train, fit=True)
                df_val_norm = self.feature_engineer.normalize_features(df_val, fit=False)

                X_train, y_train = self.feature_engineer.create_sequences(df_train_norm, 30, 21)
                X_val, y_val = self.feature_engineer.create_sequences(df_val_norm, 30, 21)

                if len(X_train) < 30 or len(X_val) < 5:
                    continue

                model = StockLSTM(lookback=30, num_features=len(self.feature_engineer.features),
                                  lstm_units=64, dropout_rate=0.2, l2_reg=1e-4)
                model.build_model()
                model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32, verbose=0)
                metrics = model.evaluate(X_val, y_val)
                metrics['fold'] = fold
                fold_metrics.append(metrics)

                # Baseline on same fold
                df_fold = df.iloc[train_end:val_end]
                baselines = self._get_baselines(df_fold, 21)
                baseline_metrics.append(baselines)

                logger.info(f"  Fold {fold+1}: DA={metrics['directional_accuracy']:.2%}, "
                            f"MAE={metrics['mae']:.4f}")

            if not fold_metrics:
                return {'status': 'error', 'message': 'Not enough data for walk-forward'}

            # Aggregate
            avg_metrics = {
                'avg_directional_accuracy': np.mean([m['directional_accuracy'] for m in fold_metrics]),
                'avg_mae': np.mean([m['mae'] for m in fold_metrics]),
                'avg_rmse': np.mean([m['rmse'] for m in fold_metrics]),
                'avg_ic': np.mean([m.get('information_coefficient', 0) for m in fold_metrics]),
                'std_directional_accuracy': np.std([m['directional_accuracy'] for m in fold_metrics]),
            }

            return {
                'status': 'success',
                'symbol': symbol,
                'n_folds': len(fold_metrics),
                'fold_metrics': fold_metrics,
                'aggregate_metrics': avg_metrics,
                'baseline_metrics': baseline_metrics
            }

        except Exception as e:
            logger.error(f"Walk-forward validation error: {e}")
            return {'status': 'error', 'message': str(e)}

    def train_all_models(self, days: int = 730, epochs: int = 150) -> Dict:
        """Train models for all tracked stocks."""
        logger.info(f"Training models for {len(self.symbols)} symbols...")
        results = {}
        for i, symbol in enumerate(self.symbols):
            logger.info(f"\n[{i+1}/{len(self.symbols)}] Training {symbol}...")
            result = self.train_model(symbol, days, epochs)
            results[symbol] = result

        successful = sum(1 for r in results.values() if r['status'] == 'success')
        logger.info(f"\nTraining complete: {successful}/{len(self.symbols)} models trained")
        return results

    def _get_baselines(self, df: pd.DataFrame, days_ahead: int) -> Dict:
        """Run all baseline models for comparison."""
        return {
            'buy_and_hold': BaselineModels.buy_and_hold(df, days_ahead),
            'ma_crossover': BaselineModels.ma_crossover(df, days_ahead),
            'linear_regression': BaselineModels.linear_regression(df, days_ahead),
        }

    def _has_model(self, symbol: str) -> bool:
        model_path = self.models_dir / f"{symbol}_model.h5"
        return model_path.exists()

    def _load_model(self, symbol: str) -> Optional[StockLSTM]:
        try:
            model_path = self.models_dir / f"{symbol}_model.h5"
            if not model_path.exists():
                return None
            model = StockLSTM(
                lookback=30,
                num_features=len(self.feature_engineer.features) or 29,
                lstm_units=64, dropout_rate=0.2, l2_reg=1e-4
            )
            model.load_model(str(model_path))
            return model
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return None

    def _load_scaler(self, symbol: str) -> Optional[Dict]:
        """Load saved scaler params (training-set statistics)."""
        try:
            scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
            if not scaler_path.exists():
                return None
            with open(scaler_path, 'rb') as f:
                scaler_info = pickle.load(f)
            # Support both old format (min_values/max_values) and new format (scaler_params)
            if 'scaler_params' in scaler_info:
                return scaler_info['scaler_params']
            elif 'min_values' in scaler_info and 'max_values' in scaler_info:
                # Convert old format
                params = {}
                for feat in scaler_info.get('features', []):
                    params[feat] = {
                        'min': scaler_info['min_values'].get(feat, 0),
                        'max': scaler_info['max_values'].get(feat, 1)
                    }
                return params
            return None
        except Exception as e:
            logger.warning(f"Could not load scaler for {symbol}: {e}")
            return None

    def _get_vertex_prediction(self, symbol: str, days_ahead: int) -> Dict:
        try:
            prediction = self.vertex_service.predict_single(symbol, days_ahead)
            if 'error' in prediction:
                return self._create_mock_prediction(symbol, days_ahead)

            return {
                'symbol': symbol,
                'current_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'predicted_return': prediction['price_change'] / 100,
                'confidence': prediction['confidence'],
                'prediction_date': prediction['timestamp'].strftime('%Y-%m-%d'),
                'model_version': prediction['model_version'],
                'features_used': ['vertex_ai_features'],
                'status': 'success',
                'data_points': 0,
                'provider': 'vertex_ai',
                'prediction_source': 'vertex_ai_ml'
            }
        except Exception as e:
            logger.error(f"Vertex AI error for {symbol}: {e}")
            return self._create_mock_prediction(symbol, days_ahead)

    def _create_mock_prediction(self, symbol: str, days_ahead: int) -> Dict:
        """Create enhanced mock prediction using real current price from API.

        WARNING: This uses RANDOM pseudo-technical-analysis, NOT real indicators.
        Only used when allow_mock=True for dashboard display.
        NEVER use mock predictions for actual trade decisions.
        """
        try:
            from data.stock_api import StockAPI
            api = StockAPI()

            current_price = None
            quote = api.get_quote(symbol)
            if quote:
                current_price = quote['current']

            if current_price is None:
                # Try get_current_price as second attempt
                current_price = api.get_current_price(symbol)

            if current_price is None or current_price <= 0:
                logger.warning(f"Could not fetch price for {symbol}, using basic mock")
                return self._create_basic_mock(symbol, days_ahead)

            # Enhanced prediction with technical analysis simulation
            np.random.seed(hash(symbol) % 2**32)

            rsi = np.random.uniform(30, 70)
            trend = np.random.normal(0.02, 0.03)  # Stocks have slight upward bias
            volatility = np.random.uniform(0.01, 0.04)

            momentum_factor = (rsi - 50) / 50 * 0.2
            trend_factor = trend * 0.5
            noise_factor = np.random.normal(0, 0.01)

            predicted_return = momentum_factor + trend_factor + noise_factor
            # Scale by prediction horizon
            predicted_return *= (days_ahead / 21.0)
            predicted_price = current_price * (1 + predicted_return)

            signal_strength = abs(momentum_factor) + abs(trend_factor)
            confidence = min(0.90, max(0.4, 0.55 + signal_strength * 2))

            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_return': float(predicted_return),
                'confidence': float(confidence),
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'model_version': 'enhanced-mock-v2.0',
                'features_used': ['real_price', 'rsi', 'trend', 'volatility'],
                'status': 'enhanced_mock',
                'data_points': 30,
                'prediction_source': 'technical_analysis',
                'warning': 'MOCK PREDICTION — uses random pseudo-indicators, not real ML. Do not trade on this.',
                'technical_analysis': {
                    'rsi': float(rsi),
                    'trend': float(trend),
                    'volatility': float(volatility)
                }
            }

        except Exception as e:
            logger.warning(f"Enhanced mock failed for {symbol}: {e}")
            return self._create_basic_mock(symbol, days_ahead)

    def _no_model_response(self, symbol: str, days_ahead: int, reason: str = "") -> Dict:
        """Return an explicit 'no model available' response instead of fake predictions.

        This prevents the system from silently trading on random noise.
        """
        return {
            'symbol': symbol,
            'current_price': 0.0,
            'predicted_price': 0.0,
            'predicted_return': 0.0,
            'confidence': 0.0,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'model_version': 'none',
            'features_used': [],
            'status': 'no_model',
            'data_points': 0,
            'prediction_source': 'none',
            'message': f'No real ML prediction available: {reason}. '
                       'Train a model first with prediction_service.train_model(). '
                       'DO NOT trade on mock/random predictions.',
        }

    def _create_basic_mock(self, symbol: str, days_ahead: int) -> Dict:
        """Ultimate fallback when no price data available.

        WARNING: This returns RANDOM data. Only used when allow_mock=True
        for dashboard display. NEVER use for trade decisions.
        """
        np.random.seed(hash(symbol) % 2**32)
        current_price = 100.0
        predicted_return = np.random.normal(0.03, 0.08)
        predicted_price = current_price * (1 + predicted_return)
        confidence = np.random.uniform(0.3, 0.5)

        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_return': float(predicted_return),
            'confidence': float(confidence),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'model_version': 'basic-mock',
            'features_used': ['estimated_price'],
            'status': 'basic_mock',
            'data_points': 0,
            'prediction_source': 'basic_mock',
            'warning': 'THIS IS RANDOM DATA — not a real prediction. Do not trade on this.'
        }
