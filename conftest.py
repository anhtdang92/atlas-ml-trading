"""Shared test fixtures for ATLAS Stock ML Trading Dashboard."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock


@pytest.fixture
def sample_ohlcv():
    """Generate 300-day OHLCV sample data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    price = np.random.randn(n).cumsum() + 150
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': price + np.random.randn(n) * 0.5,
        'high': price + abs(np.random.randn(n)) * 2,
        'low': price - abs(np.random.randn(n)) * 2,
        'close': price,
        'volume': np.random.rand(n) * 1_000_000 + 500_000
    })


@pytest.fixture
def short_ohlcv():
    """Generate 60-day OHLCV sample data (edge case)."""
    np.random.seed(99)
    n = 60
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    price = np.random.randn(n).cumsum() + 100
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'SHORT',
        'open': price + np.random.randn(n) * 0.5,
        'high': price + abs(np.random.randn(n)) * 2,
        'low': price - abs(np.random.randn(n)) * 2,
        'close': price,
        'volume': np.random.rand(n) * 500_000 + 100_000
    })


@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    from ml.feature_engineering import FeatureEngineer
    return FeatureEngineer()


@pytest.fixture
def featured_df(feature_engineer, sample_ohlcv):
    """DataFrame with all features calculated."""
    return feature_engineer.calculate_features(sample_ohlcv)


@pytest.fixture
def mock_stock_api():
    """Mock StockAPI for tests that shouldn't hit yfinance."""
    api = MagicMock()
    api.get_current_price.return_value = 185.0
    api.get_quote.return_value = {
        'current': 185.0,
        'open': 183.0,
        'high': 187.0,
        'low': 182.0,
        'close': 185.0,
        'volume': 50_000_000,
        'change_pct': 1.5,
        'prev_close': 182.0,
    }
    api.get_batch_quotes.return_value = {
        'AAPL': {'current': 185.0, 'change_pct': 1.5, 'high': 187.0, 'low': 182.0, 'volume': 50_000_000},
        'MSFT': {'current': 420.0, 'change_pct': 0.8, 'high': 425.0, 'low': 418.0, 'volume': 30_000_000},
    }
    return api


@pytest.fixture
def mock_predictions():
    """Mock prediction results for testing."""
    return [
        {
            'symbol': 'AAPL',
            'current_price': 185.0,
            'predicted_price': 192.0,
            'predicted_return': 0.038,
            'confidence': 0.72,
            'prediction_date': '2026-03-24',
            'status': 'success',
            'prediction_source': 'local_ml',
        },
        {
            'symbol': 'MSFT',
            'current_price': 420.0,
            'predicted_price': 410.0,
            'predicted_return': -0.024,
            'confidence': 0.65,
            'prediction_date': '2026-03-24',
            'status': 'success',
            'prediction_source': 'technical_analysis',
        },
    ]
