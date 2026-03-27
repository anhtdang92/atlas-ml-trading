"""
Comprehensive ML Pipeline Tests for ATLAS Stock ML Intelligence System

Tests cover:
- Feature engineering (technical indicators, sequences, normalization)
- Stock universe validation (categories, metadata, duplicates)
- Portfolio rebalancer (allocations, risk controls, orders)
- Data validation (OHLCV schema, predictions, price/volume constraints)

Usage:
    python -m pytest tests/unit/test_ml_pipeline.py -v
"""

import os
import sys
import re
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Project root on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.feature_engineering import FeatureEngineer
from data.stock_api import STOCK_UNIVERSE, get_all_symbols, get_stock_info

# Check tensorflow availability for conditional skipping
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv():
    """Generate a realistic OHLCV DataFrame with 300 business days."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime(2026, 3, 20), periods=300, freq="B")
    price = np.abs(np.random.randn(300).cumsum() + 150) + 50
    return pd.DataFrame({
        "timestamp": dates,
        "open": price + np.random.randn(300) * 0.5,
        "high": price + np.abs(np.random.randn(300)) * 2,
        "low": price - np.abs(np.random.randn(300)) * 2,
        "close": price,
        "volume": (np.random.rand(300) * 1_000_000 + 500_000).astype(int),
    })


@pytest.fixture
def short_ohlcv():
    """Generate a short OHLCV DataFrame (60 rows) for edge-case tests."""
    np.random.seed(99)
    dates = pd.date_range(end=datetime(2026, 3, 20), periods=60, freq="B")
    price = np.abs(np.random.randn(60).cumsum() + 200) + 80
    return pd.DataFrame({
        "timestamp": dates,
        "open": price + np.random.randn(60) * 0.3,
        "high": price + np.abs(np.random.randn(60)) * 1.5,
        "low": price - np.abs(np.random.randn(60)) * 1.5,
        "close": price,
        "volume": (np.random.rand(60) * 500_000 + 200_000).astype(int),
    })


@pytest.fixture
def feature_engineer():
    """Return a fresh FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def featured_df(feature_engineer, sample_ohlcv):
    """Return a DataFrame with all features already calculated."""
    return feature_engineer.calculate_features(sample_ohlcv)


@pytest.fixture
def all_symbols():
    """Return flat list of all symbols from stock universe."""
    return get_all_symbols()


# ---------------------------------------------------------------------------
# TestFeatureEngineering
# ---------------------------------------------------------------------------

EXPECTED_FEATURES = [
    "close", "Log_Volume",
    "MA_10", "MA_20", "MA_50", "MA_200",
    "Price_to_MA50", "Price_to_MA200", "MA_50_200_Cross",
    "RSI",
    "MACD", "MACD_Signal", "MACD_Histogram",
    "BB_Width", "BB_Position",
    "Volume_MA_20", "Volume_ROC", "Volume_Ratio",
    "Daily_Return", "Momentum_14", "Momentum_30", "ROC_10",
    "Volatility_14", "Volatility_30", "ATR_14",
    "Vol_Regime", "Dist_52w_High", "Price_Zscore", "Trend_Strength",
    "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
    "Relative_Strength",
]


class TestFeatureEngineering:
    """Tests for the FeatureEngineer class."""

    def test_calculate_features_output_columns(self, feature_engineer, sample_ohlcv):
        """Verify all 34 expected feature columns are present after calculation."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        for col in EXPECTED_FEATURES:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_calculate_features_no_nans(self, featured_df):
        """After bfill/ffill there should be no NaN values in feature columns."""
        for col in EXPECTED_FEATURES:
            nan_count = featured_df[col].isna().sum()
            assert nan_count == 0, f"Column '{col}' has {nan_count} NaN values"

    def test_create_sequences_shapes(self, feature_engineer, sample_ohlcv):
        """X shape must be (n, 30, 34) and y shape must be (n,)."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        X, y = feature_engineer.create_sequences(df, lookback=30, prediction_horizon=21)
        assert X.ndim == 3, "X must be 3-dimensional"
        assert y.ndim == 1, "y must be 1-dimensional"
        assert X.shape[0] == y.shape[0], "X and y sample counts must match"
        assert X.shape[1] == 30, "Lookback dimension must be 30"
        assert X.shape[2] == 34, "Feature dimension must be 34"

    def test_create_sequences_lookback_parameter(self, feature_engineer, sample_ohlcv):
        """Different lookback values should produce correct timestep dimensions."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        for lookback in [10, 20, 50]:
            X, y = feature_engineer.create_sequences(df, lookback=lookback)
            assert X.shape[1] == lookback
            assert X.shape[0] == y.shape[0]

    def test_normalize_features_range(self, feature_engineer, sample_ohlcv):
        """All normalized feature values should lie in [0, 1]."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        df_norm = feature_engineer.normalize_features(df)
        for col in feature_engineer.features:
            assert df_norm[col].min() >= -1e-9, f"{col} below 0"
            assert df_norm[col].max() <= 1.0 + 1e-9, f"{col} above 1"

    def test_rsi_range(self, featured_df):
        """RSI values must be between 0 and 100 (inclusive)."""
        rsi = featured_df["RSI"]
        assert rsi.min() >= 0.0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100.0, f"RSI above 100: {rsi.max()}"

    def test_bollinger_band_ordering(self, feature_engineer, sample_ohlcv):
        """BB_Lower <= BB_Middle <= BB_Upper for rows past warm-up period."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        stable = df.iloc[20:]
        assert (stable["BB_Lower"] <= stable["BB_Middle"]).all(), \
            "BB_Lower exceeds BB_Middle"
        assert (stable["BB_Middle"] <= stable["BB_Upper"]).all(), \
            "BB_Middle exceeds BB_Upper"

    def test_empty_dataframe_handling(self, feature_engineer):
        """Empty DataFrame should not raise an unhandled exception."""
        empty_df = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        try:
            result = feature_engineer.calculate_features(empty_df)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError):
            pass  # Acceptable to raise on empty input

    def test_feature_count_matches_config(self, feature_engineer, sample_ohlcv):
        """The feature list must contain exactly 34 features."""
        feature_engineer.calculate_features(sample_ohlcv)
        assert len(feature_engineer.features) == 34, \
            f"Expected 34 features, got {len(feature_engineer.features)}"


# ---------------------------------------------------------------------------
# TestStockUniverse
# ---------------------------------------------------------------------------

class TestStockUniverse:
    """Tests for the STOCK_UNIVERSE data and helper functions."""

    def test_total_symbol_count(self, all_symbols):
        """The universe should contain exactly 33 symbols."""
        assert len(all_symbols) == 33, f"Expected 33, got {len(all_symbols)}"

    def test_category_counts(self):
        """Each category must have the documented number of symbols."""
        expected = {"tech": 7, "sector_leaders": 9, "defensive": 4, "etfs": 8, "growth": 5}
        for category, count in expected.items():
            actual = len(STOCK_UNIVERSE[category])
            assert actual == count, \
                f"'{category}': expected {count}, got {actual}"

    def test_no_duplicate_symbols(self, all_symbols):
        """No symbol should appear in multiple categories."""
        assert len(all_symbols) == len(set(all_symbols)), \
            "Duplicate symbols found across categories"

    def test_all_symbols_have_metadata(self):
        """Every symbol must have 'name', 'sector', and 'color' metadata."""
        required_keys = {"name", "sector", "color"}
        for category, stocks in STOCK_UNIVERSE.items():
            for symbol, meta in stocks.items():
                missing = required_keys - set(meta.keys())
                assert not missing, \
                    f"{symbol} in '{category}' missing: {missing}"


# ---------------------------------------------------------------------------
# TestPortfolioRebalancer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestPortfolioRebalancer:
    """Tests for PortfolioRebalancer with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _setup_rebalancer(self, tmp_path):
        """Create a PortfolioRebalancer with mocked PredictionService/StockAPI."""
        config_file = str(tmp_path / "rebalancing_config.json")

        with patch("ml.portfolio_rebalancer.PredictionService") as mock_ps, \
             patch("ml.portfolio_rebalancer.StockAPI"):

            np.random.seed(42)
            mock_predictions = {}
            for sym in get_all_symbols():
                mock_predictions[sym] = {
                    "symbol": sym,
                    "predicted_return": np.random.uniform(-0.05, 0.10),
                    "confidence": np.random.uniform(0.4, 0.95),
                }

            # Real function so inspect.signature detects 'symbols' parameter
            def mock_get_all_predictions(symbols=None, days_ahead=21):
                return mock_predictions

            mock_service = MagicMock()
            mock_service.get_all_predictions = mock_get_all_predictions
            mock_ps.return_value = mock_service

            from ml.portfolio_rebalancer import PortfolioRebalancer
            self.rebalancer = PortfolioRebalancer(
                paper_trading=True, config_file=config_file
            )

    def test_allocation_sums_to_one(self):
        """Target allocations must sum to 1.0 (within floating point tolerance)."""
        target = self.rebalancer.get_target_allocation(use_ml=True)
        total = sum(target.values())
        assert abs(total - 1.0) < 1e-6, f"Sum is {total}, expected 1.0"

    def test_risk_controls_max_position(self):
        """No single position should exceed 15% after risk controls."""
        target = self.rebalancer.get_target_allocation(use_ml=True)
        for symbol, weight in target.items():
            assert weight <= 0.15 + 1e-9, \
                f"{symbol} weight {weight:.4f} exceeds 15%"

    def test_risk_controls_min_position(self):
        """After clamping, all positions should meet the minimum threshold."""
        raw = {sym: 0.001 for sym in get_all_symbols()}
        controlled = self.rebalancer._apply_risk_controls(raw)
        # When all inputs are equal and clamped to MIN, renormalization
        # produces equal weights that are >= 1/N
        expected_min = (1.0 / len(raw)) - 1e-6
        for symbol, weight in controlled.items():
            assert weight >= expected_min, \
                f"{symbol} weight {weight:.4f} below minimum"

    def test_min_trade_size(self):
        """No generated order should have an amount below $100."""
        orders = self.rebalancer.calculate_rebalancing_orders(
            portfolio_value=25000.0
        )
        for order in orders:
            assert order["amount_usd"] >= self.rebalancer.MIN_TRADE_SIZE, \
                f"{order['symbol']}: ${order['amount_usd']:.2f} < $100 minimum"

    def test_rebalancing_orders_structure(self):
        """Each order dict must contain the required keys."""
        required_keys = {
            "symbol", "type", "amount_usd", "net_amount_usd",
            "fee_usd", "current_weight", "target_weight",
            "weight_change", "timestamp",
        }
        orders = self.rebalancer.calculate_rebalancing_orders(
            portfolio_value=25000.0
        )
        for order in orders:
            missing = required_keys - set(order.keys())
            assert not missing, f"Order missing keys: {missing}"

    def test_paper_trading_default(self):
        """Paper trading should be enabled by default."""
        assert self.rebalancer.paper_trading is True


# ---------------------------------------------------------------------------
# TestDataValidation
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Tests for data schema and integrity constraints."""

    def test_ohlcv_schema(self, sample_ohlcv):
        """Historical data must contain the required OHLCV columns."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(sample_ohlcv.columns)
        assert not missing, f"OHLCV data missing columns: {missing}"

    def test_prediction_output_schema(self):
        """A prediction dict must contain symbol, predicted_return, confidence."""
        required_keys = {"symbol", "predicted_return", "confidence"}
        prediction = {
            "symbol": "AAPL",
            "predicted_return": 0.05,
            "confidence": 0.82,
            "timestamp": datetime.now().isoformat(),
        }
        missing = required_keys - set(prediction.keys())
        assert not missing, f"Prediction missing keys: {missing}"

    def test_price_positivity(self, sample_ohlcv):
        """All price columns must be positive."""
        for col in ["open", "high", "low", "close"]:
            assert (sample_ohlcv[col] > 0).all(), \
                f"'{col}' contains non-positive values"

    def test_volume_non_negative(self, sample_ohlcv):
        """Volume must be non-negative for every row."""
        assert (sample_ohlcv["volume"] >= 0).all(), \
            "Volume contains negative values"


# ---------------------------------------------------------------------------
# TestLSTMModel (skipped when TensorFlow is unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestLSTMModel:
    """Tests for the StockLSTM model class (requires TensorFlow)."""

    def test_model_instantiation(self):
        """StockLSTM should instantiate without errors."""
        from ml.lstm_model import StockLSTM
        model = StockLSTM(input_shape=(30, 25))
        assert model is not None

    def test_model_output_shape(self):
        """Model prediction output batch dim should match input batch dim."""
        from ml.lstm_model import StockLSTM
        model = StockLSTM(input_shape=(30, 25))
        dummy_input = np.random.randn(5, 30, 25).astype(np.float32)
        output = model.predict(dummy_input)
        assert output.shape[0] == 5, "Batch size mismatch"


# ---------------------------------------------------------------------------
# TestFeatureEngineeringEdgeCases
# ---------------------------------------------------------------------------

class TestFeatureEngineeringEdgeCases:
    """Edge-case tests for FeatureEngineer robustness."""

    def test_constant_price_series(self, feature_engineer):
        """Constant-price series should produce a valid DataFrame.

        RSI and BB_Position are mathematically undefined (0/0) when price
        never changes, so NaN in those columns is acceptable.
        """
        n = 200
        dates = pd.date_range(end=datetime(2026, 3, 20), periods=n, freq="B")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1_000_000),
        })
        result = feature_engineer.calculate_features(df)
        degenerate = {"RSI", "BB_Position"}
        for col in feature_engineer.features:
            if col not in degenerate:
                assert result[col].isna().sum() == 0, \
                    f"NaN in '{col}' for constant price series"

    def test_sequences_with_short_data(self, feature_engineer):
        """With 120 rows, sequences should still be created (fewer samples)."""
        np.random.seed(99)
        dates = pd.date_range(end=datetime(2026, 3, 20), periods=120, freq="B")
        price = np.abs(np.random.randn(120).cumsum() + 200) + 80
        short_df = pd.DataFrame({
            "timestamp": dates,
            "open": price + np.random.randn(120) * 0.3,
            "high": price + np.abs(np.random.randn(120)) * 1.5,
            "low": price - np.abs(np.random.randn(120)) * 1.5,
            "close": price,
            "volume": (np.random.rand(120) * 500_000 + 200_000).astype(int),
        })
        df = feature_engineer.calculate_features(short_df)
        X, y = feature_engineer.create_sequences(
            df, lookback=30, prediction_horizon=21
        )
        assert X.shape[0] > 0, "No sequences created from short data"
        assert X.shape[1] == 30
        assert X.shape[2] == 34

    def test_normalize_idempotent_range(self, feature_engineer, sample_ohlcv):
        """Normalizing twice should still keep values within [0, 1]."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        df_norm = feature_engineer.normalize_features(df)
        df_norm2 = feature_engineer.normalize_features(df_norm)
        for col in feature_engineer.features:
            assert df_norm2[col].min() >= -1e-9
            assert df_norm2[col].max() <= 1.0 + 1e-9

    def test_create_sequences_returns_numpy(self, feature_engineer, sample_ohlcv):
        """create_sequences must return numpy ndarrays."""
        df = feature_engineer.calculate_features(sample_ohlcv)
        X, y = feature_engineer.create_sequences(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_features_attribute_lifecycle(self, feature_engineer, sample_ohlcv):
        """Features list should be empty before and populated after calculation."""
        assert len(feature_engineer.features) == 0
        feature_engineer.calculate_features(sample_ohlcv)
        assert len(feature_engineer.features) == 34


# ---------------------------------------------------------------------------
# TestStockUniverseHelpers
# ---------------------------------------------------------------------------

class TestStockUniverseHelpers:
    """Tests for stock_api helper functions."""

    def test_get_all_symbols_returns_list(self):
        """get_all_symbols should return a list of strings."""
        symbols = get_all_symbols()
        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)

    def test_get_stock_info_known_symbol(self):
        """get_stock_info for AAPL must return correct metadata."""
        info = get_stock_info("AAPL")
        assert info is not None
        assert info["name"] == "Apple"
        assert "sector" in info and "color" in info

    def test_get_stock_info_unknown_symbol(self):
        """get_stock_info for an unknown symbol must return None."""
        assert get_stock_info("ZZZZZ") is None

    def test_color_format(self):
        """All color values should be valid hex color codes (#RRGGBB)."""
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for category, stocks in STOCK_UNIVERSE.items():
            for symbol, meta in stocks.items():
                assert hex_re.match(meta["color"]), \
                    f"{symbol} invalid color: {meta['color']}"
