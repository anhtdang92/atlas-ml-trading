"""
End-to-end integration test for the ATLAS ML pipeline.

Runs the full pipeline on synthetic OHLCV data without requiring TensorFlow:
  1. Generate synthetic price data
  2. Feature engineering (calculate_features + create_sequences)
  3. Baseline models (BuyAndHold, MeanReversion, Momentum, LinearRegression, XGBoost)
  4. compare_baselines()
  5. BacktestTearsheet metrics
  6. StatisticalValidator.bootstrap_sharpe()
  7. FeatureImportanceAnalyzer (correlation + mutual_information)

Usage:
    python -m pytest tests/integration/test_e2e_pipeline.py -v
    python tests/integration/test_e2e_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_engineering import FeatureEngineer
from ml.baseline_models import (
    BuyAndHoldBaseline,
    MeanReversionBaseline,
    MomentumBaseline,
    LinearRegressionBaseline,
    XGBoostBaseline,
    compare_baselines,
)
from ml.backtest_tearsheet import BacktestTearsheet
from ml.statistical_tests import StatisticalValidator
from ml.feature_importance import FeatureImportanceAnalyzer

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_ohlcv():
    """Generate 300 days of realistic synthetic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(end="2026-03-20", periods=n)

    # Random walk with slight upward drift
    log_returns = np.random.normal(0.0003, 0.015, n)
    close = 150.0 * np.exp(np.cumsum(log_returns))
    spread = np.abs(np.random.randn(n)) * 1.5

    return pd.DataFrame({
        "timestamp": dates,
        "open": close + np.random.randn(n) * 0.5,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": (np.random.rand(n) * 1_000_000 + 500_000).astype(int),
    })


@pytest.fixture(scope="module")
def feature_engineer():
    return FeatureEngineer()


@pytest.fixture(scope="module")
def features_df(synthetic_ohlcv, feature_engineer):
    """Run calculate_features on synthetic data (cached per module)."""
    return feature_engineer.calculate_features(synthetic_ohlcv)


@pytest.fixture(scope="module")
def sequences(features_df, feature_engineer):
    """Create LSTM sequences from feature DataFrame."""
    X, y = feature_engineer.create_sequences(features_df, lookback=30, prediction_horizon=21)
    return X, y


# ---------------------------------------------------------------------------
# Step 1 & 2: Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_features_dataframe_not_empty(self, features_df):
        assert len(features_df) > 0

    def test_warmup_rows_dropped(self, synthetic_ohlcv, features_df):
        assert len(features_df) < len(synthetic_ohlcv)

    def test_expected_feature_columns_present(self, features_df, feature_engineer):
        for feat in feature_engineer.features:
            assert feat in features_df.columns, f"Missing feature: {feat}"

    def test_no_nan_in_features(self, features_df, feature_engineer):
        subset = features_df[feature_engineer.features]
        assert subset.isna().sum().sum() == 0

    def test_sequences_shape(self, sequences, feature_engineer):
        X, y = sequences
        assert X.ndim == 3, "X should be (samples, timesteps, features)"
        assert X.shape[1] == 30, "lookback should be 30"
        assert X.shape[2] == len(feature_engineer.features)
        assert y.ndim == 1
        assert len(X) == len(y)
        assert len(X) > 0


# ---------------------------------------------------------------------------
# Step 3 & 4: Baseline models
# ---------------------------------------------------------------------------

class TestBaselineModels:
    def test_buy_and_hold(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        model = BuyAndHoldBaseline().fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape
        # All predictions should be the training mean
        assert np.allclose(preds, preds[0])

    def test_mean_reversion(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        model = MeanReversionBaseline().fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape

    def test_momentum(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        model = MomentumBaseline().fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape

    def test_linear_regression(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        model = LinearRegressionBaseline().fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape
        # Should not be constant (unlike buy-and-hold)
        assert np.std(preds) > 0

    def test_xgboost(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        model = XGBoostBaseline().fit(X[:split], y[:split])
        preds = model.predict(X[split:])
        assert preds.shape == y[split:].shape
        assert np.std(preds) > 0

    def test_compare_baselines(self, sequences):
        X, y = sequences
        split = int(len(X) * 0.8)
        results = compare_baselines(X[:split], y[:split], X[split:], y[split:])
        assert isinstance(results, dict)
        assert len(results) >= 3  # At least 3 baselines
        for name, metrics in results.items():
            assert "mse" in metrics
            assert "rmse" in metrics
            assert "directional_accuracy" in metrics
            assert 0.0 <= metrics["directional_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Step 5: Backtest tearsheet
# ---------------------------------------------------------------------------

class TestBacktestTearsheet:
    def test_compute_metrics(self):
        np.random.seed(99)
        n = 252
        dates = pd.bdate_range(end="2026-03-20", periods=n)
        daily_returns = np.random.normal(0.0004, 0.012, n)
        portfolio_values = 10_000 * np.cumprod(1 + daily_returns)

        ts = BacktestTearsheet(portfolio_values, dates)
        metrics = ts.compute_metrics()

        assert isinstance(metrics, dict)
        for key in [
            "total_return", "cagr", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "win_rate", "trading_days",
        ]:
            assert key in metrics, f"Missing metric: {key}"

        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["trading_days"], int)
        assert metrics["trading_days"] == n - 1  # pct_change drops first
        assert -1.0 <= metrics["max_drawdown"] <= 0.0


# ---------------------------------------------------------------------------
# Step 6: Statistical validator
# ---------------------------------------------------------------------------

class TestStatisticalValidator:
    def test_bootstrap_sharpe(self):
        np.random.seed(123)
        returns = np.random.normal(0.0004, 0.012, 252)

        validator = StatisticalValidator(random_state=42)
        result = validator.bootstrap_sharpe(returns, n_bootstrap=2000)

        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
        assert isinstance(result["distribution"], np.ndarray)
        assert len(result["distribution"]) == 2000


# ---------------------------------------------------------------------------
# Step 7: Feature importance analyzer
# ---------------------------------------------------------------------------

class TestFeatureImportanceAnalyzer:
    def test_correlation_importance(self, sequences, feature_engineer):
        X, y = sequences
        analyzer = FeatureImportanceAnalyzer(random_state=42)
        result = analyzer.correlation_importance(X, y, feature_engineer.features)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "correlation_score" in result.columns
        assert len(result) == len(feature_engineer.features)
        assert (result["correlation_score"] >= 0).all()

    def test_mutual_information_importance(self, sequences, feature_engineer):
        X, y = sequences
        analyzer = FeatureImportanceAnalyzer(random_state=42)
        result = analyzer.mutual_information_importance(X, y, feature_engineer.features)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "mutual_information" in result.columns
        assert len(result) == len(feature_engineer.features)
        assert (result["mutual_information"] >= 0).all()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
