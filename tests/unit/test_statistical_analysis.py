"""
Tests for statistical_tests.py, feature_importance.py, and backtest_tearsheet.py.

Covers StatisticalValidator, FeatureImportanceAnalyzer, and BacktestTearsheet
using synthetic data with fixed seeds. No TensorFlow required.

Run with: python -m pytest tests/unit/test_statistical_analysis.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from ml.statistical_tests import StatisticalValidator
from ml.feature_importance import FeatureImportanceAnalyzer
from ml.backtest_tearsheet import BacktestTearsheet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Fixed-seed random generator for reproducible synthetic data."""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_returns(rng):
    """200 days of synthetic daily returns."""
    return rng.randn(200) * 0.02


@pytest.fixture
def synthetic_predictions(rng):
    """True values and two sets of predictions (good and random)."""
    y_true = rng.randn(100) * 0.05
    good_preds = y_true * 0.6 + rng.randn(100) * 0.02
    random_preds = rng.randn(100) * 0.05
    return y_true, good_preds, random_preds


@pytest.fixture
def feature_data(rng):
    """Synthetic feature data for importance analysis (3D sequences)."""
    n_train, n_test, timesteps, n_features = 200, 50, 30, 10
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_train = rng.randn(n_train, timesteps, n_features).astype(np.float32)
    X_test = rng.randn(n_test, timesteps, n_features).astype(np.float32)
    # Target depends on features 0 and 3
    y_train = 0.5 * X_train[:, -1, 0] + 0.3 * X_train[:, -1, 3] + rng.randn(n_train) * 0.05
    y_test = 0.5 * X_test[:, -1, 0] + 0.3 * X_test[:, -1, 3] + rng.randn(n_test) * 0.05
    return X_train, y_train, X_test, y_test, feature_names


@pytest.fixture
def validator():
    """StatisticalValidator with fixed seed."""
    return StatisticalValidator(random_state=42)


# ---------------------------------------------------------------------------
# TestStatisticalValidator
# ---------------------------------------------------------------------------

class TestStatisticalValidator:
    """Tests for ml.statistical_tests.StatisticalValidator."""

    def test_bootstrap_metric_returns_expected_keys(self, validator, synthetic_returns):
        """bootstrap_metric should return dict with required keys."""
        result = validator.bootstrap_metric(
            synthetic_returns, np.mean, n_bootstrap=500
        )
        expected_keys = {"point_estimate", "ci_lower", "ci_upper", "ci_level",
                         "std", "p_value", "n_bootstrap", "distribution"}
        assert expected_keys.issubset(result.keys())

    def test_bootstrap_metric_ci_contains_estimate(self, validator, synthetic_returns):
        """The point estimate should lie within the confidence interval."""
        result = validator.bootstrap_metric(
            synthetic_returns, np.mean, n_bootstrap=2000
        )
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_bootstrap_sharpe_structure(self, validator, synthetic_returns):
        """bootstrap_sharpe should include metric_name and standard CI fields."""
        result = validator.bootstrap_sharpe(synthetic_returns, n_bootstrap=500)
        assert result["metric_name"] == "Annualized Sharpe Ratio"
        assert "ci_lower" in result and "ci_upper" in result

    def test_bootstrap_sharpe_zero_vol(self, validator):
        """Sharpe should be 0 when returns have zero standard deviation."""
        flat_returns = np.full(50, 0.001)
        result = validator.bootstrap_sharpe(flat_returns, n_bootstrap=200)
        assert result["point_estimate"] == 0.0

    def test_bootstrap_directional_accuracy(self, validator, synthetic_predictions):
        """DA for correlated predictions should be above 0.5."""
        y_true, good_preds, _ = synthetic_predictions
        result = validator.bootstrap_directional_accuracy(
            y_true, good_preds, n_bootstrap=1000
        )
        assert result["point_estimate"] > 0.5
        assert "p_value_vs_random" in result
        assert "significantly_above_random" in result

    def test_da_random_predictions_near_50(self, validator, synthetic_predictions):
        """DA for purely random predictions should be near 50%."""
        y_true, _, random_preds = synthetic_predictions
        result = validator.bootstrap_directional_accuracy(
            y_true, random_preds, n_bootstrap=1000
        )
        assert 0.3 <= result["point_estimate"] <= 0.7

    def test_diebold_mariano_test(self, validator, synthetic_predictions):
        """DM test should return expected keys and valid p-value."""
        y_true, good_preds, random_preds = synthetic_predictions
        errors_good = y_true - good_preds
        errors_random = y_true - random_preds
        result = validator.diebold_mariano_test(errors_good, errors_random)
        assert "dm_statistic" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert isinstance(result["model_1_better"], (bool, np.bool_))

    def test_diebold_mariano_absolute_loss(self, validator, synthetic_predictions):
        """DM test with absolute loss function should work."""
        y_true, good_preds, random_preds = synthetic_predictions
        result = validator.diebold_mariano_test(
            y_true - good_preds, y_true - random_preds, loss_fn="absolute"
        )
        assert "p_value" in result

    def test_diebold_mariano_invalid_loss(self, validator):
        """DM test with unknown loss_fn should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_fn"):
            validator.diebold_mariano_test(
                np.zeros(10), np.zeros(10), loss_fn="cubic"
            )

    def test_compare_models(self, validator, synthetic_predictions):
        """compare_models should return a DataFrame sorted by DA."""
        y_true, good_preds, random_preds = synthetic_predictions
        df = validator.compare_models(
            y_true, {"Good": good_preds, "Random": random_preds}, n_bootstrap=300
        )
        assert isinstance(df, pd.DataFrame)
        assert "model" in df.columns
        assert len(df) == 2
        # Should be sorted descending by DA
        assert df.iloc[0]["da"] >= df.iloc[1]["da"]

    def test_rolling_stability(self, validator, synthetic_predictions):
        """rolling_stability should return DataFrame with expected columns."""
        y_true, good_preds, _ = synthetic_predictions
        df = validator.rolling_stability(y_true, good_preds, window=20)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"index", "rolling_da", "rolling_rmse",
                                    "rolling_mean_return", "rolling_vol"}
        assert len(df) == len(y_true) - 20 + 1

    def test_rolling_stability_small_data(self, validator):
        """rolling_stability should adapt window for small datasets."""
        y_true = np.array([0.01, -0.02, 0.03, 0.01, -0.01,
                           0.02, -0.03, 0.01, 0.04, -0.02])
        y_pred = np.array([0.005, -0.01, 0.02, 0.005, 0.0,
                           0.01, -0.02, 0.005, 0.03, -0.01])
        df = validator.rolling_stability(y_true, y_pred, window=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_identical_predictions_da(self, validator):
        """DA should be 100% when predictions perfectly match true values."""
        y_true = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        result = validator.bootstrap_directional_accuracy(
            y_true, y_true, n_bootstrap=100
        )
        assert result["point_estimate"] == 1.0


# ---------------------------------------------------------------------------
# TestFeatureImportanceAnalyzer
# ---------------------------------------------------------------------------

class TestFeatureImportanceAnalyzer:
    """Tests for ml.feature_importance.FeatureImportanceAnalyzer."""

    def test_correlation_importance(self, feature_data):
        """correlation_importance should rank features correlated with target higher."""
        X_train, y_train, X_test, y_test, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.correlation_importance(X_test, y_test, names)
        assert isinstance(result, pd.DataFrame)
        assert "correlation_score" in result.columns
        assert len(result) == len(names)

    def test_mutual_information_importance(self, feature_data):
        """mutual_information_importance should return non-negative MI scores."""
        X_train, y_train, _, _, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.mutual_information_importance(X_train, y_train, names)
        assert (result["mutual_information"] >= 0).all()

    def test_permutation_importance(self, feature_data):
        """permutation_importance_analysis should return per-feature scores."""
        X_train, y_train, X_test, y_test, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.permutation_importance_analysis(
            X_train, y_train, X_test, y_test, names, n_repeats=3
        )
        assert "importance_mean" in result.columns
        assert len(result) == len(names)

    def test_tree_importance(self, feature_data):
        """tree_importance should return feature importances summing to ~1."""
        X_train, y_train, _, _, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.tree_importance(X_train, y_train, names, n_estimators=20)
        total = result["rf_importance"].sum()
        assert 0.99 <= total <= 1.01

    def test_full_analysis_consensus(self, feature_data):
        """full_analysis should produce a consensus ranking with all features."""
        X_train, y_train, X_test, y_test, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        consensus = analyzer.full_analysis(X_train, y_train, X_test, y_test, names)
        assert "consensus_score" in consensus.columns
        assert "rank" in consensus.columns
        assert len(consensus) == len(names)

    def test_get_top_features(self, feature_data):
        """get_top_features should return a list of feature name strings."""
        X_train, y_train, X_test, y_test, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        analyzer.full_analysis(X_train, y_train, X_test, y_test, names)
        top = analyzer.get_top_features(n=3)
        assert isinstance(top, list)
        assert len(top) == 3
        assert all(isinstance(f, str) for f in top)

    def test_get_top_features_before_analysis_raises(self):
        """get_top_features should raise if full_analysis has not been run."""
        analyzer = FeatureImportanceAnalyzer()
        with pytest.raises(ValueError, match="Run full_analysis"):
            analyzer.get_top_features()

    def test_to_markdown_table(self, feature_data):
        """to_markdown_table should return a valid markdown string."""
        X_train, y_train, X_test, y_test, names = feature_data
        analyzer = FeatureImportanceAnalyzer()
        analyzer.full_analysis(X_train, y_train, X_test, y_test, names)
        md = analyzer.to_markdown_table(top_n=5)
        assert "| Rank |" in md
        assert md.count("\n") >= 6  # header + separator + 5 rows


# ---------------------------------------------------------------------------
# TestBacktestTearsheet
# ---------------------------------------------------------------------------

class TestBacktestTearsheet:
    """Tests for ml.backtest_tearsheet.BacktestTearsheet."""

    @pytest.fixture
    def tearsheet(self, rng):
        """Create a tearsheet from synthetic portfolio data."""
        n_days = 252
        dates = pd.bdate_range(end="2026-03-20", periods=n_days)
        daily_returns = rng.normal(0.0004, 0.012, n_days)
        portfolio_values = 10000 * np.cumprod(1 + daily_returns)
        bench_returns = rng.normal(0.0003, 0.011, n_days)
        benchmark_values = 10000 * np.cumprod(1 + bench_returns)
        return BacktestTearsheet(
            portfolio_values, dates, benchmark_values, name="Test Strategy"
        )

    def test_compute_metrics_keys(self, tearsheet):
        """compute_metrics should return all expected metric keys."""
        metrics = tearsheet.compute_metrics()
        required = {"total_return", "cagr", "annualized_volatility",
                     "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                     "max_drawdown", "win_rate", "trading_days"}
        assert required.issubset(metrics.keys())

    def test_benchmark_metrics_present(self, tearsheet):
        """When benchmark is provided, benchmark-specific metrics should exist."""
        metrics = tearsheet.compute_metrics()
        assert "beta" in metrics
        assert "alpha" in metrics
        assert "information_ratio" in metrics

    def test_no_benchmark(self, rng):
        """Tearsheet without benchmark should still compute core metrics."""
        n_days = 100
        dates = pd.bdate_range(end="2026-01-15", periods=n_days)
        values = 10000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days))
        ts = BacktestTearsheet(values, dates, benchmark_values=None)
        metrics = ts.compute_metrics()
        assert "sharpe_ratio" in metrics
        assert "beta" not in metrics

    def test_total_return_sign(self, rng):
        """A monotonically increasing portfolio should have positive total return."""
        dates = pd.bdate_range(start="2025-11-01", end="2026-02-01", freq="B")
        n_days = len(dates)
        values = np.linspace(10000, 11000, n_days)
        ts = BacktestTearsheet(values, dates)
        metrics = ts.compute_metrics()
        assert metrics["total_return"] > 0

    def test_max_drawdown_negative(self, tearsheet):
        """Max drawdown should be non-positive."""
        metrics = tearsheet.compute_metrics()
        assert metrics["max_drawdown"] <= 0

    def test_to_markdown(self, tearsheet):
        """to_markdown should produce a valid markdown string."""
        md = tearsheet.to_markdown()
        assert "| Metric | Value |" in md
        assert "Sharpe Ratio" in md
