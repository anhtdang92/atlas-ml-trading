"""
Tests for ablation_study.py, hyperparameter_tuning.py, and baseline_models.py.

Covers AblationStudy initialization, WalkForwardSplitter, HyperparameterTuner
scaffolding, and all five baseline models using synthetic data.
TensorFlow-dependent code is mocked where necessary.

Run with: python -m pytest tests/unit/test_model_comparison.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure matplotlib mock is available if not installed
try:
    import matplotlib
except ImportError:
    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.pyplot"] = MagicMock()
    sys.modules["matplotlib.ticker"] = MagicMock()

from ml.hyperparameter_tuning import WalkForwardSplitter, HyperparameterTuner
from ml.baseline_models import (
    BuyAndHoldBaseline,
    MeanReversionBaseline,
    MomentumBaseline,
    LinearRegressionBaseline,
    XGBoostBaseline,
    compare_baselines,
    _evaluate_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Fixed-seed random generator."""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_3d(rng):
    """Synthetic 3D data mimicking LSTM input (samples, timesteps, features)."""
    n_train, n_test = 200, 50
    timesteps, features = 30, 25
    X_train = rng.randn(n_train, timesteps, features).astype(np.float32)
    y_train = rng.randn(n_train).astype(np.float32) * 0.05
    X_test = rng.randn(n_test, timesteps, features).astype(np.float32)
    y_test = rng.randn(n_test).astype(np.float32) * 0.05
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# TestWalkForwardSplitter
# ---------------------------------------------------------------------------

class TestWalkForwardSplitter:
    """Tests for ml.hyperparameter_tuning.WalkForwardSplitter."""

    def test_basic_split(self):
        """Should produce the requested number of folds."""
        splitter = WalkForwardSplitter(n_splits=3, val_ratio=0.15)
        folds = splitter.split(300)
        assert len(folds) == 3

    def test_no_overlap(self):
        """Train and validation indices within each fold must not overlap."""
        splitter = WalkForwardSplitter(n_splits=3, val_ratio=0.15)
        folds = splitter.split(300)
        for train_idx, val_idx in folds:
            assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_temporal_order(self):
        """Validation indices must come after training indices."""
        splitter = WalkForwardSplitter(n_splits=3, val_ratio=0.15)
        folds = splitter.split(300)
        for train_idx, val_idx in folds:
            assert train_idx[-1] < val_idx[0]

    def test_small_dataset_fallback(self):
        """Very small dataset should fall back to a single 80/20 split."""
        splitter = WalkForwardSplitter(n_splits=5, val_ratio=0.15)
        folds = splitter.split(20)
        assert len(folds) >= 1
        train_idx, val_idx = folds[0]
        assert len(train_idx) > 0
        assert len(val_idx) > 0

    def test_single_split(self):
        """n_splits=1 should produce exactly one fold."""
        splitter = WalkForwardSplitter(n_splits=1, val_ratio=0.2)
        folds = splitter.split(100)
        assert len(folds) == 1

    def test_indices_within_bounds(self):
        """All indices must be in range [0, n_samples)."""
        n = 150
        splitter = WalkForwardSplitter(n_splits=3, val_ratio=0.15)
        folds = splitter.split(n)
        for train_idx, val_idx in folds:
            assert train_idx.min() >= 0
            assert val_idx.max() < n


# ---------------------------------------------------------------------------
# TestHyperparameterTuner
# ---------------------------------------------------------------------------

class TestHyperparameterTuner:
    """Tests for ml.hyperparameter_tuning.HyperparameterTuner (no TF needed)."""

    def test_init(self):
        """Tuner should initialize with empty state."""
        tuner = HyperparameterTuner()
        assert tuner._best_params is None
        assert tuner._backend is None

    def test_get_study_summary_empty(self):
        """get_study_summary before tuning should return empty DataFrame."""
        tuner = HyperparameterTuner()
        df = tuner.get_study_summary()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_plot_optimization_history_empty(self):
        """plot_optimization_history before tuning should still return a figure."""
        tuner = HyperparameterTuner()
        fig = tuner.plot_optimization_history()
        assert fig is not None

    def test_tune_requires_tensorflow(self):
        """tune() should raise ImportError when TensorFlow is unavailable."""
        tuner = HyperparameterTuner()
        X = np.zeros((50, 10, 5))
        y = np.zeros(50)
        with patch("ml.hyperparameter_tuning.HAS_TENSORFLOW", False):
            with pytest.raises(ImportError, match="TensorFlow"):
                tuner.tune(X, y, n_trials=1)

    def test_study_summary_with_mock_trials(self):
        """get_study_summary should work with manually injected trial data."""
        tuner = HyperparameterTuner()
        tuner._backend = "random_search"
        tuner._all_trials = [
            {"params": {"lstm_units": 64, "dropout_rate": 0.2}, "score": 0.55},
            {"params": {"lstm_units": 128, "dropout_rate": 0.3}, "score": 0.60},
        ]
        df = tuner.get_study_summary()
        assert len(df) == 2
        assert df.iloc[0]["score"] >= df.iloc[1]["score"]


# ---------------------------------------------------------------------------
# TestAblationStudy
# ---------------------------------------------------------------------------

class TestAblationStudy:
    """Tests for ml.ablation_study.AblationStudy (initialization and helpers)."""

    def test_init(self, synthetic_3d):
        """AblationStudy should initialize and store data dimensions."""
        from ml.ablation_study import AblationStudy
        X_train, y_train, X_test, y_test = synthetic_3d
        X_val = X_test[:25]
        y_val = y_test[:25]
        study = AblationStudy(X_train, y_train, X_val, y_val, X_test, y_test)
        assert study.lookback == 30
        assert study.n_features == 25
        assert len(study._results) == 0

    def test_get_results_df_empty(self, synthetic_3d):
        """get_results_df should return empty DataFrame before any runs."""
        from ml.ablation_study import AblationStudy
        X_train, y_train, X_test, y_test = synthetic_3d
        study = AblationStudy(X_train, y_train, X_test[:25], y_test[:25],
                              X_test, y_test)
        df = study.get_results_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_evaluate_helper(self):
        """Module-level _evaluate should compute correct directional accuracy."""
        from ml.ablation_study import _evaluate
        y_true = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        y_pred = np.array([0.05, -0.1, 0.2, 0.05, 0.01])
        metrics = _evaluate(y_true, y_pred)
        assert "rmse" in metrics
        assert "directional_accuracy" in metrics
        # 4 out of 5 directions correct
        assert metrics["directional_accuracy"] == pytest.approx(0.8)

    def test_run_single_unknown_architecture(self, synthetic_3d):
        """run_single with an unknown name should raise ValueError."""
        from ml.ablation_study import AblationStudy
        X_train, y_train, X_test, y_test = synthetic_3d
        study = AblationStudy(X_train, y_train, X_test[:25], y_test[:25],
                              X_test, y_test)
        with pytest.raises(ValueError, match="Unknown architecture"):
            study.run_single("NonExistentModel")

    def test_print_report_empty(self, synthetic_3d):
        """print_report on empty results should return informative string."""
        from ml.ablation_study import AblationStudy
        X_train, y_train, X_test, y_test = synthetic_3d
        study = AblationStudy(X_train, y_train, X_test[:25], y_test[:25],
                              X_test, y_test)
        report = study.print_report()
        assert report == "No results available."


# ---------------------------------------------------------------------------
# TestBaselineModels
# ---------------------------------------------------------------------------

class TestBaselineModels:
    """Tests for all baseline models in ml.baseline_models."""

    def test_buy_and_hold(self, synthetic_3d):
        """BuyAndHold should predict the training mean for every sample."""
        X_train, y_train, X_test, y_test = synthetic_3d
        model = BuyAndHoldBaseline()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.allclose(preds, np.mean(y_train))
        assert model.name == "Buy & Hold (Mean Return)"

    def test_mean_reversion(self, synthetic_3d):
        """MeanReversion should produce varying predictions based on recent returns."""
        X_train, y_train, X_test, y_test = synthetic_3d
        model = MeanReversionBaseline(reversion_speed=0.5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        # Predictions should not all be identical (unlike buy-and-hold)
        assert not np.allclose(preds, preds[0])

    def test_momentum(self, synthetic_3d):
        """Momentum should produce varying predictions from momentum feature."""
        X_train, y_train, X_test, y_test = synthetic_3d
        model = MomentumBaseline(momentum_weight=0.3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert not np.allclose(preds, preds[0])

    def test_linear_regression(self, synthetic_3d):
        """LinearRegressionBaseline should fit and predict without error."""
        X_train, y_train, X_test, y_test = synthetic_3d
        model = LinearRegressionBaseline(use_ridge=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert model.name == "Ridge Regression"

    def test_linear_regression_plain(self, synthetic_3d):
        """LinearRegressionBaseline with use_ridge=False should work."""
        X_train, y_train, X_test, y_test = synthetic_3d
        model = LinearRegressionBaseline(use_ridge=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert model.name == "Linear Regression"

    @pytest.mark.skipif(
        not XGBoostBaseline.__module__,  # always true; real guard below
        reason="xgboost not installed",
    )
    def test_xgboost(self, synthetic_3d):
        """XGBoostBaseline should fit and predict without error."""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed")
        X_train, y_train, X_test, y_test = synthetic_3d
        model = XGBoostBaseline(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert model.name == "XGBoost"

    def test_evaluate_predictions(self):
        """_evaluate_predictions should compute valid metrics."""
        y_true = np.array([0.05, -0.03, 0.02, -0.01, 0.04])
        y_pred = np.array([0.04, -0.02, 0.01, 0.01, 0.03])
        metrics = _evaluate_predictions(y_true, y_pred)
        assert metrics["rmse"] >= 0
        assert 0.0 <= metrics["directional_accuracy"] <= 1.0
        # 4/5 directions correct
        assert metrics["directional_accuracy"] == pytest.approx(0.8)

    def test_evaluate_predictions_perfect(self):
        """Perfect predictions should yield RMSE=0 and DA=1."""
        y = np.array([0.1, -0.2, 0.3])
        metrics = _evaluate_predictions(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["directional_accuracy"] == pytest.approx(1.0)
        assert metrics["r_squared"] == pytest.approx(1.0)

    def test_compare_baselines(self, synthetic_3d):
        """compare_baselines should return results for all available models."""
        X_train, y_train, X_test, y_test = synthetic_3d
        results = compare_baselines(X_train, y_train, X_test, y_test)
        assert isinstance(results, dict)
        assert "Buy & Hold (Mean Return)" in results
        assert "Mean Reversion" in results
        assert "Momentum" in results
        for name, metrics in results.items():
            assert "directional_accuracy" in metrics
            assert "rmse" in metrics

    def test_compare_baselines_with_lstm(self, synthetic_3d):
        """compare_baselines should include LSTM entry when predictions given."""
        X_train, y_train, X_test, y_test = synthetic_3d
        lstm_preds = y_test + np.random.randn(len(y_test)) * 0.01
        results = compare_baselines(X_train, y_train, X_test, y_test, lstm_preds)
        assert "LSTM" in results

    def test_buy_and_hold_2d_input(self, rng):
        """BuyAndHold should handle 2D input gracefully."""
        X = rng.randn(30, 10)
        y = rng.randn(30) * 0.05
        model = BuyAndHoldBaseline().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 30

    def test_mean_reversion_2d_input(self, rng):
        """MeanReversion should handle 2D input."""
        X = rng.randn(30, 10)
        y = rng.randn(30) * 0.05
        model = MeanReversionBaseline().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 30
