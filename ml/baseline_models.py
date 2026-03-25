"""
Baseline Models for Stock Return Prediction

Provides simple baseline models to compare against the LSTM:
1. BuyAndHoldBaseline - Always predicts positive market return
2. MeanReversionBaseline - Predicts return toward historical mean
3. MomentumBaseline - Predicts continuation of recent trend
4. LinearRegressionBaseline - Sklearn linear regression on features
5. XGBoostBaseline - Gradient boosted trees (if xgboost installed)

These baselines establish a performance floor. Any ML model that
cannot beat these baselines is not adding value.

Usage:
    from ml.baseline_models import compare_baselines
    results = compare_baselines(X_train, y_train, X_test, y_test)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression + financial metrics."""
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))

    pred_direction = y_pred > 0
    true_direction = y_true > 0
    directional_accuracy = float(np.mean(pred_direction == true_direction))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "r_squared": r_squared,
    }


class BuyAndHoldBaseline:
    """Predicts the historical average return (always bullish).

    This is the simplest possible baseline: assume the market
    continues its long-run average return.
    """

    def __init__(self) -> None:
        self.mean_return: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BuyAndHoldBaseline":
        self.mean_return = float(np.mean(y_train))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_return)

    @property
    def name(self) -> str:
        return "Buy & Hold (Mean Return)"


class MeanReversionBaseline:
    """Predicts returns will revert toward the long-run mean.

    If recent returns are above average, predicts lower future returns
    and vice versa. Uses the last value in each sequence as the
    recent return proxy.
    """

    # Feature names matching FeatureEngineer output order
    DAILY_RETURN_FEATURE = "Daily_Return"
    MOMENTUM_FEATURE = "Momentum_14"

    def __init__(self, reversion_speed: float = 0.5) -> None:
        self.mean_return: float = 0.0
        self.reversion_speed = reversion_speed
        self._daily_return_idx: int = 18  # Updated by fit() if feature names provided

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "MeanReversionBaseline":
        self.mean_return = float(np.mean(y_train))
        if feature_names and self.DAILY_RETURN_FEATURE in feature_names:
            self._daily_return_idx = feature_names.index(self.DAILY_RETURN_FEATURE)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            idx = min(self._daily_return_idx, X.shape[2] - 1)
            recent_returns = X[:, -1, idx]
        else:
            recent_returns = X[:, -1] if X.ndim == 2 else np.zeros(len(X))

        predictions = self.mean_return + self.reversion_speed * (self.mean_return - recent_returns)
        return predictions

    @property
    def name(self) -> str:
        return "Mean Reversion"


class MomentumBaseline:
    """Predicts continuation of recent price trend.

    Uses recent momentum to predict future returns.
    """

    def __init__(self, momentum_weight: float = 0.3) -> None:
        self.momentum_weight = momentum_weight
        self.mean_return: float = 0.0
        self._momentum_idx: int = 19  # Updated by fit() if feature names provided

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "MomentumBaseline":
        self.mean_return = float(np.mean(y_train))
        if feature_names and MeanReversionBaseline.MOMENTUM_FEATURE in feature_names:
            self._momentum_idx = feature_names.index(MeanReversionBaseline.MOMENTUM_FEATURE)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            idx = min(self._momentum_idx, X.shape[2] - 1)
            momentum = X[:, -1, idx]
        else:
            momentum = X[:, -1] if X.ndim == 2 else np.zeros(len(X))

        predictions = self.mean_return + self.momentum_weight * momentum
        return predictions

    @property
    def name(self) -> str:
        return "Momentum"


class LinearRegressionBaseline:
    """Linear regression on flattened feature sequences.

    Flattens the (timesteps, features) input and fits a simple
    linear regression. This tests whether the LSTM's ability to
    capture temporal patterns adds value over a linear model.
    """

    def __init__(self, use_ridge: bool = True, alpha: float = 1.0) -> None:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        self.model = Ridge(alpha=alpha) if use_ridge else LinearRegression()
        self.use_ridge = use_ridge
        self.alpha = alpha

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearRegressionBaseline":
        X_flat = X_train.reshape(len(X_train), -1) if X_train.ndim == 3 else X_train
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1) if X.ndim == 3 else X
        return self.model.predict(X_flat)

    @property
    def name(self) -> str:
        return f"{'Ridge' if self.use_ridge else 'Linear'} Regression"


class XGBoostBaseline:
    """XGBoost gradient boosted trees baseline.

    Tree-based models are strong baselines for tabular data.
    If XGBoost beats the LSTM, it suggests the temporal patterns
    captured by the LSTM are not adding value.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ) -> None:
        if not HAS_XGBOOST:
            raise ImportError("xgboost required: pip install xgboost")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        X_flat = X_train.reshape(len(X_train), -1) if X_train.ndim == 3 else X_train
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1) if X.ndim == 3 else X
        return self.model.predict(X_flat)

    @property
    def name(self) -> str:
        return "XGBoost"


def compare_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lstm_predictions: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare all available baselines against test data.

    Args:
        X_train: Training input sequences (samples, timesteps, features).
        y_train: Training target returns.
        X_test: Test input sequences.
        y_test: Test target returns.
        lstm_predictions: Optional LSTM predictions to include in comparison.

    Returns:
        Dictionary mapping model name to evaluation metrics.
    """
    results: Dict[str, Dict[str, float]] = {}

    baselines = [
        BuyAndHoldBaseline(),
        MeanReversionBaseline(),
        MomentumBaseline(),
    ]

    if HAS_SKLEARN:
        baselines.append(LinearRegressionBaseline())

    if HAS_XGBOOST:
        baselines.append(XGBoostBaseline())

    for model in baselines:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = _evaluate_predictions(y_test, preds)
            results[model.name] = metrics
            logger.info(f"{model.name}: DA={metrics['directional_accuracy']:.2%}, RMSE={metrics['rmse']:.6f}")
        except Exception as e:
            logger.warning(f"Baseline {model.name} failed: {e}")

    if lstm_predictions is not None:
        metrics = _evaluate_predictions(y_test, lstm_predictions)
        results["LSTM"] = metrics

    return results


def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON - Baseline vs LSTM")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'Dir. Acc.':>10} {'R²':>10}")
    print(f"{'-'*65}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["directional_accuracy"], reverse=True)

    for name, metrics in sorted_results:
        marker = " <-- best" if name == sorted_results[0][0] else ""
        print(
            f"{name:<25} "
            f"{metrics['rmse']:>10.6f} "
            f"{metrics['mae']:>10.6f} "
            f"{metrics['directional_accuracy']:>9.2%} "
            f"{metrics['r_squared']:>10.4f}"
            f"{marker}"
        )

    print(f"{'='*80}")

    # Check if LSTM beats baselines
    if "LSTM" in results:
        lstm_da = results["LSTM"]["directional_accuracy"]
        baseline_das = [v["directional_accuracy"] for k, v in results.items() if k != "LSTM"]
        best_baseline_da = max(baseline_das) if baseline_das else 0
        if lstm_da > best_baseline_da:
            improvement = (lstm_da - best_baseline_da) / best_baseline_da * 100
            print(f"\nLSTM beats best baseline by {improvement:.1f}% in directional accuracy")
        else:
            print(f"\nWarning: LSTM does NOT beat the best baseline in directional accuracy")


def main() -> None:
    """Demo comparison with synthetic data."""
    print("Baseline Model Comparison Demo")
    print("=" * 40)

    np.random.seed(42)
    n_train, n_test = 400, 100
    timesteps, features = 30, 25

    X_train = np.random.randn(n_train, timesteps, features)
    y_train = np.random.randn(n_train) * 0.05
    X_test = np.random.randn(n_test, timesteps, features)
    y_test = np.random.randn(n_test) * 0.05

    # Simulate LSTM predictions (slightly better than random)
    lstm_preds = y_test + np.random.randn(n_test) * 0.03

    results = compare_baselines(X_train, y_train, X_test, y_test, lstm_preds)
    print_comparison(results)


if __name__ == "__main__":
    main()
