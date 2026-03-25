"""
Unit tests for the ATLAS Stock ML Intelligence API (ml/api.py).

Tests all FastAPI endpoints using TestClient with mocked external dependencies
(yfinance, TensorFlow/PredictionService) so tests run offline and fast.

Usage:
    python -m pytest tests/unit/test_api.py -v
    python tests/unit/test_api.py
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mock heavy optional deps before importing the API module
# ---------------------------------------------------------------------------
# TensorFlow has a deep module tree; mock every sub-module that the codebase
# touches so that import chains succeed without the real package.
_mock_tf = MagicMock()
# layers.Layer and callbacks.Callback must be real types for class inheritance
_mock_tf.keras.layers.Layer = type("Layer", (), {})
_mock_tf.keras.callbacks.Callback = type("Callback", (), {})
for _submod in [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.layers",
    "tensorflow.keras.models",
    "tensorflow.keras.callbacks",
    "tensorflow.keras.optimizers",
    "tensorflow.keras.backend",
    "tensorflow.keras.regularizers",
    "tensorflow.keras.losses",
    "tensorflow.keras.metrics",
    "tensorflow.keras.utils",
    "tensorflow.keras.mixed_precision",
    "tensorflow.config",
    "tensorflow.data",
    "tensorflow.function",
    "tensorflow.distribute",
    "tensorflow.summary",
]:
    sys.modules[_submod] = _mock_tf

# lstm_model_gpu.py does `from keras import layers, callbacks` (standalone keras)
_mock_keras = MagicMock()
_mock_keras.layers.Layer = type("Layer", (), {})
_mock_keras.callbacks.Callback = type("Callback", (), {})
for _kmod in ["keras", "keras.layers", "keras.callbacks", "keras.models",
               "keras.optimizers", "keras.backend", "keras.regularizers"]:
    sys.modules[_kmod] = _mock_keras

# Also mock yfinance and optional GCP modules that may be imported transitively
sys.modules.setdefault("yfinance", MagicMock())
sys.modules.setdefault("google.cloud.bigquery", MagicMock())
sys.modules.setdefault("google.cloud.storage", MagicMock())
sys.modules.setdefault("google.cloud.aiplatform", MagicMock())

from fastapi.testclient import TestClient

# Patch PredictionService so the app can start without TensorFlow models
_mock_prediction_result = {
    "predicted_return": 0.032,
    "confidence": 0.74,
    "model_type": "lstm",
    "status": "ok",
    "reason": None,
}


def _make_mock_prediction_service():
    svc = MagicMock()
    svc.get_prediction.return_value = _mock_prediction_result.copy()
    return svc


# Import the app and wire up the mock prediction service
import ml.api as api_module
from ml.api import app

client = TestClient(app, raise_server_exceptions=False)

# Inject a mock prediction service so /predict does not 503
api_module.prediction_service = _make_mock_prediction_service()
api_module.feature_engineer = MagicMock()


# ── Helpers ───────────────────────────────────────────────────────────────

def _fake_yfinance_history(*_args, **_kwargs):
    """Return a realistic DataFrame mimicking yf.Ticker(...).history()."""
    dates = pd.bdate_range(end="2026-03-20", periods=252)
    np.random.seed(42)
    close = np.random.randn(252).cumsum() + 180.0
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(252) * 0.3,
            "High": close + np.abs(np.random.randn(252)) * 1.5,
            "Low": close - np.abs(np.random.randn(252)) * 1.5,
            "Close": close,
            "Volume": (np.random.rand(252) * 1e6 + 5e5).astype(int),
        },
        index=dates,
    )


# ── Health endpoint ───────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_status_is_healthy(self):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_uptime_is_positive(self):
        data = client.get("/api/v1/health").json()
        assert data["uptime_seconds"] >= 0


# ── Models endpoint ───────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_models_returns_200(self):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200

    def test_models_response_has_list(self):
        data = client.get("/api/v1/models").json()
        assert isinstance(data["models"], list)

    def test_models_has_total_count(self):
        data = client.get("/api/v1/models").json()
        assert "total_count" in data
        assert isinstance(data["total_count"], int)


# ── Predict endpoint ─────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_valid_symbols_returns_200(self):
        resp = client.post("/api/v1/predict", json={"symbols": ["AAPL", "MSFT"]})
        assert resp.status_code == 200

    def test_predict_response_has_predictions(self):
        data = client.post(
            "/api/v1/predict", json={"symbols": ["AAPL"]}
        ).json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["symbol"] == "AAPL"

    def test_predict_empty_symbols_returns_422(self):
        # Pydantic validates min_length=1 on symbols, returning 422
        resp = client.post("/api/v1/predict", json={"symbols": []})
        assert resp.status_code == 422

    def test_predict_missing_body_returns_422(self):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_predict_invalid_symbol_returns_400(self):
        # Symbol with digits triggers _validate_symbol HTTPException 400
        resp = client.post("/api/v1/predict", json={"symbols": ["123BAD"]})
        assert resp.status_code == 400

    def test_predict_includes_horizon(self):
        data = client.post(
            "/api/v1/predict", json={"symbols": ["GOOGL"], "horizon_days": 10}
        ).json()
        assert data["predictions"][0]["horizon_days"] == 10

    def test_predict_503_when_no_service(self):
        original = api_module.prediction_service
        try:
            api_module.prediction_service = None
            resp = client.post("/api/v1/predict", json={"symbols": ["AAPL"]})
            assert resp.status_code == 503
        finally:
            api_module.prediction_service = original


# ── Features endpoint ────────────────────────────────────────────────────

class TestFeaturesEndpoint:
    @patch("ml.api.yf", create=True)
    def test_features_returns_200(self, mock_yf):
        """Mock yfinance so /features/{symbol} returns computed features."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_yfinance_history()

        # Patch the yfinance import inside the endpoint
        fake_features_df = _fake_yfinance_history()
        fake_features_df.columns = [c.lower() for c in fake_features_df.columns]
        latest_row = fake_features_df.iloc[-1]
        feature_dict = {k: round(float(v), 6) for k, v in latest_row.items()}

        api_module.feature_engineer.calculate_features.return_value = fake_features_df

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            with patch("ml.api.yf", create=True) as inner_yf:
                inner_yf.Ticker.return_value = mock_ticker
                # The endpoint imports yfinance locally, so patch builtins import
                import builtins
                original_import = builtins.__import__

                def patched_import(name, *args, **kwargs):
                    if name == "yfinance":
                        mod = MagicMock()
                        mod.Ticker.return_value = mock_ticker
                        return mod
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", side_effect=patched_import):
                    resp = client.get("/api/v1/features/AAPL")

        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert "features" in data
        assert "feature_count" in data


# ── Backtest endpoint ────────────────────────────────────────────────────

class TestBacktestEndpoint:
    def test_backtest_returns_200(self):
        """Mock yfinance to test backtest endpoint."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_yfinance_history()

        import builtins
        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "yfinance":
                mod = MagicMock()
                mod.Ticker.return_value = mock_ticker
                return mod
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=patched_import):
            resp = client.get("/api/v1/backtest/AAPL")

        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert "metrics" in data
        assert "total_return_pct" in data["metrics"]
        assert "sharpe_ratio" in data["metrics"]

    def test_backtest_invalid_symbol(self):
        resp = client.get("/api/v1/backtest/123XYZ")
        assert resp.status_code == 400


# ── CORS headers ─────────────────────────────────────────────────────────

class TestCORSHeaders:
    def test_cors_allow_origin_present(self):
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert "access-control-allow-origin" in resp.headers


# ── Rate limit headers ───────────────────────────────────────────────────

class TestRateLimitHeaders:
    def test_rate_limit_headers_present(self):
        resp = client.get("/api/v1/health")
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-reset" in resp.headers

    def test_rate_limit_values_are_numeric(self):
        resp = client.get("/api/v1/health")
        assert resp.headers["x-ratelimit-limit"].isdigit()
        assert resp.headers["x-ratelimit-remaining"].isdigit()
        assert resp.headers["x-ratelimit-reset"].isdigit()


# ── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
