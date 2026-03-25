"""
Unit tests for ML service modules.

Covers:
- PredictionService (ml/prediction_service.py)
- HybridPredictionService (ml/hybrid_prediction_service.py)
- ExperimentTracker (ml/experiment_tracker.py)
- HistoricalDataFetcher (ml/historical_data_fetcher.py)
- AlpacaTradingClient (ml/alpaca_trading.py)

All external dependencies (TensorFlow, yfinance, GCP) are mocked.
Run with: python -m pytest tests/unit/test_services.py -v
"""

import os
import sys
import json
import tempfile
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock TensorFlow and Keras BEFORE any ml.* imports so that
# lstm_model_gpu.py (which defines classes inheriting from layers.Layer
# at module level) does not crash when TensorFlow is absent.
# ---------------------------------------------------------------------------
_tf_mock = MagicMock()
_keras_mock = MagicMock()
_layers_mock = MagicMock()
# Make layers.Layer a real class so subclassing works at import time
_layers_mock.Layer = type("Layer", (), {})

for mod_name in [
    "tensorflow", "tensorflow.keras", "keras", "keras.layers",
    "keras.callbacks", "tensorflow.keras.layers",
    "tensorflow.keras.callbacks",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Ensure the layers mock has a proper Layer base class
sys.modules["keras"].layers = _layers_mock
sys.modules["tensorflow"].keras = sys.modules["keras"]
sys.modules["keras.layers"] = _layers_mock

# Also mock GCP modules that may be imported at top-level
for mod_name in [
    "google.cloud", "google.cloud.bigquery",
    "google.cloud.aiplatform", "google.cloud.storage",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# ---------------------------------------------------------------------------
# Now safe to import ml modules
# ---------------------------------------------------------------------------
from ml.prediction_service import PredictionService, DEFAULT_SYMBOLS  # noqa: E402
from ml.hybrid_prediction_service import HybridPredictionService  # noqa: E402
from ml.experiment_tracker import ExperimentTracker  # noqa: E402
from ml.historical_data_fetcher import HistoricalDataFetcher  # noqa: E402
from ml.alpaca_trading import AlpacaTradingClient, OrderValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_models_dir(tmp_path):
    """Temporary models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return str(d)


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary results directory for experiment tracker."""
    d = tmp_path / "results"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_dataframe():
    """A minimal OHLCV DataFrame resembling yfinance output."""
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    np.random.seed(42)
    close = 150.0 + np.cumsum(np.random.randn(250) * 0.5)
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": "AAPL",
        "open": close - np.random.uniform(0.1, 1.0, 250),
        "high": close + np.random.uniform(0.1, 1.0, 250),
        "low": close - np.random.uniform(0.1, 1.5, 250),
        "close": close,
        "volume": np.random.randint(1_000_000, 50_000_000, 250).astype(float),
    })


# ===================================================================
# TestPredictionService
# ===================================================================

class TestPredictionService:
    """Tests for ml.prediction_service.PredictionService."""

    def _make_service(self, models_dir):
        """Build a PredictionService with mocked sub-components."""
        with patch.object(PredictionService, "__init__", lambda self_inner, **kw: None):
            svc = PredictionService()
        svc.models_dir = Path(models_dir)
        svc.models_dir.mkdir(exist_ok=True)
        svc.data_fetcher = MagicMock()
        svc.feature_engineer = MagicMock()
        svc.models = {}
        svc.scalers = {}
        svc.symbols = ["AAPL", "MSFT"]
        svc.provider = "local"
        svc.vertex_service = None
        return svc

    def test_get_prediction_no_model(self, tmp_models_dir):
        """get_prediction returns no_model status when no .h5 file exists."""
        svc = self._make_service(tmp_models_dir)
        result = svc.get_prediction("AAPL", days_ahead=21, allow_mock=False)
        assert result["status"] == "no_model"
        assert result["symbol"] == "AAPL"
        assert result["confidence"] == 0.0
        assert result["predicted_return"] == 0.0
        assert "prediction_source" in result

    def test_get_prediction_returns_expected_keys(self, tmp_models_dir):
        """Even a no_model response has the standard dict keys."""
        svc = self._make_service(tmp_models_dir)
        result = svc.get_prediction("MSFT", allow_mock=False)
        expected_keys = {
            "symbol", "current_price", "predicted_price", "predicted_return",
            "confidence", "status", "prediction_source",
        }
        assert expected_keys.issubset(result.keys())

    def test_has_model_false(self, tmp_models_dir):
        """_has_model returns False when no model file is present."""
        svc = self._make_service(tmp_models_dir)
        assert svc._has_model("AAPL") is False

    def test_has_model_true(self, tmp_models_dir):
        """_has_model returns True when an .h5 file exists."""
        svc = self._make_service(tmp_models_dir)
        model_file = Path(tmp_models_dir) / "AAPL_model.h5"
        model_file.write_text("fake")
        assert svc._has_model("AAPL") is True

    def test_get_all_predictions_returns_list(self, tmp_models_dir):
        """get_all_predictions returns a list of dicts."""
        svc = self._make_service(tmp_models_dir)
        svc.symbols = ["AAPL", "MSFT"]
        results = svc.get_all_predictions(allow_mock=False)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r["status"] == "no_model" for r in results)

    def test_train_model_no_tensorflow(self, tmp_models_dir):
        """train_model gracefully returns error when TensorFlow/LSTM is missing."""
        svc = self._make_service(tmp_models_dir)
        with patch("ml.prediction_service.HAS_LSTM", False):
            result = svc.train_model("AAPL")
        assert result["status"] == "error"
        assert "TensorFlow" in result["message"]

    def test_get_prediction_uses_cache_param(self, tmp_models_dir):
        """use_cache parameter is accepted without error."""
        svc = self._make_service(tmp_models_dir)
        r1 = svc.get_prediction("AAPL", use_cache=True, allow_mock=False)
        r2 = svc.get_prediction("AAPL", use_cache=False, allow_mock=False)
        assert r1["status"] == "no_model"
        assert r2["status"] == "no_model"


# ===================================================================
# TestHybridPredictionService
# ===================================================================

class TestHybridPredictionService:
    """Tests for ml.hybrid_prediction_service.HybridPredictionService."""

    def _make_service(self, models_dir):
        """Build a HybridPredictionService with mocked internals."""
        with patch.object(HybridPredictionService, "__init__", lambda self_inner, **kw: None):
            svc = HybridPredictionService()
        svc.models_dir = models_dir
        svc.models_status = {}
        svc.enhanced_service = MagicMock()
        svc.vertex_service = None
        svc.local_ml_models = {}
        return svc

    def test_init(self, tmp_models_dir):
        """Service initialises with mocked dependencies."""
        svc = self._make_service(tmp_models_dir)
        assert svc.vertex_service is None
        assert isinstance(svc.local_ml_models, dict)

    def test_fallback_no_model(self, tmp_models_dir):
        """Without any model source, returns no_model status."""
        svc = self._make_service(tmp_models_dir)
        result = svc.get_prediction("AAPL", allow_mock=False)
        assert result["status"] == "no_model"
        assert result["prediction_source"] == "none"
        assert result["confidence"] == 0.0

    def test_allow_mock_true_returns_mock(self, tmp_models_dir):
        """With allow_mock=True, falls back to enhanced_service mock."""
        svc = self._make_service(tmp_models_dir)
        svc.enhanced_service.get_prediction.return_value = {
            "symbol": "AAPL",
            "status": "enhanced_mock",
            "predicted_return": 0.02,
            "confidence": 0.5,
        }
        result = svc.get_prediction("AAPL", allow_mock=True)
        assert result["prediction_type"] == "mock"
        assert result["prediction_source"] == "technical_analysis"
        assert "warning" in result

    def test_allow_mock_false_no_fallback(self, tmp_models_dir):
        """With allow_mock=False, does NOT call the enhanced_service mock."""
        svc = self._make_service(tmp_models_dir)
        result = svc.get_prediction("AAPL", allow_mock=False)
        assert result["status"] == "no_model"
        svc.enhanced_service.get_prediction.assert_not_called()

    def test_get_prediction_summary(self, tmp_models_dir):
        """get_prediction_summary returns expected keys."""
        svc = self._make_service(tmp_models_dir)
        summary = svc.get_prediction_summary()
        expected = {
            "vertex_ai_available",
            "local_ml_models_available",
            "local_ml_models_count",
            "supported_symbols",
            "mock_predictions_default",
            "system_status",
        }
        assert expected == set(summary.keys())
        assert summary["mock_predictions_default"] is False
        assert summary["system_status"] == "hybrid_operational"

    def test_get_all_predictions(self, tmp_models_dir):
        """get_all_predictions returns a dict keyed by symbol."""
        svc = self._make_service(tmp_models_dir)
        results = svc.get_all_predictions(symbols=["AAPL", "MSFT"], allow_mock=False)
        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "MSFT" in results
        assert all(v["status"] == "no_model" for v in results.values())

    def test_vertex_takes_priority(self, tmp_models_dir):
        """When vertex_service returns success, it takes priority."""
        svc = self._make_service(tmp_models_dir)
        svc.vertex_service = MagicMock()
        svc.vertex_service.get_prediction.return_value = {
            "status": "success",
            "symbol": "AAPL",
            "predicted_return": 0.05,
            "confidence": 0.8,
        }
        result = svc.get_prediction("AAPL", allow_mock=False)
        assert result["prediction_source"] == "vertex_ai_ml"
        assert result["prediction_type"] == "real_ml"


# ===================================================================
# TestExperimentTracker
# ===================================================================

class TestExperimentTracker:
    """Tests for ml.experiment_tracker.ExperimentTracker."""

    def test_init_creates_directory(self, tmp_results_dir):
        sub = os.path.join(tmp_results_dir, "sub", "experiments")
        tracker = ExperimentTracker(results_dir=sub)
        assert Path(sub).exists()
        assert tracker.csv_path == Path(sub) / "experiment_log.csv"

    def test_log_experiment_writes_data(self, tmp_results_dir):
        tracker = ExperimentTracker(results_dir=tmp_results_dir)

        with tracker.start_run(symbol="AAPL", params={"lr": 0.001, "epochs": 50}) as run:
            run.log_metrics({"mse": 0.002, "directional_accuracy": 0.62})

        assert tracker.csv_path.exists()
        runs = tracker.get_runs(symbol="AAPL")
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].metrics["mse"] == 0.002

    def test_get_experiment_history_returns_list(self, tmp_results_dir):
        tracker = ExperimentTracker(results_dir=tmp_results_dir)

        for i in range(3):
            with tracker.start_run(symbol="MSFT", params={"trial": i}) as run:
                run.log_metrics({"directional_accuracy": 0.5 + i * 0.05})

        history = tracker.get_runs(symbol="MSFT")
        assert isinstance(history, list)
        assert len(history) == 3

    def test_summary_statistics(self, tmp_results_dir):
        tracker = ExperimentTracker(results_dir=tmp_results_dir)

        with tracker.start_run(symbol="AAPL", params={"lr": 0.001}) as run:
            run.log_metrics({"directional_accuracy": 0.65, "rmse": 0.01})
        with tracker.start_run(symbol="AAPL", params={"lr": 0.01}) as run:
            run.log_metrics({"directional_accuracy": 0.70, "rmse": 0.009})

        summary = tracker.get_summary()
        assert summary["total_runs"] == 2
        assert summary["completed"] == 2
        assert summary["failed"] == 0
        assert summary["symbols_trained"] == 1
        assert "AAPL" in summary["per_symbol"]
        assert summary["per_symbol"]["AAPL"]["total_runs"] == 2

    def test_failed_run_tracked(self, tmp_results_dir):
        tracker = ExperimentTracker(results_dir=tmp_results_dir)

        with pytest.raises(ValueError):
            with tracker.start_run(symbol="TSLA") as run:
                raise ValueError("simulated error")

        runs = tracker.get_runs(status="failed")
        assert len(runs) == 1
        assert runs[0].symbol == "TSLA"

    def test_json_artifact_created(self, tmp_results_dir):
        tracker = ExperimentTracker(results_dir=tmp_results_dir)

        with tracker.start_run(symbol="NVDA") as run:
            run.log_metrics({"mae": 0.005})

        json_files = list(Path(tmp_results_dir).glob("run_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["symbol"] == "NVDA"


# ===================================================================
# TestHistoricalDataFetcher
# ===================================================================

class TestHistoricalDataFetcher:
    """Tests for ml.historical_data_fetcher.HistoricalDataFetcher."""

    @patch("ml.historical_data_fetcher.StockAPI")
    @patch("ml.historical_data_fetcher.bigquery", None)
    def test_init_without_bigquery(self, mock_api_cls):
        """Gracefully degrades when BigQuery is not available."""
        fetcher = HistoricalDataFetcher(project_id="test-project")
        assert fetcher.bq_client is None
        assert fetcher.project_id == "test-project"

    @patch("ml.historical_data_fetcher.StockAPI")
    @patch("ml.historical_data_fetcher.bigquery", None)
    def test_project_id_defaults_to_env(self, mock_api_cls):
        """project_id falls back to GOOGLE_CLOUD_PROJECT env var."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "env-project"}):
            fetcher = HistoricalDataFetcher()
            assert fetcher.project_id == "env-project"

    @patch("ml.historical_data_fetcher.StockAPI")
    @patch("ml.historical_data_fetcher.bigquery", None)
    def test_fetch_historical_data_with_mock(self, mock_api_cls):
        """fetch_historical_data returns a DataFrame via mocked yfinance."""
        # Use recent dates so the date-filter inside fetch_historical_data keeps rows
        dates = pd.date_range(end=datetime.now(), periods=250, freq="B")
        np.random.seed(42)
        close = 150.0 + np.cumsum(np.random.randn(250) * 0.5)
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "AAPL",
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(1_000_000, 50_000_000, 250).astype(float),
        })
        mock_api = MagicMock()
        mock_api.get_historical_data.return_value = df
        mock_api_cls.return_value = mock_api

        fetcher = HistoricalDataFetcher(project_id="test")
        result = fetcher.fetch_historical_data("AAPL", days=365)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns

    @patch("ml.historical_data_fetcher.StockAPI")
    @patch("ml.historical_data_fetcher.bigquery", None)
    def test_fetch_returns_none_on_empty(self, mock_api_cls):
        """Returns None when API returns empty data."""
        mock_api = MagicMock()
        mock_api.get_historical_data.return_value = pd.DataFrame()
        mock_api_cls.return_value = mock_api

        fetcher = HistoricalDataFetcher(project_id="test")
        result = fetcher.fetch_historical_data("FAKE", days=365)
        assert result is None


# ===================================================================
# TestAlpacaTradingClient
# ===================================================================

class TestAlpacaTradingClient:
    """Tests for ml.alpaca_trading.AlpacaTradingClient."""

    def _make_client(self, paper=True):
        return AlpacaTradingClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=paper,
            live_trading_confirmed=(not paper),
        )

    def test_init_paper_mode(self):
        """Default construction uses paper trading endpoint."""
        client = self._make_client(paper=True)
        assert client.paper is True
        assert "paper" in client.base_url

    def test_live_requires_confirmation(self):
        """Live trading without confirmation raises ValueError."""
        with pytest.raises(ValueError, match="explicit confirmation"):
            AlpacaTradingClient(
                api_key="key", secret_key="secret",
                paper=False, live_trading_confirmed=False,
            )

    def test_missing_credentials(self):
        """Empty credentials raise ValueError."""
        with pytest.raises(ValueError, match="required"):
            AlpacaTradingClient(api_key="", secret_key="")

    def test_validate_order_min_trade_size(self):
        """Orders below MIN_TRADE_SIZE are rejected."""
        client = self._make_client()
        err = client._validate_order("AAPL", "BUY", 50.0, 100_000.0)
        assert err is not None
        assert "minimum" in err.lower()

    def test_validate_order_max_position(self):
        """Orders exceeding 10% of portfolio are rejected."""
        client = self._make_client()
        err = client._validate_order("AAPL", "BUY", 15_000.0, 100_000.0)
        assert err is not None
        assert "10%" in err

    def test_validate_order_passes(self):
        """Valid order passes validation (returns None)."""
        client = self._make_client()
        err = client._validate_order("AAPL", "BUY", 5_000.0, 100_000.0)
        assert err is None

    def test_validate_order_invalid_side(self):
        """Invalid order side is rejected."""
        client = self._make_client()
        err = client._validate_order("AAPL", "HOLD", 1_000.0, 100_000.0)
        assert err is not None
        assert "Invalid" in err

    def test_validate_order_empty_symbol(self):
        """Empty symbol is rejected."""
        client = self._make_client()
        err = client._validate_order("", "BUY", 1_000.0, 100_000.0)
        assert err is not None

    def test_execute_rebalancing_empty(self):
        """Empty order list returns empty results."""
        client = self._make_client()
        results = client.execute_rebalancing_orders([])
        assert results == []

    @patch.object(AlpacaTradingClient, "get_account")
    @patch.object(AlpacaTradingClient, "place_order")
    @patch.object(AlpacaTradingClient, "_get_latest_price")
    def test_execute_rebalancing_orders(self, mock_price, mock_place, mock_acct):
        """Rebalancing orders are submitted via place_order."""
        mock_acct.return_value = {"portfolio_value": 100_000.0}
        mock_price.return_value = 150.0
        mock_place.return_value = {
            "id": "order-1", "status": "accepted", "symbol": "AAPL",
            "qty": "6.666667", "side": "buy",
        }

        client = self._make_client()
        orders = [{"symbol": "AAPL", "type": "BUY", "amount_usd": 1_000.0}]
        results = client.execute_rebalancing_orders(orders)

        assert len(results) == 1
        assert results[0]["status"] == "accepted"
        mock_place.assert_called_once()

    @patch.object(AlpacaTradingClient, "get_account")
    def test_execute_rebalancing_rejects_small_order(self, mock_acct):
        """Orders below min trade size are rejected during rebalancing."""
        mock_acct.return_value = {"portfolio_value": 100_000.0}
        client = self._make_client()
        orders = [{"symbol": "AAPL", "type": "BUY", "amount_usd": 50.0}]
        results = client.execute_rebalancing_orders(orders)
        assert len(results) == 1
        assert results[0]["status"] == "rejected"

    def test_from_config_with_env_vars(self):
        """from_config loads credentials from environment variables."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "env-key",
            "ALPACA_SECRET_KEY": "env-secret",
        }):
            client = AlpacaTradingClient.from_config(paper=True)
        assert client.api_key == "env-key"
        assert client.secret_key == "env-secret"
        assert client.paper is True

    def test_from_config_no_creds_raises(self):
        """from_config raises ValueError when no credentials found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="not found"):
                AlpacaTradingClient.from_config(
                    config_path="/nonexistent/secrets.yaml", paper=True,
                )
