"""
ATLAS Stock ML Intelligence API

Production REST API for the ATLAS ML Trading Strategy system.
Serves LSTM-based stock price predictions, feature engineering data,
and backtest metrics via FastAPI.

Endpoints:
    GET  /api/v1/health            - Service health check
    GET  /api/v1/models            - List available trained models
    POST /api/v1/predict           - Generate ML predictions for symbols
    GET  /api/v1/features/{symbol} - Computed technical indicators
    GET  /api/v1/backtest/{symbol} - Historical backtest metrics

Author: Anh Dang
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atlas.api")

# ---------------------------------------------------------------------------
# Graceful imports from the ATLAS codebase
# ---------------------------------------------------------------------------
try:
    from data.stock_api import STOCK_UNIVERSE, get_all_symbols, get_stock_info
    HAS_STOCK_API = True
except ImportError:
    logger.warning("data.stock_api not available — symbol lookups disabled")
    HAS_STOCK_API = False
    STOCK_UNIVERSE = {}

    def get_all_symbols() -> List[str]:
        return []

    def get_stock_info(symbol: str) -> Optional[Dict]:
        return None

try:
    from ml.feature_engineering import FeatureEngineer
    HAS_FEATURES = True
except ImportError:
    logger.warning("Feature engineering module not available")
    HAS_FEATURES = False

try:
    from ml.prediction_service import PredictionService
    HAS_PREDICTION = True
except ImportError:
    logger.warning("Prediction service not available (TensorFlow may be missing)")
    HAS_PREDICTION = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_VERSION = "1.0.0"
APP_TITLE = "ATLAS Stock ML Intelligence API"
MODELS_DIR = Path("models")
RATE_LIMIT_PER_MINUTE = 60
STARTUP_TIME = time.time()

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Stock ticker symbols to predict",
        json_schema_extra={"example": ["AAPL", "MSFT", "GOOGL"]},
    )
    horizon_days: int = Field(
        default=21,
        ge=1,
        le=90,
        description="Prediction horizon in trading days",
    )


class SymbolPrediction(BaseModel):
    symbol: str
    predicted_return: Optional[float] = None
    confidence: Optional[float] = None
    horizon_days: int
    model_type: str = "lstm"
    timestamp: str
    status: str = "ok"
    message: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[SymbolPrediction]
    model_version: str = API_VERSION
    generated_at: str


class ModelInfo(BaseModel):
    symbol: str
    filename: str
    size_bytes: int
    last_modified: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    models_dir: str
    total_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    prediction_service: bool
    feature_engine: bool
    uptime_seconds: float
    timestamp: str
    symbols_tracked: int


class FeatureResponse(BaseModel):
    symbol: str
    feature_count: int
    record_count: int
    features: Dict[str, Any]
    computed_at: str


class BacktestResponse(BaseModel):
    symbol: str
    period: str
    metrics: Dict[str, Any]
    computed_at: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=(
        "AI-powered stock trading predictions using LSTM neural networks. "
        "Part of the ATLAS Stock ML Intelligence System. "
        "Provides real-time ML predictions, technical feature computation, "
        "and historical backtest analytics for ~33 tracked equities."
    ),
    version=API_VERSION,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness"},
        {"name": "models", "description": "Trained model inventory"},
        {"name": "predictions", "description": "ML price predictions"},
        {"name": "features", "description": "Technical indicator computation"},
        {"name": "backtest", "description": "Historical backtest analytics"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state (initialized on startup)
# ---------------------------------------------------------------------------
prediction_service: Optional[Any] = None
feature_engineer: Optional[Any] = None


# ---------------------------------------------------------------------------
# Middleware — rate-limit headers
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """Attach informational rate-limit headers to every response."""
    response: Response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT_PER_MINUTE - 1)
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
    return response


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize ML services and log configuration on startup."""
    global prediction_service, feature_engineer

    logger.info("=" * 60)
    logger.info(f"  {APP_TITLE} v{API_VERSION}")
    logger.info("=" * 60)
    logger.info(f"Models directory : {MODELS_DIR.resolve()}")
    logger.info(f"Stock API loaded : {HAS_STOCK_API}")
    logger.info(f"Feature engine   : {HAS_FEATURES}")
    logger.info(f"Prediction svc   : {HAS_PREDICTION}")
    logger.info(f"Symbols tracked  : {len(get_all_symbols())}")

    if HAS_FEATURES:
        feature_engineer = FeatureEngineer()
        logger.info("FeatureEngineer initialized")

    if HAS_PREDICTION:
        try:
            prediction_service = PredictionService(
                models_dir=str(MODELS_DIR), provider="local"
            )
            logger.info("PredictionService initialized (local provider)")
        except Exception as exc:
            logger.warning(f"PredictionService init failed: {exc}")

    logger.info("Startup complete — ready to serve requests")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_symbol(symbol: str) -> str:
    """Normalize and validate a ticker symbol."""
    symbol = symbol.upper().strip()
    if not symbol.isalpha() or len(symbol) > 5:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    return symbol


def _list_model_files() -> List[Path]:
    """Return all model artifact files (.h5, .keras, .pkl) in MODELS_DIR."""
    if not MODELS_DIR.exists():
        return []
    patterns = ["*.h5", "*.keras", "*.pkl", "*.joblib"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(MODELS_DIR.glob(pat))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Service health check",
)
async def health_check():
    """Return service health, dependency status, and uptime."""
    model_files = _list_model_files()
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        model_loaded=len(model_files) > 0,
        prediction_service=prediction_service is not None,
        feature_engine=feature_engineer is not None,
        uptime_seconds=round(time.time() - STARTUP_TIME, 2),
        timestamp=_now_iso(),
        symbols_tracked=len(get_all_symbols()),
    )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    tags=["models"],
    summary="List available trained models",
)
async def list_models():
    """List all trained model artifacts with metadata."""
    model_files = _list_model_files()
    models = []
    for fp in model_files:
        stat = fp.stat()
        # Extract symbol from filename convention: model_AAPL.h5 or AAPL_lstm.keras
        stem = fp.stem.upper()
        symbol = stem.replace("MODEL_", "").replace("_LSTM", "").split("_")[0]
        models.append(
            ModelInfo(
                symbol=symbol,
                filename=fp.name,
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            )
        )
    return ModelsResponse(
        models=models,
        models_dir=str(MODELS_DIR.resolve()),
        total_count=len(models),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["predictions"],
    summary="Generate ML predictions for symbols",
    responses={
        503: {"model": ErrorResponse, "description": "Prediction service unavailable"},
    },
)
async def predict(body: PredictRequest):
    """Generate LSTM-based price predictions for the requested symbols.

    Returns predicted return percentage and confidence score for each symbol
    over the specified horizon. If no trained model exists, the response
    includes an explicit status rather than a silent fallback.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. TensorFlow may not be installed.",
        )

    predictions: List[SymbolPrediction] = []
    for raw_symbol in body.symbols:
        symbol = _validate_symbol(raw_symbol)
        try:
            result = prediction_service.get_prediction(
                symbol=symbol,
                days_ahead=body.horizon_days,
                use_cache=True,
                allow_mock=False,
            )
            predictions.append(
                SymbolPrediction(
                    symbol=symbol,
                    predicted_return=result.get("predicted_return"),
                    confidence=result.get("confidence"),
                    horizon_days=body.horizon_days,
                    model_type=result.get("model_type", "lstm"),
                    timestamp=_now_iso(),
                    status=result.get("status", "ok"),
                    message=result.get("reason"),
                )
            )
        except Exception as exc:
            logger.error(f"Prediction failed for {symbol}: {exc}")
            predictions.append(
                SymbolPrediction(
                    symbol=symbol,
                    horizon_days=body.horizon_days,
                    timestamp=_now_iso(),
                    status="error",
                    message=str(exc),
                )
            )

    return PredictResponse(
        predictions=predictions,
        generated_at=_now_iso(),
    )


@app.get(
    "/api/v1/features/{symbol}",
    response_model=FeatureResponse,
    tags=["features"],
    summary="Get computed technical features for a symbol",
    responses={
        503: {"model": ErrorResponse, "description": "Feature engine unavailable"},
    },
)
async def get_features(symbol: str):
    """Compute and return the full set of technical indicators for a symbol.

    Fetches 1 year of OHLCV data via yfinance, computes ~29 features
    (moving averages, RSI, MACD, Bollinger Bands, volume, momentum,
    volatility, ATR), and returns the most recent values.
    """
    symbol = _validate_symbol(symbol)

    if feature_engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Feature engineering service not available.",
        )

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for symbol {symbol}",
            )

        df_features = feature_engineer.calculate_features(df)
        latest = df_features.iloc[-1]

        # Convert to serializable dict, dropping NaN
        feature_dict = {
            k: round(float(v), 6) for k, v in latest.items() if not _is_nan(v)
        }

        return FeatureResponse(
            symbol=symbol,
            feature_count=len(feature_dict),
            record_count=len(df_features),
            features=feature_dict,
            computed_at=_now_iso(),
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="yfinance not installed — cannot fetch market data.",
        )
    except Exception as exc:
        logger.error(f"Feature computation failed for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/api/v1/backtest/{symbol}",
    response_model=BacktestResponse,
    tags=["backtest"],
    summary="Get backtest metrics for a symbol",
    responses={
        503: {"model": ErrorResponse, "description": "Backtest service unavailable"},
    },
)
async def get_backtest(symbol: str, period: str = "1y"):
    """Run a basic backtest for the symbol and return performance metrics.

    Computes buy-and-hold baseline metrics over the requested period.
    Full ML-vs-baseline backtesting requires trained models.
    """
    symbol = _validate_symbol(symbol)

    try:
        import yfinance as yf
        import numpy as np

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for symbol {symbol}",
            )

        closes = df["Close"]
        returns = closes.pct_change().dropna()

        total_return = float((closes.iloc[-1] / closes.iloc[0]) - 1)
        annualized_vol = float(returns.std() * (252 ** 0.5))
        sharpe = float(total_return / annualized_vol) if annualized_vol > 0 else 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())

        info = get_stock_info(symbol)

        return BacktestResponse(
            symbol=symbol,
            period=period,
            metrics={
                "total_return_pct": round(total_return * 100, 2),
                "annualized_volatility_pct": round(annualized_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "trading_days": len(returns),
                "start_date": str(df.index[0].date()),
                "end_date": str(df.index[-1].date()),
                "start_price": round(float(closes.iloc[0]), 2),
                "end_price": round(float(closes.iloc[-1]), 2),
                "name": info.get("name") if info else symbol,
                "sector": info.get("sector") if info else None,
            },
            computed_at=_now_iso(),
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="yfinance not installed — cannot fetch market data.",
        )
    except Exception as exc:
        logger.error(f"Backtest failed for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nan(value) -> bool:
    """Check if a value is NaN without requiring numpy."""
    try:
        return value != value  # NaN != NaN is True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ml.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
