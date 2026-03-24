"""
Data Validation Schemas for Stock ML Pipeline

Validates data at system boundaries (yfinance input, model output)
using pandera schemas. Catches data quality issues early before
they corrupt model training or predictions.

Usage:
    from ml.validation.data_schemas import validate_ohlcv, validate_prediction
    validated_df = validate_ohlcv(raw_df)
    validate_prediction(prediction_dict)
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema, Index
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False


# --- Schema Definitions (pandera) ---

if HAS_PANDERA:
    OHLCVSchema = DataFrameSchema(
        columns={
            "timestamp": Column(
                dtype="datetime64[ns]",
                nullable=False,
                description="Trading day timestamp",
            ),
            "symbol": Column(
                dtype="object",
                checks=Check.str_length(min_value=1, max_value=10),
                nullable=False,
                description="Stock ticker symbol",
            ),
            "open": Column(
                dtype="float64",
                checks=[Check.gt(0), Check.lt(1e6)],
                nullable=False,
                description="Opening price",
            ),
            "high": Column(
                dtype="float64",
                checks=[Check.gt(0), Check.lt(1e6)],
                nullable=False,
                description="High price",
            ),
            "low": Column(
                dtype="float64",
                checks=[Check.gt(0), Check.lt(1e6)],
                nullable=False,
                description="Low price",
            ),
            "close": Column(
                dtype="float64",
                checks=[Check.gt(0), Check.lt(1e6)],
                nullable=False,
                description="Closing price",
            ),
            "volume": Column(
                dtype="float64",
                checks=Check.ge(0),
                nullable=False,
                description="Trading volume",
            ),
        },
        checks=[
            Check(lambda df: (df["high"] >= df["low"]).all(), error="High must be >= Low"),
            Check(lambda df: (df["high"] >= df["open"]).all(), error="High must be >= Open"),
            Check(lambda df: (df["high"] >= df["close"]).all(), error="High must be >= Close"),
            Check(lambda df: (df["low"] <= df["open"]).all(), error="Low must be <= Open"),
            Check(lambda df: (df["low"] <= df["close"]).all(), error="Low must be <= Close"),
        ],
        coerce=True,
    )

    FeatureSchema = DataFrameSchema(
        columns={
            "RSI": Column(
                dtype="float64",
                checks=[Check.ge(0), Check.le(100)],
                nullable=False,
                description="RSI must be between 0-100",
            ),
            "BB_Position": Column(
                dtype="float64",
                nullable=False,
                description="Bollinger Band position",
            ),
            "Volume_Ratio": Column(
                dtype="float64",
                checks=Check.ge(0),
                nullable=False,
                description="Volume ratio must be non-negative",
            ),
        },
        strict=False,  # Allow additional columns
        coerce=True,
    )


# --- Fallback validation (no pandera) ---

def _validate_ohlcv_basic(df: pd.DataFrame) -> List[str]:
    """Basic OHLCV validation without pandera."""
    errors: List[str] = []

    required_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    if df.empty:
        errors.append("DataFrame is empty")
        return errors

    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            errors.append(f"Non-positive values in '{col}'")

    if (df["volume"] < 0).any():
        errors.append("Negative volume values")

    if (df["high"] < df["low"]).any():
        errors.append("High < Low detected")

    if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
        errors.append("NaN values in price/volume columns")

    return errors


def _validate_prediction_basic(pred: Dict[str, Any]) -> List[str]:
    """Basic prediction dict validation."""
    errors: List[str] = []

    required_keys = {
        "symbol", "current_price", "predicted_price",
        "predicted_return", "confidence", "status",
    }
    missing = required_keys - set(pred.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")
        return errors

    if pred["current_price"] <= 0:
        errors.append(f"current_price must be positive, got {pred['current_price']}")

    if pred["predicted_price"] <= 0:
        errors.append(f"predicted_price must be positive, got {pred['predicted_price']}")

    if not 0 <= pred["confidence"] <= 1:
        errors.append(f"confidence must be in [0,1], got {pred['confidence']}")

    if abs(pred["predicted_return"]) > 1.0:
        errors.append(f"predicted_return seems extreme: {pred['predicted_return']}")

    return errors


# --- Public API ---

def validate_ohlcv(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate OHLCV DataFrame from yfinance.

    Args:
        df: Raw OHLCV DataFrame.
        raise_on_error: If True, raises ValueError on validation failure.

    Returns:
        Validated (and possibly coerced) DataFrame.

    Raises:
        ValueError: If validation fails and raise_on_error is True.
    """
    if HAS_PANDERA:
        try:
            validated = OHLCVSchema.validate(df)
            logger.info(f"OHLCV validation passed ({len(df)} rows)")
            return validated
        except pa.errors.SchemaError as e:
            msg = f"OHLCV validation failed: {e}"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg) from e
            return df
    else:
        errors = _validate_ohlcv_basic(df)
        if errors:
            msg = f"OHLCV validation failed: {'; '.join(errors)}"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
        else:
            logger.info(f"OHLCV validation passed ({len(df)} rows)")
        return df


def validate_features(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate feature-engineered DataFrame."""
    if HAS_PANDERA:
        try:
            validated = FeatureSchema.validate(df)
            logger.info(f"Feature validation passed ({len(df)} rows, {len(df.columns)} cols)")
            return validated
        except pa.errors.SchemaError as e:
            msg = f"Feature validation failed: {e}"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg) from e
            return df
    else:
        errors: List[str] = []
        if "RSI" in df.columns:
            if (df["RSI"] < 0).any() or (df["RSI"] > 100).any():
                errors.append("RSI values outside [0, 100]")
        if errors:
            msg = f"Feature validation failed: {'; '.join(errors)}"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
        else:
            logger.info(f"Feature validation passed ({len(df)} rows)")
        return df


def validate_prediction(pred: Dict[str, Any], raise_on_error: bool = True) -> bool:
    """Validate a prediction output dictionary.

    Args:
        pred: Prediction dictionary from PredictionService.
        raise_on_error: If True, raises ValueError on failure.

    Returns:
        True if valid.
    """
    errors = _validate_prediction_basic(pred)
    if errors:
        msg = f"Prediction validation failed for {pred.get('symbol', '?')}: {'; '.join(errors)}"
        logger.error(msg)
        if raise_on_error:
            raise ValueError(msg)
        return False

    logger.debug(f"Prediction validation passed for {pred.get('symbol')}")
    return True


def validate_sequences(
    X: np.ndarray,
    y: np.ndarray,
    expected_lookback: int = 30,
    expected_features: int = 25,
    raise_on_error: bool = True,
) -> bool:
    """Validate LSTM input/output sequences.

    Args:
        X: Input array, expected shape (n, lookback, features).
        y: Target array, expected shape (n,).
        expected_lookback: Expected lookback window.
        expected_features: Expected feature count.
    """
    errors: List[str] = []

    if X.ndim != 3:
        errors.append(f"X must be 3D, got {X.ndim}D with shape {X.shape}")
    elif X.shape[1] != expected_lookback:
        errors.append(f"X lookback is {X.shape[1]}, expected {expected_lookback}")
    elif X.shape[2] != expected_features:
        errors.append(f"X has {X.shape[2]} features, expected {expected_features}")

    if y.ndim != 1:
        errors.append(f"y must be 1D, got {y.ndim}D with shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        errors.append(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

    if np.isnan(X).any():
        nan_pct = np.isnan(X).mean() * 100
        errors.append(f"X contains {nan_pct:.1f}% NaN values")

    if np.isnan(y).any():
        errors.append(f"y contains {np.isnan(y).sum()} NaN values")

    if errors:
        msg = f"Sequence validation failed: {'; '.join(errors)}"
        logger.error(msg)
        if raise_on_error:
            raise ValueError(msg)
        return False

    logger.info(f"Sequence validation passed: X{X.shape}, y{y.shape}")
    return True
