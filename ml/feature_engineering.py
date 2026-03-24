"""
Feature Engineering for Stock Price Prediction

Creates technical indicators and features from raw OHLCV data for LSTM model input.
Designed for position trading (weeks-months timeframe).

Features Created (29 total):
1. Moving Averages (MA): 10, 20, 50, 200-day + relative ratios
2. Relative Strength Index (RSI): 14-day
3. MACD: 12/26/9 signal line crossover
4. Bollinger Bands: 20-day, 2 std deviations
5. Volume Indicators: Log volume, Volume MA, ROC, ratio
6. Price Momentum: Daily returns, 14-day and 30-day momentum
7. Volatility: 14-day and 30-day rolling standard deviation, ATR
8. Market Regime: VIX proxy (rolling vol of SPY), sector-relative performance

Architecture Decision:
- Features normalized with train-set-only scaler to prevent data leakage
- Missing values filled forward then backward (natural time-series order)
- Features scaled per-symbol to handle different price ranges
- First 50 rows dropped after feature calc (unreliable rolling windows)
- 30-day lookback for position trading

Usage:
    fe = FeatureEngineer()
    df = fe.calculate_features(raw_df)
    X, y = fe.create_sequences(df, lookback=30)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features from raw OHLCV stock data for position trading."""

    EXPECTED_FEATURE_COUNT: int = 29
    # Minimum rows of data needed before rolling indicators are reliable
    WARMUP_ROWS: int = 50

    def __init__(self) -> None:
        self.features: List[str] = []
        self._scaler_params: Optional[Dict] = None

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages (10, 20, 50, 200-day)."""
        df['MA_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['MA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['MA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['MA_200'] = df['close'].rolling(window=200, min_periods=1).mean()

        # Price relative to MAs as percentage deviation (scale-invariant)
        df['Price_to_MA50'] = (df['close'] - df['MA_50']) / df['MA_50']
        df['Price_to_MA200'] = (df['close'] - df['MA_200']) / df['MA_200']
        df['MA_50_200_Cross'] = (df['MA_50'] - df['MA_200']) / df['MA_200']

        logger.info("Calculated moving averages (10, 20, 50, 200-day)")
        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        logger.info(f"Calculated RSI ({period}-day)")
        return df

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (12/26/9)."""
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        logger.info("Calculated MACD (12/26/9)")
        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands (20-day, 2 std dev)."""
        df['BB_Middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=period, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        # Bollinger %B - where price is relative to bands (0=lower, 1=upper)
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Width'] = bb_range / df['BB_Middle']
        df['BB_Position'] = np.where(bb_range > 0,
                                     (df['close'] - df['BB_Lower']) / bb_range,
                                     0.5)
        logger.info("Calculated Bollinger Bands")
        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators with log transform."""
        # Log-transform volume (highly skewed distribution)
        df['Log_Volume'] = np.log1p(df['volume'])
        df['Volume_MA_20'] = df['Log_Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_ROC'] = df['Log_Volume'].pct_change(periods=10)
        # Volume relative to average (spike detection)
        df['Volume_Ratio'] = np.where(df['Volume_MA_20'] > 0,
                                      df['Log_Volume'] / df['Volume_MA_20'],
                                      1.0)
        logger.info("Calculated volume indicators (log-transformed)")
        return df

    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price momentum indicators for position trading."""
        df['Daily_Return'] = df['close'].pct_change()
        df['Momentum_14'] = df['close'].pct_change(periods=14)
        df['Momentum_30'] = df['close'].pct_change(periods=30)
        # Rate of change
        df['ROC_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        logger.info("Calculated momentum indicators")
        return df

    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        returns = df['close'].pct_change()
        df['Volatility_14'] = returns.rolling(window=14, min_periods=1).std()
        df['Volatility_30'] = returns.rolling(window=30, min_periods=1).std()
        # ATR (Average True Range) - good for position sizing
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
        logger.info("Calculated volatility indicators")
        return df

    def calculate_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime proxies from single-stock data.

        Adds volatility regime and mean-reversion signals that approximate
        macro context without requiring external data feeds.
        """
        returns = df['close'].pct_change()

        # Volatility regime: ratio of short-term to long-term vol
        # >1 = high vol regime (risk-off), <1 = low vol regime (risk-on)
        vol_short = returns.rolling(window=5, min_periods=1).std()
        vol_long = returns.rolling(window=60, min_periods=1).std()
        df['Vol_Regime'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)

        # Distance from 52-week high (drawdown proxy)
        rolling_high = df['close'].rolling(window=252, min_periods=1).max()
        df['Dist_52w_High'] = (df['close'] - rolling_high) / rolling_high

        # Mean reversion signal: z-score of price relative to 50-day MA
        ma50 = df['close'].rolling(window=50, min_periods=1).mean()
        std50 = df['close'].rolling(window=50, min_periods=1).std()
        df['Price_Zscore'] = np.where(std50 > 0,
                                      (df['close'] - ma50) / std50,
                                      0.0)

        # Trend strength: ADX approximation via directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        atr = df.get('ATR_14', true_range_fallback(df))
        plus_di = pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean()
        minus_di = pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean()
        di_sum = plus_di + minus_di
        df['Trend_Strength'] = np.where(di_sum > 0,
                                        np.abs(plus_di - minus_di) / di_sum,
                                        0.0)

        logger.info("Calculated market regime features")
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for stock prediction."""
        logger.info(f"Calculating features for {len(df)} records...")
        df = df.copy()

        df = self.calculate_moving_averages(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_momentum(df)
        df = self.calculate_volatility(df)
        df = self.calculate_market_regime(df)

        # Fill NaN: forward fill first (natural), then backfill for start-of-series
        df = df.ffill().bfill()

        # Drop warmup rows where rolling indicators are unreliable
        if len(df) > self.WARMUP_ROWS:
            df = df.iloc[self.WARMUP_ROWS:].reset_index(drop=True)
            logger.info(f"Dropped first {self.WARMUP_ROWS} warmup rows")

        self.features = [
            'close', 'Log_Volume',
            'MA_10', 'MA_20', 'MA_50', 'MA_200',
            'Price_to_MA50', 'Price_to_MA200', 'MA_50_200_Cross',
            'RSI',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position',
            'Volume_MA_20', 'Volume_ROC', 'Volume_Ratio',
            'Daily_Return', 'Momentum_14', 'Momentum_30', 'ROC_10',
            'Volatility_14', 'Volatility_30', 'ATR_14',
            'Vol_Regime', 'Dist_52w_High', 'Price_Zscore', 'Trend_Strength'
        ]

        logger.info(f"Feature engineering complete! Created {len(self.features)} features")
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        lookback: int = 30,
        prediction_horizon: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.

        For position trading: 30-day lookback to predict 21-day (~1 month) returns.

        Args:
            df: DataFrame with features
            lookback: Number of days to look back (default: 30)
            prediction_horizon: Days ahead to predict (default: 21 trading days)

        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target values (samples,) - predicted returns
        """
        logger.info(f"Creating sequences: lookback={lookback}, horizon={prediction_horizon}")

        feature_data = df[self.features].values

        # Calculate future returns (target variable)
        df = df.copy()
        df['Future_Return'] = df['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        df = df.dropna(subset=['Future_Return'])

        X, y = [], []

        for i in range(len(df) - lookback):
            sequence = feature_data[i:i+lookback]
            target = df['Future_Return'].iloc[i+lookback]
            X.append(sequence)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences")
        logger.info(f"   Input shape: {X.shape} (samples, timesteps, features)")
        logger.info(f"   Output shape: {y.shape} (samples,)")

        return X, y

    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        scaler_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Normalize features to [0, 1] range.

        Args:
            df: DataFrame with feature columns
            fit: If True, compute scaler params from this data (training set).
                 If False, use provided scaler_params (validation/test set).
            scaler_params: Pre-computed min/max from training set. Required when fit=False.

        Returns:
            Normalized DataFrame
        """
        logger.info("Normalizing features...")
        df_normalized = df.copy()

        if fit:
            self._scaler_params = {}
            for feature in self.features:
                if feature in df.columns:
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    self._scaler_params[feature] = {'min': min_val, 'max': max_val}
        elif scaler_params is not None:
            self._scaler_params = scaler_params
        elif self._scaler_params is None:
            raise ValueError("No scaler params available. Call with fit=True first on training data.")

        for feature in self.features:
            if feature in df.columns and feature in self._scaler_params:
                min_val = self._scaler_params[feature]['min']
                max_val = self._scaler_params[feature]['max']
                if max_val > min_val:
                    df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)
                    # Clip to [0, 1] for val/test data that may exceed training range
                    df_normalized[feature] = df_normalized[feature].clip(0, 1)
                else:
                    df_normalized[feature] = 0

        logger.info("Features normalized")
        return df_normalized

    def get_scaler_params(self) -> Optional[Dict]:
        """Return current scaler parameters for saving/reuse."""
        return self._scaler_params


def true_range_fallback(df: pd.DataFrame) -> pd.Series:
    """Calculate ATR fallback for trend strength when ATR_14 not yet available."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=14, min_periods=1).mean()


def main():
    """Test feature engineering on sample data."""
    print("=" * 60)
    print("Testing Feature Engineering for Stocks")
    print("=" * 60)

    from datetime import datetime
    dates = pd.date_range(end=datetime.now(), periods=300, freq='B')  # Business days
    np.random.seed(42)
    price = np.random.randn(300).cumsum() + 150
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(300) * 0.5,
        'high': price + abs(np.random.randn(300)) * 2,
        'low': price - abs(np.random.randn(300)) * 2,
        'close': price,
        'volume': np.random.rand(300) * 1000000 + 500000
    })

    fe = FeatureEngineer()
    df_features = fe.calculate_features(df)

    print(f"\nFeatures created: {fe.features}")
    print(f"Total features: {len(fe.features)}")

    # Test train-only normalization
    split_idx = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:split_idx]
    df_val = df_features.iloc[split_idx:]

    df_train_norm = fe.normalize_features(df_train, fit=True)
    df_val_norm = fe.normalize_features(df_val, fit=False)
    print(f"\nTrain normalized range: [{df_train_norm[fe.features].min().min():.3f}, {df_train_norm[fe.features].max().max():.3f}]")
    print(f"Val normalized range (clipped): [{df_val_norm[fe.features].min().min():.3f}, {df_val_norm[fe.features].max().max():.3f}]")

    X, y = fe.create_sequences(df_features, lookback=30, prediction_horizon=21)
    print(f"\nSequences created:")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"\nReady for LSTM training!")


if __name__ == "__main__":
    main()
