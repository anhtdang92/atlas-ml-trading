"""
Feature Engineering for Stock Price Prediction

Creates technical indicators and features from raw OHLCV data for LSTM model input.
Designed for position trading (weeks-months timeframe).

Features Created:
1. Moving Averages (MA): 10, 20, 50, 200-day
2. Relative Strength Index (RSI): 14-day
3. MACD: 12/26/9 signal line crossover
4. Bollinger Bands: 20-day, 2 std deviations
5. Volume Indicators: Volume MA, OBV trend, Volume Rate of Change
6. Price Momentum: Daily returns, 14-day and 30-day momentum
7. Volatility: 14-day and 30-day rolling standard deviation
8. Price relative to moving averages (position signals)

Architecture Decision:
- All features normalized to [0, 1] range for LSTM stability
- Missing values filled forward (common in time series)
- Features scaled per-symbol to handle different price ranges
- 30-day lookback for position trading

Usage:
    fe = FeatureEngineer()
    df = fe.calculate_features(raw_df)
    X, y = fe.create_sequences(df, lookback=30)
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features from raw OHLCV stock data for position trading."""

    def __init__(self):
        self.features = []

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages (10, 20, 50, 200-day)."""
        df['MA_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['MA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['MA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['MA_200'] = df['close'].rolling(window=200, min_periods=1).mean()

        # Price relative to MAs (useful signals)
        df['Price_to_MA50'] = df['close'] / df['MA_50']
        df['Price_to_MA200'] = df['close'] / df['MA_200']
        df['MA_50_200_Cross'] = df['MA_50'] / df['MA_200']  # Golden/Death cross signal

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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        logger.info("Calculated Bollinger Bands")
        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        df['Volume_MA_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_ROC'] = df['volume'].pct_change(periods=10)
        # Volume relative to average (spike detection)
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA_20']
        logger.info("Calculated volume indicators")
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

        # Fill NaN values
        df = df.bfill().ffill()

        self.features = [
            'close', 'volume',
            'MA_10', 'MA_20', 'MA_50', 'MA_200',
            'Price_to_MA50', 'Price_to_MA200', 'MA_50_200_Cross',
            'RSI',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position',
            'Volume_MA_20', 'Volume_ROC', 'Volume_Ratio',
            'Daily_Return', 'Momentum_14', 'Momentum_30', 'ROC_10',
            'Volatility_14', 'Volatility_30', 'ATR_14'
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

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range."""
        logger.info("Normalizing features...")
        df_normalized = df.copy()

        for feature in self.features:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:
                    df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df_normalized[feature] = 0

        logger.info("Features normalized")
        return df_normalized


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

    X, y = fe.create_sequences(df_features, lookback=30, prediction_horizon=21)
    print(f"\nSequences created:")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"\nReady for LSTM training!")


if __name__ == "__main__":
    main()
