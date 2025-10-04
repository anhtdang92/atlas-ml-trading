"""
Feature Engineering for Crypto Price Prediction

Creates technical indicators and features from raw OHLCV data for LSTM model input.

Features Created:
1. Moving Averages (MA): 7-day, 14-day, 30-day
2. Relative Strength Index (RSI): 14-day
3. Volume Indicators: Volume MA, Volume Rate of Change
4. Price Momentum: Daily returns, 7-day momentum
5. Volatility: 7-day rolling standard deviation

Architecture Decision:
- All features normalized to [0, 1] range for LSTM stability
- Missing values filled forward (common in time series)
- Features scaled per-symbol to handle different price ranges

Usage:
    fe = FeatureEngineer()
    df = fe.calculate_features(raw_df)
    X, y = fe.create_sequences(df, lookback=7)
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create features from raw OHLCV data.
    
    Technical Indicators Explained:
    
    1. Moving Average (MA): Average price over N days
       - Smooths out noise
       - Shows trend direction
       - 7-day: Short-term trend
       - 14-day: Medium-term trend
       - 30-day: Long-term trend
    
    2. RSI (Relative Strength Index): Momentum oscillator (0-100)
       - > 70: Overbought (might drop)
       - < 30: Oversold (might rise)
       - Measures speed/magnitude of price changes
    
    3. Volume Indicators: Trading activity
       - High volume + price up = strong uptrend
       - High volume + price down = strong downtrend
       - Low volume = weak signal
    
    4. Momentum: Rate of price change
       - Positive = upward pressure
       - Negative = downward pressure
       - Magnitude = strength of move
    
    5. Volatility: Price variation
       - High volatility = risky, opportunity
       - Low volatility = stable, boring
       - Important for risk management
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.features = []
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with MA_7, MA_14, MA_30 columns
        """
        df['MA_7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['MA_14'] = df['close'].rolling(window=14, min_periods=1).mean()
        df['MA_30'] = df['close'].rolling(window=30, min_periods=1).mean()
        
        logger.info("✅ Calculated moving averages (7, 14, 30-day)")
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        RSI Formula:
        1. Calculate price changes (deltas)
        2. Separate gains and losses
        3. Calculate average gain and average loss over period
        4. RS = average gain / average loss
        5. RSI = 100 - (100 / (1 + RS))
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default: 14 days)
            
        Returns:
            DataFrame with RSI column (0-100)
        """
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        df['RSI'] = rsi
        
        logger.info(f"✅ Calculated RSI ({period}-day)")
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Indicators:
        - Volume_MA_7: 7-day average volume
        - Volume_ROC: Rate of change in volume
        
        Args:
            df: DataFrame with 'volume' column
            
        Returns:
            DataFrame with volume indicators
        """
        df['Volume_MA_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
        df['Volume_ROC'] = df['volume'].pct_change(periods=7)
        
        logger.info("✅ Calculated volume indicators")
        return df
    
    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price momentum indicators.
        
        Indicators:
        - Daily_Return: (today - yesterday) / yesterday
        - Momentum_7: 7-day rate of change
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with momentum indicators
        """
        df['Daily_Return'] = df['close'].pct_change()
        df['Momentum_7'] = df['close'].pct_change(periods=7)
        
        logger.info("✅ Calculated momentum indicators")
        return df
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility (risk) indicators.
        
        Indicators:
        - Volatility_7: 7-day rolling standard deviation of returns
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with volatility column
        """
        returns = df['close'].pct_change()
        df['Volatility_7'] = returns.rolling(window=7, min_periods=1).std()
        
        logger.info("✅ Calculated volatility indicators")
        return df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with all features added
        """
        logger.info(f"🔧 Calculating features for {len(df)} records...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate all features
        df = self.calculate_moving_averages(df)
        df = self.calculate_rsi(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_momentum(df)
        df = self.calculate_volatility(df)
        
        # Fill NaN values (from initial rolling windows)
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Store feature list
        self.features = [
            'close', 'volume',
            'MA_7', 'MA_14', 'MA_30',
            'RSI',
            'Volume_MA_7', 'Volume_ROC',
            'Daily_Return', 'Momentum_7',
            'Volatility_7'
        ]
        
        logger.info(f"✅ Feature engineering complete! Created {len(self.features)} features")
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        lookback: int = 7,
        prediction_horizon: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        LSTM needs sequences: Use past 7 days to predict next 7 days.
        
        Example:
            Input (X): Days 1-7 features
            Output (y): Day 14 return (7 days ahead)
            
            Input (X): Days 2-8 features  
            Output (y): Day 15 return
            
            ... and so on
        
        Args:
            df: DataFrame with features
            lookback: Number of days to look back (default: 7)
            prediction_horizon: Days ahead to predict (default: 7)
            
        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target values (samples,) - predicted returns
        """
        logger.info(f"📊 Creating sequences: lookback={lookback}, horizon={prediction_horizon}")
        
        # Extract feature columns
        feature_data = df[self.features].values
        
        # Calculate future returns (target variable)
        df['Future_Return'] = df['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        
        # Remove rows without future data
        df = df.dropna(subset=['Future_Return'])
        
        X, y = [], []
        
        for i in range(len(df) - lookback):
            # Input: past 'lookback' days of features
            sequence = feature_data[i:i+lookback]
            
            # Output: future return after 'prediction_horizon' days
            target = df['Future_Return'].iloc[i+lookback]
            
            X.append(sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Created {len(X)} sequences")
        logger.info(f"   Input shape: {X.shape} (samples, timesteps, features)")
        logger.info(f"   Output shape: {y.shape} (samples,)")
        
        return X, y
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range.
        
        Why normalize?
        - LSTM works better with normalized inputs
        - Prevents large values from dominating
        - Faster convergence during training
        
        Method: Min-Max scaling per feature
        
        Args:
            df: DataFrame with features
            
        Returns:
            Normalized DataFrame
        """
        logger.info("🔄 Normalizing features...")
        
        df_normalized = df.copy()
        
        for feature in self.features:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                
                if max_val > min_val:
                    df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df_normalized[feature] = 0
        
        logger.info("✅ Features normalized")
        return df_normalized


def main():
    """Test feature engineering on sample data."""
    print("="*60)
    print("🧪 Testing Feature Engineering")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000
    })
    
    # Test feature calculation
    fe = FeatureEngineer()
    df_features = fe.calculate_features(df)
    
    print(f"\n✅ Features created: {fe.features}")
    print(f"\nSample of engineered features:")
    print(df_features[fe.features].tail())
    
    # Test sequence creation
    X, y = fe.create_sequences(df_features, lookback=7, prediction_horizon=7)
    
    print(f"\n✅ Sequences created:")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"\n📝 Ready for LSTM training!")


if __name__ == "__main__":
    from datetime import datetime
    main()

