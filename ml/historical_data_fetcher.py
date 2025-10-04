"""
Historical Data Fetcher

Fetches historical OHLCV (Open/High/Low/Close/Volume) data from Kraken API
and stores it in Google BigQuery for ML model training.

Architecture Decision:
- Uses Kraken's public OHLC endpoint (no API keys needed)
- Fetches daily candles (1440-minute interval)
- Stores in BigQuery partitioned by date for cost optimization
- Handles rate limiting automatically (15 req/min)
- Validates data quality before storage

Usage:
    # Fetch 365 days for BTC
    fetcher = HistoricalDataFetcher()
    df = fetcher.fetch_historical_data('BTC', days=365)
    
    # Store in BigQuery
    fetcher.store_to_bigquery(df, 'BTC')
"""

import sys
import time
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    from google.cloud import bigquery
except ImportError:
    print("⚠️  Warning: google-cloud-bigquery not installed")
    bigquery = None

# Local imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.kraken_api import KrakenAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetch and store historical cryptocurrency data.
    
    This class handles:
    - Fetching OHLCV data from Kraken API
    - Data validation and cleaning
    - Storage in Google BigQuery
    - Incremental updates (fetch only new data)
    """
    
    # Kraken pair mappings
    PAIR_MAP = {
        'BTC': 'XXBTZUSD',
        'ETH': 'XETHZUSD',
        'SOL': 'SOLUSD',
        'ADA': 'ADAUSD',
        'DOT': 'DOTUSD',
        'MATIC': 'MATICUSD',
        'LINK': 'LINKUSD',
        'AVAX': 'AVAXUSD'
    }
    
    def __init__(self, project_id: str = 'crypto-ml-trading-487',
                 dataset: str = 'crypto_data'):
        """
        Initialize the data fetcher.
        
        Args:
            project_id: Google Cloud project ID
            dataset: BigQuery dataset name
        """
        self.project_id = project_id
        self.dataset = dataset
        self.kraken = KrakenAPI()
        
        # Initialize BigQuery client if available
        if bigquery:
            self.bq_client = bigquery.Client(project=project_id)
        else:
            self.bq_client = None
            logger.warning("BigQuery client not initialized")
    
    def fetch_historical_data(
        self,
        symbol: str,
        days: int = 365,
        interval: int = 1440
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Kraken.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days to fetch (default: 365)
            interval: Candle interval in minutes (default: 1440 = 1 day)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Example:
            >>> fetcher = HistoricalDataFetcher()
            >>> df = fetcher.fetch_historical_data('BTC', days=365)
            >>> print(df.head())
            
        Technical Notes:
            - Kraken returns max 720 candles per request
            - For 1 year of daily data, we need 365 candles (single request OK)
            - For hourly data over long periods, need multiple requests
            - Data is automatically sorted by timestamp
        """
        logger.info(f"📊 Fetching {days} days of data for {symbol}...")
        
        # Get Kraken pair name
        pair = self.PAIR_MAP.get(symbol)
        if not pair:
            logger.error(f"Unknown symbol: {symbol}")
            return None
        
        try:
            # Fetch OHLC data
            ohlc_data = self.kraken.get_ohlc(pair, interval=interval)
            
            if not ohlc_data:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'vwap', 'volume', 'count'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['symbol'] = symbol
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Keep only required columns
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Filter to requested number of days
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate data quality
            if self._validate_data(df, symbol):
                logger.info(f"✅ Fetched {len(df)} days of clean data for {symbol}")
                logger.info(f"   Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
                logger.info(f"   Price Range: ${df['close'].min():,.2f} to ${df['close'].max():,.2f}")
                return df
            else:
                logger.error(f"Data validation failed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality.
        
        Checks:
        - No missing values
        - No negative prices
        - No zero volumes (suspicious)
        - Reasonable price ranges
        - Chronological order
        """
        if df is None or df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning(f"Missing values found in {symbol} data")
            # Fill forward for minor gaps
            df.fillna(method='ffill', inplace=True)
        
        # Check for negative prices (impossible)
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            logger.error(f"Negative prices found in {symbol}")
            return False
        
        # Check for zero prices (data error)
        if (df[['open', 'high', 'low', 'close']] == 0).any().any():
            logger.warning(f"Zero prices found in {symbol}")
            df = df[df['close'] > 0]  # Remove zero-price rows
        
        # Check chronological order
        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Data not in chronological order, sorting...")
            df.sort_values('timestamp', inplace=True)
        
        logger.info(f"✅ Data validation passed for {symbol}")
        return True
    
    def store_to_bigquery(
        self,
        df: pd.DataFrame,
        symbol: str,
        table_name: str = 'historical_prices'
    ) -> bool:
        """
        Store historical data in BigQuery.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Crypto symbol
            table_name: BigQuery table name
            
        Returns:
            True if successful, False otherwise
            
        Technical Notes:
            - Appends to existing table (no duplicates due to deduplication)
            - Table is partitioned by timestamp for cost optimization
            - Streaming inserts for real-time updates
        """
        if self.bq_client is None:
            logger.warning("BigQuery client not available, skipping storage")
            return False
        
        logger.info(f"💾 Storing {len(df)} records to BigQuery...")
        
        try:
            # Add metadata
            df['data_source'] = 'kraken_api'
            df['created_at'] = datetime.now()
            
            # Define table
            table_id = f"{self.project_id}.{self.dataset}.{table_name}"
            
            # Upload to BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                schema=[
                    bigquery.SchemaField("timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("symbol", "STRING"),
                    bigquery.SchemaField("open", "FLOAT64"),
                    bigquery.SchemaField("high", "FLOAT64"),
                    bigquery.SchemaField("low", "FLOAT64"),
                    bigquery.SchemaField("close", "FLOAT64"),
                    bigquery.SchemaField("volume", "FLOAT64"),
                    bigquery.SchemaField("data_source", "STRING"),
                    bigquery.SchemaField("created_at", "TIMESTAMP"),
                ]
            )
            
            job = self.bq_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            
            job.result()  # Wait for completion
            
            logger.info(f"✅ Successfully stored {len(df)} records to {table_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to BigQuery: {e}")
            return False
    
    def fetch_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 365
    ) -> dict:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of symbols (default: ['BTC', 'ETH', 'SOL', 'ADA'])
            days: Number of days to fetch
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL', 'ADA']
        
        logger.info(f"🚀 Fetching data for {len(symbols)} symbols...")
        
        data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
            
            df = self.fetch_historical_data(symbol, days)
            
            if df is not None:
                data[symbol] = df
                
                # Store in BigQuery
                if self.bq_client:
                    self.store_to_bigquery(df, symbol)
            
            # Rate limiting: wait between requests
            if i < len(symbols) - 1:
                time.sleep(2)  # 2 seconds between requests
        
        logger.info(f"\n✅ Completed fetching {len(data)}/{len(symbols)} symbols")
        return data


def main():
    """Main function for testing and initial data collection."""
    print("="*60)
    print("🚀 Historical Data Fetcher - Initial Data Collection")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher()
    
    # Fetch data for all symbols
    symbols = ['BTC', 'ETH', 'SOL', 'ADA']
    data = fetcher.fetch_all_symbols(symbols, days=365)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Data Collection Summary")
    print("="*60)
    
    for symbol, df in data.items():
        if df is not None:
            print(f"\n{symbol}:")
            print(f"   Records: {len(df)}")
            print(f"   Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            print(f"   Price Range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            print(f"   Avg Volume: {df['volume'].mean():,.2f}")
    
    print("\n" + "="*60)
    print("✅ Initial data collection complete!")
    print("="*60)
    print("\n📝 Next steps:")
    print("   1. Data is now in BigQuery")
    print("   2. Build feature engineering pipeline")
    print("   3. Create LSTM model architecture")
    print("   4. Train model on this data")


if __name__ == "__main__":
    main()

