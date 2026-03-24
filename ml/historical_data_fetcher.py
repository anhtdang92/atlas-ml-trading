"""
Historical Data Fetcher for Stocks

Fetches historical OHLCV data from Yahoo Finance (via yfinance)
and stores it in Google BigQuery for ML model training.

Architecture:
- Uses yfinance (free, no API key needed)
- Fetches daily data for 2+ years
- Stores in BigQuery partitioned by date
- Validates data quality before storage
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
    print("Warning: google-cloud-bigquery not installed")
    bigquery = None

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stock_api import StockAPI, get_all_symbols

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """Fetch and store historical stock data."""

    def __init__(self, project_id: str = 'stock-ml-trading-487',
                 dataset: str = 'stock_data'):
        self.project_id = project_id
        self.dataset = dataset
        self.stock_api = StockAPI()

        if bigquery:
            try:
                self.bq_client = bigquery.Client(project=project_id)
            except Exception:
                self.bq_client = None
                logger.warning("BigQuery client not initialized")
        else:
            self.bq_client = None
            logger.warning("BigQuery client not available")

    def fetch_historical_data(
        self,
        symbol: str,
        days: int = 730,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'SPY')
            days: Number of days to fetch (default: 730 = 2 years)
            interval: Data interval (default: 1d)

        Returns:
            DataFrame with timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {days} days of data for {symbol}...")

        # Map days to yfinance period
        if days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"

        try:
            df = self.stock_api.get_historical_data(symbol, period=period, interval=interval)

            if df is None or df.empty:
                logger.error(f"No data returned for {symbol}")
                return None

            # Filter to requested days
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            df = df.sort_values('timestamp').reset_index(drop=True)

            if self._validate_data(df, symbol):
                logger.info(f"Fetched {len(df)} days of data for {symbol}")
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
        """Validate data quality."""
        if df is None or df.empty:
            logger.error("DataFrame is empty")
            return False

        if df.isnull().any().any():
            logger.warning(f"Missing values found in {symbol} data, forward filling")
            filled = df.ffill()
            df.loc[:] = filled

        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            logger.error(f"Negative prices found in {symbol}")
            return False

        if (df[['open', 'high', 'low', 'close']] == 0).any().any():
            logger.warning(f"Zero prices found in {symbol}")
            mask = df['close'] > 0
            rows_to_drop = df.index[~mask]
            df.drop(rows_to_drop, inplace=True)

        if not df['timestamp'].is_monotonic_increasing:
            logger.warning("Data not in chronological order, sorting...")
            df.sort_values('timestamp', inplace=True)

        logger.info(f"Data validation passed for {symbol}")
        return True

    def store_to_bigquery(
        self,
        df: pd.DataFrame,
        symbol: str,
        table_name: str = 'historical_prices'
    ) -> bool:
        """Store historical data in BigQuery."""
        if self.bq_client is None:
            logger.warning("BigQuery client not available, skipping storage")
            return False

        logger.info(f"Storing {len(df)} records to BigQuery...")

        try:
            df = df.copy()
            df['data_source'] = 'yfinance'
            df['created_at'] = datetime.now()

            table_id = f"{self.project_id}.{self.dataset}.{table_name}"

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
            job.result()

            logger.info(f"Successfully stored {len(df)} records to {table_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing to BigQuery: {e}")
            return False

    def fetch_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 730
    ) -> dict:
        """Fetch historical data for multiple symbols."""
        if symbols is None:
            symbols = get_all_symbols()

        logger.info(f"Fetching data for {len(symbols)} symbols...")

        data = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

            df = self.fetch_historical_data(symbol, days)

            if df is not None:
                data[symbol] = df
                if self.bq_client:
                    self.store_to_bigquery(df, symbol)

            if i < len(symbols) - 1:
                time.sleep(0.5)  # Brief pause between requests

        logger.info(f"\nCompleted fetching {len(data)}/{len(symbols)} symbols")
        return data


def main():
    """Main function for testing and initial data collection."""
    print("=" * 60)
    print("Historical Data Fetcher - Stock Data Collection")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    fetcher = HistoricalDataFetcher()

    # Fetch a subset for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    data = fetcher.fetch_all_symbols(symbols, days=365)

    print("\n" + "=" * 60)
    print("Data Collection Summary")
    print("=" * 60)

    for symbol, df in data.items():
        if df is not None:
            print(f"\n{symbol}:")
            print(f"   Records: {len(df)}")
            print(f"   Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            print(f"   Price Range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            print(f"   Avg Volume: {df['volume'].mean():,.0f}")

    print("\nData collection complete!")


if __name__ == "__main__":
    main()
