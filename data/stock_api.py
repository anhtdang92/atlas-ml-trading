"""
Stock Market Data API Client

Handles all stock market data fetching using yfinance (free, no API key required).
Supports stocks, ETFs, and indices with historical OHLCV data.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    print("Warning: yfinance not installed. Install with: pip install yfinance")
    HAS_YFINANCE = False

logger = logging.getLogger(__name__)


# Stock universe organized by category
STOCK_UNIVERSE = {
    'tech': {
        'AAPL': {'name': 'Apple', 'sector': 'Technology', 'color': '#A2AAAD'},
        'MSFT': {'name': 'Microsoft', 'sector': 'Technology', 'color': '#00A4EF'},
        'GOOGL': {'name': 'Alphabet', 'sector': 'Technology', 'color': '#4285F4'},
        'AMZN': {'name': 'Amazon', 'sector': 'Technology', 'color': '#FF9900'},
        'NVDA': {'name': 'NVIDIA', 'sector': 'Technology', 'color': '#76B900'},
        'META': {'name': 'Meta', 'sector': 'Technology', 'color': '#0668E1'},
        'TSLA': {'name': 'Tesla', 'sector': 'Technology', 'color': '#CC0000'},
    },
    'sector_leaders': {
        'JPM': {'name': 'JPMorgan Chase', 'sector': 'Financials', 'color': '#003087'},
        'UNH': {'name': 'UnitedHealth', 'sector': 'Healthcare', 'color': '#002677'},
        'XOM': {'name': 'ExxonMobil', 'sector': 'Energy', 'color': '#ED1B2F'},
        'CAT': {'name': 'Caterpillar', 'sector': 'Industrials', 'color': '#FFCD11'},
        'PG': {'name': 'Procter & Gamble', 'sector': 'Consumer Staples', 'color': '#003DA5'},
        'HD': {'name': 'Home Depot', 'sector': 'Consumer Disc.', 'color': '#F96302'},
        'NEE': {'name': 'NextEra Energy', 'sector': 'Utilities', 'color': '#003865'},
        'AMT': {'name': 'American Tower', 'sector': 'Real Estate', 'color': '#00529B'},
        'LIN': {'name': 'Linde', 'sector': 'Materials', 'color': '#004F9F'},
    },
    'etfs': {
        'SPY': {'name': 'S&P 500 ETF', 'sector': 'Index ETF', 'color': '#1f77b4'},
        'QQQ': {'name': 'Nasdaq 100 ETF', 'sector': 'Index ETF', 'color': '#ff7f0e'},
        'DIA': {'name': 'Dow Jones ETF', 'sector': 'Index ETF', 'color': '#2ca02c'},
        'IWM': {'name': 'Russell 2000 ETF', 'sector': 'Small Cap ETF', 'color': '#d62728'},
        'XLK': {'name': 'Tech Sector ETF', 'sector': 'Sector ETF', 'color': '#9467bd'},
        'XLF': {'name': 'Financial Sector ETF', 'sector': 'Sector ETF', 'color': '#8c564b'},
        'XLE': {'name': 'Energy Sector ETF', 'sector': 'Sector ETF', 'color': '#e377c2'},
        'XLV': {'name': 'Healthcare Sector ETF', 'sector': 'Sector ETF', 'color': '#7f7f7f'},
        'ARKK': {'name': 'ARK Innovation ETF', 'sector': 'Growth ETF', 'color': '#bcbd22'},
    },
    'growth': {
        'PLTR': {'name': 'Palantir', 'sector': 'Technology', 'color': '#101010'},
        'CRWD': {'name': 'CrowdStrike', 'sector': 'Cybersecurity', 'color': '#FF0000'},
        'SNOW': {'name': 'Snowflake', 'sector': 'Cloud Data', 'color': '#29B5E8'},
        'SQ': {'name': 'Block', 'sector': 'Fintech', 'color': '#006AFF'},
        'COIN': {'name': 'Coinbase', 'sector': 'Fintech', 'color': '#0052FF'},
    }
}


def get_all_symbols() -> List[str]:
    """Get flat list of all tracked stock symbols."""
    symbols = []
    for category in STOCK_UNIVERSE.values():
        symbols.extend(category.keys())
    return symbols


def get_stock_info(symbol: str) -> Optional[Dict]:
    """Get stock metadata (name, sector, color) for a symbol."""
    for category in STOCK_UNIVERSE.values():
        if symbol in category:
            return category[symbol]
    return None


def get_symbols_by_category(category: str) -> List[str]:
    """Get symbols for a specific category."""
    if category in STOCK_UNIVERSE:
        return list(STOCK_UNIVERSE[category].keys())
    return []


class StockAPI:
    """Client for fetching stock market data via yfinance.

    Provides stock market data via Yahoo Finance (yfinance).
    All data is free and requires no API keys.
    """

    def __init__(self):
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 60  # seconds

    def _get_ticker(self, symbol: str) -> Any:
        """Get yfinance Ticker object."""
        return yf.Ticker(symbol)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry exists and has not expired."""
        if cache_key not in self._cache:
            return False
        if cache_key not in self._cache_time:
            return False
        elapsed = time.time() - self._cache_time[cache_key]
        return elapsed < self._cache_ttl

    def _get_cached(self, cache_key: str) -> Any:
        """Return cached value if valid, otherwise None sentinel."""
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for '{cache_key}'")
            return self._cache[cache_key]
        logger.debug(f"Cache miss for '{cache_key}'")
        return None

    def _set_cached(self, cache_key: str, value: Any) -> None:
        """Store a value in the cache with the current timestamp."""
        self._cache[cache_key] = value
        self._cache_time[cache_key] = time.time()

    def _fetch_with_retry(self, func, description: str = "yfinance call",
                          max_retries: int = 3, initial_backoff: float = 1.0):
        """Execute a callable with retry and exponential backoff.

        Args:
            func: Zero-argument callable that performs the yfinance fetch.
            description: Human-readable label for log messages.
            max_retries: Maximum number of attempts (default 3).
            initial_backoff: Seconds to wait after the first failure (doubles each retry).

        Returns:
            The return value of *func* on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        backoff = initial_backoff
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"{description}: attempt {attempt}/{max_retries} failed "
                        f"({e}), retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        f"{description}: all {max_retries} attempts failed ({e})"
                    )
        raise last_exception

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a stock.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Current price as float, or None on error
        """
        cache_key = f"price:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ticker = self._get_ticker(symbol)
            hist = self._fetch_with_retry(
                lambda: ticker.history(period='1d'),
                description=f"get_current_price({symbol})"
            )
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self._set_cached(cache_key, price)
                return price
            return None
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get quote data for a stock (price, change, volume, etc.).

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with current, open, high, low, volume, change_pct
        """
        cache_key = f"quote:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ticker = self._get_ticker(symbol)
            hist = self._fetch_with_retry(
                lambda: ticker.history(period='5d'),
                description=f"get_quote({symbol})"
            )

            if hist.empty or len(hist) < 1:
                return None

            current = float(hist['Close'].iloc[-1])
            open_price = float(hist['Open'].iloc[-1])
            high = float(hist['High'].iloc[-1])
            low = float(hist['Low'].iloc[-1])
            volume = float(hist['Volume'].iloc[-1])

            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else open_price
            change_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0

            quote = {
                'current': current,
                'open': open_price,
                'high': high,
                'low': low,
                'volume': volume,
                'prev_close': prev_close,
                'change_pct': change_pct
            }
            self._set_cached(cache_key, quote)
            return quote
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols efficiently.

        Args:
            symbols: List of stock tickers

        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}

        # Check cache first; collect symbols that still need fetching
        uncached_symbols = []
        for symbol in symbols:
            cache_key = f"quote:{symbol}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[symbol] = cached
            else:
                uncached_symbols.append(symbol)

        if not uncached_symbols:
            return results

        try:
            # yfinance supports batch downloads
            data = yf.download(uncached_symbols, period='5d', group_by='ticker', progress=False)

            for symbol in uncached_symbols:
                try:
                    if len(uncached_symbols) == 1:
                        df = data
                    else:
                        df = data[symbol]

                    if df.empty or len(df) < 1:
                        continue

                    current = float(df['Close'].iloc[-1])
                    open_price = float(df['Open'].iloc[-1])
                    high = float(df['High'].iloc[-1])
                    low = float(df['Low'].iloc[-1])
                    volume = float(df['Volume'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else open_price
                    change_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0

                    quote = {
                        'current': current,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'volume': volume,
                        'prev_close': prev_close,
                        'change_pct': change_pct
                    }
                    results[symbol] = quote
                    self._set_cached(f"quote:{symbol}", quote)
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error in batch quote fetch: {e}")
            # Fallback to individual fetches
            for symbol in uncached_symbols:
                quote = self.get_quote(symbol)
                if quote:
                    results[symbol] = quote

        return results

    def get_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a stock.

        Args:
            symbol: Stock ticker
            period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            DataFrame with timestamp, open, high, low, close, volume columns
        """
        try:
            ticker = self._get_ticker(symbol)
            df = self._fetch_with_retry(
                lambda: ticker.history(period=period, interval=interval),
                description=f"get_historical_data({symbol}, {period}, {interval})"
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Standardize column names
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            df['symbol'] = symbol
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Remove timezone info for consistency
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_ohlc(self, symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get OHLC data (alias for get_historical_data for compatibility)."""
        return self.get_historical_data(symbol, period=period, interval=interval)

    def get_market_status(self) -> Dict:
        """Check if the US stock market is currently open."""
        now = datetime.now()
        # Simple check - market is open Mon-Fri 9:30-16:00 ET
        weekday = now.weekday()
        hour = now.hour

        is_weekday = weekday < 5
        is_market_hours = 9 <= hour < 16

        return {
            'is_open': is_weekday and is_market_hours,
            'next_open': 'Monday 9:30 AM ET' if weekday >= 5 else 'Tomorrow 9:30 AM ET',
            'timestamp': now.isoformat()
        }

    def get_stock_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data for a stock (P/E, market cap, etc.)."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            return {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None


# Create a singleton instance for easy importing
stock_api = StockAPI() if HAS_YFINANCE else None
