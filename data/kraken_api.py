"""
Kraken API Client Module

Handles all interactions with the Kraken API, including:
- Public endpoints (prices, OHLC data, trading pairs)
- Private endpoints (account balance, trades) - requires API keys
"""

import time
import logging
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KrakenAPI:
    """Client for interacting with Kraken API.
    
    Supports both public (no authentication) and private (requires API keys)
    endpoints with automatic retry logic and rate limiting.
    """
    
    BASE_URL = "https://api.kraken.com"
    
    # Rate limiting: 15 requests per minute for public endpoints
    RATE_LIMIT_CALLS = 15
    RATE_LIMIT_WINDOW = 60  # seconds
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize Kraken API client.
        
        Args:
            api_key: Kraken API key (optional, for private endpoints)
            api_secret: Kraken API secret (optional, for private endpoints)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken-ML-Trading-Dashboard/1.0'
        })
        
        # Track API calls for rate limiting
        self._api_calls = []
    
    def _check_rate_limit(self) -> None:
        """Check if we're within rate limits, sleep if necessary."""
        now = time.time()
        
        # Remove calls older than the rate limit window
        self._api_calls = [t for t in self._api_calls if now - t < self.RATE_LIMIT_WINDOW]
        
        # If we've hit the limit, wait
        if len(self._api_calls) >= self.RATE_LIMIT_CALLS:
            sleep_time = self.RATE_LIMIT_WINDOW - (now - self._api_calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._api_calls = []
        
        self._api_calls.append(now)
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Optional[Dict]:
        """Make HTTP request to Kraken API with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., '/0/public/Ticker')
            params: Query parameters
            retries: Number of retry attempts on failure
            
        Returns:
            API response as dictionary, or None on error
        """
        self._check_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('error'):
                    logger.error(f"API error: {data['error']}")
                    return None
                
                logger.info(f"Successfully fetched {endpoint}")
                return data['result']
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        
        logger.error(f"Failed to fetch {endpoint} after {retries} attempts")
        return None
    
    def get_server_time(self) -> Optional[Dict]:
        """Get Kraken server time.
        
        Returns:
            Dictionary with 'unixtime' and 'rfc1123' keys
        """
        return self._make_request('/0/public/Time')
    
    def get_ticker(self, pairs: List[str]) -> Optional[Dict]:
        """Get ticker information for specified trading pairs.
        
        Args:
            pairs: List of trading pairs (e.g., ['XXBTZUSD', 'XETHZUSD'])
            
        Returns:
            Dictionary with ticker data for each pair
        """
        pair_str = ','.join(pairs)
        return self._make_request('/0/public/Ticker', params={'pair': pair_str})
    
    def get_ohlc(
        self,
        pair: str,
        interval: int = 1440,
        since: Optional[int] = None
    ) -> Optional[List]:
        """Get OHLC (candlestick) data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return data since given timestamp
            
        Returns:
            List of OHLC data arrays: [timestamp, open, high, low, close, vwap, volume, count]
        """
        params = {'pair': pair, 'interval': interval}
        if since:
            params['since'] = since
        
        result = self._make_request('/0/public/OHLC', params=params)
        
        if result:
            # Get the pair key (first key that's not 'last')
            pair_keys = [k for k in result.keys() if k != 'last']
            if pair_keys:
                return result[pair_keys[0]]
        
        return None
    
    def get_asset_pairs(self, pairs: Optional[List[str]] = None) -> Optional[Dict]:
        """Get information about tradable asset pairs.
        
        Args:
            pairs: Optional list of specific pairs to query
            
        Returns:
            Dictionary of asset pair information
        """
        params = {'pair': ','.join(pairs)} if pairs else None
        return self._make_request('/0/public/AssetPairs', params=params)
    
    def get_tradable_usd_pairs(self) -> List[str]:
        """Get list of all tradable USD pairs.
        
        Returns:
            List of USD trading pair names
        """
        pairs = self.get_asset_pairs()
        
        if not pairs:
            return []
        
        # Filter for USD pairs only
        usd_pairs = [
            pair for pair in pairs.keys()
            if 'USD' in pair and not pair.endswith('.d')
        ]
        
        return sorted(usd_pairs)
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            
        Returns:
            Current price as float, or None on error
        """
        ticker = self.get_ticker([pair])
        
        if ticker:
            # Find the matching key
            matching_keys = [k for k in ticker.keys() if pair in k or k in pair]
            if matching_keys:
                return float(ticker[matching_keys[0]]['c'][0])
        
        return None
    
    def get_24h_stats(self, pair: str) -> Optional[Dict[str, float]]:
        """Get 24-hour statistics for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            
        Returns:
            Dictionary with open, high, low, close, volume, and change percentage
        """
        ticker = self.get_ticker([pair])
        
        if ticker:
            matching_keys = [k for k in ticker.keys() if pair in k or k in pair]
            if matching_keys:
                data = ticker[matching_keys[0]]
                current = float(data['c'][0])
                open_price = float(data['o'])
                change_pct = ((current - open_price) / open_price) * 100 if open_price else 0
                
                return {
                    'current': current,
                    'open': open_price,
                    'high': float(data['h'][1]),
                    'low': float(data['l'][1]),
                    'volume': float(data['v'][1]),
                    'change_pct': change_pct
                }
        
        return None


# Create a singleton instance for easy importing
kraken = KrakenAPI()

