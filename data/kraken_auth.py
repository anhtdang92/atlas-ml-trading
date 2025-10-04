"""
Kraken Authenticated API Client

Handles private endpoints requiring API key authentication.
"""

import time
import hmac
import hashlib
import base64
import urllib.parse
import logging
from typing import Dict, Optional, Any
import requests
import yaml

logger = logging.getLogger(__name__)


class KrakenAuthClient:
    """Authenticated Kraken API client for private endpoints."""
    
    BASE_URL = "https://api.kraken.com"
    API_VERSION = "0"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize authenticated client.
        
        Args:
            api_key: Kraken API key (if None, loads from secrets.yaml)
            api_secret: Kraken API secret (if None, loads from secrets.yaml)
        """
        if api_key and api_secret:
            self.api_key = api_key
            self.api_secret = api_secret
        else:
            self._load_credentials()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken-ML-Trading-Dashboard/1.0'
        })
    
    def _load_credentials(self) -> None:
        """Load API credentials from secrets.yaml file."""
        try:
            with open('config/secrets.yaml', 'r') as f:
                config = yaml.safe_load(f)
                self.api_key = config['kraken']['api_key']
                self.api_secret = config['kraken']['api_secret']
        except FileNotFoundError:
            raise Exception("secrets.yaml not found. Please create it from secrets.yaml.example")
        except KeyError as e:
            raise Exception(f"Missing configuration key: {e}")
    
    def _get_kraken_signature(
        self,
        urlpath: str,
        data: Dict[str, Any],
        secret: str
    ) -> str:
        """Generate Kraken API signature.
        
        Args:
            urlpath: API endpoint path
            data: POST data dictionary
            secret: API secret key
            
        Returns:
            Base64-encoded signature
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _make_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Optional[Dict]:
        """Make authenticated API request.
        
        Args:
            endpoint: Private API endpoint (e.g., 'Balance')
            data: Optional POST data
            retries: Number of retry attempts
            
        Returns:
            API response or None on error
        """
        if data is None:
            data = {}
        
        # Add nonce (timestamp in microseconds)
        data['nonce'] = int(time.time() * 1000000)
        
        urlpath = f"/{self.API_VERSION}/private/{endpoint}"
        
        # Generate signature
        signature = self._get_kraken_signature(urlpath, data, self.api_secret)
        
        headers = {
            'API-Key': self.api_key,
            'API-Sign': signature
        }
        
        url = f"{self.BASE_URL}{urlpath}"
        
        for attempt in range(retries):
            try:
                response = self.session.post(url, data=data, headers=headers, timeout=10)
                response.raise_for_status()
                result = response.json()
                
                if result.get('error'):
                    logger.error(f"API error: {result['error']}")
                    return None
                
                logger.info(f"Successfully fetched {endpoint}")
                return result.get('result')
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        
        return None
    
    def get_account_balance(self) -> Optional[Dict[str, str]]:
        """Get account balance for all assets.
        
        Returns:
            Dictionary mapping asset names to balance strings
        """
        return self._make_request('Balance')
    
    def get_trade_balance(self, asset: str = 'ZUSD') -> Optional[Dict]:
        """Get trade balance info.
        
        Args:
            asset: Base asset (default: ZUSD = USD)
            
        Returns:
            Dictionary with equivalent balance, trade balance, margin, etc.
        """
        return self._make_request('TradeBalance', {'asset': asset})
    
    def get_open_orders(self) -> Optional[Dict]:
        """Get list of open orders.
        
        Returns:
            Dictionary of open orders
        """
        return self._make_request('OpenOrders')
    
    def get_closed_orders(self, start: Optional[int] = None) -> Optional[Dict]:
        """Get list of closed orders.
        
        Args:
            start: Starting timestamp (optional)
            
        Returns:
            Dictionary of closed orders
        """
        data = {}
        if start:
            data['start'] = start
        return self._make_request('ClosedOrders', data)
    
    def get_trades_history(self, start: Optional[int] = None) -> Optional[Dict]:
        """Get trades history.
        
        Args:
            start: Starting timestamp (optional)
            
        Returns:
            Dictionary of trades
        """
        data = {}
        if start:
            data['start'] = start
        return self._make_request('TradesHistory', data)


# Convenience function for quick balance check
def get_portfolio_summary() -> Optional[Dict]:
    """Get portfolio summary with balances and values.
    
    Returns:
        Dictionary with portfolio information
    """
    try:
        client = KrakenAuthClient()
        
        # Get account balance
        balance = client.get_account_balance()
        if not balance:
            return None
        
        # Get trade balance (USD equivalent)
        trade_balance = client.get_trade_balance()
        
        return {
            'balances': balance,
            'trade_balance': trade_balance
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return None

