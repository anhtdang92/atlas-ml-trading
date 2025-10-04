#!/usr/bin/env python3
"""
Kraken API Test Script
Tests both public and private endpoints (when API keys are provided)

Usage:
    # Test public endpoints (no keys needed)
    python kraken_test.py
    
    # Test with API keys (set environment variables first)
    export KRAKEN_API_KEY="your_api_key"
    export KRAKEN_API_SECRET="your_api_secret"
    python kraken_test.py --with-auth
"""

import sys
import argparse
from typing import Dict, List, Optional
import json
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not found.")
    print("Install it with: pip install requests")
    sys.exit(1)


class KrakenAPITester:
    """Test Kraken API endpoints."""
    
    BASE_URL = "https://api.kraken.com"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize API tester.
        
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
    
    def test_public_endpoint(self) -> bool:
        """Test public endpoint - Get server time."""
        print("\n" + "="*60)
        print("🧪 TEST 1: Server Time (Public Endpoint)")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.BASE_URL}/0/public/Time")
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"❌ API Error: {data['error']}")
                return False
            
            server_time = datetime.fromtimestamp(data['result']['unixtime'])
            print(f"✅ Connection successful!")
            print(f"   Server Time: {server_time}")
            print(f"   RFC1123: {data['result']['rfc1123']}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_ticker(self, pairs: List[str]) -> bool:
        """Get ticker information for cryptocurrency pairs.
        
        Args:
            pairs: List of trading pairs (e.g., ['XXBTZUSD', 'XETHZUSD'])
        """
        print("\n" + "="*60)
        print(f"🧪 TEST 2: Ticker Information (Public Endpoint)")
        print("="*60)
        
        try:
            pair_str = ','.join(pairs)
            response = self.session.get(
                f"{self.BASE_URL}/0/public/Ticker",
                params={'pair': pair_str}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"❌ API Error: {data['error']}")
                return False
            
            print(f"✅ Successfully fetched ticker data for {len(data['result'])} pairs:\n")
            
            for pair_name, ticker_data in data['result'].items():
                price = float(ticker_data['c'][0])  # Current price
                volume = float(ticker_data['v'][1])  # 24h volume
                high = float(ticker_data['h'][1])  # 24h high
                low = float(ticker_data['l'][1])  # 24h low
                
                print(f"   {pair_name}:")
                print(f"      Current Price: ${price:,.2f}")
                print(f"      24h High:      ${high:,.2f}")
                print(f"      24h Low:       ${low:,.2f}")
                print(f"      24h Volume:    {volume:,.2f}")
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_ohlc_data(self, pair: str, interval: int = 1440) -> bool:
        """Get OHLC (candlestick) data.
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            interval: Time interval in minutes (default: 1440 = 1 day)
        """
        print("\n" + "="*60)
        print(f"🧪 TEST 3: OHLC Data for {pair} (Public Endpoint)")
        print("="*60)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/0/public/OHLC",
                params={'pair': pair, 'interval': interval}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"❌ API Error: {data['error']}")
                return False
            
            # Get the pair key from result
            pair_key = list(data['result'].keys())[0] if data['result'] else None
            if not pair_key or pair_key == 'last':
                print("❌ No OHLC data found")
                return False
            
            ohlc_data = data['result'][pair_key]
            print(f"✅ Successfully fetched {len(ohlc_data)} OHLC records\n")
            print(f"   Last 5 records:")
            print(f"   {'Timestamp':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<12}")
            print(f"   {'-'*80}")
            
            for record in ohlc_data[-5:]:
                timestamp = datetime.fromtimestamp(record[0]).strftime('%Y-%m-%d %H:%M')
                open_price, high, low, close, _, volume, _ = record[1:8]
                print(f"   {timestamp:<20} ${float(open_price):<11,.2f} ${float(high):<11,.2f} "
                      f"${float(low):<11,.2f} ${float(close):<11,.2f} {float(volume):<11,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_tradable_pairs(self) -> bool:
        """Get list of tradable asset pairs."""
        print("\n" + "="*60)
        print(f"🧪 TEST 4: Tradable Asset Pairs (Public Endpoint)")
        print("="*60)
        
        try:
            response = self.session.get(f"{self.BASE_URL}/0/public/AssetPairs")
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"❌ API Error: {data['error']}")
                return False
            
            # Filter for USD pairs only
            usd_pairs = {k: v for k, v in data['result'].items() 
                        if 'USD' in k and not k.endswith('.d')}
            
            print(f"✅ Found {len(data['result'])} total pairs")
            print(f"   {len(usd_pairs)} USD pairs available\n")
            print("   Popular USD pairs:")
            
            popular = ['XXBTZUSD', 'XETHZUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD', 'MATICUSD']
            for pair in popular:
                if pair in usd_pairs:
                    pair_info = usd_pairs[pair]
                    print(f"      {pair:<15} (Alt name: {pair_info.get('altname', 'N/A')})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test Kraken API connectivity')
    parser.add_argument('--with-auth', action='store_true', 
                       help='Test private endpoints (requires API keys in env vars)')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 KRAKEN API TEST SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize tester
    if args.with_auth:
        import os
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not api_key or not api_secret:
            print("\n⚠️  Warning: --with-auth specified but API keys not found")
            print("   Set environment variables: KRAKEN_API_KEY and KRAKEN_API_SECRET")
            print("   Continuing with public endpoints only...\n")
            tester = KrakenAPITester()
        else:
            print("\n✅ API keys found - will test private endpoints")
            tester = KrakenAPITester(api_key, api_secret)
    else:
        print("\n📢 Testing PUBLIC endpoints only (no API keys required)")
        print("   To test private endpoints, use: python kraken_test.py --with-auth\n")
        tester = KrakenAPITester()
    
    # Run public endpoint tests
    results = []
    
    results.append(("Server Time", tester.test_public_endpoint()))
    results.append(("Ticker Data", tester.get_ticker(['XXBTZUSD', 'XETHZUSD', 'SOLUSD'])))
    results.append(("OHLC Data", tester.get_ohlc_data('XXBTZUSD')))
    results.append(("Tradable Pairs", tester.get_tradable_pairs()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name:<25} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n   Total: {passed_count}/{total_count} tests passed")
    print("="*60)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Kraken API is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Open index.html in your browser to see the live dashboard")
        print("   2. When ready for trading, add your API keys to test private endpoints")
        print("   3. Run: python kraken_test.py --with-auth")
    else:
        print("\n⚠️  Some tests failed. Please check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()

