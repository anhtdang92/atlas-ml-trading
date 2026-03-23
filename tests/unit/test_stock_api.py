#!/usr/bin/env python3
"""
Stock API Test Script

Tests the yfinance-based stock data API:
- Current price fetching
- Quote data (price, change, volume)
- Batch quote fetching
- Historical OHLCV data
- Stock fundamentals
- Market status

Usage:
    python tests/unit/test_stock_api.py
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.stock_api import StockAPI, STOCK_UNIVERSE, get_all_symbols, get_stock_info


def test_current_price():
    """Test fetching current price for a stock."""
    print("\n" + "=" * 60)
    print("TEST 1: Current Price (Single Stock)")
    print("=" * 60)

    try:
        api = StockAPI()
        price = api.get_current_price('AAPL')

        if price and price > 0:
            print(f"  AAPL Current Price: ${price:,.2f}")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No price returned")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_quote():
    """Test fetching quote data."""
    print("\n" + "=" * 60)
    print("TEST 2: Quote Data (MSFT)")
    print("=" * 60)

    try:
        api = StockAPI()
        quote = api.get_quote('MSFT')

        if quote:
            print(f"  Current: ${quote['current']:,.2f}")
            print(f"  Open: ${quote['open']:,.2f}")
            print(f"  High: ${quote['high']:,.2f}")
            print(f"  Low: ${quote['low']:,.2f}")
            print(f"  Volume: {quote['volume']:,.0f}")
            print(f"  Change: {quote['change_pct']:+.2f}%")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No quote returned")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_batch_quotes():
    """Test batch quote fetching."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Quotes (FAANG+)")
    print("=" * 60)

    try:
        api = StockAPI()
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        quotes = api.get_batch_quotes(symbols)

        if quotes and len(quotes) > 0:
            for symbol, quote in quotes.items():
                print(f"  {symbol}: ${quote['current']:,.2f} ({quote['change_pct']:+.2f}%)")
            print(f"\n  Fetched {len(quotes)}/{len(symbols)} quotes")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No quotes returned")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_historical_data():
    """Test historical data fetching."""
    print("\n" + "=" * 60)
    print("TEST 4: Historical Data (SPY, 6 months)")
    print("=" * 60)

    try:
        api = StockAPI()
        df = api.get_historical_data('SPY', period='6mo')

        if df is not None and not df.empty:
            print(f"  Records: {len(df)}")
            print(f"  Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            print(f"  Price Range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            print(f"  Columns: {list(df.columns)}")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No data returned")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_stock_universe():
    """Test stock universe configuration."""
    print("\n" + "=" * 60)
    print("TEST 5: Stock Universe Configuration")
    print("=" * 60)

    try:
        all_symbols = get_all_symbols()
        print(f"  Total symbols tracked: {len(all_symbols)}")
        print(f"  Categories: {list(STOCK_UNIVERSE.keys())}")

        for category, stocks in STOCK_UNIVERSE.items():
            print(f"  {category}: {len(stocks)} stocks - {list(stocks.keys())[:5]}...")

        # Test get_stock_info
        info = get_stock_info('AAPL')
        if info:
            print(f"\n  AAPL info: {info}")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No info for AAPL")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_market_status():
    """Test market status check."""
    print("\n" + "=" * 60)
    print("TEST 6: Market Status")
    print("=" * 60)

    try:
        api = StockAPI()
        status = api.get_market_status()

        if status:
            print(f"  Market Open: {status['is_open']}")
            print(f"  Timestamp: {status['timestamp']}")
            print("  PASSED")
            return True
        else:
            print("  FAILED: No status returned")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("STOCK API TEST SUITE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Data Source: Yahoo Finance (yfinance)")
    print("No API key required\n")

    results = []
    results.append(("Current Price", test_current_price()))
    results.append(("Quote Data", test_quote()))
    results.append(("Batch Quotes", test_batch_quotes()))
    results.append(("Historical Data", test_historical_data()))
    results.append(("Stock Universe", test_stock_universe()))
    results.append(("Market Status", test_market_status()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"   {test_name:<25} {status}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n   Total: {passed_count}/{total_count} tests passed")
    print("=" * 60)

    if passed_count == total_count:
        print("\nAll tests passed! Stock API is working correctly.")
    else:
        print("\nSome tests failed. Check yfinance installation: pip install yfinance")
        sys.exit(1)


if __name__ == "__main__":
    main()
