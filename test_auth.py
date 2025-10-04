#!/usr/bin/env python3
"""
Test Kraken API Authentication and Display Portfolio
"""

from data.kraken_auth import KrakenAuthClient
import json


def main():
    print("="*60)
    print("🔐 Testing Kraken API Authentication")
    print("="*60)
    
    try:
        # Initialize client
        print("\n📡 Connecting to Kraken...")
        client = KrakenAuthClient()
        
        # Test 1: Get account balance
        print("\n" + "="*60)
        print("💰 Fetching Account Balance...")
        print("="*60)
        
        balance = client.get_account_balance()
        
        if balance:
            print("✅ Successfully authenticated!\n")
            print("Your Holdings:")
            print("-" * 60)
            
            # Filter out zero balances
            holdings = {k: v for k, v in balance.items() if float(v) > 0}
            
            if holdings:
                for asset, amount in holdings.items():
                    # Clean up asset names (Kraken uses X/Z prefixes)
                    clean_asset = asset.replace('X', '').replace('Z', '')
                    if clean_asset.startswith('USD'):
                        clean_asset = 'USD'
                    
                    print(f"   {clean_asset:<10} {float(amount):>15,.8f}")
            else:
                print("   No holdings found (account may be empty)")
                
        else:
            print("❌ Failed to fetch balance")
            print("\nPossible issues:")
            print("   - API keys are incorrect")
            print("   - API key doesn't have 'Query Funds' permission")
            print("   - Network connection issue")
            return
        
        # Test 2: Get trade balance (USD equivalent)
        print("\n" + "="*60)
        print("📊 Fetching Trade Balance...")
        print("="*60)
        
        trade_balance = client.get_trade_balance()
        
        if trade_balance:
            print("✅ Trade balance retrieved!\n")
            print(f"   Equivalent Balance (USD): ${float(trade_balance.get('eb', 0)):,.2f}")
            print(f"   Trade Balance (USD):      ${float(trade_balance.get('tb', 0)):,.2f}")
            print(f"   Margin Amount:            ${float(trade_balance.get('m', 0)):,.2f}")
            print(f"   Unrealized P&L:           ${float(trade_balance.get('n', 0)):,.2f}")
            print(f"   Cost Basis:               ${float(trade_balance.get('c', 0)):,.2f}")
            print(f"   Current Valuation:        ${float(trade_balance.get('v', 0)):,.2f}")
            print(f"   Equity:                   ${float(trade_balance.get('e', 0)):,.2f}")
            print(f"   Free Margin:              ${float(trade_balance.get('mf', 0)):,.2f}")
        
        # Test 3: Get open orders
        print("\n" + "="*60)
        print("📋 Checking Open Orders...")
        print("="*60)
        
        open_orders = client.get_open_orders()
        
        if open_orders:
            if open_orders.get('open'):
                print(f"✅ You have {len(open_orders['open'])} open order(s)")
            else:
                print("✅ No open orders")
        
        print("\n" + "="*60)
        print("🎉 Authentication Successful!")
        print("="*60)
        print("\n✅ Your API keys are working correctly!")
        print("✅ You can now view your real portfolio in the dashboard")
        print("\n💡 Next: Restart the Streamlit dashboard to see your real portfolio")
        
    except FileNotFoundError:
        print("\n❌ Error: config/secrets.yaml not found")
        print("   Please create it with your API keys")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("   1. API keys are correctly pasted in config/secrets.yaml")
        print("   2. API key has 'Query Funds' permission on Kraken")
        print("   3. No extra spaces or quotes in the keys")


if __name__ == "__main__":
    main()

