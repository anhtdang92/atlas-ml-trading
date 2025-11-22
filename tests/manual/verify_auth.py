
import sys
import os

# Add project root to path
# Add project root to path (2 levels up from tests/manual/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

try:
    from data.kraken_auth import KrakenAuthClient
    
    print("🔄 Attempting to connect to Kraken with configured keys...")
    client = KrakenAuthClient()
    
    # Try to fetch balance (requires valid keys)
    balance = client.get_account_balance()
    
    if balance is not None:
        print("✅ SUCCESS: Connected to Kraken!")
        print(f"   Found {len(balance)} assets in portfolio.")
        # Print a few assets to prove it works, but don't expose too much
        print("   Top assets found: " + ", ".join(list(balance.keys())[:3]))
    else:
        print("❌ FAILURE: Could not fetch balance. Check your API keys.")
        sys.exit(1)

except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
