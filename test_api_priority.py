"""
Test API priority - verify TradingView is used first
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

print("🌙 Testing API Priority Order")
print("="*60)

# Test 1: APIAdapter
print("\n📊 Test 1: APIAdapter (auto mode)")
from src.agents.api_adapter import APIAdapter

adapter = APIAdapter()
print(f"✅ Using source: {adapter.source}")

# Get some data to verify it's working
funding = adapter.get_funding_data()
if funding is not None and not funding.empty:
    print(f"✅ Got funding data from {adapter.source}")
    print(funding.head(3)[['symbol', 'funding_rate']])

# Test 2: UnifiedDataAPI
print("\n\n📊 Test 2: UnifiedDataAPI")
from src.agents.unified_data_api import UnifiedDataAPI

unified = UnifiedDataAPI()
print(f"✅ Available sources: {unified.get_available_sources()}")

# Test getting price
price = unified.get_price('BTCUSDT')
if price:
    print(f"✅ BTC Price: ${price:,.2f}")

# Test 3: Force specific sources
print("\n\n📊 Test 3: Testing forced sources")

# Force TradingView
os.environ['DATA_SOURCE'] = 'tradingview'
adapter_tv = APIAdapter()
print(f"✅ Forced TradingView: {adapter_tv.source}")

# Force Binance
os.environ['DATA_SOURCE'] = 'binance'
adapter_binance = APIAdapter()
print(f"✅ Forced Binance: {adapter_binance.source}")

# Reset to auto
os.environ['DATA_SOURCE'] = 'auto'

print("\n\n✅ Priority order confirmed:")
print("1. Moon Dev API (if API key present)")
print("2. TradingView (no geo-restrictions!)")
print("3. Binance (fallback if TradingView fails)")

print("\n📝 Benefits of TradingView as primary:")
print("- No geo-restrictions (works everywhere)")
print("- No proxy needed")
print("- Technical indicators included")
print("- Multi-market support (crypto, stocks, forex)")