"""
Test TradingView historical data retrieval
"""

import pandas as pd
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🌙 Testing TradingView Historical Data")
print("=" * 60)

# Test 1: Using tradingview-ta (no historical data)
print("\n📊 Test 1: tradingview-ta library")
print("Note: This library only provides CURRENT data, not historical")
from src.agents.tradingview_api import TradingViewAPI

tv_api = TradingViewAPI()
price_data = tv_api.get_price_data('BTCUSDT')
if price_data:
    print(f"Current BTC Price: ${price_data['close']:,.2f}")
    print("⚠️  Only current values available")

# Test 2: Using authenticated API (with historical data)
print("\n\n📊 Test 2: Authenticated TradingView API")
print("This requires the Node.js server to be running")

try:
    # First check if server is running
    import requests
    response = requests.get('http://localhost:5000')
    
    if response.status_code == 200:
        print("✅ TradingView server is running")
        
        # Create authenticated client
        from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI
        
        auth_api = TradingViewAuthenticatedAPI()
        
        # Get historical data
        print("\n🔍 Fetching 1-hour historical data for BTC...")
        hist_data = auth_api.get_historical_data(
            symbol='BTCUSDT',
            timeframe='60',  # 1 hour
            bars=50,  # Last 50 bars
            exchange='BINANCE'
        )
        
        if not hist_data.empty:
            print(f"✅ Got {len(hist_data)} historical bars")
            print("\nFirst 5 bars:")
            print(hist_data[['time', 'open', 'high', 'low', 'close', 'volume']].head())
            
            print("\nLast 5 bars:")
            print(hist_data[['time', 'open', 'high', 'low', 'close', 'volume']].tail())
            
            # Calculate some stats
            print("\n📊 Statistics:")
            print(f"Date Range: {hist_data['time'].min()} to {hist_data['time'].max()}")
            print(f"Price Range: ${hist_data['low'].min():,.2f} - ${hist_data['high'].max():,.2f}")
            print(f"Average Volume: {hist_data['volume'].mean():,.0f}")
            
            # Test different timeframes
            print("\n\n🔍 Testing different timeframes...")
            timeframes = {
                '5': '5 minute',
                '15': '15 minute',
                '240': '4 hour',
                '1D': 'Daily'
            }
            
            for tf, name in timeframes.items():
                try:
                    df = auth_api.get_historical_data('BTCUSDT', tf, bars=10)
                    if not df.empty:
                        print(f"✅ {name}: Got {len(df)} bars")
                except:
                    print(f"❌ {name}: Failed")
        else:
            print("❌ No historical data received")
            
        # Close connection
        auth_api.close()
        
except requests.ConnectionError:
    print("❌ TradingView server is not running!")
    print("📝 To start the server:")
    print("   cd tradingview-server")
    print("   npm start")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n\n📝 Summary:")
print("- tradingview-ta: Current data only (no authentication needed)")
print("- Authenticated API: Historical data available with multiple timeframes")
print("- Timeframes: 1, 5, 15, 30, 60 (minutes), 240 (4h), 1D, 1W, 1M")
print("- Max bars: Depends on your TradingView subscription level")