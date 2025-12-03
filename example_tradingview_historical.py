"""
🌙 Example: Using TradingView for Historical Data
Shows how to retrieve historical OHLCV data from TradingView
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🌙 TradingView Historical Data Example")
print("=" * 60)

# Make sure the TradingView server is running first
print("\n📝 Prerequisites:")
print("1. TradingView server must be running:")
print("   cd tradingview-server && npm start")
print("2. Credentials in .env file")
print("\n" + "=" * 60)

try:
    # Check if server is running
    import requests
    response = requests.get('http://localhost:5000')
    if response.status_code != 200:
        raise Exception("Server not running")
    
    print("\n✅ TradingView server is running")
    
    # Method 1: Using the adapter (recommended for compatibility)
    print("\n📊 Method 1: Using TradingView Adapter")
    from src.agents.tradingview_adapter import TradingViewAdapter
    
    adapter = TradingViewAdapter()
    
    # Get historical data
    symbol = 'BTCUSDT'
    interval = '1h'
    bars = 50
    
    print(f"\n🔍 Fetching {bars} {interval} bars for {symbol}...")
    df = adapter.get_ohlcv_data(symbol, interval, bars)
    
    if not df.empty:
        print(f"✅ Got {len(df)} bars of data")
        print("\nFirst 5 bars:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head())
        
        # Calculate some technical indicators
        print("\n📊 Simple Analysis:")
        print(f"Current Price: ${df.iloc[-1]['close']:,.2f}")
        print(f"24h Change: ${df.iloc[-1]['close'] - df.iloc[-24]['close']:,.2f} ({((df.iloc[-1]['close'] / df.iloc[-24]['close']) - 1) * 100:.2f}%)")
        print(f"High (50 bars): ${df['high'].max():,.2f}")
        print(f"Low (50 bars): ${df['low'].min():,.2f}")
        print(f"Avg Volume: {df['volume'].mean():,.0f}")
        
        if 'rsi' in df.columns and pd.notna(df.iloc[-1]['rsi']):
            print(f"RSI: {df.iloc[-1]['rsi']:.2f}")
        if 'recommendation' in df.columns:
            print(f"Recommendation: {df.iloc[-1]['recommendation']}")
    
    # Method 2: Direct API usage (more control)
    print("\n\n📊 Method 2: Direct Authenticated API")
    from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI
    
    auth_api = TradingViewAuthenticatedAPI()
    
    # Test different timeframes
    timeframes = {
        '5': '5 minute',
        '15': '15 minute',
        '60': '1 hour',
        '240': '4 hour',
        '1D': 'Daily'
    }
    
    print("\n🔍 Testing multiple timeframes:")
    for tf, name in timeframes.items():
        try:
            df = auth_api.get_historical_data('ETHUSDT', tf, bars=20)
            if not df.empty:
                latest = df.iloc[-1]
                print(f"✅ {name}: ${latest['close']:,.2f} ({len(df)} bars)")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    # Example: Building a simple strategy with historical data
    print("\n\n📊 Example Strategy: Simple Moving Average")
    
    # Get more historical data
    df = auth_api.get_historical_data('BTCUSDT', '60', bars=200)
    
    if not df.empty and len(df) >= 50:
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        print(f"\nBTC/USDT Analysis:")
        print(f"Price: ${latest['close']:,.2f}")
        print(f"SMA 20: ${latest['sma_20']:,.2f}")
        print(f"SMA 50: ${latest['sma_50']:,.2f}")
        
        # Simple signal
        if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
            print("📈 Signal: Golden Cross (Bullish)")
        elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
            print("📉 Signal: Death Cross (Bearish)")
        elif latest['sma_20'] > latest['sma_50']:
            print("📊 Signal: Bullish Trend")
        else:
            print("📊 Signal: Bearish Trend")
    
    # Clean up
    auth_api.close()
    
except requests.ConnectionError:
    print("\n❌ ERROR: TradingView server is not running!")
    print("Please start the server first:")
    print("  cd tradingview-server")
    print("  npm start")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n\n📝 Summary:")
print("✅ TradingView provides real historical OHLCV data")
print("✅ Multiple timeframes supported (1m to 1M)")
print("✅ Works globally without geo-restrictions")
print("✅ Authenticated access provides better rate limits")
print("✅ Can be used for backtesting strategies")