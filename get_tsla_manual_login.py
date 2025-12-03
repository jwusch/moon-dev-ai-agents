"""
Get TSLA data with manual login handling
"""

import pandas as pd
from datetime import datetime
import sys
import os
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🌙 Fetching TSLA Historical Data (Manual Login)")
print("=" * 60)

# Direct server communication
server_url = 'http://localhost:8888'

print(f"📡 Server URL: {server_url}")

# Check health
try:
    health = requests.get(f'{server_url}/health')
    print(f"✅ Server health: {health.status_code}")
except Exception as e:
    print(f"❌ Server not reachable: {e}")
    exit(1)

print("\n⚠️  Captcha Issue Detected!")
print("The TradingView login requires manual intervention.")
print("\nOptions:")
print("1. The Node.js server console should show a login prompt")
print("2. Check the server logs for manual login instructions")
print("3. You may need to restart the server with manual login mode")

print("\n📝 Alternative: Using current data only (no auth required)")

# Use basic API for current data
try:
    from tradingview_ta import TA_Handler, Interval
    
    print("\n📊 Fetching current TSLA data...")
    handler = TA_Handler(
        symbol="TSLA",
        exchange="NASDAQ", 
        screener="america",
        interval=Interval.INTERVAL_1_DAY
    )
    
    analysis = handler.get_analysis()
    
    # Create a DataFrame with current data
    current_data = {
        'time': [datetime.now()],
        'open': [analysis.indicators.get('open', 0)],
        'high': [analysis.indicators.get('high', 0)],
        'low': [analysis.indicators.get('low', 0)],
        'close': [analysis.indicators.get('close', 0)],
        'volume': [analysis.indicators.get('volume', 0)]
    }
    
    df = pd.DataFrame(current_data)
    
    print(f"\n✅ Current TSLA Data:")
    print(df)
    
    # Add indicators
    print(f"\n📊 Technical Indicators:")
    print(f"RSI: {analysis.indicators.get('RSI', 'N/A')}")
    print(f"MACD: {analysis.indicators.get('MACD.macd', 'N/A')}")
    print(f"Recommendation: {analysis.summary.get('RECOMMENDATION', 'N/A')}")
    
    # Moving averages
    print(f"\n📊 Moving Averages:")
    print(f"SMA 20: ${analysis.indicators.get('SMA20', 0):.2f}")
    print(f"SMA 50: ${analysis.indicators.get('SMA50', 0):.2f}")
    print(f"SMA 100: ${analysis.indicators.get('SMA100', 0):.2f}")
    print(f"SMA 200: ${analysis.indicators.get('SMA200', 0):.2f}")
    
    # Pivot points
    print(f"\n📊 Pivot Points:")
    for key, value in analysis.indicators.items():
        if 'Pivot' in key and value:
            print(f"{key}: ${value:.2f}")
    
    print("\n⚠️  Note: For 100-day historical data:")
    print("1. Resolve the captcha issue in the Node.js server")
    print("2. Or restart the server with: NODE_ENV=development npm start")
    print("3. This may allow manual login in the server console")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n📝 Summary:")
print("- TradingView server is running on port 8888")
print("- Authentication is blocked by captcha")
print("- Current data is available without authentication")
print("- Historical data requires resolving the captcha issue")