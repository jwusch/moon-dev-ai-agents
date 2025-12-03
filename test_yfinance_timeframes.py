"""
🔍 Test YFinance Intraday Data Capabilities
Check what timeframes are available for PEP

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("🔍 Testing YFinance Intraday Data Capabilities\n")
print("="*70)

# Test different intervals
pep = yf.Ticker("PEP")

intervals = {
    "1m": "1 minute",
    "2m": "2 minutes", 
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "60m": "60 minutes",
    "90m": "90 minutes",
    "1h": "1 hour",
    "1d": "1 day"
}

print("Available data for each interval:\n")

for interval, description in intervals.items():
    try:
        # YFinance limits for intraday data:
        # 1m: max 7 days
        # 5m: max 60 days  
        # 15m: max 60 days
        # 30m: max 60 days
        # 60m/1h: max 730 days
        
        if interval == "1m":
            period = "7d"
        elif interval in ["2m", "5m", "15m", "30m"]:
            period = "60d"
        elif interval in ["60m", "90m", "1h"]:
            period = "730d"
        else:
            period = "max"
            
        df = pep.history(period=period, interval=interval)
        
        if len(df) > 0:
            print(f"✅ {interval} ({description}):")
            print(f"   Available days: {len(df)} bars")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Sample data: Open=${df['Open'].iloc[-1]:.2f}, Close=${df['Close'].iloc[-1]:.2f}")
            
            # Check if we have volume data
            if df['Volume'].sum() > 0:
                print(f"   Volume data: ✅ Available")
            else:
                print(f"   Volume data: ❌ Not available")
        else:
            print(f"❌ {interval}: No data available")
            
    except Exception as e:
        print(f"❌ {interval}: Error - {str(e)}")
        
    print()

# Test optimal approach for backtesting
print("="*70)
print("📊 RECOMMENDED APPROACH FOR INTRADAY CAMARILLA\n")

print("""
Based on YFinance limitations:

1. **15-minute bars** (Best for backtesting):
   - 60 days of data available
   - ~1,920 bars (32 bars/day × 60 days)
   - Good balance of data quantity and granularity
   - Camarilla levels update every 32 bars

2. **5-minute bars** (More signals):
   - 60 days of data available  
   - ~5,760 bars (96 bars/day × 60 days)
   - More trading opportunities
   - Higher transaction costs

3. **1-minute bars** (Scalping):
   - Only 7 days available
   - ~2,730 bars (390 bars/day × 7 days)
   - Too limited for proper backtesting
   - Best for live trading only

Recommendation: Use 15-minute bars for backtesting
""")

# Show how to calculate daily Camarilla from intraday
print("="*70)
print("💡 INTRADAY CAMARILLA CALCULATION\n")

# Get 15m data
df_15m = pep.history(period="5d", interval="15m")

if len(df_15m) > 0:
    # Group by day
    df_15m['Date'] = df_15m.index.date
    
    # Calculate daily OHLC from intraday
    daily_stats = df_15m.groupby('Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    print("Daily OHLC from 15-minute data:")
    print(daily_stats.tail())
    
    # Show how Camarilla levels would update
    print("\n📍 Camarilla levels would update:")
    print("   - Every day at market open (new daily levels)")
    print("   - Intraday trades use same day's levels")
    print("   - VWAP resets each day")
    print("   - More trading opportunities per day")