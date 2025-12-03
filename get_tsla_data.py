"""
🚀 Get TSLA OHLCV Data using YFinance
Reliable data without authentication hassles!

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

from agents.yfinance_adapter import YFinanceAdapter
import pandas as pd
from datetime import datetime, timedelta
import os

def get_tsla_data():
    """Get TSLA historical data for the past 6 months using YFinance"""
    
    print("🚀 Getting TSLA 6-Month OHLCV Data")
    print("=" * 50)
    
    # Initialize YFinance adapter
    yf = YFinanceAdapter()
    print("✅ Using YFinance - no authentication needed!")
    
    # Get 6 months of daily data
    symbol = 'TSLA'
    bars = 180  # ~6 months of trading days
    
    print(f"\n📡 Requesting {bars} days of {symbol} data...")
    
    try:
        # Get OHLCV data
        data = yf.get_ohlcv_data(symbol, '1d', bars)
        
        if data.empty:
            print("❌ No data retrieved")
            return None
            
        print(f"📊 Bars received: {len(data)}")
        print(f"✅ Symbol: {symbol}")
        print(f"📈 Timeframe: Daily (1D)")
        
        # Show first 5 and last 5 records
        print("\n📋 TSLA Data Sample:")
        print("-" * 80)
        
        print("First 5 records:")
        for idx, row in data.head(5).iterrows():
            date_str = row['timestamp'].strftime('%Y-%m-%d')
            print(f"  {date_str}: O=${row['open']:.2f} H=${row['high']:.2f} L=${row['low']:.2f} C=${row['close']:.2f} V={row['volume']:,.0f}")
        
        if len(data) > 5:
            print("  ...")
            print("Last 5 records:")
            for idx, row in data.tail(5).iterrows():
                date_str = row['timestamp'].strftime('%Y-%m-%d')
                print(f"  {date_str}: O=${row['open']:.2f} H=${row['high']:.2f} L=${row['low']:.2f} C=${row['close']:.2f} V={row['volume']:,.0f}")
        
        # Summary stats
        print(f"\n📊 6-Month Summary:")
        print(f"   Current Price: ${data['close'].iloc[-1]:.2f}")
        print(f"   6-Month High: ${data['high'].max():.2f}")
        print(f"   6-Month Low: ${data['low'].min():.2f}")
        
        # Calculate return
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        print(f"   6-Month Return: {change_pct:+.1f}%")
        
        # Additional YFinance benefits
        print(f"\n✨ YFinance Benefits:")
        print(f"   • No session tokens needed")
        print(f"   • Always works - no expiration")
        print(f"   • Free and unlimited")
        
        # Save to CSV for convenience
        filename = f"TSLA_6months_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Format for backtesting
        save_data = pd.DataFrame({
            'Date': data['timestamp'],
            'Open': data['open'],
            'High': data['high'],
            'Low': data['low'],
            'Close': data['close'],
            'Volume': data['volume']
        })
        save_data.set_index('Date', inplace=True)
        
        save_data.to_csv(filename)
        print(f"\n💾 Data saved to: {filename}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error getting data: {e}")
        return None

def compare_with_session_server():
    """Optional: Compare YFinance with session server if running"""
    print("\n\n📊 Comparison with TradingView Session Server")
    print("=" * 50)
    
    try:
        import requests
        response = requests.get("http://localhost:8891/health", timeout=1)
        if response.status_code == 200:
            print("✅ Session server is running")
            print("❌ But we don't need it anymore!")
            print("🚀 YFinance is more reliable and doesn't expire")
    except:
        print("✅ Session server not running - and that's OK!")
        print("🚀 YFinance works without any server")

if __name__ == "__main__":
    # Get TSLA data using YFinance
    data = get_tsla_data()
    
    # Optional comparison
    compare_with_session_server()
    
    if data is not None:
        print("\n✅ Successfully retrieved TSLA data with YFinance!")
    else:
        print("\n❌ Failed to retrieve data")