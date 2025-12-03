"""
Test TSLA with basic TradingView API (no auth needed)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🌙 Testing TSLA with Basic TradingView API")
print("=" * 60)

try:
    from src.agents.tradingview_api import TradingViewAPI
    
    # Create API instance
    tv_api = TradingViewAPI(default_exchange='NASDAQ')
    
    # For stocks, we need to use direct TA_Handler
    from tradingview_ta import TA_Handler, Interval
    
    print("\n📊 Fetching TSLA current data...")
    handler = TA_Handler(
        symbol="TSLA",
        exchange="NASDAQ",
        screener="america",  # Use america screener for US stocks
        interval=Interval.INTERVAL_1_DAY
    )
    
    analysis = handler.get_analysis()
    
    if analysis:
        indicators = analysis.indicators
        summary = analysis.summary
        
        print("\n✅ TSLA Current Data:")
        print(f"Price: ${indicators.get('close', 0):.2f}")
        print(f"Open: ${indicators.get('open', 0):.2f}")
        print(f"High: ${indicators.get('high', 0):.2f}")
        print(f"Low: ${indicators.get('low', 0):.2f}")
        print(f"Volume: {indicators.get('volume', 0):,.0f}")
        
        # Technical indicators
        print(f"\n📈 Technical Indicators:")
        print(f"RSI: {indicators.get('RSI', 'N/A')}")
        print(f"MACD: {indicators.get('MACD.macd', 'N/A')}")
        print(f"Signal: {indicators.get('MACD.signal', 'N/A')}")
        
        # Recommendation
        print(f"\n🎯 Analysis:")
        print(f"Recommendation: {summary.get('RECOMMENDATION', 'N/A')}")
        print(f"Buy signals: {summary.get('BUY', 0)}")
        print(f"Sell signals: {summary.get('SELL', 0)}")
        print(f"Neutral signals: {summary.get('NEUTRAL', 0)}")
        
        # Moving averages
        print(f"\n📊 Moving Averages:")
        sma20 = indicators.get('SMA20')
        sma50 = indicators.get('SMA50')
        sma200 = indicators.get('SMA200')
        
        if sma20:
            print(f"SMA 20: ${sma20:.2f}")
        if sma50:
            print(f"SMA 50: ${sma50:.2f}")
        if sma200:
            print(f"SMA 200: ${sma200:.2f}")
        
        print("\n⚠️  Note: This is current data only. For historical 100-day data:")
        print("1. Start the TradingView server: cd tradingview-server && npm start")
        print("2. Run: python get_tsla_data.py")
        
    else:
        print("❌ No data received")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()