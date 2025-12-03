"""
Test TradingView initialization error
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing TradingView initialization...")

try:
    print("1. Importing TradingViewAdapter...")
    from src.agents.tradingview_adapter import TradingViewAdapter
    print("✅ Import successful")
    
    print("\n2. Creating TradingViewAdapter instance...")
    adapter = TradingViewAdapter()
    print("✅ Adapter created successfully")
    
    print("\n3. Testing get_price method...")
    price = adapter.get_price('BTCUSDT')
    print(f"✅ BTC Price: ${price:,.2f}" if price else "❌ Price is None")
    
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print(f"📍 Error type: {type(e).__name__}")
    
    import traceback
    print("\n📍 Full traceback:")
    traceback.print_exc()

print("\n4. Testing tradingview-ta directly...")
try:
    from tradingview_ta import TA_Handler, Interval, Exchange
    print("✅ tradingview-ta imports work")
    
    handler = TA_Handler(
        symbol="BTCUSDT",
        exchange="BINANCE",
        screener="crypto",
        interval=Interval.INTERVAL_1_HOUR,
        timeout=10
    )
    
    print("✅ TA_Handler created")
    
    analysis = handler.get_analysis()
    price = analysis.indicators.get('close')
    print(f"✅ Direct TradingView test: BTC = ${price:,.2f}")
    
except Exception as e:
    print(f"❌ Direct test failed: {e}")
    import traceback
    traceback.print_exc()