"""
🌙 Test TradingView Integration
Demonstrates using TradingView as a data source
"""

import os
import sys

# Force TradingView as data source
os.environ['DATA_SOURCE'] = 'tradingview'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.api_adapter import APIAdapter
from src.agents.unified_data_api import UnifiedDataAPI

def test_api_adapter():
    """Test APIAdapter with TradingView"""
    print("🌙 Testing APIAdapter with TradingView")
    print("="*60)
    
    # Create adapter (will use TradingView due to env var)
    api = APIAdapter()
    
    # Test liquidation data
    print("\n📊 Test 1: Liquidation Data")
    liq_data = api.get_liquidation_data(limit=5)
    if liq_data is not None and not liq_data.empty:
        print(f"✅ Got {len(liq_data)} liquidation estimates")
        print("Sample data:")
        print(liq_data.head(3))
    
    # Test funding rates
    print("\n📊 Test 2: Funding Rates")
    funding = api.get_funding_data()
    if funding is not None and not funding.empty:
        print(f"✅ Got funding rates for {len(funding)} markets")
        print("Top 3 by funding rate:")
        print(funding.nlargest(3, 'funding_rate')[['symbol', 'funding_rate', 'recommendation']])
    
    # Test OI data
    print("\n📊 Test 3: Open Interest")
    oi_data = api.get_oi_data()
    if oi_data is not None and not oi_data.empty:
        print(f"✅ Got OI estimates")
        print(oi_data)

def test_unified_api():
    """Test UnifiedDataAPI"""
    print("\n\n🌙 Testing Unified Data API")
    print("="*60)
    
    # Create unified API
    api = UnifiedDataAPI()
    
    # Show what's available
    print(f"Available sources: {api.get_available_sources()}")
    
    # Get technical analysis
    print("\n📊 Technical Analysis for Major Cryptos")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in symbols:
        ta = api.get_technical_analysis(symbol, '1h')
        if ta:
            print(f"\n{symbol}:")
            print(f"  Price: ${ta['indicators'].get('close', 0):,.2f}")
            print(f"  Recommendation: {ta['summary']['RECOMMENDATION']}")
            print(f"  RSI: {ta['indicators'].get('RSI', 'N/A')}")
            print(f"  MACD: {ta['indicators'].get('MACD.macd', 'N/A')}")
            print(f"  Volume: {ta['indicators'].get('volume', 'N/A'):,.0f}")

def test_multi_timeframe():
    """Test multiple timeframes"""
    print("\n\n🌙 Testing Multiple Timeframes")
    print("="*60)
    
    from src.agents.tradingview_api import TradingViewAPI
    
    tv = TradingViewAPI()
    symbol = 'BTCUSDT'
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    print(f"\n{symbol} Analysis Across Timeframes:")
    print("-" * 50)
    
    for tf in timeframes:
        rec = tv.get_recommendation(symbol, interval=tf)
        analysis = tv.get_analysis(symbol, interval=tf)
        if analysis:
            indicators = analysis['indicators']
            print(f"{tf:>5}: {rec:>12} | RSI: {indicators.get('RSI', 0):>6.2f} | "
                  f"Buy: {analysis['summary']['BUY']:>2} Sell: {analysis['summary']['SELL']:>2}")

def test_stock_market():
    """Test stock market data"""
    print("\n\n🌙 Testing Stock Market Data")
    print("="*60)
    
    from src.agents.tradingview_api import TradingViewAPI
    
    # Create stock market API
    tv = TradingViewAPI(screener="america", default_exchange="NASDAQ")
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("\nStock Market Analysis:")
    print("-" * 70)
    print(f"{'Symbol':<8} {'Price':>10} {'Rec':>12} {'RSI':>8} {'Volume':>15}")
    print("-" * 70)
    
    for symbol in stocks:
        analysis = tv.get_analysis(symbol, interval='1d')
        if analysis:
            ind = analysis['indicators']
            print(f"{symbol:<8} ${ind.get('close', 0):>9.2f} "
                  f"{analysis['summary']['RECOMMENDATION']:>12} "
                  f"{ind.get('RSI', 0):>8.2f} "
                  f"{ind.get('volume', 0):>15,.0f}")

if __name__ == "__main__":
    print("🌙 Moon Dev TradingView Integration Test")
    print("📊 Using TradingView for all market data")
    print("="*60)
    
    # Test 1: API Adapter
    test_api_adapter()
    
    # Test 2: Unified API
    test_unified_api()
    
    # Test 3: Multi-timeframe
    test_multi_timeframe()
    
    # Test 4: Stocks
    test_stock_market()
    
    print("\n\n✅ TradingView integration complete!")
    print("\n📝 To use TradingView as your data source:")
    print("1. Set environment variable: DATA_SOURCE=tradingview")
    print("2. Or use UnifiedDataAPI which auto-selects best source")
    print("3. TradingView provides:")
    print("   - Real-time technical analysis")
    print("   - 200+ indicators")
    print("   - Multi-timeframe analysis")
    print("   - Stocks, crypto, forex support")
    print("   - No API key required!")