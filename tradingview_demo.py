"""
🌙 TradingView API Demo
Shows how to use TradingView as a data source
"""

from src.agents.tradingview_api import TradingViewAPI
import time

def demo_basic():
    """Basic TradingView functionality"""
    print("🌙 TradingView API Demo")
    print("="*60)
    
    # Initialize API
    tv = TradingViewAPI()
    
    # Single symbol analysis
    print("\n📊 1. Single Symbol Analysis - BTC")
    analysis = tv.get_analysis('BTCUSDT', interval='1h')
    if analysis:
        print(f"Symbol: {analysis['symbol']}")
        print(f"Price: ${analysis['indicators'].get('close', 0):,.2f}")
        print(f"Recommendation: {analysis['summary']['RECOMMENDATION']}")
        print(f"Buy Signals: {analysis['summary']['BUY']}")
        print(f"Sell Signals: {analysis['summary']['SELL']}")
        print(f"Neutral Signals: {analysis['summary']['NEUTRAL']}")
        
        # Some key indicators
        indicators = analysis['indicators']
        print(f"\nKey Indicators:")
        print(f"RSI: {indicators.get('RSI', 'N/A')}")
        print(f"MACD: {indicators.get('MACD.macd', 'N/A')}")
        print(f"ATR: {indicators.get('ATR', 'N/A')}")
        print(f"Volume: {indicators.get('volume', 'N/A'):,.0f}")
    
    # Wait to avoid rate limit
    time.sleep(2)
    
    # Different timeframe
    print("\n📊 2. Different Timeframe - ETH Daily")
    eth_daily = tv.get_analysis('ETHUSDT', interval='1d')
    if eth_daily:
        print(f"ETH Daily: {eth_daily['summary']['RECOMMENDATION']}")
        print(f"Price: ${eth_daily['indicators'].get('close', 0):,.2f}")
    
    time.sleep(2)
    
    # Stock market
    print("\n📊 3. Stock Market - AAPL")
    tv_stocks = TradingViewAPI(screener="america", default_exchange="NASDAQ")
    aapl = tv_stocks.get_analysis('AAPL', interval='1d')
    if aapl:
        print(f"AAPL: {aapl['summary']['RECOMMENDATION']}")
        print(f"Price: ${aapl['indicators'].get('close', 0):,.2f}")
        print(f"P/E Ratio: {aapl['indicators'].get('P.E', 'N/A')}")

def demo_indicators():
    """Show available indicators"""
    print("\n\n📊 Available Indicators Example")
    print("="*60)
    
    tv = TradingViewAPI()
    
    # Get all indicators for BTC
    indicators = tv.get_indicators('BTCUSDT')
    
    if indicators:
        print(f"\nTotal indicators available: {len(indicators)}")
        
        # Categorize indicators
        price_data = ['open', 'high', 'low', 'close', 'volume']
        moving_averages = [k for k in indicators.keys() if 'MA' in k or 'EMA' in k or 'SMA' in k]
        oscillators = ['RSI', 'MACD.macd', 'MACD.signal', 'Stoch.K', 'Stoch.D', 'CCI20', 'W%R']
        
        print("\n📈 Price Data:")
        for ind in price_data:
            if ind in indicators:
                print(f"  {ind}: {indicators[ind]}")
        
        print("\n📊 Moving Averages:")
        for ind in moving_averages[:5]:  # Show first 5
            print(f"  {ind}: {indicators.get(ind, 'N/A')}")
        
        print("\n📉 Oscillators:")
        for ind in oscillators:
            if ind in indicators:
                print(f"  {ind}: {indicators.get(ind, 'N/A')}")

def demo_recommendations():
    """Show recommendation logic"""
    print("\n\n📊 Multi-Timeframe Recommendations")
    print("="*60)
    
    tv = TradingViewAPI()
    symbol = 'BTCUSDT'
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    print(f"\n{symbol} Recommendations Across Timeframes:")
    print("-" * 40)
    
    for tf in timeframes:
        try:
            rec = tv.get_recommendation(symbol, interval=tf)
            print(f"{tf:>5}: {rec if rec else 'N/A'}")
            time.sleep(1.5)  # Rate limit
        except Exception as e:
            print(f"{tf:>5}: Error - {str(e)[:30]}...")

if __name__ == "__main__":
    print("🌙 Moon Dev TradingView Integration Demo")
    print("📊 Free alternative to paid data APIs")
    print("="*60)
    
    # Run demos
    demo_basic()
    demo_indicators()
    demo_recommendations()
    
    print("\n\n✅ TradingView Integration Summary:")
    print("1. No API key required")
    print("2. 200+ technical indicators")
    print("3. Multi-timeframe analysis")
    print("4. Stocks, crypto, forex support")
    print("5. Buy/Sell recommendations")
    print("6. Real-time data")
    print("\n⚠️ Note: Respect rate limits (1 request/second)")
    print("\n💡 Use cases:")
    print("- Technical analysis signals")
    print("- Market screening")
    print("- Indicator-based strategies")
    print("- Multi-asset monitoring")