"""
🚀 Explore yfinance Data Capabilities
See all available timeframes and data types

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def explore_timeframes():
    """Show all available timeframes in yfinance"""
    print("📊 yfinance Available Timeframes")
    print("=" * 50)
    
    # Valid intervals
    intervals = {
        "Intraday": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
        "Daily+": ["1d", "5d", "1wk", "1mo", "3mo"]
    }
    
    print("\n🕐 INTRADAY INTERVALS:")
    for interval in intervals["Intraday"]:
        print(f"   • {interval:<4} - ", end="")
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            print("Max 7 days of data")
        elif interval in ["60m", "90m", "1h"]:
            print("Max 60 days of data")
    
    print("\n📅 DAILY+ INTERVALS:")
    for interval in intervals["Daily+"]:
        print(f"   • {interval:<4} - Unlimited historical data")
    
    return intervals

def test_intraday_data(symbol="TSLA"):
    """Test fetching intraday data at different intervals"""
    print(f"\n🧪 Testing Intraday Data for {symbol}")
    print("=" * 50)
    
    ticker = yf.Ticker(symbol)
    
    # Test different intraday intervals
    test_intervals = ["1m", "5m", "15m", "1h"]
    
    for interval in test_intervals:
        try:
            # Intraday data limitations:
            # 1m, 2m, 5m, 15m, 30m = max 7 days
            # 60m, 90m, 1h = max 60 days
            
            if interval in ["1m", "5m", "15m"]:
                period = "5d"  # Last 5 days
            else:
                period = "1mo"  # Last month
            
            print(f"\n📈 {interval} interval (last {period}):")
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                print(f"   • Bars retrieved: {len(data)}")
                print(f"   • First: {data.index[0]}")
                print(f"   • Last: {data.index[-1]}")
                print(f"   • Latest close: ${data['Close'].iloc[-1]:.2f}")
                
                # Show sample data
                print(f"\n   Sample {interval} bars:")
                sample = data.tail(3)
                for idx, row in sample.iterrows():
                    print(f"   {idx}: O=${row['Open']:.2f} H=${row['High']:.2f} L=${row['Low']:.2f} C=${row['Close']:.2f} V={row['Volume']:,.0f}")
            else:
                print("   ❌ No data retrieved")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def show_data_types(symbol="TSLA"):
    """Show all types of data available from yfinance"""
    print(f"\n🎯 All Data Types Available for {symbol}")
    print("=" * 50)
    
    ticker = yf.Ticker(symbol)
    
    print("\n📊 PRICE DATA:")
    print("   • history() - OHLCV data at any interval")
    print("   • info - Current price, market cap, 52wk high/low, etc")
    
    print("\n📈 FUNDAMENTAL DATA:")
    fundamentals = {
        "financials": "Income statement (annual)",
        "quarterly_financials": "Income statement (quarterly)", 
        "balance_sheet": "Balance sheet (annual)",
        "quarterly_balance_sheet": "Balance sheet (quarterly)",
        "cashflow": "Cash flow (annual)",
        "quarterly_cashflow": "Cash flow (quarterly)",
        "earnings": "Earnings history",
        "quarterly_earnings": "Quarterly earnings",
        "recommendations": "Analyst recommendations",
        "dividends": "Dividend history",
        "splits": "Stock split history"
    }
    
    for attr, desc in fundamentals.items():
        try:
            data = getattr(ticker, attr)
            if data is not None and (isinstance(data, pd.DataFrame) and not data.empty or isinstance(data, pd.Series) and len(data) > 0):
                print(f"   ✓ {attr:<25} - {desc}")
            else:
                print(f"   ✗ {attr:<25} - No data")
        except:
            print(f"   ✗ {attr:<25} - Not available")
    
    print("\n📰 OTHER DATA:")
    other_data = {
        "news": "Recent news articles",
        "options": "Option chain data",
        "institutional_holders": "Institutional ownership",
        "major_holders": "Major shareholders",
        "calendar": "Earnings dates, ex-dividend dates",
        "isin": "International Securities ID",
        "sustainability": "ESG scores"
    }
    
    for attr, desc in other_data.items():
        try:
            data = getattr(ticker, attr)
            if data is not None:
                print(f"   ✓ {attr:<25} - {desc}")
        except:
            print(f"   ✗ {attr:<25} - Not available")

def get_multiple_symbols():
    """Download data for multiple symbols efficiently"""
    print("\n🚀 Bulk Download Multiple Symbols")
    print("=" * 50)
    
    symbols = ["TSLA", "AAPL", "NVDA", "SPY", "BTC-USD", "ETH-USD"]
    
    print(f"📊 Downloading 5m data for: {', '.join(symbols)}")
    
    # Bulk download - much faster than individual calls
    data = yf.download(
        tickers=symbols,
        period="1d",
        interval="5m",
        group_by="ticker",
        progress=False
    )
    
    print(f"\n✅ Downloaded data shape: {data.shape}")
    
    # Show latest price for each
    print("\n💰 Latest 5m close prices:")
    for symbol in symbols:
        try:
            if len(symbols) > 1:
                latest_close = data[symbol]['Close'].iloc[-1]
            else:
                latest_close = data['Close'].iloc[-1]
            print(f"   {symbol:<10} ${latest_close:.2f}")
        except:
            print(f"   {symbol:<10} N/A")

def show_limitations():
    """Show yfinance limitations"""
    print("\n⚠️ yfinance Limitations")
    print("=" * 50)
    
    print("\n🕐 INTRADAY DATA LIMITS:")
    print("   • 1m data: Max 7 days back")
    print("   • 5m data: Max 60 days back")  
    print("   • 15m data: Max 60 days back")
    print("   • 1h data: Max 730 days (2 years)")
    
    print("\n🌍 MARKET COVERAGE:")
    print("   • US Stocks: ✅ Excellent")
    print("   • Crypto: ✅ Good (via Yahoo Finance)")
    print("   • Forex: ✅ Major pairs")
    print("   • International: ⚠️ Limited to what Yahoo has")
    
    print("\n⚡ RATE LIMITS:")
    print("   • No official rate limit")
    print("   • But be respectful (~1-2 requests/second)")
    print("   • Bulk download is more efficient")
    
    print("\n💡 TIPS:")
    print("   • Use bulk download for multiple symbols")
    print("   • Cache/save data to avoid re-downloading")
    print("   • For real-time data, use other providers")

def main():
    # Explore all capabilities
    intervals = explore_timeframes()
    test_intraday_data("TSLA")
    show_data_types("TSLA")
    get_multiple_symbols()
    show_limitations()
    
    print("\n✅ yfinance is perfect for:")
    print("   • Backtesting strategies")
    print("   • Getting free historical data")
    print("   • Bulk symbol analysis")
    print("   • Quick prototyping")
    
    print("\n💡 Next step: Want to see a specific example?")

if __name__ == "__main__":
    main()