"""
🧪 Test Data Download Issue
Figure out why some symbols are getting insufficient data
"""

import yfinance as yf
from datetime import datetime
from termcolor import colored

def test_symbol_data(symbol):
    """Test data download for a symbol"""
    print(f"\n🔍 Testing {symbol}...")
    
    ticker = yf.Ticker(symbol)
    
    # Test different periods
    periods = ['1mo', '3mo', '1y', '2y', '5y', 'max']
    
    for period in periods:
        try:
            df = ticker.history(period=period, interval='1d')
            if len(df) > 0:
                days = (df.index[-1] - df.index[0]).days
                print(f"   {period}: {len(df)} points over {days} days")
            else:
                print(f"   {period}: No data")
        except Exception as e:
            print(f"   {period}: Error - {str(e)[:50]}")
    
    # Check what comprehensive backtest actually gets
    try:
        print("\n   Testing 'max' period in detail...")
        df_max = ticker.history(period='max', interval='1d')
        if len(df_max) > 0:
            print(f"   First date: {df_max.index[0].date()}")
            print(f"   Last date: {df_max.index[-1].date()}")
            print(f"   Total points: {len(df_max)}")
            
            # Check for gaps
            date_diff = df_max.index.to_series().diff()
            gaps = date_diff[date_diff > pd.Timedelta(days=10)]
            if len(gaps) > 0:
                print(f"   ⚠️  Found {len(gaps)} gaps > 10 days")
        else:
            print("   ❌ No data returned for 'max' period")
            
    except Exception as e:
        print(f"   ❌ Error with max period: {e}")

def main():
    """Test problematic symbols"""
    
    print(colored("🧪 TESTING DATA DOWNLOAD ISSUES", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Test symbols that were marked as insufficient
    test_symbols = ['C', 'BAC', 'DASH', 'EVGO', 'CENN']
    
    for symbol in test_symbols:
        test_symbol_data(symbol)
    
    print(colored("\n📊 ANALYSIS:", 'yellow'))
    print("The issue might be:")
    print("1. Some symbols have corporate actions that split their history")
    print("2. The cache might be returning limited data")
    print("3. Yahoo Finance API might limit data for some requests")

if __name__ == "__main__":
    import pandas as pd
    main()