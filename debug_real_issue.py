"""
🐛 Debug the REAL issue
Why are symbols with plenty of data being marked as insufficient?
"""

from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester
import pandas as pd
from termcolor import colored

def debug_backtest_process(symbol):
    """Debug exactly what happens during backtest"""
    
    print(colored(f"\n🐛 DEBUGGING {symbol}", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Step 1: Create backtester
    print("Step 1: Creating backtester...")
    backtester = CachedComprehensiveBacktester(symbol)
    
    # Step 2: Download data
    print("\nStep 2: Downloading data...")
    df = backtester.download_maximum_data()
    
    if df is None:
        print("❌ df is None!")
        return
    
    print(f"✅ Downloaded {len(df)} data points")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 3: Check what would trigger "insufficient data"
    print("\nStep 3: Checking thresholds...")
    print(f"   len(df) < 500? {len(df) < 500}")
    print(f"   len(df) < 250? {len(df) < 250}")
    
    # Step 4: Try to run actual backtest
    print("\nStep 4: Running comprehensive_backtest...")
    try:
        results = backtester.comprehensive_backtest(df)
        print("✅ Backtest completed successfully!")
        print(f"   Strategy return: {results.strategy_total_return_pct:.2f}%")
        print(f"   Excess return: {results.excess_return_pct:.2f}%")
    except Exception as e:
        print(f"❌ Backtest failed with error: {str(e)}")
        print("\nFull error:")
        import traceback
        traceback.print_exc()
        
    # Step 5: Check data quality
    print("\nStep 5: Checking data quality...")
    print(f"   NaN values: {df.isna().sum().sum()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   First few rows:")
    print(df.head())

def main():
    # Test the problematic symbols
    problem_symbols = ['C', 'BAC', 'DASH']
    
    for symbol in problem_symbols:
        debug_backtest_process(symbol)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()