"""
🔍 Diagnose Cache Issues
Check why some symbols are getting insufficient data
"""

from aegs_permanent_cache import AEGSPermanentCache
from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester
import yfinance as yf
from termcolor import colored

def diagnose_symbol(symbol):
    """Diagnose data issues for a symbol"""
    
    print(colored(f"\n🔍 Diagnosing {symbol}", 'cyan'))
    print("-" * 40)
    
    # 1. Check direct yfinance
    print("1. Direct yfinance download:")
    ticker = yf.Ticker(symbol)
    df_direct = ticker.history(period="max", interval="1d")
    print(f"   Data points: {len(df_direct)}")
    if len(df_direct) > 0:
        print(f"   Date range: {df_direct.index[0].date()} to {df_direct.index[-1].date()}")
    
    # 2. Check cache
    print("\n2. AEGS Cache:")
    cache = AEGSPermanentCache()
    cache_key = f"{symbol}_max_1d"
    
    if cache_key in cache.metadata:
        info = cache.metadata[cache_key]
        print(f"   Cached data points: {info['rows']}")
        print(f"   Cached on: {info['timestamp']}")
        print(f"   Date range: {info['start_date']} to {info['end_date']}")
    else:
        print("   Not in cache")
    
    # 3. Check what backtester gets
    print("\n3. Backtester download:")
    backtester = CachedComprehensiveBacktester(symbol)
    df_backtest = backtester.download_maximum_data()
    print(f"   Data points: {len(df_backtest) if df_backtest is not None else 0}")
    
    # 4. Recommendation
    print("\n4. Analysis:")
    if len(df_direct) > 500 and (df_backtest is None or len(df_backtest) < 500):
        print("   ⚠️  Cache might have stale/limited data!")
        print("   💡 Recommendation: Clear cache entry and re-download")
        
        # Option to fix
        if cache_key in cache.metadata:
            print(f"\n   To fix, run: python -c \"from aegs_permanent_cache import AEGSPermanentCache; c = AEGSPermanentCache(); c.clear_symbol_cache('{symbol}')\"")
    else:
        print("   ✅ Data looks good")

def main():
    """Diagnose problematic symbols"""
    
    print(colored("🔍 CACHE DIAGNOSTIC TOOL", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Symbols that were marked as insufficient
    problem_symbols = ['C', 'BAC', 'DASH', 'CENN', 'EVGO']
    
    for symbol in problem_symbols:
        diagnose_symbol(symbol)

if __name__ == "__main__":
    main()