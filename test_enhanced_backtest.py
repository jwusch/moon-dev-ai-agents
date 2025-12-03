"""
🧪 Test Enhanced Backtester
Test with symbols that previously failed
"""

from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

def test_symbol(symbol):
    """Test enhanced backtester with a symbol"""
    
    print(colored(f"\n🧪 Testing {symbol} with Enhanced Multi-Period Discovery", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    try:
        # Create backtester
        backtester = CachedComprehensiveBacktester(symbol, use_cache=True)
        
        # Download data
        df = backtester.download_maximum_data()
        
        if df is None or len(df) == 0:
            print(f"❌ No data available for {symbol}")
            return
        
        print(f"✅ Data loaded: {len(df)} points")
        
        # Run comprehensive backtest
        results = backtester.comprehensive_backtest(df)
        
        print(colored(f"\n✅ SUCCESS! {symbol} backtested with enhanced method", 'green'))
        print(f"📊 Results:")
        print(f"   Strategy Return: {results.strategy_total_return_pct:.2f}%")
        print(f"   Buy & Hold Return: {results.buy_hold_total_return_pct:.2f}%")
        print(f"   Excess Return: {results.excess_return_pct:.2f}%")
        print(f"   Win Rate: {results.win_rate:.1f}%")
        print(f"   Sharpe Ratio: {results.strategy_sharpe:.2f}")
        
    except Exception as e:
        print(colored(f"❌ Failed: {str(e)}", 'red'))
        if "No profitable alpha sources" in str(e):
            print("   Even with multiple periods, no profitable patterns found")
            print("   This stock may be genuinely unsuitable for mean reversion")

def main():
    # Test symbols that previously failed
    test_symbols = ['C', 'BAC', 'DASH']
    
    print(colored("🧪 TESTING ENHANCED MULTI-PERIOD BACKTESTER", 'cyan', attrs=['bold']))
    print("This version tests multiple time periods to find alpha sources:")
    print("- First 2 years")
    print("- Years 3-5")
    print("- Recent 2 years")
    print("- Mid-period")
    print("- Full history")
    print("- Relaxed criteria as fallback")
    
    for symbol in test_symbols:
        test_symbol(symbol)

if __name__ == "__main__":
    main()