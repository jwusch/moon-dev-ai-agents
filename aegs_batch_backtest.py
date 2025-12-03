"""
🔥💎 AEGS BATCH BACKTEST - Test Multiple Symbols 💎🔥
Test a list of symbols and automatically build your goldmine registry

Usage: 
1. Edit the SYMBOLS_TO_TEST list below
2. Run: python aegs_batch_backtest.py
"""

from run_aegs_backtest import run_aegs_backtest
from termcolor import colored
import time
import json
from datetime import datetime

# EDIT THIS LIST WITH SYMBOLS YOU WANT TO TEST
SYMBOLS_TO_TEST = [
    # Example symbols - replace with your own
    ("NVDA", "Tech Stock"),
    ("TSLA", "EV Stock"),
    ("PLTR", "Tech Stock"),
    ("HOOD", "Fintech"),
    ("DOGE-USD", "Cryptocurrency"),
    ("SHIB-USD", "Cryptocurrency"),
    # Add more symbols here...
]

def batch_backtest():
    """Run AEGS backtest on multiple symbols"""
    
    print(colored("🔥💎 AEGS BATCH BACKTEST SYSTEM 💎🔥", 'cyan', attrs=['bold']))
    print("=" * 80)
    print(f"Testing {len(SYMBOLS_TO_TEST)} symbols...")
    print("=" * 80)
    
    results_summary = []
    goldmines_found = []
    high_potential_found = []
    
    for i, (symbol, category) in enumerate(SYMBOLS_TO_TEST, 1):
        print(colored(f"\n\n📊 TESTING {i}/{len(SYMBOLS_TO_TEST)}: {symbol}", 'yellow', attrs=['bold']))
        print("-" * 60)
        
        try:
            # Run backtest
            results = run_aegs_backtest(symbol, category)
            
            if results:
                summary = {
                    'symbol': symbol,
                    'category': category,
                    'excess_return': results.excess_return_pct,
                    'strategy_return': results.strategy_total_return_pct,
                    'win_rate': results.win_rate,
                    'trades': results.total_trades,
                    'sharpe': results.strategy_sharpe
                }
                
                results_summary.append(summary)
                
                # Track goldmines
                if results.excess_return_pct > 1000:
                    goldmines_found.append((symbol, results.excess_return_pct))
                elif results.excess_return_pct > 100:
                    high_potential_found.append((symbol, results.excess_return_pct))
            
            # Brief pause between tests
            time.sleep(2)
            
        except Exception as e:
            print(colored(f"❌ Error testing {symbol}: {str(e)}", 'red'))
            continue
    
    # Final summary
    print("\n\n" + "=" * 80)
    print(colored("🏆 BATCH BACKTEST COMPLETE - SUMMARY", 'yellow', attrs=['bold']))
    print("=" * 80)
    
    # Sort by excess return
    results_summary.sort(key=lambda x: x['excess_return'], reverse=True)
    
    # Display results table
    print(f"\n{'Symbol':<12} {'Category':<20} {'Excess %':>10} {'Win Rate':>10} {'Trades':>8} {'Sharpe':>8}")
    print("-" * 78)
    
    for r in results_summary:
        excess = r['excess_return']
        
        # Color based on performance
        if excess > 1000:
            color = 'red'
            marker = '💎'
        elif excess > 100:
            color = 'yellow'
            marker = '🚀'
        elif excess > 0:
            color = 'green'
            marker = '✅'
        else:
            color = 'white'
            marker = '❌'
        
        row = f"{r['symbol']:<12} {r['category']:<20} {excess:>9.0f}% {r['win_rate']:>9.1f}% {r['trades']:>8} {r['sharpe']:>8.2f} {marker}"
        print(colored(row, color))
    
    # Goldmine summary
    if goldmines_found:
        print(colored(f"\n\n🔥💎 GOLDMINES DISCOVERED ({len(goldmines_found)}):", 'red', attrs=['bold']))
        for symbol, excess in goldmines_found:
            print(colored(f"   {symbol}: {excess:+.0f}% excess return", 'red'))
    
    if high_potential_found:
        print(colored(f"\n🚀 HIGH POTENTIAL FOUND ({len(high_potential_found)}):", 'yellow', attrs=['bold']))
        for symbol, excess in high_potential_found:
            print(colored(f"   {symbol}: {excess:+.0f}% excess return", 'yellow'))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'aegs_batch_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'test_date': timestamp,
            'symbols_tested': len(SYMBOLS_TO_TEST),
            'goldmines_found': len(goldmines_found),
            'high_potential_found': len(high_potential_found),
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {filename}")
    
    # Final recommendation
    total_positive = sum(1 for r in results_summary if r['excess_return'] > 0)
    success_rate = total_positive / len(results_summary) * 100 if results_summary else 0
    
    print(f"\n📊 SUCCESS RATE: {success_rate:.1f}% ({total_positive}/{len(results_summary)})")
    
    if goldmines_found:
        print(colored("\n🎯 IMMEDIATE ACTION: Deploy capital on goldmine pullbacks!", 'red', attrs=['bold']))
    elif high_potential_found:
        print(colored("\n🎯 ACTION: Add high potential symbols to watch list!", 'yellow'))
    else:
        print(colored("\n💡 Continue searching for better opportunities", 'blue'))

def quick_test_list():
    """Quick test of popular symbols"""
    
    popular_symbols = [
        # Top tech stocks
        ("AAPL", "Tech Stock"),
        ("MSFT", "Tech Stock"),
        ("GOOGL", "Tech Stock"),
        ("AMZN", "Tech Stock"),
        ("META", "Tech Stock"),
        
        # Popular volatility plays
        ("NVDA", "Tech Stock"),
        ("TSLA", "EV Stock"),
        ("AMD", "Tech Stock"),
        ("PLTR", "Tech Stock"),
        ("SOFI", "Fintech"),
        
        # Crypto related
        ("DOGE-USD", "Cryptocurrency"),
        ("SHIB-USD", "Cryptocurrency"),
        ("ADA-USD", "Cryptocurrency"),
        ("XRP-USD", "Cryptocurrency"),
        
        # New meme potentials
        ("HOOD", "Fintech"),
        ("RBLX", "Gaming"),
        ("DASH", "Delivery"),
        ("ABNB", "Travel"),
        ("COIN", "Crypto Stock")
    ]
    
    return popular_symbols

if __name__ == "__main__":
    # Uncomment to use the quick test list
    # SYMBOLS_TO_TEST = quick_test_list()
    
    batch_backtest()