"""
📚 VIEW BACKTEST HISTORY
Quick utility to see which symbols have been tested
"""

from src.agents.backtest_history import BacktestHistory
from datetime import datetime
from termcolor import colored
import json

def view_history():
    """Display backtest history"""
    
    history = BacktestHistory()
    
    print(colored("📚 AEGS BACKTEST HISTORY", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    # Get summary
    summary = history.get_summary()
    print(f"\n📊 Summary:")
    print(f"   Total symbols tested: {summary['total_symbols_tested']}")
    print(f"   Tested today: {summary['tested_today']}")
    print(f"   Tested this week: {summary['tested_this_week']}")
    print(f"   Tested this month: {summary['tested_this_month']}")
    print(f"   Retest interval: {summary['retest_interval_days']} days")
    
    # Show recent tests
    if history.history['backtest_history']:
        print(f"\n📅 Recent Tests:")
        
        # Sort by last tested date
        sorted_symbols = sorted(
            history.history['backtest_history'].items(),
            key=lambda x: x[1]['last_tested'],
            reverse=True
        )
        
        # Show top 20
        for symbol, data in sorted_symbols[:20]:
            last_tested = data['last_tested']
            test_count = data['test_count']
            
            # Get most recent result
            if data['results']:
                latest_result = data['results'][-1]
                excess = latest_result['excess_return']
                
                if excess > 1000:
                    status = colored("💎 GOLDMINE", 'red')
                elif excess > 100:
                    status = colored("🚀 HIGH", 'yellow')
                elif excess > 0:
                    status = colored("✅ POSITIVE", 'green')
                else:
                    status = "❌ NEGATIVE"
                
                print(f"   {symbol}: Last tested {last_tested} | {status} | "
                      f"Excess: {excess:+.0f}% | Tests: {test_count}")
            else:
                print(f"   {symbol}: Last tested {last_tested} | Tests: {test_count}")
    else:
        print("\n📭 No backtest history yet")
    
    # Show registry stats
    try:
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        total_registered = registry['metadata']['total_symbols']
        goldmines = registry['metadata']['goldmine_count']
        
        print(f"\n💎 Registry Status:")
        print(f"   Registered symbols: {total_registered}")
        print(f"   Goldmines found: {goldmines}")
    except:
        pass

if __name__ == "__main__":
    view_history()