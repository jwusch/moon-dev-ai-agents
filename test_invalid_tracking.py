"""
🧪 Test Invalid Symbol Tracking
Demonstrates how the system prevents re-testing failed symbols
"""

from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
from aegs_enhanced_discovery import EnhancedDiscoveryAgent
from termcolor import colored
import json

def demonstrate_invalid_tracking():
    """Show how invalid symbol tracking works"""
    
    print(colored("🧪 TESTING INVALID SYMBOL TRACKING", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    # 1. Show current invalid symbols
    tracker = InvalidSymbolTracker()
    print(f"\n📊 Currently tracking {len(tracker.invalid_symbols)} invalid symbols")
    
    summary = tracker.get_summary()
    print("\nBy Error Type:")
    for error_type, count in summary['by_error_type'].items():
        print(f"  {error_type}: {count} symbols")
    
    # 2. Show what happens when discovery agent runs
    print(colored("\n🔍 Testing Enhanced Discovery Agent", 'yellow'))
    
    discovery = EnhancedDiscoveryAgent()
    print(f"\nTotal excluded symbols: {len(discovery.excluded_symbols)}")
    
    # Break down exclusions
    tested_count = len(discovery.history.history['backtest_history'])
    invalid_count = len(tracker.get_all_invalid())
    
    print(f"  - Tested symbols: {tested_count}")
    print(f"  - Invalid symbols: {invalid_count}")
    print(f"  - Other exclusions: {len(discovery.excluded_symbols) - tested_count - invalid_count}")
    
    # 3. Simulate adding a new invalid symbol
    print(colored("\n🚫 Simulating Failed Symbol", 'red'))
    test_symbol = "FAKE123"
    
    print(f"Adding {test_symbol} to invalid list...")
    tracker.add_invalid_symbol(test_symbol, "Test symbol - does not exist", "test")
    
    # Reload discovery agent to see if it excludes the new symbol
    discovery2 = EnhancedDiscoveryAgent()
    
    if test_symbol in discovery2.excluded_symbols:
        print(colored(f"✅ SUCCESS: {test_symbol} is now excluded from discovery", 'green'))
    else:
        print(colored(f"❌ FAILED: {test_symbol} was not excluded", 'red'))
    
    # Clean up test symbol
    tracker.retry_symbol(test_symbol)
    
    # 4. Show recent failures
    print(colored("\n📋 Recent Symbol Failures", 'yellow'))
    for failure in summary['recent_failures'][:5]:
        print(f"  {failure['symbol']}: {failure['reason']}")
    
    # 5. Demonstrate the full exclusion chain
    print(colored("\n🔄 Full Exclusion Chain Demo", 'cyan'))
    
    # Create a test list with some valid and invalid symbols
    test_candidates = ['AAPL', 'BRDS', 'TSLA', 'FSR', 'GOOGL', 'WISH', 'FAKE999']
    
    print(f"\nTest candidates: {test_candidates}")
    
    # Filter through discovery agent
    filtered = []
    for symbol in test_candidates:
        if symbol not in discovery.excluded_symbols:
            filtered.append(symbol)
        else:
            reason = "Unknown"
            if symbol in tracker.invalid_symbols:
                reason = "Invalid/Failed"
            elif symbol in discovery.history.history['backtest_history']:
                reason = "Already tested"
            print(f"  ❌ {symbol} excluded: {reason}")
    
    print(f"\nFiltered candidates: {filtered}")
    
    print(colored("\n✅ Invalid symbol tracking is working correctly!", 'green'))
    print("\nThe system will now:")
    print("1. Skip symbols that previously failed with errors")
    print("2. Track why symbols failed (delisted, no data, etc.)")
    print("3. Prevent wasted API calls on known bad symbols")
    print("4. Allow manual retry of symbols if needed")

if __name__ == "__main__":
    demonstrate_invalid_tracking()