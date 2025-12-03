"""
🧹 Clean up symbols that are valid but not suitable for AEGS
"""

from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
from termcolor import colored

def cleanup_not_suitable():
    """Remove symbols that are valid but just not volatile enough for AEGS"""
    
    print(colored("🧹 CLEANING UP NOT-SUITABLE SYMBOLS", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    tracker = InvalidSymbolTracker()
    
    # These are valid symbols, just not suitable for AEGS mean reversion
    not_suitable = ['C', 'BAC', 'DASH', 'EBON', 'CENN', 'EVGO']
    
    removed = []
    
    for symbol in not_suitable:
        if symbol in tracker.invalid_symbols:
            info = tracker.invalid_symbols[symbol]
            if 'insufficient data' in info['reason'] or 'Backtest failed' in info['reason']:
                print(f"✅ Removing {symbol} - it's valid but not suitable for AEGS")
                tracker.retry_symbol(symbol)
                removed.append(symbol)
    
    print(f"\n📊 Removed {len(removed)} symbols: {removed}")
    print("\nThese symbols are:")
    print("- Valid and tradeable ✅")
    print("- Have sufficient data ✅")
    print("- Just not volatile enough for AEGS strategy ❌")
    
    # Show what remains
    summary = tracker.get_summary()
    print(f"\nRemaining invalid symbols: {summary['total']}")
    print("These are truly delisted/invalid:")
    for symbol in tracker.invalid_symbols:
        print(f"  - {symbol}: {tracker.invalid_symbols[symbol]['reason']}")

if __name__ == "__main__":
    cleanup_not_suitable()