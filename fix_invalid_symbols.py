"""
🔧 Fix Invalid Symbols
Remove legitimate symbols that were incorrectly marked as invalid
"""

from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
from termcolor import colored
import yfinance as yf

def fix_invalid_symbols():
    """Remove legitimate symbols from invalid list"""
    
    print(colored("🔧 FIXING INVALID SYMBOLS", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    tracker = InvalidSymbolTracker()
    
    # Symbols to check and potentially remove
    check_symbols = ['C', 'BAC', 'DASH', 'EBON', 'CENN', 'EVGO']
    
    removed = []
    kept = []
    
    for symbol in check_symbols:
        if symbol in tracker.invalid_symbols:
            print(f"\n🔍 Checking {symbol}...")
            
            # Verify if it's truly invalid
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if len(hist) > 0:
                    price = hist['Close'].iloc[-1]
                    print(f"   ✅ {symbol} is valid! Current price: ${price:.2f}")
                    tracker.retry_symbol(symbol)
                    removed.append(symbol)
                else:
                    print(f"   ❌ {symbol} has no recent data - keeping as invalid")
                    kept.append(symbol)
                    
            except Exception as e:
                print(f"   ❌ {symbol} error: {str(e)[:50]} - keeping as invalid")
                kept.append(symbol)
    
    print(colored(f"\n📊 SUMMARY:", 'yellow'))
    print(f"Removed from invalid list: {removed}")
    print(f"Kept as invalid: {kept}")
    
    # Show updated summary
    summary = tracker.get_summary()
    print(f"\nTotal invalid symbols now: {summary['total']}")

if __name__ == "__main__":
    fix_invalid_symbols()