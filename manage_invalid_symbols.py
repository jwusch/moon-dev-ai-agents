"""
🚫 Invalid Symbol Manager
View and manage symbols that failed during discovery/backtesting
"""

import argparse
from termcolor import colored
from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
import json

def main():
    parser = argparse.ArgumentParser(description='Manage invalid symbols')
    parser.add_argument('--show', action='store_true', help='Show all invalid symbols')
    parser.add_argument('--summary', action='store_true', help='Show summary of invalid symbols')
    parser.add_argument('--retry', help='Remove symbol from invalid list to retry')
    parser.add_argument('--cleanup', type=int, help='Remove entries older than N days')
    parser.add_argument('--export', help='Export invalid symbols to file')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = InvalidSymbolTracker()
    
    if args.show:
        print(colored("🚫 ALL INVALID SYMBOLS", 'red', attrs=['bold']))
        print("=" * 80)
        
        for symbol, info in sorted(tracker.invalid_symbols.items()):
            print(f"\n{symbol}:")
            print(f"  Reason: {info['reason']}")
            print(f"  Error Type: {info['error_type']}")
            print(f"  First Failed: {info['first_failed']}")
            print(f"  Fail Count: {info.get('fail_count', 1)}")
            if 'last_failed' in info:
                print(f"  Last Failed: {info['last_failed']}")
    
    elif args.summary:
        summary = tracker.get_summary()
        
        print(colored("📊 INVALID SYMBOL SUMMARY", 'yellow', attrs=['bold']))
        print("=" * 80)
        print(f"\nTotal Invalid Symbols: {summary['total']}")
        
        print("\nBy Error Type:")
        for error_type, count in summary['by_error_type'].items():
            print(f"  {error_type}: {count}")
        
        print("\nRecent Failures:")
        for failure in summary['recent_failures']:
            print(f"  {failure['symbol']} ({failure['date']}): {failure['reason']}")
    
    elif args.retry:
        if tracker.retry_symbol(args.retry):
            print(colored(f"✅ Removed {args.retry} from invalid list", 'green'))
        else:
            print(colored(f"❌ {args.retry} not found in invalid list", 'red'))
    
    elif args.cleanup:
        removed = tracker.cleanup_old_entries(args.cleanup)
        if removed:
            print(colored(f"🧹 Removed {len(removed)} old entries: {', '.join(removed)}", 'yellow'))
        else:
            print(colored("No old entries to remove", 'green'))
    
    elif args.export:
        data = {
            'invalid_symbols': tracker.invalid_symbols,
            'summary': tracker.get_summary()
        }
        with open(args.export, 'w') as f:
            json.dump(data, f, indent=2)
        print(colored(f"💾 Exported {len(tracker.invalid_symbols)} invalid symbols to {args.export}", 'green'))
    
    else:
        # Default: show summary
        summary = tracker.get_summary()
        print(colored(f"🚫 {summary['total']} Invalid Symbols Tracked", 'yellow'))
        print("Use --show to see all, --summary for details")

if __name__ == "__main__":
    main()