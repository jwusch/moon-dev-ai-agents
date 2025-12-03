#!/usr/bin/env python3
"""
🔧 Fix AEGS JSON Parsing Errors
Clear invalid symbols that have JSON parsing errors so they can be retested
"""

import json
import os
from datetime import datetime
from termcolor import colored

def fix_json_errors():
    """Remove symbols with JSON parsing errors from invalid list"""
    
    print(colored("🔧 Fixing AEGS JSON Parsing Errors", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Load invalid symbols
    invalid_file = 'aegs_invalid_symbols.json'
    if not os.path.exists(invalid_file):
        print("❌ No invalid symbols file found")
        return
    
    with open(invalid_file, 'r') as f:
        data = json.load(f)
    
    invalid_symbols = data.get('invalid_symbols', {})
    original_count = len(invalid_symbols)
    
    # Find symbols with JSON parsing errors
    json_error_symbols = []
    for symbol, info in invalid_symbols.items():
        reason = info.get('reason', '')
        if 'Extra data: line' in reason and 'char' in reason:
            json_error_symbols.append(symbol)
    
    print(f"\n📊 Found {len(json_error_symbols)} symbols with JSON parsing errors:")
    for symbol in json_error_symbols[:10]:  # Show first 10
        print(f"   • {symbol}: {invalid_symbols[symbol]['reason'][:60]}...")
    
    if len(json_error_symbols) > 10:
        print(f"   ... and {len(json_error_symbols) - 10} more")
    
    # Remove these symbols from invalid list
    print(f"\n🗑️ Removing {len(json_error_symbols)} symbols from invalid list...")
    
    for symbol in json_error_symbols:
        del invalid_symbols[symbol]
    
    # Update the file
    data['invalid_symbols'] = invalid_symbols
    data['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['metadata']['total_invalid'] = len(invalid_symbols)
    
    with open(invalid_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(colored(f"\n✅ Removed {len(json_error_symbols)} symbols from invalid list", 'green'))
    print(f"   Invalid symbols: {original_count} → {len(invalid_symbols)}")
    
    # Also clear from backtest history
    history_file = 'aegs_backtest_history.json'
    if os.path.exists(history_file):
        print("\n📊 Clearing failed backtests from history...")
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        backtest_history = history_data.get('backtest_history', {})
        cleared_count = 0
        
        for symbol in json_error_symbols:
            if symbol in backtest_history:
                # Check if all tests failed
                results = backtest_history[symbol].get('results', [])
                if all(r.get('excess_return', 0) == 0 for r in results):
                    del backtest_history[symbol]
                    cleared_count += 1
        
        if cleared_count > 0:
            history_data['backtest_history'] = backtest_history
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            print(colored(f"✅ Cleared {cleared_count} failed symbols from backtest history", 'green'))
    
    print(colored("\n🎯 Fix complete! These symbols can now be retested.", 'green', attrs=['bold']))
    print("\n💡 Next steps:")
    print("1. The JSON parsing error is likely in the strategy configuration")
    print("2. Run a single symbol test to identify the exact issue")
    print("3. Check if QQQ_ensemble_strategy.json or SPY_ensemble_strategy.json were modified")

def check_strategy_files():
    """Check if strategy files are corrupted"""
    print(colored("\n🔍 Checking strategy files...", 'yellow'))
    print("-"*80)
    
    strategy_files = ['QQQ_ensemble_strategy.json', 'SPY_ensemble_strategy.json']
    
    for filename in strategy_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                    data = json.loads(content)
                
                # Check structure
                if 'symbol' in data and 'strategies' in data:
                    print(f"✅ {filename}: Valid structure")
                else:
                    print(f"⚠️  {filename}: Unexpected structure")
                    
            except json.JSONDecodeError as e:
                print(f"❌ {filename}: JSON parse error - {e}")
            except Exception as e:
                print(f"❌ {filename}: Error - {e}")

def main():
    """Run the fix"""
    fix_json_errors()
    check_strategy_files()

if __name__ == "__main__":
    main()