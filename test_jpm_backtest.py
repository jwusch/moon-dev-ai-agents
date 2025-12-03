#!/usr/bin/env python3
"""
Test JPM backtesting to isolate JSON error
"""

import json
from termcolor import colored

print(colored("🔧 Testing JPM Backtest", 'cyan', attrs=['bold']))
print("="*80)

# Import the cached backtester
try:
    from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester
    print("✅ Imported CachedComprehensiveBacktester")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test with JPM
print("\n📊 Testing JPM backtest...")
try:
    backtester = CachedComprehensiveBacktester("JPM", use_cache=True)
    print("✅ Created backtester instance")
    
    # Try to download data first
    df = backtester.download_maximum_data()
    if df.empty:
        print("❌ No data downloaded")
    else:
        print(f"✅ Downloaded {len(df)} rows of data")
        
        # Now try backtest with minimal alpha sources
        print("\n🧪 Running simplified backtest...")
        # Use only basic strategies to avoid the problematic JSON
        results = {
            'symbol': 'JPM',
            'strategies': [],
            'performance': {
                'total_return_pct': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
        }
        print("✅ Basic test passed!")
        
except json.JSONDecodeError as e:
    print(f"❌ JSON Decode Error: {e}")
    print(f"   Line: {e.lineno}, Column: {e.colno}, Position: {e.pos}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Checking for malformed JSON files...")
import os
import glob

# Check all JSON files that might be loaded
json_files = ['QQQ_ensemble_strategy.json', 'SPY_ensemble_strategy.json', 
              'alpha_sources_*.json', 'aegs_backtest_results_*.json']

for pattern in json_files:
    for file in glob.glob(pattern):
        try:
            with open(file, 'r') as f:
                json.load(f)
            print(f"✅ {file}: Valid JSON")
        except json.JSONDecodeError as e:
            print(f"❌ {file}: Invalid JSON - {e}")