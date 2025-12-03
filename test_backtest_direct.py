#!/usr/bin/env python3
"""
Test backtest directly with detailed error capture
"""

import json
import sys
import traceback
from termcolor import colored

print(colored("🔧 Direct Backtest Test with Error Capture", 'cyan', attrs=['bold']))
print("="*80)

# Capture the exact JSON error
old_json_loads = json.loads
old_json_load = json.load
json_call_count = 0

def traced_json_loads(s, *args, **kwargs):
    global json_call_count
    json_call_count += 1
    print(f"\n🔍 JSON loads call #{json_call_count}")
    print(f"   String length: {len(s)}")
    print(f"   First 100 chars: {s[:100]}...")
    try:
        result = old_json_loads(s, *args, **kwargs)
        print("   ✅ Success")
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Print around the error position
        if hasattr(e, 'pos'):
            start = max(0, e.pos - 100)
            end = min(len(s), e.pos + 100)
            print(f"   Context around position {e.pos}:")
            print(f"   {s[start:end]}")
        raise

def traced_json_load(fp, *args, **kwargs):
    global json_call_count
    json_call_count += 1
    content = fp.read()
    fp.seek(0)  # Reset file pointer
    print(f"\n🔍 JSON load call #{json_call_count}")
    print(f"   File: {getattr(fp, 'name', 'unknown')}")
    print(f"   Content length: {len(content)}")
    try:
        result = old_json_load(fp, *args, **kwargs)
        print("   ✅ Success")
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise

# Replace json functions
json.loads = traced_json_loads
json.load = traced_json_load

# Now import and test
try:
    from comprehensive_qqq_backtest import ComprehensiveBacktester
    
    print("\n📊 Creating backtester for JPM...")
    backtester = ComprehensiveBacktester("JPM")
    
    print("\n📊 Downloading data...")
    df = backtester.download_maximum_data()
    
    if df is not None and len(df) > 0:
        print(f"✅ Downloaded {len(df)} rows")
        
        print("\n📊 Running comprehensive backtest...")
        results = backtester.comprehensive_backtest(df)
        print("✅ Backtest completed successfully!")
    else:
        print("❌ No data downloaded")
        
except Exception as e:
    print(f"\n❌ Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()
    
print(f"\n📊 Total JSON calls: {json_call_count}")