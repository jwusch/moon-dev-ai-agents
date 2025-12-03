#!/usr/bin/env python3
"""
Deep debug of JSON error - step by step
"""

import sys
import json
from termcolor import colored

print(colored("🔍 Deep Debug of JSON Error", 'cyan', attrs=['bold']))
print("="*80)

# Step 1: Import comprehensive_qqq_backtest
print("\n1️⃣ Testing import of comprehensive_qqq_backtest...")
try:
    import comprehensive_qqq_backtest
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Import comprehensive_qqq_backtest_enhanced
print("\n2️⃣ Testing import of comprehensive_qqq_backtest_enhanced...")
try:
    import comprehensive_qqq_backtest_enhanced
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Import comprehensive_qqq_backtest_cached
print("\n3️⃣ Testing import of comprehensive_qqq_backtest_cached...")
try:
    import comprehensive_qqq_backtest_cached
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Try creating instances
print("\n4️⃣ Testing instance creation...")
try:
    # Try the basic one first
    print("   Creating ComprehensiveBacktester...")
    basic = comprehensive_qqq_backtest.ComprehensiveBacktester("JPM")
    print("   ✅ Basic instance created")
except Exception as e:
    print(f"   ❌ Basic instance failed: {e}")
    import traceback
    traceback.print_exc()

try:
    # Try the enhanced one
    print("\n   Creating EnhancedComprehensiveBacktester...")
    enhanced = comprehensive_qqq_backtest_enhanced.EnhancedComprehensiveBacktester("JPM")
    print("   ✅ Enhanced instance created")
except Exception as e:
    print(f"   ❌ Enhanced instance failed: {e}")
    import traceback
    traceback.print_exc()

try:
    # Try the cached one
    print("\n   Creating CachedComprehensiveBacktester...")
    cached = comprehensive_qqq_backtest_cached.CachedComprehensiveBacktester("JPM", use_cache=False)
    print("   ✅ Cached instance created")
except Exception as e:
    print(f"   ❌ Cached instance failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Check if it's a global variable issue
print("\n5️⃣ Checking module globals...")
for module_name, module in [
    ('comprehensive_qqq_backtest', comprehensive_qqq_backtest),
    ('comprehensive_qqq_backtest_enhanced', comprehensive_qqq_backtest_enhanced),
    ('comprehensive_qqq_backtest_cached', comprehensive_qqq_backtest_cached)
]:
    print(f"\n   {module_name}:")
    for attr in dir(module):
        if not attr.startswith('_'):
            try:
                value = getattr(module, attr)
                if isinstance(value, (dict, list)) and str(value).startswith('{'):
                    print(f"      {attr}: {type(value).__name__} (might be JSON-like)")
            except:
                pass

print("\n✅ Debug complete!")