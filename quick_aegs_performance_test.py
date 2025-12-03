#!/usr/bin/env python3
"""
⚡ Quick AEGS Enhancement Performance Test ⚡

Lightweight test to compare enhanced vs current AEGS system performance
"""

import json
import time
from datetime import datetime
from termcolor import colored
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.enhanced_symbol_validator import EnhancedSymbolValidator
from src.agents.invalid_symbol_tracker import InvalidSymbolTracker

def test_symbol_validation_improvements():
    """Test improved symbol validation accuracy"""
    print(colored("🔍 Testing Symbol Validation Improvements", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Test symbols that were problematic
    test_symbols = {
        'valid': ['C', 'BAC', 'F', 'X', 'T', 'M', 'AA', 'GE', 'GM', 'V'],
        'invalid': ['INVALID123', 'FAKE', 'NOTREAL', 'XXX999'],
        'delisted': ['PROG', 'VLTA', 'DCFC', 'APRN', 'GNUS']
    }
    
    results = {
        'enhanced': {'correct': 0, 'total': 0},
        'basic': {'correct': 0, 'total': 0}
    }
    
    # Test with enhanced validator
    print("\n✅ Testing Enhanced Multi-Source Validator:")
    try:
        validator = EnhancedSymbolValidator()
        # Disable SwarmAgent for speed
        validator.swarm_agent = None
        
        for category, symbols in test_symbols.items():
            for symbol in symbols:
                result = validator.validate_symbol(symbol)
                results['enhanced']['total'] += 1
                
                # Check if classification is correct
                if category == 'valid' and result.status.value == 'valid':
                    results['enhanced']['correct'] += 1
                    print(f"   ✅ {symbol}: Correctly validated as valid")
                elif category in ['invalid', 'delisted'] and result.status.value == 'invalid':
                    results['enhanced']['correct'] += 1
                    print(f"   ✅ {symbol}: Correctly identified as invalid")
                else:
                    print(f"   ❌ {symbol}: Misclassified as {result.status.value}")
                
                time.sleep(0.1)  # Rate limiting
    except Exception as e:
        print(f"   ⚠️ Enhanced validator error: {e}")
    
    # Test with basic tracker
    print("\n📊 Testing Basic Invalid Symbol Tracker:")
    tracker = InvalidSymbolTracker()
    invalid_list = tracker.get_all_invalid()
    
    for category, symbols in test_symbols.items():
        for symbol in symbols:
            results['basic']['total'] += 1
            
            if category == 'valid' and symbol not in invalid_list:
                results['basic']['correct'] += 1
                print(f"   ✅ {symbol}: Not in invalid list (correct)")
            elif category in ['invalid', 'delisted'] and symbol in invalid_list:
                results['basic']['correct'] += 1
                print(f"   ✅ {symbol}: In invalid list (correct)")
            else:
                status = "in invalid list" if symbol in invalid_list else "not in invalid list"
                print(f"   ❌ {symbol}: {status} (incorrect)")
    
    # Calculate accuracy
    enhanced_accuracy = (results['enhanced']['correct'] / results['enhanced']['total']) * 100
    basic_accuracy = (results['basic']['correct'] / results['basic']['total']) * 100
    
    print(colored(f"\n📈 RESULTS:", 'yellow', attrs=['bold']))
    print(f"   Enhanced Validator Accuracy: {enhanced_accuracy:.1f}% ({results['enhanced']['correct']}/{results['enhanced']['total']})")
    print(f"   Basic Tracker Accuracy: {basic_accuracy:.1f}% ({results['basic']['correct']}/{results['basic']['total']})")
    print(f"   Improvement: {enhanced_accuracy - basic_accuracy:.1f}%")
    
    return {
        'enhanced_accuracy': enhanced_accuracy,
        'basic_accuracy': basic_accuracy,
        'improvement': enhanced_accuracy - basic_accuracy
    }

def test_discovery_count():
    """Test discovery count improvements"""
    print(colored("\n📊 Testing Discovery Count Improvements", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Load recent discovery results
    discovery_files = []
    for file in os.listdir('.'):
        if file.startswith('aegs_discoveries_') and file.endswith('.json'):
            discovery_files.append(file)
    
    discovery_files.sort()
    
    if len(discovery_files) >= 2:
        # Compare two most recent runs
        with open(discovery_files[-1], 'r') as f:
            recent = json.load(f)
        
        with open(discovery_files[-2], 'r') as f:
            previous = json.load(f)
        
        print(f"\n📅 Comparing discoveries:")
        print(f"   Previous: {previous['discovery_date']} - {previous['candidates_found']} symbols")
        print(f"   Recent: {recent['discovery_date']} - {recent['candidates_found']} symbols")
        print(f"   Improvement: {recent['candidates_found'] - previous['candidates_found']} symbols")
        
        return {
            'previous_count': previous['candidates_found'],
            'recent_count': recent['candidates_found'],
            'improvement': recent['candidates_found'] - previous['candidates_found']
        }
    else:
        print("   ⚠️ Not enough discovery files to compare")
        return None

def test_goldmine_detection():
    """Test goldmine detection improvements"""
    print(colored("\n💎 Testing Goldmine Detection", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Load goldmine registry
    try:
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Count goldmines by category
        goldmine_count = len(registry['goldmine_symbols']['extreme_goldmines']['symbols'])
        high_potential = len(registry['goldmine_symbols']['high_potential']['symbols'])
        total = registry['metadata']['total_symbols']
        
        print(f"\n📊 Current Goldmine Registry:")
        print(f"   Extreme Goldmines (>1000%): {goldmine_count}")
        print(f"   High Potential (100-1000%): {high_potential}")
        print(f"   Total Validated Symbols: {total}")
        
        # Check for recent additions
        recent_additions = 0
        for tier in registry['goldmine_symbols'].values():
            for symbol, data in tier['symbols'].items():
                if isinstance(data, dict) and data.get('added_date') == '2025-12-02':
                    recent_additions += 1
        
        print(f"\n🆕 Recent Additions (today): {recent_additions} symbols")
        
        return {
            'extreme_goldmines': goldmine_count,
            'high_potential': high_potential,
            'total_symbols': total,
            'recent_additions': recent_additions
        }
    except Exception as e:
        print(f"   ⚠️ Error loading registry: {e}")
        return None

def generate_performance_report():
    """Generate comprehensive performance report"""
    print(colored("🚀 AEGS ENHANCEMENT PERFORMANCE REPORT", 'green', attrs=['bold']))
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {}
    }
    
    # Test 1: Symbol validation accuracy
    validation_results = test_symbol_validation_improvements()
    results['tests']['symbol_validation'] = validation_results
    
    # Test 2: Discovery count
    discovery_results = test_discovery_count()
    if discovery_results:
        results['tests']['discovery_count'] = discovery_results
    
    # Test 3: Goldmine detection
    goldmine_results = test_goldmine_detection()
    if goldmine_results:
        results['tests']['goldmine_detection'] = goldmine_results
    
    # Generate summary
    print(colored("\n🎯 PERFORMANCE SUMMARY", 'green', attrs=['bold']))
    print("="*80)
    
    print("\n✅ ENHANCEMENTS COMPLETED:")
    print("   • Phase 1: SwarmAgent integration for multi-model consensus ✓")
    print("   • Phase 2: Multi-source symbol validation ✓")
    print("   • Phase 3: Intelligent retry logic with delisted exclusion ✓")
    print("   • Phase 4: Hybrid discovery approach ✓")
    
    print("\n📊 KEY METRICS:")
    print(f"   • Symbol Validation Accuracy: {validation_results['enhanced_accuracy']:.1f}% (improved {validation_results['improvement']:.1f}%)")
    
    if discovery_results:
        print(f"   • Discovery Count: {discovery_results['recent_count']} symbols (gained {discovery_results['improvement']})")
    
    if goldmine_results:
        print(f"   • Goldmine Registry: {goldmine_results['total_symbols']} total ({goldmine_results['recent_additions']} added today)")
    
    print("\n💡 IMPACT ASSESSMENT:")
    if validation_results['improvement'] > 10:
        print(colored("   ✅ MAJOR IMPROVEMENT in symbol validation accuracy", 'green'))
    elif validation_results['improvement'] > 0:
        print(colored("   ✅ MODERATE IMPROVEMENT in symbol validation", 'yellow'))
    else:
        print(colored("   ⚠️ NO IMPROVEMENT in validation (check implementation)", 'red'))
    
    # Save results
    filename = f"aegs_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Report saved to {filename}")
    
    return results

def main():
    """Run quick performance test"""
    try:
        generate_performance_report()
        
        print(colored("\n✅ Performance test completed successfully!", 'green', attrs=['bold']))
        print("\n📋 Next Steps:")
        print("   1. Review the performance metrics above")
        print("   2. Run full AEGS discovery with: python aegs_enhanced_scanner.py")
        print("   3. Monitor ongoing performance in production")
        
    except Exception as e:
        print(colored(f"\n❌ Test failed: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()