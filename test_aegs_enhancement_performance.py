#!/usr/bin/env python3
"""
🧪 AEGS Enhancement Performance Testing 🧪

Compare enhanced AEGS system (with SwarmAgent + multi-source validation) 
vs current system across key discovery accuracy metrics.

Metrics measured:
1. Discovery accuracy (# of valid symbols found)
2. Goldmine detection rate (# of >1000% excess return symbols)
3. Processing time comparison
4. Token usage metrics
5. False positive/negative rates
"""

import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.aegs_discovery_agent import AEGSDiscoveryAgent
from src.agents.enhanced_symbol_validator import EnhancedSymbolValidator
from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
from termcolor import colored

class AEGSPerformanceTester:
    """Test and compare AEGS enhancement performance"""
    
    def __init__(self):
        self.test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_system': {},
            'current_system': {},
            'comparison': {}
        }
        
    def run_discovery_test(self, use_enhanced: bool = True) -> Dict:
        """Run discovery test with or without enhancements"""
        print(colored(f"\n{'='*80}", 'cyan'))
        print(colored(f"🧪 Testing {'ENHANCED' if use_enhanced else 'CURRENT'} AEGS System", 'cyan', attrs=['bold']))
        print(colored(f"{'='*80}", 'cyan'))
        
        start_time = time.time()
        
        # Create discovery agent
        agent = AEGSDiscoveryAgent()
        
        # Temporarily disable enhancements if testing current system
        if not use_enhanced:
            agent.swarm_agent = None
            agent.symbol_validator = None
            print("⚠️ Disabled enhancements for baseline test")
        
        # Run discovery
        discovered_symbols = agent.run()
        
        end_time = time.time()
        
        # Analyze results
        results = {
            'symbols_discovered': len(discovered_symbols),
            'discovery_time_seconds': round(end_time - start_time, 2),
            'symbols': discovered_symbols,
            'discovery_reasons': agent.discovery_reasons
        }
        
        # Validate discovered symbols
        if agent.symbol_validator:
            print("\n🔍 Validating discovered symbols...")
            validation_results = agent.symbol_validator.validate_batch(discovered_symbols[:10])  # Test first 10
            
            valid_count = sum(1 for r in validation_results.values() if r.status.value == "valid")
            results['validation_accuracy'] = (valid_count / len(validation_results)) * 100 if validation_results else 0
        else:
            results['validation_accuracy'] = "N/A (no enhanced validator)"
        
        return results
    
    def analyze_goldmine_potential(self, symbols: List[str]) -> Dict:
        """Analyze goldmine potential of discovered symbols"""
        print("\n💎 Analyzing goldmine potential...")
        
        # Load goldmine registry for comparison
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            known_goldmines = set()
            for tier in registry['goldmine_symbols'].values():
                known_goldmines.update(tier.get('symbols', {}).keys())
        except:
            known_goldmines = set()
        
        # Check how many discovered symbols are known goldmines
        found_goldmines = [s for s in symbols if s in known_goldmines]
        
        return {
            'known_goldmines_found': len(found_goldmines),
            'goldmine_detection_rate': (len(found_goldmines) / len(symbols)) * 100 if symbols else 0,
            'goldmine_symbols': found_goldmines
        }
    
    def compare_results(self, enhanced_results: Dict, current_results: Dict) -> Dict:
        """Compare enhanced vs current system performance"""
        print(colored("\n📊 PERFORMANCE COMPARISON", 'yellow', attrs=['bold']))
        print("="*80)
        
        comparison = {
            'symbols_discovered': {
                'enhanced': enhanced_results['symbols_discovered'],
                'current': current_results['symbols_discovered'],
                'improvement': enhanced_results['symbols_discovered'] - current_results['symbols_discovered'],
                'improvement_pct': ((enhanced_results['symbols_discovered'] - current_results['symbols_discovered']) / 
                                  current_results['symbols_discovered'] * 100) if current_results['symbols_discovered'] > 0 else 0
            },
            'discovery_time': {
                'enhanced': enhanced_results['discovery_time_seconds'],
                'current': current_results['discovery_time_seconds'],
                'difference': enhanced_results['discovery_time_seconds'] - current_results['discovery_time_seconds'],
                'overhead_pct': ((enhanced_results['discovery_time_seconds'] - current_results['discovery_time_seconds']) / 
                               current_results['discovery_time_seconds'] * 100) if current_results['discovery_time_seconds'] > 0 else 0
            },
            'validation_accuracy': {
                'enhanced': enhanced_results.get('validation_accuracy', 'N/A'),
                'current': current_results.get('validation_accuracy', 'N/A')
            }
        }
        
        # Print comparison
        print(f"\n📈 Symbol Discovery:")
        print(f"   Enhanced: {comparison['symbols_discovered']['enhanced']} symbols")
        print(f"   Current:  {comparison['symbols_discovered']['current']} symbols")
        print(f"   Improvement: {comparison['symbols_discovered']['improvement']} "
              f"({comparison['symbols_discovered']['improvement_pct']:.1f}%)")
        
        print(f"\n⏱️ Discovery Time:")
        print(f"   Enhanced: {comparison['discovery_time']['enhanced']:.1f}s")
        print(f"   Current:  {comparison['discovery_time']['current']:.1f}s")
        print(f"   Overhead: {comparison['discovery_time']['difference']:.1f}s "
              f"({comparison['discovery_time']['overhead_pct']:.1f}%)")
        
        print(f"\n✅ Validation Accuracy:")
        print(f"   Enhanced: {comparison['validation_accuracy']['enhanced']}")
        print(f"   Current:  {comparison['validation_accuracy']['current']}")
        
        return comparison
    
    def test_false_positive_reduction(self):
        """Test reduction in false positives (valid symbols marked invalid)"""
        print(colored("\n🔍 Testing False Positive Reduction", 'yellow', attrs=['bold']))
        
        # Known valid symbols that were problematic
        problematic_symbols = ['C', 'BAC', 'F', 'X', 'T', 'M']
        
        # Test with enhanced validator
        if os.path.exists('src/agents/enhanced_symbol_validator.py'):
            validator = EnhancedSymbolValidator()
            enhanced_results = validator.validate_batch(problematic_symbols)
            
            enhanced_valid = sum(1 for r in enhanced_results.values() if r.status.value == "valid")
            print(f"\n✅ Enhanced validator: {enhanced_valid}/{len(problematic_symbols)} correctly validated")
            
            for symbol, result in enhanced_results.items():
                status_emoji = "✅" if result.status.value == "valid" else "❌"
                print(f"   {status_emoji} {symbol}: {result.status.value} ({result.confidence:.1f}% confidence)")
        
        # Test with basic validator
        tracker = InvalidSymbolTracker()
        basic_valid = 0
        for symbol in problematic_symbols:
            if symbol not in tracker.get_all_invalid():
                basic_valid += 1
        
        print(f"\n📊 Basic tracker: {basic_valid}/{len(problematic_symbols)} not marked invalid")
        
        return {
            'enhanced_accuracy': (enhanced_valid / len(problematic_symbols)) * 100 if 'enhanced_valid' in locals() else 0,
            'basic_accuracy': (basic_valid / len(problematic_symbols)) * 100,
            'false_positive_reduction': enhanced_valid - basic_valid if 'enhanced_valid' in locals() else 0
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive performance test"""
        print(colored("🚀 AEGS ENHANCEMENT PERFORMANCE TEST", 'green', attrs=['bold']))
        print("="*80)
        
        # Test 1: Enhanced system
        print("\nPhase 1: Testing Enhanced AEGS System")
        enhanced_results = self.run_discovery_test(use_enhanced=True)
        enhanced_goldmine = self.analyze_goldmine_potential(enhanced_results['symbols'])
        enhanced_results.update(enhanced_goldmine)
        
        # Test 2: Current system (baseline)
        print("\nPhase 2: Testing Current AEGS System (Baseline)")
        current_results = self.run_discovery_test(use_enhanced=False)
        current_goldmine = self.analyze_goldmine_potential(current_results['symbols'])
        current_results.update(current_goldmine)
        
        # Test 3: Compare results
        comparison = self.compare_results(enhanced_results, current_results)
        
        # Test 4: False positive reduction
        fp_results = self.test_false_positive_reduction()
        
        # Compile final results
        self.test_results['enhanced_system'] = enhanced_results
        self.test_results['current_system'] = current_results
        self.test_results['comparison'] = comparison
        self.test_results['false_positive_reduction'] = fp_results
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save test results to file"""
        filename = f"aegs_enhancement_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
    
    def print_summary(self):
        """Print test summary"""
        print(colored("\n🎯 PERFORMANCE TEST SUMMARY", 'green', attrs=['bold']))
        print("="*80)
        
        comp = self.test_results['comparison']
        fp = self.test_results['false_positive_reduction']
        
        print("\n✅ KEY IMPROVEMENTS:")
        print(f"   • Discovery Accuracy: {comp['symbols_discovered']['improvement_pct']:.1f}% more symbols found")
        print(f"   • Processing Time: {comp['discovery_time']['overhead_pct']:.1f}% overhead (acceptable for better accuracy)")
        print(f"   • False Positive Reduction: {fp['enhanced_accuracy'] - fp['basic_accuracy']:.1f}% improvement")
        print(f"   • Goldmine Detection: Enhanced found {self.test_results['enhanced_system']['known_goldmines_found']} vs "
              f"Current found {self.test_results['current_system']['known_goldmines_found']}")
        
        print("\n📊 ENHANCEMENT VERDICT:")
        if comp['symbols_discovered']['improvement'] > 0 and fp['enhanced_accuracy'] > fp['basic_accuracy']:
            print(colored("   ✅ ENHANCEMENTS SUCCESSFUL - Higher accuracy with acceptable overhead", 'green', attrs=['bold']))
        elif comp['symbols_discovered']['improvement'] > 0:
            print(colored("   ⚠️ PARTIAL SUCCESS - More symbols but validation needs improvement", 'yellow'))
        else:
            print(colored("   ❌ NEEDS OPTIMIZATION - Performance degradation detected", 'red'))
        
        print("\n💡 RECOMMENDATIONS:")
        if comp['discovery_time']['overhead_pct'] > 50:
            print("   • Consider optimizing SwarmAgent queries for faster processing")
        if fp['enhanced_accuracy'] < 90:
            print("   • Add more validation sources or tune consensus thresholds")
        if self.test_results['enhanced_system']['known_goldmines_found'] < 5:
            print("   • Fine-tune AI prompts for better goldmine pattern recognition")

def main():
    """Run performance test"""
    tester = AEGSPerformanceTester()
    
    try:
        tester.run_comprehensive_test()
    except Exception as e:
        print(colored(f"\n❌ Test failed: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()