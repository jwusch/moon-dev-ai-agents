#!/usr/bin/env python3
"""
Quick test of symbol consensus swarm with just a few symbols
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from symbol_consensus_swarm import SymbolConsensusSwarm
from termcolor import colored

def quick_test():
    """Test with just 2 symbols"""
    
    print(colored("🔍 Quick Symbol Consensus Test", 'cyan', attrs=['bold']))
    print("=" * 50)
    
    # Initialize swarm
    swarm = SymbolConsensusSwarm()
    
    # Test just 2 symbols
    test_symbols = [
        'DCFC',    # Likely delisted
        'AAPL'     # Obviously valid (control)
    ]
    
    print(f"Testing {len(test_symbols)} symbols...")
    
    # Research symbols one by one
    results = {}
    
    for symbol in test_symbols:
        try:
            print(colored(f"\n🔍 Researching {symbol}...", 'yellow'))
            result = swarm.research_symbol(symbol)
            results[symbol] = result
            
            # Show quick summary
            consensus = result['consensus_analysis']
            print(colored(f"   Result: {consensus['status']} on {consensus['exchange']}", 'green' if consensus['tradable'] else 'red'))
            
        except Exception as e:
            print(colored(f"   ❌ Error: {str(e)}", 'red'))
            break
    
    # Print summary
    if results:
        swarm.print_summary(results)
        swarm.save_results(results, "quick_test_results.json")

if __name__ == "__main__":
    quick_test()