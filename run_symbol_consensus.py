#!/usr/bin/env python3
"""
Run symbol consensus swarm on the problematic AEGS symbols
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

def main():
    """Run consensus on all the problematic symbols from AEGS discovery"""
    
    print(colored("🔍💎 SYMBOL CONSENSUS SWARM - FULL BATCH 💎🔍", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    # Initialize swarm
    swarm = SymbolConsensusSwarm()
    
    # All the problematic symbols from your AEGS discovery
    problematic_symbols = [
        'DCFC',    # Already tested - should be delisted
        'NAKD',    # Likely delisted
        'PTRA',    # Likely delisted
        'VLTA',    # Likely delisted
        'ODDITY',  # Likely delisted
        'GNUS',    # Likely delisted
        'REV',     # Likely delisted
        'HYZN',    # Likely delisted
        'EXPR',    # Likely delisted
        'CEI',     # Likely delisted
        'PROG',    # Likely delisted
        'APRN',    # Likely delisted
        'ARVL'     # Likely delisted
    ]
    
    print(f"Researching {len(problematic_symbols)} problematic symbols with AI swarm...")
    print("This will help identify which are truly delisted vs. moved to OTC")
    print()
    
    # Research all symbols
    results = swarm.research_symbol_batch(problematic_symbols)
    
    # Print summary
    swarm.print_summary(results)
    
    # Save results
    timestamp = "20251202_problematic_symbols"
    filename = f"symbol_research_{timestamp}.json"
    swarm.save_results(results, filename)
    
    print(colored("\\n🎯 KEY INSIGHTS:", 'yellow', attrs=['bold']))
    print("✅ Truly tradable symbols can be added back to AEGS discovery")
    print("🟡 OTC symbols need special handling (.PK, .OB suffixes)")  
    print("❌ Delisted symbols should be permanently excluded")
    print("📊 This data will improve AEGS discovery efficiency!")

if __name__ == "__main__":
    main()