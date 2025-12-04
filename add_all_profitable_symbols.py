#!/usr/bin/env python3

import json
import os
from datetime import datetime

def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_profitable_symbols_from_file(filepath, min_return=0.0):
    """Extract all profitable symbols from a result file"""
    data = load_json_file(filepath)
    if not data:
        return []
    
    # Check for different data structures
    symbols_data = []
    if 'profitable_symbols' in data:
        symbols_data = data['profitable_symbols']
    elif 'top_performers' in data:
        symbols_data = data['top_performers']
    else:
        return []
    
    profitable = []
    for symbol_data in symbols_data:
        symbol = symbol_data['symbol']
        strategy_return = symbol_data.get('strategy_return', 0)
        
        if strategy_return > min_return:
            profitable.append({
                'symbol': symbol,
                'strategy_return': strategy_return,
                'total_trades': symbol_data.get('total_trades', 0),
                'win_rate': symbol_data.get('win_rate', 0),
                'excess_return': symbol_data.get('excess_return', 0)
            })
    
    return profitable

def categorize_symbol(symbol_data):
    """Categorize symbol based on performance"""
    strategy_return = symbol_data['strategy_return']
    
    if strategy_return >= 100:
        return "extreme_goldmines"
    elif strategy_return >= 30:
        return "high_potential"
    else:
        return "positive"

def main():
    print("🔍 Extracting all profitable symbols from alphabet scan results...")
    
    # Find all result files
    result_files = [
        'nasdaq_a_brute_force_aegs_results_20251203_191806.json',
        'nasdaq_bc_brute_force_aegs_results_20251203_192837.json', 
        'nasdaq_dz_brute_force_aegs_results_20251203_194051.json'
    ]
    
    all_profitable = []
    scan_summary = {}
    
    # Extract profitable symbols from each scan
    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"📊 Processing {result_file}...")
            profitable = get_profitable_symbols_from_file(result_file)
            all_profitable.extend(profitable)
            
            # Track by scan type
            if 'nasdaq_a' in result_file:
                scan_summary['a_symbols'] = len(profitable)
            elif 'nasdaq_bc' in result_file:
                scan_summary['bc_symbols'] = len(profitable)
            elif 'nasdaq_dz' in result_file:
                scan_summary['dz_symbols'] = len(profitable)
                
            print(f"  ✅ Found {len(profitable)} profitable symbols")
    
    print(f"\n📈 Total profitable symbols found: {len(all_profitable)}")
    print(f"   A symbols: {scan_summary.get('a_symbols', 0)}")
    print(f"   B&C symbols: {scan_summary.get('bc_symbols', 0)}")
    print(f"   D-Z symbols: {scan_summary.get('dz_symbols', 0)}")
    
    # Load existing goldmine registry
    registry_file = 'aegs_goldmine_registry.json'
    if os.path.exists(registry_file):
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {
            "goldmine_symbols": {
                "extreme_goldmines": {},
                "high_potential": {}, 
                "positive": {}
            },
            "categories": {},
            "metadata": {}
        }
    
    # Track new additions by category
    new_additions = {
        "extreme_goldmines": 0,
        "high_potential": 0,
        "positive": 0
    }
    
    # Add all profitable symbols to registry
    for symbol_data in all_profitable:
        symbol = symbol_data['symbol']
        category = categorize_symbol(symbol_data)
        
        # Check if symbol already exists in any category
        exists = False
        for cat in registry['goldmine_symbols']:
            if symbol in registry['goldmine_symbols'][cat]:
                exists = True
                break
        
        if not exists:
            registry['goldmine_symbols'][category][symbol] = {
                'strategy_return': symbol_data['strategy_return'],
                'total_trades': symbol_data['total_trades'],
                'win_rate': symbol_data['win_rate'],
                'excess_return': symbol_data['excess_return'],
                'added_date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'alphabet_brute_force_scan'
            }
            new_additions[category] += 1
    
    # Update metadata
    total_symbols = sum(len(registry['goldmine_symbols'][cat]) for cat in registry['goldmine_symbols'])
    
    registry['metadata'] = {
        'total_symbols': total_symbols,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'complete_alphabet_scan': {
            'a_symbols_profitable': scan_summary.get('a_symbols', 0),
            'bc_symbols_profitable': scan_summary.get('bc_symbols', 0), 
            'dz_symbols_profitable': scan_summary.get('dz_symbols', 0),
            'total_alphabet_profitable': len(all_profitable)
        },
        'new_additions_by_category': new_additions
    }
    
    # Save updated registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n🎯 Updated AEGS Goldmine Registry:")
    print(f"   💎 Extreme Goldmines (≥100%): {len(registry['goldmine_symbols']['extreme_goldmines'])} (+{new_additions['extreme_goldmines']})")
    print(f"   🌟 High Potential (30-99%): {len(registry['goldmine_symbols']['high_potential'])} (+{new_additions['high_potential']})")
    print(f"   ✅ Positive (1-29%): {len(registry['goldmine_symbols']['positive'])} (+{new_additions['positive']})")
    print(f"   📊 Total Registry Size: {total_symbols}")
    print(f"\n✅ All {len(all_profitable)} profitable symbols now included in registry!")

if __name__ == "__main__":
    main()