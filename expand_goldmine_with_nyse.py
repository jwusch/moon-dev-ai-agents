#!/usr/bin/env python3
"""
🔥 EXPAND GOLDMINE WITH NYSE WINNERS 🔥
Add the 16 profitable NYSE blue chips to our AEGS goldmine registry
"""

import json
from datetime import datetime

def load_nyse_results():
    """Load the latest NYSE blue chips results"""
    try:
        # Find the latest NYSE results file
        import glob
        nyse_files = glob.glob("nyse_bluechips_aegs_scan_*.json")
        if not nyse_files:
            print("❌ No NYSE results file found!")
            return None
        
        latest_file = max(nyse_files)
        print(f"📄 Loading NYSE results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            nyse_data = json.load(f)
        
        return nyse_data['profitable_symbols']
        
    except Exception as e:
        print(f"❌ Error loading NYSE results: {e}")
        return None

def load_current_goldmine():
    """Load current goldmine registry"""
    try:
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        return registry
    except Exception as e:
        print(f"❌ Error loading goldmine registry: {e}")
        return None

def categorize_nyse_symbol(symbol_data):
    """Categorize NYSE symbol based on performance"""
    strategy_return = symbol_data['strategy_return']
    
    if strategy_return >= 100:
        return "extreme_goldmines"
    elif strategy_return >= 30:
        return "high_potential" 
    else:
        return "positive"

def add_nyse_to_goldmine(nyse_symbols, registry):
    """Add NYSE symbols to the goldmine registry"""
    
    print(f"🔥 Adding {len(nyse_symbols)} NYSE symbols to goldmine registry")
    
    # Track additions
    additions_by_category = {
        "extreme_goldmines": 0,
        "high_potential": 0,
        "positive": 0
    }
    
    # Add each NYSE symbol
    for symbol_data in nyse_symbols:
        symbol = symbol_data['symbol']
        category = categorize_nyse_symbol(symbol_data)
        
        # Check if symbol already exists
        exists = False
        for cat in registry['goldmine_symbols']:
            if symbol in registry['goldmine_symbols'][cat]:
                print(f"⚠️  {symbol} already exists in {cat} category")
                exists = True
                break
        
        if not exists:
            # Add to appropriate category
            registry['goldmine_symbols'][category][symbol] = {
                'strategy_return': symbol_data['strategy_return'],
                'total_trades': symbol_data['total_trades'],
                'win_rate': symbol_data['win_rate'],
                'excess_return': symbol_data['excess_return'],
                'added_date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'nyse_bluechips_scan',
                'exchange': 'NYSE'
            }
            additions_by_category[category] += 1
            print(f"✅ Added {symbol} to {category} (+{symbol_data['strategy_return']:.1f}%)")
    
    return additions_by_category

def update_metadata(registry, additions):
    """Update registry metadata"""
    
    # Calculate new totals
    total_symbols = sum(len(registry['goldmine_symbols'][cat]) for cat in registry['goldmine_symbols'])
    
    # Update metadata
    if 'metadata' not in registry:
        registry['metadata'] = {}
    
    registry['metadata'].update({
        'total_symbols': total_symbols,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nyse_expansion': {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'symbols_added': sum(additions.values()),
            'additions_by_category': additions,
            'source': 'NYSE Blue Chips AEGS Scan'
        }
    })
    
    # Update complete alphabet scan metadata if it exists
    if 'complete_alphabet_scan' in registry['metadata']:
        registry['metadata']['complete_alphabet_scan']['total_alphabet_profitable'] += sum(additions.values())

def main():
    print("🔥💎 EXPANDING GOLDMINE WITH NYSE WINNERS 💎🔥")
    print("=" * 60)
    
    # Load NYSE results
    nyse_symbols = load_nyse_results()
    if not nyse_symbols:
        return
    
    print(f"📊 Found {len(nyse_symbols)} profitable NYSE symbols")
    
    # Load current goldmine
    registry = load_current_goldmine()
    if not registry:
        return
    
    current_total = sum(len(registry['goldmine_symbols'][cat]) for cat in registry['goldmine_symbols'])
    print(f"📈 Current goldmine size: {current_total} symbols")
    
    # Add NYSE symbols
    additions = add_nyse_to_goldmine(nyse_symbols, registry)
    
    # Update metadata
    update_metadata(registry, additions)
    
    # Save updated registry
    with open('aegs_goldmine_registry.json', 'w') as f:
        json.dump(registry, f, indent=2)
    
    # Summary
    new_total = sum(len(registry['goldmine_symbols'][cat]) for cat in registry['goldmine_symbols'])
    
    print(f"\n🎉 GOLDMINE EXPANSION COMPLETE!")
    print(f"📊 Updated Registry Stats:")
    print(f"   💎 Extreme Goldmines (≥100%): {len(registry['goldmine_symbols']['extreme_goldmines'])} (+{additions['extreme_goldmines']})")
    print(f"   🌟 High Potential (30-99%): {len(registry['goldmine_symbols']['high_potential'])} (+{additions['high_potential']})")
    print(f"   ✅ Positive (1-29%): {len(registry['goldmine_symbols']['positive'])} (+{additions['positive']})")
    print(f"   📈 Total Registry: {current_total} → {new_total} (+{sum(additions.values())})")
    
    print(f"\n🏆 NYSE Blue Chip Winners Added:")
    for symbol_data in sorted(nyse_symbols, key=lambda x: x['strategy_return'], reverse=True):
        symbol = symbol_data['symbol']
        ret = symbol_data['strategy_return']
        trades = symbol_data['total_trades']
        print(f"   {symbol:<5} +{ret:5.1f}% ({trades} trades)")
    
    print(f"\n✅ Multi-exchange goldmine now contains NYSE + NASDAQ winners!")

if __name__ == "__main__":
    main()