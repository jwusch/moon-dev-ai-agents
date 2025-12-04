#!/usr/bin/env python3
"""
🧬⚡ FAST BIOTECH BLITZ ⚡🧬
Quick biotech scan using mostly cached data + top liquid biotech stocks
"""

import time
import json
from datetime import datetime
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner

def get_top_biotech_targets():
    """Get top liquid biotech stocks that are likely cached + active"""
    
    # Focus on large/mid cap biotech that are heavily traded and likely cached
    top_biotech = [
        # Large Cap Biotech (most liquid)
        'MRNA', 'BNTX', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'INCY', 'BMRN',
        
        # Mid Cap Biotech (good liquidity)  
        'CRSP', 'EDIT', 'NTLA', 'BEAM', 'FOLD', 'ARWR', 'IONS', 'EXAS', 'RARE',
        
        # Active Small/Mid Biotech (known for volatility)
        'OCGN', 'NVAX', 'CYTK', 'DVAX', 'CRVS', 'MCRB', 'RDHL', 'KPTI', 'ACRX',
        
        # High-Volume Biotech ETF Components
        'ACRS', 'AGIO', 'CGEN', 'CGEM', 'DRNA', 'NRXP', 'GNPX', 'CTIC', 'CTXR'
    ]
    
    print(f"🧬 Top biotech targets: {len(top_biotech)} liquid symbols")
    return top_biotech

def execute_fast_biotech_scan():
    """Execute fast biotech scan focusing on speed"""
    
    print("🧬⚡ FAST BIOTECH BLITZ ⚡🧬")
    print("=" * 50)
    print("🚀 Fast scan prioritizing cached biotech data")
    
    # Get targets
    biotech_targets = get_top_biotech_targets()
    
    # Initialize scanner
    scanner = MultiExchangeAEGSScanner()
    
    print(f"\n🧬 Scanning {len(biotech_targets)} top biotech symbols...")
    
    # Execute fast scan (no artificial delays)
    start_time = time.time()
    results = scanner.scan_multi_exchange(custom_symbols=biotech_targets)
    scan_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"fast_biotech_blitz_{timestamp}.json"
    
    biotech_results = {
        'blitz_date': datetime.now().isoformat(),
        'blitz_type': 'FAST_BIOTECH_BLITZ',
        'targets_scanned': len(biotech_targets),
        'scan_duration_minutes': scan_time / 60,
        'cache_stats': scanner.cache.cache_stats,
        'results': results,
        'profitable_goldmines': len(results['profitable'])
    }
    
    with open(filename, 'w') as f:
        json.dump(biotech_results, f, indent=2)
    
    # Analysis
    analyze_biotech_results(results, scan_time, filename)
    
    return biotech_results

def analyze_biotech_results(results, scan_time, filename):
    """Analyze fast biotech results"""
    
    profitable = results['profitable']
    
    print(f"\n🧬⚡ FAST BIOTECH BLITZ COMPLETE! ⚡🧬")
    print(f"⏱️  Scan time: {scan_time/60:.1f} minutes")
    print(f"🧬 Biotech goldmines: {len(profitable)}")
    
    if profitable:
        # Sort by return
        profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        # Biotech categories
        monsters = [p for p in profitable if p['strategy_return'] >= 100]
        beasts = [p for p in profitable if 50 <= p['strategy_return'] < 100]
        solid = [p for p in profitable if 20 <= p['strategy_return'] < 50]
        decent = [p for p in profitable if p['strategy_return'] < 20]
        
        print(f"\n🎯 BIOTECH GOLDMINE BREAKDOWN:")
        print(f"   💀 BIOTECH MONSTERS (≥100%): {len(monsters)}")
        print(f"   🔥 BIOTECH BEASTS (50-99%): {len(beasts)}")
        print(f"   ⚡ SOLID BIOTECH (20-49%): {len(solid)}")
        print(f"   ✅ DECENT BIOTECH (<20%): {len(decent)}")
        
        print(f"\n🏆 TOP BIOTECH PERFORMERS:")
        for i, result in enumerate(profitable[:15], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            
            if ret >= 100:
                emoji = "💀"
            elif ret >= 50:
                emoji = "🔥"
            elif ret >= 20:
                emoji = "⚡"
            else:
                emoji = "✅"
            
            print(f"   {i:2}. {emoji} {symbol:<6} +{ret:6.1f}% ({trades} trades)")
        
        if monsters:
            print(f"\n💀 BIOTECH MONSTERS:")
            for monster in monsters:
                symbol = monster['symbol']
                ret = monster['strategy_return']
                print(f"   💀 {symbol}: +{ret:.1f}% - BIOTECH LEGEND!")
    
    else:
        print("❌ No profitable biotech stocks found")
    
    print(f"\n💾 Results: {filename}")
    return profitable

def expand_goldmine_with_biotech(biotech_results):
    """Auto-expand goldmine with biotech discoveries"""
    
    try:
        # Load current goldmine
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        profitable = biotech_results['results']['profitable']
        additions = {'extreme_goldmines': 0, 'high_potential': 0, 'positive': 0}
        
        print(f"\n🔄 Expanding goldmine with {len(profitable)} biotech symbols...")
        
        for symbol_data in profitable:
            symbol = symbol_data['symbol']
            ret = symbol_data['strategy_return']
            
            # Check if exists
            exists = any(symbol in registry['goldmine_symbols'][cat] 
                        for cat in registry['goldmine_symbols'])
            
            if not exists:
                # Categorize
                if ret >= 100:
                    category = "extreme_goldmines"
                elif ret >= 30:
                    category = "high_potential"
                else:
                    category = "positive"
                
                # Add to registry
                registry['goldmine_symbols'][category][symbol] = {
                    'strategy_return': symbol_data['strategy_return'],
                    'total_trades': symbol_data['total_trades'],
                    'win_rate': symbol_data['win_rate'],
                    'excess_return': symbol_data['excess_return'],
                    'added_date': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'fast_biotech_blitz',
                    'sector': 'biotech'
                }
                additions[category] += 1
                print(f"   ✅ {symbol} → {category} (+{ret:.1f}%)")
            else:
                print(f"   ⚠️  {symbol} already exists")
        
        # Update metadata
        total_symbols = sum(len(registry['goldmine_symbols'][cat]) 
                          for cat in registry['goldmine_symbols'])
        
        registry['metadata'] = registry.get('metadata', {})
        registry['metadata'].update({
            'total_symbols': total_symbols,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fast_biotech_expansion': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbols_added': sum(additions.values()),
                'additions': additions,
                'source': 'Fast Biotech Blitz'
            }
        })
        
        # Save updated registry
        with open('aegs_goldmine_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"\n🎉 BIOTECH EXPANSION COMPLETE!")
        print(f"   💀 Extreme: +{additions['extreme_goldmines']}")
        print(f"   🌟 High Potential: +{additions['high_potential']}")
        print(f"   ✅ Positive: +{additions['positive']}")
        print(f"   📈 Total Added: {sum(additions.values())}")
        print(f"   🏆 New Registry Size: {total_symbols} symbols")
        
    except Exception as e:
        print(f"❌ Error expanding goldmine: {e}")

def main():
    """Execute fast biotech blitz"""
    print("🧬 Starting Fast Biotech Blitz...")
    results = execute_fast_biotech_scan()
    
    if results['profitable_goldmines'] > 0:
        print(f"\n🔥 Expanding goldmine with {results['profitable_goldmines']} biotech discoveries!")
        expand_goldmine_with_biotech(results)
    else:
        print("\n📊 No biotech goldmines found")
    
    return results

if __name__ == "__main__":
    main()