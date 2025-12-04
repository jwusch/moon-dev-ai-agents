#!/usr/bin/env python3
"""
⚡ QUICK AMEX SMALL CAP BLITZ ⚡ 
Fast scan of the best AMEX small cap volatility candidates
"""

import time
import json
from datetime import datetime
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner

def get_prime_amex_targets():
    """Get the highest-quality AMEX small cap targets (active and liquid)"""
    
    # Top-tier AMEX small caps - actively traded and volatile
    prime_targets = [
        # Biotech/Pharma - High volatility
        'ABUS', 'ACAD', 'ACHV', 'ACTG', 'ADAP', 'ADMA', 'ADVM', 'AEMD', 'AFMD',
        'AGEN', 'AIMD', 'ALDX', 'ALEC', 'ALGN', 'ALKS', 'ALNY', 'AMGN', 'AMRN',
        
        # Energy/Materials - Momentum plays
        'AREC', 'ARRY', 'ARTW', 'ARVN', 'ASMB', 'ASPS', 'ASRT', 'ATHE', 'ATNI', 
        'ATOS', 'ATRC', 'AUPH', 'AVCO', 'AVDL', 'AVGR', 'AVIR', 'AVRO', 'AXDX',
        
        # Tech/Software - Small cap growth
        'AXSM', 'AYTU', 'AZPN', 'AZRX', 'BAND', 'BBCP', 'BBIG', 'BCDA', 'BCEL',
        'BCLI', 'BDSX', 'BEAT', 'BFRI', 'BGCP', 'BGNE', 'BHAT', 'BIIB', 'BIMI',
        
        # Healthcare/Services
        'BIOC', 'BIOX', 'BITF', 'BIVI', 'BKKT', 'BLBD', 'BLCM', 'BLFS', 'BLNK',
        'BLUE', 'BLUW', 'BMEA', 'BMEX', 'BMRN', 'BNGO', 'BNTX', 'BODY', 'BOLT',
        
        # Financial/REIT small caps  
        'BOXL', 'BPMC', 'BPTH', 'BRAC', 'BRDS', 'BREZ', 'BRID', 'BRLI', 'BRQS',
        'BSGM', 'BTAI', 'BTBT', 'BWAY', 'BYFC', 'BYND', 'BZUN', 'CAAS', 'CABA'
    ]
    
    print(f"🎯 Prime AMEX targets loaded: {len(prime_targets)} symbols")
    return prime_targets

def execute_quick_blitz():
    """Execute fast AMEX small cap scan"""
    
    print("⚡💀 QUICK AMEX SMALL CAP BLITZ 💀⚡")
    print("=" * 60)
    print("🚀 Fast scan of prime AMEX volatility candidates")
    
    # Get targets
    amex_targets = get_prime_amex_targets()
    
    # Initialize scanner  
    scanner = MultiExchangeAEGSScanner()
    
    print(f"\n🏴‍☠️ Scanning {len(amex_targets)} prime AMEX small caps...")
    
    # Execute scan
    start_time = time.time()
    results = scanner.scan_multi_exchange(custom_symbols=amex_targets)
    scan_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"quick_amex_blitz_{timestamp}.json"
    
    blitz_results = {
        'blitz_date': datetime.now().isoformat(),
        'blitz_type': 'QUICK_AMEX_SMALL_CAP',
        'targets_scanned': len(amex_targets),
        'scan_duration_minutes': scan_time / 60,
        'cache_stats': scanner.cache.cache_stats,
        'results': results,
        'profitable_goldmines': len(results['profitable'])
    }
    
    with open(filename, 'w') as f:
        json.dump(blitz_results, f, indent=2)
    
    # Analysis
    analyze_quick_results(results, scan_time, filename)
    
    return blitz_results

def analyze_quick_results(results, scan_time, filename):
    """Analyze quick blitz results"""
    
    profitable = results['profitable']
    
    print(f"\n⚡💀 QUICK AMEX BLITZ COMPLETE! 💀⚡")
    print(f"⏱️  Scan time: {scan_time/60:.1f} minutes")  
    print(f"💰 Small cap goldmines found: {len(profitable)}")
    
    if profitable:
        # Sort by return
        profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        # Categorize
        monsters = [p for p in profitable if p['strategy_return'] >= 100]
        beasts = [p for p in profitable if 50 <= p['strategy_return'] < 100]  
        solid = [p for p in profitable if 20 <= p['strategy_return'] < 50]
        decent = [p for p in profitable if p['strategy_return'] < 20]
        
        print(f"\n🎯 AMEX GOLDMINE BREAKDOWN:")
        print(f"   💀 MONSTERS (≥100%): {len(monsters)}")
        print(f"   🔥 BEASTS (50-99%): {len(beasts)}")
        print(f"   ⚡ SOLID (20-49%): {len(solid)}") 
        print(f"   ✅ DECENT (<20%): {len(decent)}")
        
        print(f"\n🏆 TOP AMEX SMALL CAP PERFORMERS:")
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
            print(f"\n💀 AMEX VOLATILITY MONSTERS:")
            for monster in monsters:
                symbol = monster['symbol']
                ret = monster['strategy_return']
                print(f"   💀 {symbol}: +{ret:.1f}% - SMALL CAP LEGEND!")
    
    else:
        print("❌ No profitable AMEX small caps found")
    
    print(f"\n💾 Results saved: {filename}")
    return profitable

def main():
    """Execute quick AMEX blitz"""
    print("⚡ Starting Quick AMEX Small Cap Blitz...")
    results = execute_quick_blitz()
    
    if results['profitable_goldmines'] > 0:
        print(f"\n🔥 Ready to expand goldmine with {results['profitable_goldmines']} AMEX discoveries!")
        
        # Auto-expand goldmine
        expand_goldmine_with_amex(results)
    else:
        print("\n📊 No AMEX goldmines discovered in this scan")
    
    return results

def expand_goldmine_with_amex(blitz_results):
    """Auto-expand goldmine with AMEX discoveries"""
    
    try:
        # Load current goldmine
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        profitable = blitz_results['results']['profitable']
        additions = {'extreme_goldmines': 0, 'high_potential': 0, 'positive': 0}
        
        print(f"\n🔄 Auto-expanding goldmine with {len(profitable)} AMEX symbols...")
        
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
                    'source': 'quick_amex_blitz',
                    'exchange': 'AMEX'
                }
                additions[category] += 1
                print(f"   ✅ {symbol} → {category} (+{ret:.1f}%)")
        
        # Update metadata
        total_symbols = sum(len(registry['goldmine_symbols'][cat]) 
                          for cat in registry['goldmine_symbols'])
        
        registry['metadata'] = registry.get('metadata', {})
        registry['metadata'].update({
            'total_symbols': total_symbols,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'amex_expansion': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbols_added': sum(additions.values()),
                'additions': additions,
                'source': 'Quick AMEX Small Cap Blitz'
            }
        })
        
        # Save updated registry
        with open('aegs_goldmine_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"\n🎉 GOLDMINE EXPANDED!")
        print(f"   💀 Extreme: +{additions['extreme_goldmines']}")
        print(f"   🌟 High Potential: +{additions['high_potential']}")
        print(f"   ✅ Positive: +{additions['positive']}")
        print(f"   📈 Total Added: {sum(additions.values())}")
        print(f"   🏆 New Registry Size: {total_symbols} symbols")
        
    except Exception as e:
        print(f"❌ Error expanding goldmine: {e}")

if __name__ == "__main__":
    main()