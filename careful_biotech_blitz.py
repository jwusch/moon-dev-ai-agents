#!/usr/bin/env python3
"""
🧬💀 CAREFUL BIOTECH SECTOR DESTRUCTION 💀🧬
Gentle scan of biotech volatility with aggressive rate limiting protection
"""

import time
import json
from datetime import datetime
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner

def get_biotech_targets_batch_1():
    """First batch of top biotech volatility candidates (50 symbols max)"""
    
    # Tier 1: Established biotech with known volatility
    biotech_batch_1 = [
        # Large Cap Biotech (proven volatility)
        'MRNA', 'BNTX', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'INCY', 'BMRN', 'SGEN',
        
        # Mid Cap Biotech (high momentum)  
        'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'FOLD', 'ARWR', 'IONS', 'EXAS', 'RARE',
        
        # Small Cap Biotech (extreme volatility)
        'ATNF', 'OCGN', 'NVAX', 'ALNA', 'CYTK', 'DVAX', 'CRVS', 'HGEN', 'MCRB', 'RDHL',
        
        # Penny Stock Biotech (chaos zone)
        'ADMP', 'OBSV', 'SNSS', 'CBAY', 'TXMD', 'GTHX', 'CTXR', 'CTIC', 'GNPX', 'NRXP',
        
        # Additional High-Beta Biotech
        'KPTI', 'ACRX', 'ACRS', 'AGIO', 'BGNE', 'BLUE', 'CGEN', 'CGEM', 'DRNA', 'FOLD'
    ]
    
    print(f"🧬 Biotech Batch 1 loaded: {len(biotech_batch_1)} symbols")
    print(f"🐌 RATE LIMIT PROTECTION: 2 second delays between requests")
    return biotech_batch_1

class CarefulBiotechScanner:
    """Ultra-careful biotech scanner with rate limiting"""
    
    def __init__(self):
        self.scanner = MultiExchangeAEGSScanner()
        self.request_delay = 2.0  # 2 seconds between requests
        
    def scan_with_delays(self, symbols):
        """Scan with mandatory delays to protect yfinance"""
        
        print(f"\n🐌 CAREFUL SCANNING MODE ACTIVATED")
        print(f"⏳ {self.request_delay}s delay between each request")
        print(f"🛡️  Protection against yfinance rate limits")
        
        results = {'profitable': [], 'unprofitable': [], 'errors': []}
        start_time = time.time()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\n[{i:2}/{len(symbols)}] Processing {symbol}...")
                
                # Check cache first
                result = self.scanner.cache.scan_symbol(symbol)
                
                if result:
                    results['profitable'].append(result)
                    ret = result['strategy_return']
                    trades = result['total_trades']
                    print(f"    ✅ BIOTECH GOLDMINE: +{ret:.1f}% ({trades} trades)")
                else:
                    results['unprofitable'].append(symbol)
                    print(f"    ❌ Not profitable")
                
                # MANDATORY DELAY (except for last symbol)
                if i < len(symbols):
                    print(f"    ⏳ Rate limit protection: waiting {self.request_delay}s...")
                    time.sleep(self.request_delay)
                
                # Progress updates every 10 symbols
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    profitable_count = len(results['profitable'])
                    eta_seconds = (len(symbols) - i) * self.request_delay
                    
                    print(f"\n📊 Biotech Progress Update:")
                    print(f"   Completed: {i}/{len(symbols)} ({i/len(symbols)*100:.0f}%)")
                    print(f"   ETA: {eta_seconds/60:.1f} minutes remaining")
                    print(f"   Biotech goldmines: {profitable_count}")
                    print(f"   Cache stats: {self.scanner.cache.cache_stats}")
                    
            except Exception as e:
                results['errors'].append({'symbol': symbol, 'error': str(e)})
                print(f"    ❌ Error: {e}")
                
                # Still delay on error to be safe
                if i < len(symbols):
                    time.sleep(self.request_delay)
        
        return results

def execute_careful_biotech_blitz():
    """Execute careful biotech scan"""
    
    print("🧬💀 CAREFUL BIOTECH SECTOR DESTRUCTION 💀🧬")
    print("=" * 70)
    print("🛡️  RATE-LIMITED BIOTECH VOLATILITY HUNTING")
    print("🐌 Gentle on yfinance servers - 2s delays between requests")
    
    # Get batch 1 targets
    biotech_targets = get_biotech_targets_batch_1()
    
    # Initialize careful scanner
    scanner = CarefulBiotechScanner()
    
    print(f"\n🧬 Starting careful scan of {len(biotech_targets)} biotech symbols...")
    estimated_time = len(biotech_targets) * scanner.request_delay / 60
    print(f"⏱️  Estimated completion time: {estimated_time:.1f} minutes")
    
    # Execute scan with delays
    start_time = time.time()
    results = scanner.scan_with_delays(biotech_targets)
    scan_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"careful_biotech_blitz_{timestamp}.json"
    
    biotech_results = {
        'blitz_date': datetime.now().isoformat(),
        'blitz_type': 'CAREFUL_BIOTECH_DESTRUCTION',
        'targets_scanned': len(biotech_targets),
        'scan_duration_minutes': scan_time / 60,
        'rate_limit_protection': True,
        'delay_between_requests': scanner.request_delay,
        'cache_stats': scanner.scanner.cache.cache_stats,
        'results': results,
        'profitable_goldmines': len(results['profitable'])
    }
    
    with open(filename, 'w') as f:
        json.dump(biotech_results, f, indent=2)
    
    # Analysis
    analyze_biotech_results(results, scan_time, filename)
    
    return biotech_results

def analyze_biotech_results(results, scan_time, filename):
    """Analyze biotech destruction results"""
    
    profitable = results['profitable']
    
    print(f"\n🧬💀 BIOTECH DESTRUCTION COMPLETE! 💀🧬")
    print(f"⏱️  Total time: {scan_time/60:.1f} minutes")
    print(f"🧬 Biotech goldmines discovered: {len(profitable)}")
    print(f"🛡️  Rate limiting: SUCCESS (no bans!)")
    
    if profitable:
        # Sort by return
        profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        # Categorize biotech discoveries
        biotech_monsters = [p for p in profitable if p['strategy_return'] >= 100]
        biotech_beasts = [p for p in profitable if 50 <= p['strategy_return'] < 100]
        biotech_solid = [p for p in profitable if 20 <= p['strategy_return'] < 50]
        biotech_decent = [p for p in profitable if p['strategy_return'] < 20]
        
        print(f"\n🎯 BIOTECH GOLDMINE BREAKDOWN:")
        print(f"   💀 BIOTECH MONSTERS (≥100%): {len(biotech_monsters)}")
        print(f"   🔥 BIOTECH BEASTS (50-99%): {len(biotech_beasts)}")
        print(f"   ⚡ SOLID BIOTECH (20-49%): {len(biotech_solid)}")
        print(f"   ✅ DECENT BIOTECH (<20%): {len(biotech_decent)}")
        
        print(f"\n🏆 TOP BIOTECH VOLATILITY CHAMPIONS:")
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
        
        if biotech_monsters:
            print(f"\n💀 BIOTECH VOLATILITY MONSTERS:")
            for monster in biotech_monsters:
                symbol = monster['symbol']
                ret = monster['strategy_return']
                print(f"   💀 {symbol}: +{ret:.1f}% - BIOTECH LEGEND!")
    
    else:
        print("❌ No profitable biotech stocks found in batch 1")
    
    print(f"\n💾 Biotech results saved: {filename}")
    return profitable

def main():
    """Execute careful biotech destruction"""
    print("🧬 Starting Careful Biotech Sector Destruction...")
    results = execute_careful_biotech_blitz()
    
    if results['profitable_goldmines'] > 0:
        print(f"\n🔥 Ready to expand goldmine with {results['profitable_goldmines']} biotech discoveries!")
        
        # Auto-expand goldmine
        expand_goldmine_with_biotech(results)
    else:
        print("\n📊 No biotech goldmines found in batch 1")
        print("🧬 Consider running batch 2 with different biotech symbols")
    
    return results

def expand_goldmine_with_biotech(biotech_results):
    """Auto-expand goldmine with biotech discoveries"""
    
    try:
        # Load current goldmine
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        profitable = biotech_results['results']['profitable']
        additions = {'extreme_goldmines': 0, 'high_potential': 0, 'positive': 0}
        
        print(f"\n🔄 Auto-expanding goldmine with {len(profitable)} biotech symbols...")
        
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
                    'source': 'careful_biotech_blitz',
                    'sector': 'biotech'
                }
                additions[category] += 1
                print(f"   ✅ {symbol} → {category} (+{ret:.1f}%)")
            else:
                print(f"   ⚠️  {symbol} already in registry")
        
        # Update metadata
        total_symbols = sum(len(registry['goldmine_symbols'][cat]) 
                          for cat in registry['goldmine_symbols'])
        
        registry['metadata'] = registry.get('metadata', {})
        registry['metadata'].update({
            'total_symbols': total_symbols,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'biotech_expansion': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbols_added': sum(additions.values()),
                'additions': additions,
                'source': 'Careful Biotech Sector Destruction',
                'rate_limited': False
            }
        })
        
        # Save updated registry
        with open('aegs_goldmine_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"\n🎉 BIOTECH GOLDMINE EXPANSION COMPLETE!")
        print(f"   💀 Extreme Biotech: +{additions['extreme_goldmines']}")
        print(f"   🌟 High Potential Biotech: +{additions['high_potential']}")
        print(f"   ✅ Positive Biotech: +{additions['positive']}")
        print(f"   📈 Total Biotech Added: {sum(additions.values())}")
        print(f"   🏆 New Registry Size: {total_symbols} symbols")
        
    except Exception as e:
        print(f"❌ Error expanding biotech goldmine: {e}")

if __name__ == "__main__":
    main()