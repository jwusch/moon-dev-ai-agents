#!/usr/bin/env python3
"""
💎 GOLDMINE REFRESH SCAN 💎
Scan existing goldmine symbols for fresh AEGS signals
"""

import json
import time
from datetime import datetime
from working_cached_aegs_scanner import WorkingCacheAEGS

def load_goldmine_registry():
    """Load existing goldmine registry"""
    
    try:
        with open('aegs_goldmine_registry.json', 'r') as f:
            registry = json.load(f)
        
        print("💎 GOLDMINE REGISTRY LOADED")
        print("=" * 50)
        
        # Extract all symbols
        all_symbols = []
        category_counts = {}
        
        for category in registry['goldmine_symbols']:
            symbols = list(registry['goldmine_symbols'][category].keys())
            all_symbols.extend(symbols)
            category_counts[category] = len(symbols)
            
        print("📊 GOLDMINE BREAKDOWN:")
        total_symbols = 0
        for category, count in category_counts.items():
            total_symbols += count
            print(f"   {category.replace('_', ' ').title():<20}: {count:>3} symbols")
        
        print(f"   {'TOTAL GOLDMINES':<20}: {total_symbols:>3} symbols")
        
        return all_symbols, registry
        
    except Exception as e:
        print(f"❌ Error loading goldmine registry: {e}")
        return [], {}

def scan_goldmine_for_signals(goldmine_symbols):
    """Scan goldmine symbols for fresh AEGS signals"""
    
    print(f"\n💎 SCANNING {len(goldmine_symbols)} GOLDMINE SYMBOLS")
    print("=" * 60)
    print("🎯 Looking for fresh AEGS oversold bounce opportunities")
    print("⚡ Using cached data for speed")
    
    # Initialize scanner
    scanner = WorkingCacheAEGS()
    
    results = {
        'still_profitable': [],
        'no_longer_profitable': [],
        'errors': []
    }
    
    start_time = time.time()
    
    for i, symbol in enumerate(goldmine_symbols, 1):
        try:
            print(f"\n[{i:3}/{len(goldmine_symbols)}] Scanning {symbol}...")
            
            # Scan symbol with latest data
            result = scanner.scan_symbol(symbol)
            
            if result:
                results['still_profitable'].append(result)
                ret = result['strategy_return']
                trades = result['total_trades']
                win_rate = result['win_rate']
                print(f"    ✅ STILL PROFITABLE: +{ret:.1f}% ({trades} trades, {win_rate:.0f}% wins)")
            else:
                results['no_longer_profitable'].append(symbol)
                print(f"    ❌ No longer profitable")
            
            # Progress updates every 25 symbols
            if i % 25 == 0:
                elapsed = time.time() - start_time
                profitable_count = len(results['still_profitable'])
                unprofitable_count = len(results['no_longer_profitable'])
                
                print(f"\n📊 GOLDMINE REFRESH PROGRESS:")
                print(f"   Scanned: {i}/{len(goldmine_symbols)} ({i/len(goldmine_symbols)*100:.1f}%)")
                print(f"   Still profitable: {profitable_count}")
                print(f"   No longer profitable: {unprofitable_count}")
                print(f"   Success rate: {profitable_count/(profitable_count+unprofitable_count)*100:.1f}%")
                print(f"   Cache stats: {scanner.cache_stats}")
                
        except Exception as e:
            results['errors'].append({'symbol': symbol, 'error': str(e)})
            print(f"    ❌ Error: {e}")
    
    scan_time = time.time() - start_time
    return results, scan_time

def analyze_goldmine_refresh(results, scan_time):
    """Analyze goldmine refresh results"""
    
    still_profitable = results['still_profitable']
    no_longer_profitable = results['no_longer_profitable']
    errors = results['errors']
    
    total_scanned = len(still_profitable) + len(no_longer_profitable) + len(errors)
    
    print(f"\n💎 GOLDMINE REFRESH COMPLETE! 💎")
    print("=" * 60)
    print(f"⏱️  Scan time: {scan_time/60:.1f} minutes")
    print(f"📊 Goldmines scanned: {total_scanned}")
    print(f"✅ Still profitable: {len(still_profitable)}")
    print(f"❌ No longer profitable: {len(no_longer_profitable)}")
    print(f"⚠️  Errors: {len(errors)}")
    
    if total_scanned > 0:
        success_rate = len(still_profitable) / total_scanned * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    if still_profitable:
        # Sort by return
        still_profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        # Categorize current performance
        monsters = [p for p in still_profitable if p['strategy_return'] >= 100]
        beasts = [p for p in still_profitable if 50 <= p['strategy_return'] < 100]
        solid = [p for p in still_profitable if 20 <= p['strategy_return'] < 50]
        decent = [p for p in still_profitable if p['strategy_return'] < 20]
        
        print(f"\n🎯 CURRENT GOLDMINE PERFORMANCE:")
        print(f"   💀 MONSTERS (≥100%): {len(monsters)} goldmines")
        print(f"   🔥 BEASTS (50-99%): {len(beasts)} goldmines")
        print(f"   ⚡ SOLID (20-49%): {len(solid)} goldmines")
        print(f"   ✅ DECENT (<20%): {len(decent)} goldmines")
        
        print(f"\n🏆 TOP 20 REFRESHED GOLDMINES:")
        for i, result in enumerate(still_profitable[:20], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            win_rate = result['win_rate']
            
            if ret >= 100:
                emoji = "💀"
            elif ret >= 50:
                emoji = "🔥"
            elif ret >= 20:
                emoji = "⚡"
            else:
                emoji = "✅"
            
            print(f"   {i:2}. {emoji} {symbol:<6} +{ret:6.1f}% ({trades} trades, {win_rate:.0f}% wins)")
        
        # Highlight the absolute legends
        if monsters:
            print(f"\n💀 CURRENT GOLDMINE MONSTERS:")
            for monster in monsters[:10]:  # Top 10 monsters
                symbol = monster['symbol']
                ret = monster['strategy_return']
                trades = monster['total_trades']
                print(f"   💀 {symbol}: +{ret:.1f}% ({trades} trades) - STILL A LEGEND!")
    
    if no_longer_profitable:
        print(f"\n📉 GOLDMINES NO LONGER PROFITABLE:")
        print(f"   These {len(no_longer_profitable)} symbols may need to be removed from goldmine:")
        
        # Show first 20
        for i, symbol in enumerate(no_longer_profitable[:20], 1):
            print(f"   {i:2}. {symbol}")
        
        if len(no_longer_profitable) > 20:
            print(f"   ... and {len(no_longer_profitable) - 20} more")
    
    return still_profitable, no_longer_profitable

def save_goldmine_refresh_results(results, scan_time, still_profitable, no_longer_profitable):
    """Save goldmine refresh results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"goldmine_refresh_scan_{timestamp}.json"
    
    refresh_results = {
        'refresh_date': datetime.now().isoformat(),
        'scan_type': 'goldmine_refresh',
        'scan_duration_minutes': scan_time / 60,
        'summary': {
            'total_scanned': len(results['still_profitable']) + len(results['no_longer_profitable']) + len(results['errors']),
            'still_profitable_count': len(still_profitable),
            'no_longer_profitable_count': len(no_longer_profitable),
            'error_count': len(results['errors'])
        },
        'still_profitable': still_profitable,
        'no_longer_profitable': no_longer_profitable,
        'errors': results['errors']
    }
    
    with open(filename, 'w') as f:
        json.dump(refresh_results, f, indent=2)
    
    print(f"\n💾 Goldmine refresh results saved: {filename}")
    return filename

def main():
    """Execute goldmine refresh scan"""
    
    print("💎 GOLDMINE REFRESH SCAN INITIATED")
    print("=" * 70)
    print("🎯 Scanning existing goldmines for fresh AEGS signals")
    print("🔄 Using latest market data to verify profitability")
    
    # Load goldmine symbols
    goldmine_symbols, registry = load_goldmine_registry()
    
    if not goldmine_symbols:
        print("❌ No goldmine symbols found!")
        return
    
    # Scan goldmine for fresh signals
    results, scan_time = scan_goldmine_for_signals(goldmine_symbols)
    
    # Analyze results
    still_profitable, no_longer_profitable = analyze_goldmine_refresh(results, scan_time)
    
    # Save results
    filename = save_goldmine_refresh_results(results, scan_time, still_profitable, no_longer_profitable)
    
    print(f"\n💎🚀 GOLDMINE REFRESH MISSION COMPLETE! 🚀💎")
    print(f"   📊 {len(still_profitable)} goldmines still profitable")
    print(f"   📉 {len(no_longer_profitable)} goldmines may need review")
    print(f"   💾 Results: {filename}")

if __name__ == "__main__":
    main()