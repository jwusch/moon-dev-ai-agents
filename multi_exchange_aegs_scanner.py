#!/usr/bin/env python3
"""
🌍 MULTI-EXCHANGE AEGS SCANNER 🌍
Scan symbols across NASDAQ, NYSE, AMEX using cached approach
"""

import json
import os
import time
from working_cached_aegs_scanner import WorkingCacheAEGS
from datetime import datetime

class MultiExchangeAEGSScanner:
    """AEGS scanner for multiple exchanges"""
    
    def __init__(self, cache_dir="multi_exchange_cache"):
        self.cache = WorkingCacheAEGS(cache_dir)
        
    def get_major_index_symbols(self):
        """Get symbols from major indices"""
        
        # S&P 500 Tech Giants (Known to be liquid and active)
        sp500_tech = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 
            'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'COST', 'TMUS', 'PYPL'
        ]
        
        # NYSE Blue Chips
        nyse_bluechips = [
            'JPM', 'JNJ', 'WMT', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC',
            'ABT', 'TMO', 'CVX', 'LLY', 'ABBV', 'PFE', 'KO', 'DHR', 'VZ', 'ACN',
            'MRK', 'T', 'BMY', 'LIN', 'WFC', 'PM', 'RTX', 'NEE', 'HON', 'UPS'
        ]
        
        # High-volatility/momentum stocks (good for AEGS)
        volatility_candidates = [
            'GME', 'AMC', 'BB', 'TLRY', 'SNDL', 'NOK', 'PLTR', 'WISH', 'CLOV', 'SPCE',
            'LCID', 'RIVN', 'F', 'NIO', 'XPEV', 'LI', 'SOFI', 'HOOD', 'COIN', 'SQ'
        ]
        
        # Biotech/Small caps (often volatile)
        biotech_smallcap = [
            'MRNA', 'BNTX', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'INCY', 'BMRN', 'SGEN',
            'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'BLUE', 'FOLD', 'ARWR', 'IONS', 'EXAS'
        ]
        
        # Popular ETFs (for market context)
        major_etfs = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL'
        ]
        
        return {
            'SP500_TECH': sp500_tech,
            'NYSE_BLUE_CHIPS': nyse_bluechips, 
            'VOLATILITY': volatility_candidates,
            'BIOTECH_SMALL_CAP': biotech_smallcap,
            'MAJOR_ETFS': major_etfs
        }
    
    def get_existing_goldmine_symbols(self):
        """Get symbols from our existing goldmine registry"""
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            goldmine_symbols = []
            for category in registry['goldmine_symbols']:
                goldmine_symbols.extend(list(registry['goldmine_symbols'][category].keys()))
            
            print(f"📈 Loaded {len(goldmine_symbols)} symbols from AEGS goldmine registry")
            return goldmine_symbols
            
        except Exception as e:
            print(f"⚠️  Could not load goldmine registry: {e}")
            return []
    
    def get_comprehensive_symbol_list(self):
        """Get comprehensive symbol list from multiple sources"""
        
        all_symbols = {
            'GOLDMINE': self.get_existing_goldmine_symbols(),
        }
        
        # Add major indices
        major_indices = self.get_major_index_symbols()
        all_symbols.update(major_indices)
        
        # Create deduplicated master list
        master_list = []
        seen = set()
        
        for category, symbols in all_symbols.items():
            for symbol in symbols:
                if symbol not in seen:
                    master_list.append(symbol)
                    seen.add(symbol)
        
        print(f"\n📊 SYMBOL BREAKDOWN:")
        for category, symbols in all_symbols.items():
            print(f"   {category:<15}: {len(symbols):>3} symbols")
        print(f"   {'TOTAL UNIQUE':<15}: {len(master_list):>3} symbols")
        
        return master_list
    
    def scan_multi_exchange(self, custom_symbols=None, max_symbols=None):
        """Scan symbols across multiple exchanges"""
        
        print("🌍 MULTI-EXCHANGE AEGS SCAN")
        print("=" * 60)
        
        # Get symbol list
        if custom_symbols:
            symbols = custom_symbols
            print(f"🎯 Using custom symbol list: {len(symbols)} symbols")
        else:
            symbols = self.get_comprehensive_symbol_list()
        
        if max_symbols and len(symbols) > max_symbols:
            print(f"⚠️  Limiting to first {max_symbols} symbols")
            symbols = symbols[:max_symbols]
        
        print(f"\n🚀 Starting scan of {len(symbols)} symbols")
        print("💡 Using cached data where available")
        
        # Track results by source
        results_by_category = {
            'profitable': [],
            'unprofitable': [],
            'errors': []
        }
        
        start_time = time.time()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\n[{i:3}/{len(symbols)}] Processing {symbol}...")
                
                result = self.cache.scan_symbol(symbol)
                
                if result:
                    results_by_category['profitable'].append(result)
                    ret = result['strategy_return']
                    trades = result['total_trades']
                    win_rate = result['win_rate']
                    print(f"    ✅ PROFITABLE: +{ret:.1f}% ({trades} trades, {win_rate:.0f}% wins)")
                else:
                    results_by_category['unprofitable'].append(symbol)
                    print(f"    ❌ Not profitable")
                
                # Progress updates
                if i % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(symbols) - i) / rate / 60
                    profitable_count = len(results_by_category['profitable'])
                    
                    print(f"\n📊 Progress Update:")
                    print(f"   Completed: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")
                    print(f"   Time remaining: {eta:.1f} minutes")
                    print(f"   Cache stats: {self.cache.cache_stats}")
                    print(f"   Profitable found: {profitable_count}")
                    
            except Exception as e:
                results_by_category['errors'].append({'symbol': symbol, 'error': str(e)})
                print(f"    ❌ Error: {e}")
        
        return results_by_category
    
    def save_results(self, results, scan_type="multi_exchange"):
        """Save scan results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{scan_type}_aegs_scan_{timestamp}.json"
        
        # Prepare output
        profitable = results['profitable']
        
        output = {
            'scan_date': datetime.now().isoformat(),
            'scan_type': scan_type,
            'cache_stats': self.cache.cache_stats,
            'summary': {
                'total_scanned': len(profitable) + len(results['unprofitable']) + len(results['errors']),
                'profitable_count': len(profitable),
                'unprofitable_count': len(results['unprofitable']),
                'error_count': len(results['errors'])
            },
            'profitable_symbols': profitable,
            'errors': results['errors']
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        return filename

def main():
    """Run multi-exchange AEGS scan"""
    
    print("🌍💎 MULTI-EXCHANGE AEGS SCANNER 💎🌍")
    print("=" * 70)
    
    scanner = MultiExchangeAEGSScanner()
    
    # Choice of scan types
    print("\n🎯 Scan Options:")
    print("1. Full multi-exchange scan (all sources)")
    print("2. Major indices only (SP500 + NYSE bluechips)")
    print("3. High volatility candidates")
    print("4. Existing goldmine expansion")
    print("5. Custom symbol list")
    
    choice = input("\nSelect scan type (1-5): ").strip()
    
    if choice == "1":
        # Full scan
        results = scanner.scan_multi_exchange()
        scan_type = "full_multi_exchange"
        
    elif choice == "2":
        # Major indices only
        major_symbols = []
        indices = scanner.get_major_index_symbols()
        for category in ['SP500_TECH', 'NYSE_BLUE_CHIPS']:
            major_symbols.extend(indices[category])
        
        results = scanner.scan_multi_exchange(custom_symbols=major_symbols)
        scan_type = "major_indices"
        
    elif choice == "3":
        # High volatility
        volatility_symbols = scanner.get_major_index_symbols()['VOLATILITY']
        results = scanner.scan_multi_exchange(custom_symbols=volatility_symbols)
        scan_type = "high_volatility"
        
    elif choice == "4":
        # Goldmine expansion
        goldmine_symbols = scanner.get_existing_goldmine_symbols()
        results = scanner.scan_multi_exchange(custom_symbols=goldmine_symbols)
        scan_type = "goldmine_expansion"
        
    else:
        # Custom
        symbol_input = input("Enter symbols (comma-separated): ")
        custom_symbols = [s.strip().upper() for s in symbol_input.split(',')]
        results = scanner.scan_multi_exchange(custom_symbols=custom_symbols)
        scan_type = "custom"
    
    # Save and summarize
    filename = scanner.save_results(results, scan_type)
    
    profitable = results['profitable']
    
    print(f"\n🎉 MULTI-EXCHANGE SCAN COMPLETE!")
    print(f"💾 Results saved to: {filename}")
    print(f"📊 Summary:")
    print(f"   Profitable symbols: {len(profitable)}")
    print(f"   Cache efficiency: {scanner.cache.cache_stats}")
    
    if profitable:
        # Sort by return
        profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        print(f"\n🏆 Top 15 Multi-Exchange Performers:")
        for i, result in enumerate(profitable[:15], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            print(f"   {i:2}. {symbol:<6} +{ret:6.1f}% ({trades} trades)")

if __name__ == "__main__":
    import time
    main()