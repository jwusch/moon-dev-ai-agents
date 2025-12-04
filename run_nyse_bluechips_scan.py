#!/usr/bin/env python3
"""
🔥 NYSE BLUE CHIPS AEGS SCAN 🔥
Test our multi-exchange scanner on major NYSE stocks
"""

import time
from datetime import datetime
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner

def main():
    print("🔥💎 NYSE BLUE CHIPS AEGS SCAN 💎🔥")
    print("=" * 50)
    
    # Initialize scanner
    scanner = MultiExchangeAEGSScanner()
    
    # Get NYSE blue chip symbols
    nyse_bluechips = [
        'JPM', 'JNJ', 'WMT', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC',
        'ABT', 'TMO', 'CVX', 'LLY', 'ABBV', 'PFE', 'KO', 'DHR', 'VZ', 'ACN',
        'MRK', 'T', 'BMY', 'LIN', 'WFC', 'PM', 'RTX', 'NEE', 'HON', 'UPS'
    ]
    
    print(f"🎯 Scanning {len(nyse_bluechips)} NYSE blue chip stocks")
    print("💡 These are large, established companies - let's see if AEGS works!")
    
    # Run the scan
    start_time = time.time()
    results = scanner.scan_multi_exchange(custom_symbols=nyse_bluechips)
    scan_time = time.time() - start_time
    
    # Save results
    filename = scanner.save_results(results, "nyse_bluechips")
    
    # Analysis
    profitable = results['profitable']
    
    print(f"\n🎉 NYSE BLUE CHIPS SCAN COMPLETE!")
    print(f"⏱️  Total time: {scan_time/60:.1f} minutes")
    print(f"📊 Cache stats: {scanner.cache.cache_stats}")
    print(f"💰 Profitable symbols: {len(profitable)}/{len(nyse_bluechips)}")
    
    if profitable:
        profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        print(f"\n🏆 PROFITABLE NYSE BLUE CHIPS:")
        for i, result in enumerate(profitable, 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            win_rate = result['win_rate']
            print(f"   {i:2}. {symbol:<5} +{ret:6.1f}% ({trades} trades, {win_rate:.0f}% wins)")
    else:
        print("❌ No profitable NYSE blue chips found with AEGS strategy")
    
    print(f"\n💾 Results saved to: {filename}")
    return results

if __name__ == "__main__":
    main()