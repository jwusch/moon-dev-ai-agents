"""
🔥💎 AEGS CACHE INTEGRATION 💎🔥
Integrate YFinanceCache into AEGS system for efficient data management
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yfinance_cache_demo import YFinanceCache
from datetime import datetime
import pandas as pd
from termcolor import colored

class AEGSDataCache:
    """
    AEGS-specific data caching layer
    """
    
    def __init__(self):
        # Use dedicated AEGS cache directory
        self.cache = YFinanceCache(cache_dir="aegs_yfinance_cache")
        self.symbols_tested = set()
        
    def get_backtest_data(self, symbol, period="730d", interval="1d"):
        """Get data for backtesting with AEGS-specific defaults"""
        print(f"📦 Fetching {symbol} data for backtesting...")
        
        # Use cache with AEGS defaults
        df = self.cache.get_data(symbol, period=period, interval=interval)
        
        if len(df) > 0:
            self.symbols_tested.add(symbol)
            print(f"   ✅ Loaded {len(df)} days of data")
        
        return df
    
    def get_scan_data(self, symbol, period="1mo", interval="1h"):
        """Get data for live scanning"""
        # For scans, we want fresher data - force refresh if older than 1 hour
        df = self.cache.get_data(symbol, period=period, interval=interval, force_refresh=False)
        return df
    
    def get_discovery_data(self, symbol, period="3mo", interval="1d"):
        """Get data for discovery agent analysis"""
        return self.cache.get_data(symbol, period=period, interval=interval)
    
    def show_cache_status(self):
        """Display AEGS cache statistics"""
        stats = self.cache.get_cache_stats()
        
        print(colored("\n📊 AEGS CACHE STATUS", 'cyan', attrs=['bold']))
        print("=" * 60)
        print(f"Total cached symbols: {stats['total_items']}")
        print(f"Cache size: {stats['total_size_mb']:.1f} MB")
        print(f"Symbols tested this session: {len(self.symbols_tested)}")
        
        # Show cached symbols by type
        if stats['by_interval']:
            print("\nCached data by timeframe:")
            for interval, count in stats['by_interval'].items():
                print(f"  {interval}: {count} symbols")
        
        # Show top cached symbols
        if stats['by_symbol']:
            print("\nTop cached symbols:")
            sorted_symbols = sorted(stats['by_symbol'].items(), key=lambda x: x[1], reverse=True)
            for symbol, count in sorted_symbols[:10]:
                print(f"  {symbol}: {count} timeframes")
    
    def cleanup_old_cache(self, days=7):
        """Clean up old cache files"""
        cleared = self.cache.clear_old_cache(max_age_days=days)
        print(f"🧹 Cleared {cleared} old cache files")
        return cleared
    
    def get_bb_data_with_cache(self):
        """Get BB (BlackBerry) data using cache"""
        print(colored("\n🔍 Fetching BB (BlackBerry) data with cache", 'yellow'))
        
        # Get multiple timeframes for comprehensive analysis
        data = {}
        
        # Daily data for backtesting
        data['daily'] = self.get_backtest_data('BB', period='730d', interval='1d')
        
        # Hourly data for scanning
        data['hourly'] = self.get_scan_data('BB', period='1mo', interval='1h')
        
        # 5-minute data for intraday analysis
        data['5min'] = self.cache.get_data('BB', period='5d', interval='5m')
        
        # Show what we cached
        print(f"\n📦 BB Data Cached:")
        for timeframe, df in data.items():
            if len(df) > 0:
                print(f"   {timeframe}: {len(df)} bars, latest: ${df['Close'].iloc[-1]:.2f}")
        
        return data


def demo_aegs_cache():
    """Demonstrate AEGS cache integration"""
    
    print(colored("🔥💎 AEGS CACHE INTEGRATION DEMO 💎🔥", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Initialize AEGS cache
    aegs_cache = AEGSDataCache()
    
    # 1. Get BB data using cache
    bb_data = aegs_cache.get_bb_data_with_cache()
    
    # 2. Show cache status
    aegs_cache.show_cache_status()
    
    # 3. Test other goldmine symbols
    print(colored("\n📊 Caching other goldmine symbols...", 'yellow'))
    goldmine_symbols = ['WULF', 'MARA', 'NOK', 'EQT', 'TLRY']
    
    for symbol in goldmine_symbols:
        df = aegs_cache.get_scan_data(symbol)
        print(f"   {symbol}: Cached {len(df)} hours of data")
    
    # 4. Show updated cache status
    aegs_cache.show_cache_status()
    
    # 5. Save BB data summary
    if 'daily' in bb_data and len(bb_data['daily']) > 0:
        summary_file = f"BB_cache_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create summary with key metrics
        summary_data = {
            'Date': bb_data['daily'].index[-30:],
            'Close': bb_data['daily']['Close'][-30:],
            'Volume': bb_data['daily']['Volume'][-30:],
            'High': bb_data['daily']['High'][-30:],
            'Low': bb_data['daily']['Low'][-30:]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Saved BB summary to: {summary_file}")
    
    print(colored("\n✅ AEGS Cache Integration Complete!", 'green'))
    print("\nBenefits:")
    print("  • Faster backtesting with cached data")
    print("  • Reduced API calls to Yahoo Finance")
    print("  • Consistent data across AEGS components")
    print("  • Automatic cache management")


if __name__ == "__main__":
    demo_aegs_cache()