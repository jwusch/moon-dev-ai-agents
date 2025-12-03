"""
🔥💎 AEGS PERMANENT CACHE SYSTEM 💎🔥
Modified YFinanceCache that never deletes data
"""

import os
import json
import hashlib
from datetime import datetime
import pickle
from yfinance_cache_demo import YFinanceCache

class AEGSPermanentCache(YFinanceCache):
    """
    Enhanced cache that never deletes data automatically
    Stores everything permanently for AEGS backtesting
    """
    
    def __init__(self, cache_dir="aegs_permanent_cache"):
        super().__init__(cache_dir)
        self.never_expire = True  # Override expiration logic
        
    def is_cache_valid(self, cache_key, max_age_hours=24):
        """
        Override to always return True - cache never expires
        """
        if cache_key not in self.metadata:
            return False
        
        # Always valid if we have it
        return True
    
    def get_data(self, symbol, period="1mo", interval="1d", force_refresh=False):
        """
        Get data with permanent caching
        Only downloads if not in cache or force_refresh=True
        """
        cache_key = self.get_cache_key(symbol, period, interval)
        cache_path = self.get_cache_path(cache_key)
        
        # Check cache
        if not force_refresh and cache_key in self.metadata:
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                
                # Show cache info
                cached_info = self.metadata[cache_key]
                cached_time = cached_info['timestamp']
                rows = cached_info['rows']
                
                print(f"📦 Permanent cache hit: {symbol} {period} {interval}")
                print(f"   Cached on: {cached_time}")
                print(f"   Data points: {rows}")
                
                return df
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Fetch fresh data
        print(f"🌐 Downloading and permanently caching: {symbol} {period} {interval}")
        
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if len(df) > 0:
            # Save to permanent cache
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'rows': len(df),
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1]),
                'permanent': True  # Mark as permanent
            }
            self.save_metadata()
            
            print(f"💾 Permanently cached {len(df)} data points")
        
        return df
    
    def clear_old_cache(self, max_age_days=7):
        """
        Override to do nothing - we never clear cache
        """
        print("🔒 Permanent cache - cleanup disabled")
        return 0
    
    def clear_symbol_cache(self, symbol):
        """
        Clear cache for a specific symbol
        """
        import os
        cleared = 0
        
        # Find all cache entries for this symbol
        keys_to_remove = []
        for key in self.metadata.keys():
            if key.startswith(f"{symbol}_"):
                keys_to_remove.append(key)
        
        # Remove files and metadata
        for key in keys_to_remove:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                cleared += 1
            
            del self.metadata[key]
        
        if cleared > 0:
            self.save_metadata()
            print(f"🧹 Cleared {cleared} cache entries for {symbol}")
        
        return cleared
    
    def get_cache_inventory(self):
        """
        Show detailed inventory of cached data
        """
        inventory = {
            'by_symbol': {},
            'total_files': len(self.metadata),
            'total_size_mb': 0,
            'oldest_cache': None,
            'newest_cache': None
        }
        
        oldest_time = None
        newest_time = None
        
        for cache_key, info in self.metadata.items():
            symbol = info['symbol']
            period = info['period']
            interval = info['interval']
            timestamp = datetime.fromisoformat(info['timestamp'])
            
            # Track by symbol
            if symbol not in inventory['by_symbol']:
                inventory['by_symbol'][symbol] = []
            
            inventory['by_symbol'][symbol].append({
                'period': period,
                'interval': interval,
                'cached_on': info['timestamp'],
                'data_points': info['rows']
            })
            
            # Track oldest/newest
            if oldest_time is None or timestamp < oldest_time:
                oldest_time = timestamp
                inventory['oldest_cache'] = f"{symbol} {period} {interval} ({info['timestamp']})"
            
            if newest_time is None or timestamp > newest_time:
                newest_time = timestamp
                inventory['newest_cache'] = f"{symbol} {period} {interval} ({info['timestamp']})"
            
            # Calculate size
            cache_path = self.get_cache_path(cache_key)
            if os.path.exists(cache_path):
                size_mb = os.path.getsize(cache_path) / 1024 / 1024
                inventory['total_size_mb'] += size_mb
        
        return inventory


def update_comprehensive_backtest_to_permanent():
    """
    Update the comprehensive backtest to use permanent cache
    """
    
    print("🔧 Updating to permanent cache system...")
    
    # Read the cached backtest file
    with open('comprehensive_qqq_backtest_cached.py', 'r') as f:
        content = f.read()
    
    # Replace YFinanceCache with AEGSPermanentCache
    content = content.replace(
        'from yfinance_cache_demo import YFinanceCache',
        'from aegs_permanent_cache import AEGSPermanentCache'
    )
    content = content.replace(
        'self.cache = YFinanceCache(cache_dir="aegs_yfinance_cache")',
        'self.cache = AEGSPermanentCache(cache_dir="aegs_permanent_cache")'
    )
    
    # Save updated version
    with open('comprehensive_qqq_backtest_cached.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated to use permanent cache!")


def demo_permanent_cache():
    """
    Demonstrate permanent caching
    """
    
    print("🔥💎 AEGS PERMANENT CACHE DEMO 💎🔥")
    print("=" * 70)
    
    cache = AEGSPermanentCache()
    
    # Test symbols
    symbols = ['BB', 'WULF', 'MARA', 'NOK']
    
    print("\n1️⃣ Caching multiple timeframes permanently:")
    print("-" * 50)
    
    for symbol in symbols:
        # Cache different timeframes
        print(f"\n{symbol}:")
        
        # Daily for backtesting
        df_daily = cache.get_data(symbol, period="730d", interval="1d")
        print(f"   Daily: {len(df_daily)} bars")
        
        # Hourly for scanning
        df_hourly = cache.get_data(symbol, period="1mo", interval="1h")
        print(f"   Hourly: {len(df_hourly)} bars")
    
    print("\n2️⃣ Cache Inventory:")
    print("-" * 50)
    
    inventory = cache.get_cache_inventory()
    print(f"Total cached files: {inventory['total_files']}")
    print(f"Total size: {inventory['total_size_mb']:.1f} MB")
    print(f"Oldest: {inventory['oldest_cache']}")
    print(f"Newest: {inventory['newest_cache']}")
    
    print("\n3️⃣ Cached symbols:")
    for symbol, data in inventory['by_symbol'].items():
        print(f"\n{symbol}:")
        for item in data:
            print(f"   - {item['interval']} ({item['data_points']} points) cached on {item['cached_on'][:10]}")
    
    print("\n✅ All data permanently cached - will never expire!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='Update backtest to use permanent cache')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--inventory', action='store_true', help='Show cache inventory')
    
    args = parser.parse_args()
    
    if args.update:
        update_comprehensive_backtest_to_permanent()
    elif args.demo:
        demo_permanent_cache()
    elif args.inventory:
        cache = AEGSPermanentCache()
        inventory = cache.get_cache_inventory()
        print(json.dumps(inventory, indent=2))
    else:
        print("Use --update to switch to permanent cache, --demo to test, or --inventory to view")