"""
🎯 YFinance Caching Mechanism
Building a smart cache to reduce API calls and improve performance

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import os
import json
import hashlib
from datetime import datetime, timedelta
import pickle

class YFinanceCache:
    """
    Custom caching layer for yfinance data
    Stores actual price data, not just timezones
    """
    
    def __init__(self, cache_dir="yfinance_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Keep metadata about cached items
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_cache_key(self, symbol, period, interval):
        """Generate unique cache key"""
        key_string = f"{symbol}_{period}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, cache_key):
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cache_valid(self, cache_key, max_age_hours=24):
        """Check if cache is still valid"""
        if cache_key not in self.metadata:
            return False
        
        cached_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
        age = datetime.now() - cached_time
        
        # Different max ages for different intervals
        interval = self.metadata[cache_key].get('interval', '1d')
        if interval in ['1m', '5m', '15m', '30m']:
            max_age_hours = 1  # Intraday data expires quickly
        elif interval in ['1h', '90m']:
            max_age_hours = 6
        else:  # Daily or longer
            max_age_hours = 24
        
        return age < timedelta(hours=max_age_hours)
    
    def get_data(self, symbol, period="1mo", interval="1d", force_refresh=False):
        """Get data with caching"""
        cache_key = self.get_cache_key(symbol, period, interval)
        cache_path = self.get_cache_path(cache_key)
        
        # Check cache
        if not force_refresh and self.is_cache_valid(cache_key):
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                print(f"📦 Cache hit: {symbol} {period} {interval}")
                return df
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Fetch fresh data
        print(f"🌐 Fetching fresh data: {symbol} {period} {interval}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if len(df) > 0:
            # Save to cache
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
                'end_date': str(df.index[-1])
            }
            self.save_metadata()
        
        return df
    
    def clear_old_cache(self, max_age_days=7):
        """Clear old cache files"""
        cleared = 0
        current_time = datetime.now()
        
        for cache_key, info in list(self.metadata.items()):
            cached_time = datetime.fromisoformat(info['timestamp'])
            age = current_time - cached_time
            
            if age > timedelta(days=max_age_days):
                # Remove file
                cache_path = self.get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    cleared += 1
                
                # Remove metadata
                del self.metadata[cache_key]
        
        self.save_metadata()
        return cleared
    
    def get_cache_stats(self):
        """Get cache statistics"""
        stats = {
            'total_items': len(self.metadata),
            'total_size_mb': 0,
            'by_symbol': {},
            'by_interval': {}
        }
        
        for cache_key, info in self.metadata.items():
            # File size
            cache_path = self.get_cache_path(cache_key)
            if os.path.exists(cache_path):
                size_mb = os.path.getsize(cache_path) / 1024 / 1024
                stats['total_size_mb'] += size_mb
            
            # Count by symbol
            symbol = info['symbol']
            if symbol not in stats['by_symbol']:
                stats['by_symbol'][symbol] = 0
            stats['by_symbol'][symbol] += 1
            
            # Count by interval
            interval = info['interval']
            if interval not in stats['by_interval']:
                stats['by_interval'][interval] = 0
            stats['by_interval'][interval] += 1
        
        return stats


# Demonstration
def main():
    print("="*70)
    print("📊 YFINANCE CACHING DEMONSTRATION")
    print("="*70)
    
    # Initialize cache
    cache = YFinanceCache()
    
    # Test cached vs non-cached performance
    import time
    
    symbols = ['SPY', 'VXX', 'GLD', 'TLT', 'TSLA']
    
    print("\n1️⃣ FIRST RUN (No cache):")
    print("-"*50)
    start_time = time.time()
    for symbol in symbols:
        df = cache.get_data(symbol, period="1mo", interval="1h")
        print(f"   {symbol}: {len(df)} rows")
    first_run_time = time.time() - start_time
    print(f"\nTime taken: {first_run_time:.1f} seconds")
    
    print("\n2️⃣ SECOND RUN (With cache):")
    print("-"*50)
    start_time = time.time()
    for symbol in symbols:
        df = cache.get_data(symbol, period="1mo", interval="1h")
        print(f"   {symbol}: {len(df)} rows")
    second_run_time = time.time() - start_time
    print(f"\nTime taken: {second_run_time:.1f} seconds")
    print(f"Speed improvement: {first_run_time/second_run_time:.1f}x faster")
    
    # Show cache stats
    stats = cache.get_cache_stats()
    print("\n📊 CACHE STATISTICS:")
    print("-"*50)
    print(f"Total cached items: {stats['total_items']}")
    print(f"Total cache size: {stats['total_size_mb']:.1f} MB")
    print(f"Cached symbols: {', '.join(stats['by_symbol'].keys())}")
    print(f"Cached intervals: {stats['by_interval']}")
    
    # Test different timeframes
    print("\n3️⃣ CACHING DIFFERENT TIMEFRAMES:")
    print("-"*50)
    
    test_params = [
        ("5d", "5m", "Short-term intraday"),
        ("1mo", "15m", "Medium-term intraday"),
        ("6mo", "1d", "Long-term daily"),
    ]
    
    for period, interval, desc in test_params:
        df = cache.get_data("VXX", period=period, interval=interval)
        print(f"{desc}: {len(df)} bars cached")
    
    # Show where actual files are stored
    print("\n📁 CACHE STORAGE:")
    print("-"*50)
    print(f"Cache directory: {os.path.abspath(cache.cache_dir)}")
    print("\nCached files:")
    for f in os.listdir(cache.cache_dir):
        if f.endswith('.pkl'):
            file_path = os.path.join(cache.cache_dir, f)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  {f}: {size_kb:.1f} KB")
    
    # Show yfinance's built-in cache location
    print("\n📍 YFINANCE BUILT-IN CACHE:")
    print("-"*50)
    
    # Find yfinance cache directory
    home = os.path.expanduser("~")
    possible_paths = [
        os.path.join(home, ".cache", "py-yfinance"),  # Linux
        os.path.join(home, "AppData", "Local", "py-yfinance"),  # Windows
        os.path.join(home, "Library", "Caches", "py-yfinance"),  # macOS
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found yfinance cache: {path}")
            try:
                files = os.listdir(path)
                print(f"Contains {len(files)} files")
                for f in files[:5]:  # Show first 5
                    print(f"  - {f}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")
            except:
                pass
            break
    else:
        print("yfinance built-in cache not found in standard locations")
    
    print("\n✅ Caching demonstration complete!")
    print("\nBENEFITS:")
    print("1. Reduced API calls to Yahoo Finance")
    print("2. Faster backtesting iterations")
    print("3. Ability to work offline with cached data")
    print("4. Lower risk of rate limiting")

if __name__ == "__main__":
    main()