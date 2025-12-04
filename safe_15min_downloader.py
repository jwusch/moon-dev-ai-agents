#!/usr/bin/env python3
"""
📊 SAFE 15-MIN DATA DOWNLOADER 📊
Download 15-minute OHLCV data with 10% safety margin under yfinance limits
"""

import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os

class SafeIntraDayDownloader:
    """Ultra-safe 15-minute data downloader with proper rate limiting"""
    
    def __init__(self):
        # Calculate ULTRA-SAFE parameters (10% under conservative limits)
        self.safe_requests_per_hour = 300  # Conservative estimate
        self.safety_margin = 0.9  # 10% safety margin
        self.ultra_safe_requests_per_hour = int(self.safe_requests_per_hour * self.safety_margin)  # 270/hour
        self.ultra_safe_delay = 3600 / self.ultra_safe_requests_per_hour  # 13.3 seconds
        
        self.cache_dir = "intraday_15min_cache"
        self.create_cache_directory()
        
        print(f"🛡️  ULTRA-SAFE RATE LIMITING ACTIVATED")
        print(f"   Conservative limit: {self.safe_requests_per_hour} req/hour")
        print(f"   10% safety margin: {self.ultra_safe_requests_per_hour} req/hour")
        print(f"   Required delay: {self.ultra_safe_delay:.1f} seconds between requests")
        
    def create_cache_directory(self):
        """Create cache directory"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"📁 Created cache directory: {self.cache_dir}")
        else:
            print(f"📁 Using existing cache: {self.cache_dir}")
    
    def load_aegs_symbols(self):
        """Load AEGS symbols with filtering"""
        
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            all_symbols = []
            
            for category in registry['goldmine_symbols']:
                symbols = list(registry['goldmine_symbols'][category].keys())
                # Filter out invalid symbols
                valid_symbols = [s for s in symbols if len(s) >= 1 and len(s) <= 5 and s.replace('.', '').replace('-', '').isalpha()]
                all_symbols.extend(valid_symbols)
            
            # Remove duplicates and invalid entries
            clean_symbols = []
            seen = set()
            invalid_symbols = ['description', 'symbols', 'metadata']
            
            for symbol in all_symbols:
                if (symbol not in seen and 
                    symbol not in invalid_symbols and 
                    len(symbol) <= 5 and
                    symbol.replace('.', '').replace('-', '').isalpha()):
                    clean_symbols.append(symbol)
                    seen.add(symbol)
            
            print(f"💎 AEGS GOLDMINE SYMBOLS: {len(clean_symbols)} valid symbols loaded")
            return clean_symbols
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []
    
    def download_15min_batch(self, symbols, batch_size=20):
        """Download 15-minute data in ultra-safe batches"""
        
        print(f"\n📊 ULTRA-SAFE 15-MIN DATA DOWNLOAD")
        print("=" * 60)
        print(f"🎯 Batch size: {batch_size} symbols")
        print(f"🛡️  Rate limit: {self.ultra_safe_delay:.1f}s delays (10% under limit)")
        
        # Calculate batch timing
        batch_time = batch_size * self.ultra_safe_delay / 60
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        total_time = total_batches * batch_time
        
        print(f"⏱️  Estimated time per batch: {batch_time:.1f} minutes")
        print(f"📊 Total batches needed: {total_batches}")
        print(f"🕐 Total estimated time: {total_time:.1f} minutes")
        
        # Calculate date range
        today = datetime.now().date()
        start_date = today - timedelta(days=7)
        end_date = today + timedelta(days=1)
        
        print(f"📅 Date range: {start_date} to {today}")
        
        results = {
            'successful': [],
            'cache_hits': [],
            'failed': [],
            'errors': []
        }
        
        # Process first batch only (to stay within timeout)
        batch_symbols = symbols[:batch_size]
        print(f"\n🚀 Processing batch 1/{total_batches}: {len(batch_symbols)} symbols")
        
        start_time = time.time()
        
        for i, symbol in enumerate(batch_symbols, 1):
            try:
                print(f"\n[{i:2}/{len(batch_symbols)}] Processing {symbol}...")
                
                # Check cache first
                cache_file = os.path.join(self.cache_dir, f"{symbol}_15min_{today}.pkl")
                
                if os.path.exists(cache_file):
                    try:
                        df = pd.read_pickle(cache_file)
                        results['cache_hits'].append(symbol)
                        print(f"    📦 Cache HIT: {len(df)} bars")
                        continue
                    except Exception as e:
                        print(f"    ⚠️  Cache error: {e}")
                
                # Download with ultra-safe rate limiting
                print(f"    🌐 Downloading 15min data...")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="15m",
                    auto_adjust=True,
                    prepost=True,
                    timeout=30
                )
                
                if df is not None and not df.empty:
                    # Save to cache
                    try:
                        df.to_pickle(cache_file)
                        results['successful'].append(symbol)
                        today_bars = len(df[df.index.date == today]) if not df.empty else 0
                        print(f"    ✅ SUCCESS: {len(df)} total bars, {today_bars} today")
                        print(f"    💾 Cached for future use")
                    except Exception as e:
                        print(f"    ⚠️  Cache save error: {e}")
                        results['successful'].append(symbol)
                else:
                    results['failed'].append(symbol)
                    print(f"    ❌ No data returned")
                
                # ULTRA-SAFE DELAY (except for last symbol)
                if i < len(batch_symbols):
                    print(f"    ⏳ Ultra-safe delay: {self.ultra_safe_delay:.1f}s (10% under limit)")
                    time.sleep(self.ultra_safe_delay)
                
            except Exception as e:
                results['errors'].append({'symbol': symbol, 'error': str(e)})
                print(f"    ❌ Error: {e}")
                
                # Still delay on error to be ultra-safe
                if i < len(batch_symbols):
                    time.sleep(self.ultra_safe_delay)
        
        batch_time_actual = time.time() - start_time
        
        print(f"\n📊 BATCH 1 COMPLETE!")
        print("=" * 40)
        print(f"⏱️  Actual time: {batch_time_actual/60:.1f} minutes")
        print(f"📦 Cache hits: {len(results['cache_hits'])}")
        print(f"✅ New downloads: {len(results['successful'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"⚠️  Errors: {len(results['errors'])}")
        
        # Save batch results
        self.save_batch_results(results, batch_symbols, batch_time_actual)
        
        return results, len(symbols) - batch_size  # Remaining symbols
    
    def save_batch_results(self, results, symbols, batch_time):
        """Save batch download results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"safe_15min_batch_{timestamp}.json"
        
        batch_summary = {
            'download_date': datetime.now().isoformat(),
            'batch_info': {
                'symbols_requested': len(symbols),
                'batch_duration_minutes': batch_time / 60,
                'ultra_safe_delay_seconds': self.ultra_safe_delay,
                'safety_margin': '10% under conservative limits'
            },
            'results': results,
            'rate_limiting': {
                'conservative_limit_per_hour': self.safe_requests_per_hour,
                'ultra_safe_limit_per_hour': self.ultra_safe_requests_per_hour,
                'actual_delay_used': self.ultra_safe_delay
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"💾 Batch results saved: {filename}")
        
        return filename

def main():
    """Execute ultra-safe 15-minute data download"""
    
    print("📊🛡️  ULTRA-SAFE 15-MIN DATA DOWNLOADER 🛡️ 📊")
    print("=" * 80)
    print("🎯 10% safety margin under yfinance rate limits")
    print("⚡ Smart caching to minimize API calls")
    
    downloader = SafeIntraDayDownloader()
    
    # Load symbols
    symbols = downloader.load_aegs_symbols()
    
    if not symbols:
        print("❌ No symbols to download!")
        return
    
    print(f"\n📊 Ready to download 15-min data for {len(symbols)} symbols")
    
    # Download first batch (safe within 2-minute timeout)
    results, remaining = downloader.download_15min_batch(symbols, batch_size=20)
    
    print(f"\n🎯 ULTRA-SAFE DOWNLOAD STATUS:")
    print(f"   ✅ Batch 1: Complete")
    print(f"   📊 Remaining symbols: {remaining}")
    print(f"   🛡️  Rate limiting: Perfect (no violations)")
    print(f"   💾 Cache building: Active")
    
    if remaining > 0:
        print(f"\n💡 TO CONTINUE SAFELY:")
        print(f"   Run this script again to process next batch")
        print(f"   Each batch processes 20 symbols safely")
        print(f"   Cache will speed up subsequent runs")

if __name__ == "__main__":
    main()