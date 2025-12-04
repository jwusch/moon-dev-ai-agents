#!/usr/bin/env python3
"""
📊 DOWNLOAD TODAY'S 15-MIN OHLCV DATA 📊
Download today's 15-minute OHLCV data for all AEGS goldmine symbols
"""

import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from working_cached_aegs_scanner import WorkingCacheAEGS

class IntraDayDataDownloader:
    """Download intraday 15-minute data for AEGS symbols"""
    
    def __init__(self):
        self.cache_dir = "intraday_15min_cache"
        self.create_cache_directory()
        
    def create_cache_directory(self):
        """Create cache directory for 15-minute data"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"📁 Created intraday cache directory: {self.cache_dir}")
        else:
            print(f"📁 Using existing cache directory: {self.cache_dir}")
    
    def load_aegs_goldmine_symbols(self):
        """Load all AEGS goldmine symbols"""
        
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            # Extract all symbols
            all_symbols = []
            category_counts = {}
            
            for category in registry['goldmine_symbols']:
                symbols = list(registry['goldmine_symbols'][category].keys())
                # Filter out invalid symbols (like 'description', 'symbols')
                valid_symbols = [s for s in symbols if len(s) <= 5 and s.isalpha()]
                all_symbols.extend(valid_symbols)
                category_counts[category] = len(valid_symbols)
                
            print("💎 AEGS GOLDMINE SYMBOLS LOADED")
            print("=" * 50)
            
            total_symbols = 0
            for category, count in category_counts.items():
                total_symbols += count
                print(f"   {category.replace('_', ' ').title():<20}: {count:>3} symbols")
            
            print(f"   {'TOTAL VALID SYMBOLS':<20}: {len(all_symbols):>3} symbols")
            
            return all_symbols
            
        except Exception as e:
            print(f"❌ Error loading goldmine registry: {e}")
            return []
    
    def download_15min_data(self, symbol, max_retries=2):
        """Download today's 15-minute OHLCV data for a symbol"""
        
        # Calculate today's date range
        today = datetime.now().date()
        start_date = today - timedelta(days=7)  # Get last 7 days for context
        end_date = today + timedelta(days=1)    # Include today
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{symbol}_15min_{today}.pkl")
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                print(f"📦 Cache HIT: {symbol} ({len(df)} bars)")
                return df
            except Exception as e:
                print(f"⚠️  Cache read error for {symbol}: {e}")
        
        # Download fresh data
        for attempt in range(max_retries + 1):
            try:
                print(f"🌐 Cache MISS - Downloading 15min: {symbol}")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="15m",
                    auto_adjust=True,
                    prepost=True,
                    timeout=10
                )
                
                if df is not None and not df.empty:
                    # Filter to today's data primarily
                    today_data = df[df.index.date == today] if not df.empty else df
                    
                    # Save to cache
                    try:
                        df.to_pickle(cache_file)
                        print(f"💾 Cached {symbol}: {len(df)} total bars, {len(today_data) if not today_data.empty else 0} today")
                    except Exception as e:
                        print(f"⚠️  Cache save error: {e}")
                    
                    return df
                else:
                    print(f"⚠️  Empty data for {symbol}, retrying...")
                    if attempt < max_retries:
                        print(f"⏳ Retry {attempt + 1}/{max_retries} for {symbol} - waiting 2.0s...")
                        time.sleep(2.0)
                    
            except Exception as e:
                print(f"⚠️  Download error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    print(f"⏳ Retry {attempt + 1}/{max_retries} for {symbol} - waiting 2.0s...")
                    time.sleep(2.0)
        
        print(f"❌ Failed to download {symbol} after {max_retries + 1} attempts")
        return None
    
    def download_all_goldmine_15min_data(self, goldmine_symbols, delay_seconds=8):
        """Download 15-minute data for all goldmine symbols with rate limiting"""
        
        print(f"\n📊 DOWNLOADING 15-MIN INTRADAY DATA")
        print("=" * 60)
        print(f"🎯 Target: {len(goldmine_symbols)} AEGS goldmine symbols")
        print(f"⏰ Timeframe: 15-minute OHLCV")
        print(f"📅 Period: Last 7 days + today")
        print(f"🛡️  Rate limiting: {delay_seconds}s delays between requests")
        
        # Calculate estimated time
        estimated_minutes = (len(goldmine_symbols) * delay_seconds) / 60
        print(f"⏱️  Estimated completion: {estimated_minutes:.1f} minutes")
        
        results = {
            'successful_downloads': [],
            'failed_downloads': [],
            'cache_hits': [],
            'errors': []
        }
        
        start_time = time.time()
        
        for i, symbol in enumerate(goldmine_symbols, 1):
            try:
                print(f"\n[{i:3}/{len(goldmine_symbols)}] Processing {symbol}...")
                
                # Check if cached first
                today = datetime.now().date()
                cache_file = os.path.join(self.cache_dir, f"{symbol}_15min_{today}.pkl")
                
                if os.path.exists(cache_file):
                    try:
                        df = pd.read_pickle(cache_file)
                        results['cache_hits'].append(symbol)
                        print(f"    📦 Cache HIT: {len(df)} bars available")
                    except:
                        # Cache corrupted, download fresh
                        df = self.download_15min_data(symbol)
                        if df is not None:
                            results['successful_downloads'].append(symbol)
                        else:
                            results['failed_downloads'].append(symbol)
                else:
                    # Download fresh data
                    df = self.download_15min_data(symbol)
                    if df is not None:
                        results['successful_downloads'].append(symbol)
                    else:
                        results['failed_downloads'].append(symbol)
                
                # Rate limiting delay (except for last symbol)
                if i < len(goldmine_symbols):
                    print(f"    ⏳ Rate limit protection: waiting {delay_seconds}s...")
                    time.sleep(delay_seconds)
                
                # Progress updates every 25 symbols
                if i % 25 == 0:
                    elapsed = time.time() - start_time
                    successful = len(results['successful_downloads'])
                    cached = len(results['cache_hits'])
                    failed = len(results['failed_downloads'])
                    
                    print(f"\n📊 15-MIN DATA DOWNLOAD PROGRESS:")
                    print(f"   Completed: {i}/{len(goldmine_symbols)} ({i/len(goldmine_symbols)*100:.1f}%)")
                    print(f"   Cache hits: {cached}")
                    print(f"   New downloads: {successful}")
                    print(f"   Failed: {failed}")
                    print(f"   Success rate: {(successful + cached)/i*100:.1f}%")
                    
                    # ETA calculation
                    if i > 0:
                        rate = i / elapsed
                        eta_seconds = (len(goldmine_symbols) - i) / rate
                        print(f"   ETA: {eta_seconds/60:.1f} minutes")
                
            except Exception as e:
                results['errors'].append({'symbol': symbol, 'error': str(e)})
                print(f"    ❌ Error: {e}")
        
        download_time = time.time() - start_time
        return results, download_time
    
    def analyze_download_results(self, results, download_time):
        """Analyze 15-minute data download results"""
        
        successful = results['successful_downloads']
        cached = results['cache_hits']
        failed = results['failed_downloads']
        errors = results['errors']
        
        total_processed = len(successful) + len(cached) + len(failed) + len(errors)
        
        print(f"\n📊 15-MIN DATA DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total time: {download_time/60:.1f} minutes")
        print(f"📊 Symbols processed: {total_processed}")
        print(f"📦 Cache hits: {len(cached)}")
        print(f"🌐 New downloads: {len(successful)}")
        print(f"❌ Failed downloads: {len(failed)}")
        print(f"⚠️  Errors: {len(errors)}")
        
        if total_processed > 0:
            success_rate = (len(successful) + len(cached)) / total_processed * 100
            print(f"📈 Overall success rate: {success_rate:.1f}%")
        
        # Show sample of successful symbols
        if successful:
            print(f"\n✅ SUCCESSFUL NEW DOWNLOADS (sample):")
            for symbol in successful[:10]:
                print(f"   📊 {symbol}: Fresh 15-min data downloaded")
            if len(successful) > 10:
                print(f"   ... and {len(successful) - 10} more")
        
        # Show cache hits
        if cached:
            print(f"\n📦 CACHE HITS (sample):")
            for symbol in cached[:10]:
                print(f"   ⚡ {symbol}: Using cached 15-min data")
            if len(cached) > 10:
                print(f"   ... and {len(cached) - 10} more")
        
        # Show failures
        if failed:
            print(f"\n❌ FAILED DOWNLOADS:")
            for symbol in failed[:10]:
                print(f"   ❌ {symbol}: Download failed")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more")
    
    def save_download_summary(self, results, download_time, goldmine_symbols):
        """Save download summary"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"aegs_15min_download_summary_{timestamp}.json"
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'timeframe': '15min',
            'period': 'last_7_days_plus_today',
            'total_symbols_requested': len(goldmine_symbols),
            'download_duration_minutes': download_time / 60,
            'results_summary': {
                'cache_hits': len(results['cache_hits']),
                'successful_downloads': len(results['successful_downloads']),
                'failed_downloads': len(results['failed_downloads']),
                'errors': len(results['errors'])
            },
            'successful_symbols': results['successful_downloads'],
            'cached_symbols': results['cache_hits'],
            'failed_symbols': results['failed_downloads'],
            'error_details': results['errors']
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Download summary saved: {filename}")
        return filename

def main():
    """Execute 15-minute data download for all AEGS symbols"""
    
    print("📊💎 AEGS GOLDMINE 15-MIN DATA DOWNLOADER 💎📊")
    print("=" * 80)
    print("🎯 Downloading today's 15-minute OHLCV data for all goldmine symbols")
    print("⚡ Using yfinance cache mechanism for efficiency")
    
    # Initialize downloader
    downloader = IntraDayDataDownloader()
    
    # Load goldmine symbols
    goldmine_symbols = downloader.load_aegs_goldmine_symbols()
    
    if not goldmine_symbols:
        print("❌ No goldmine symbols found!")
        return
    
    # Download 15-minute data with rate limiting
    results, download_time = downloader.download_all_goldmine_15min_data(
        goldmine_symbols, 
        delay_seconds=8  # Conservative rate limiting
    )
    
    # Analyze results
    downloader.analyze_download_results(results, download_time)
    
    # Save summary
    summary_file = downloader.save_download_summary(results, download_time, goldmine_symbols)
    
    print(f"\n📊🚀 15-MIN DATA DOWNLOAD MISSION COMPLETE! 🚀📊")
    print(f"   📦 Cache directory: {downloader.cache_dir}")
    print(f"   💾 Summary: {summary_file}")
    print(f"   🎯 Ready for intraday AEGS analysis!")

if __name__ == "__main__":
    main()