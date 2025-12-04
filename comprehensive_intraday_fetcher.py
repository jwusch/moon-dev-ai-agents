#!/usr/bin/env python3
"""
📊 COMPREHENSIVE INTRADAY DATA FETCHER 📊
Fetch 1-minute and 15-minute data as far back as possible for goldmine symbols
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import yfinance as yf

class ComprehensiveIntraDayFetcher:
    """Fetch multiple timeframes of intraday data with intelligent caching"""
    
    def __init__(self):
        self.cache_dir = "comprehensive_intraday_cache"
        self.create_cache_directory()
        
        # Timeframe limits based on yfinance constraints
        self.timeframe_limits = {
            "1m": 7,    # 1-minute data: last 7 days only
            "15m": 60,  # 15-minute data: last 60 days
            "1h": 730,  # 1-hour data: last 2 years
            "1d": 365 * 5  # Daily data: last 5 years
        }
        
        print(f"📊 TIMEFRAME LIMITS:")
        print(f"   1-minute:  {self.timeframe_limits['1m']} days")
        print(f"   15-minute: {self.timeframe_limits['15m']} days")
        print(f"   1-hour:    {self.timeframe_limits['1h']} days")
        print(f"   Daily:     {self.timeframe_limits['1d']} days")
    
    def create_cache_directory(self):
        """Create cache directory structure"""
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"📁 Created cache directory: {self.cache_dir}")
        
        # Create subdirectories for each timeframe
        for timeframe in ["1m", "15m", "1h", "1d"]:
            subdir = os.path.join(self.cache_dir, timeframe)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
                print(f"📁 Created {timeframe} cache subdirectory")
    
    def load_goldmine_symbols(self, limit=None):
        """Load goldmine symbols with optional limit for testing"""
        
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            all_symbols = []
            for category in registry['goldmine_symbols']:
                symbols = list(registry['goldmine_symbols'][category].keys())
                # Filter valid symbols
                valid_symbols = [s for s in symbols if len(s) <= 5 and s.replace('.', '').replace('-', '').isalpha()]
                all_symbols.extend(valid_symbols)
            
            # Remove duplicates
            unique_symbols = list(set(all_symbols))
            unique_symbols.sort()
            
            if limit:
                unique_symbols = unique_symbols[:limit]
                print(f"💎 GOLDMINE SYMBOLS (LIMITED): {len(unique_symbols)} symbols (showing first {limit})")
            else:
                print(f"💎 GOLDMINE SYMBOLS: {len(unique_symbols)} symbols")
            
            return unique_symbols
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []
    
    def get_cache_filename(self, symbol, timeframe, days_back):
        """Generate cache filename"""
        return os.path.join(self.cache_dir, timeframe, f"{symbol}_{timeframe}_{days_back}d.pkl")
    
    def is_cache_valid(self, cache_file, max_age_hours=6):
        """Check if cache file is still valid"""
        
        if not os.path.exists(cache_file):
            return False
        
        # Check file age
        file_age = time.time() - os.path.getctime(cache_file)
        return file_age < (max_age_hours * 3600)
    
    def fetch_timeframe_data(self, symbol, timeframe, days_back):
        """Fetch data for specific timeframe"""
        
        # Check cache first
        cache_file = self.get_cache_filename(symbol, timeframe, days_back)
        
        if self.is_cache_valid(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if df is not None and not df.empty:
                    print(f"    📦 {timeframe} Cache HIT: {len(df)} bars")
                    return df
            except Exception as e:
                print(f"    ⚠️  {timeframe} Cache error: {e}")
        
        # Fetch fresh data
        print(f"    🌐 Fetching {timeframe} data ({days_back}d)...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=timeframe,
                auto_adjust=True,
                prepost=True,
                timeout=30
            )
            
            if df is not None and not df.empty:
                # Save to cache
                try:
                    df.to_pickle(cache_file)
                    print(f"    ✅ {timeframe}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
                    return df
                except Exception as e:
                    print(f"    ⚠️  Cache save error: {e}")
                    return df
            else:
                print(f"    ❌ {timeframe}: No data returned")
                return None
                
        except Exception as e:
            print(f"    ❌ {timeframe} fetch error: {e}")
            return None
    
    def fetch_symbol_all_timeframes(self, symbol):
        """Fetch all available timeframes for a symbol"""
        
        print(f"\n📊 Fetching all timeframes: {symbol}")
        
        results = {}
        
        for timeframe, max_days in self.timeframe_limits.items():
            df = self.fetch_timeframe_data(symbol, timeframe, max_days)
            
            if df is not None and not df.empty:
                results[timeframe] = {
                    'bars': len(df),
                    'start_date': df.index[0].date(),
                    'end_date': df.index[-1].date(),
                    'status': 'success'
                }
            else:
                results[timeframe] = {'status': 'failed'}
            
            # Rate limiting between timeframes
            time.sleep(1.5)
        
        return results
    
    def run_comprehensive_fetch(self, symbols, rate_limit_seconds=5):
        """Run comprehensive data fetch for all symbols"""
        
        print(f"\n🚀 COMPREHENSIVE INTRADAY DATA FETCH")
        print("=" * 70)
        print(f"📊 Symbols: {len(symbols)}")
        print(f"⏱️  Rate limit: {rate_limit_seconds}s between symbols")
        print(f"📁 Cache directory: {self.cache_dir}")
        
        # Estimate time
        total_requests = len(symbols) * len(self.timeframe_limits)
        estimated_minutes = (total_requests * rate_limit_seconds) / 60
        print(f"🕐 Estimated time: {estimated_minutes:.1f} minutes")
        
        all_results = {}
        start_time = time.time()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\n[{i:3}/{len(symbols)}] Processing {symbol}...")
                
                # Fetch all timeframes for symbol
                symbol_results = self.fetch_symbol_all_timeframes(symbol)
                all_results[symbol] = symbol_results
                
                # Show summary
                successful_timeframes = [tf for tf, data in symbol_results.items() if data['status'] == 'success']
                print(f"    🎯 Success: {len(successful_timeframes)}/{len(self.timeframe_limits)} timeframes")
                
                # Rate limiting between symbols
                if i < len(symbols):
                    print(f"    ⏳ Rate limit delay: {rate_limit_seconds}s")
                    time.sleep(rate_limit_seconds)
                
                # Progress updates
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (len(symbols) - i) / rate
                    
                    print(f"\n📊 PROGRESS UPDATE:")
                    print(f"   Completed: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")
                    print(f"   Elapsed: {elapsed/60:.1f} minutes")
                    print(f"   ETA: {eta/60:.1f} minutes")
                
            except Exception as e:
                print(f"    ❌ Error processing {symbol}: {e}")
                all_results[symbol] = {'error': str(e)}
                time.sleep(2)  # Still rate limit on error
        
        fetch_time = time.time() - start_time
        return all_results, fetch_time
    
    def analyze_fetch_results(self, all_results, fetch_time):
        """Analyze comprehensive fetch results"""
        
        print(f"\n📊 COMPREHENSIVE FETCH COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total time: {fetch_time/60:.1f} minutes")
        print(f"📊 Symbols processed: {len(all_results)}")
        
        # Analyze by timeframe
        timeframe_stats = {}
        
        for timeframe in self.timeframe_limits.keys():
            successful = 0
            total_bars = 0
            
            for symbol, results in all_results.items():
                if timeframe in results and results[timeframe]['status'] == 'success':
                    successful += 1
                    total_bars += results[timeframe]['bars']
            
            success_rate = successful / len(all_results) * 100 if all_results else 0
            avg_bars = total_bars / successful if successful > 0 else 0
            
            timeframe_stats[timeframe] = {
                'successful': successful,
                'success_rate': success_rate,
                'total_bars': total_bars,
                'avg_bars': avg_bars
            }
            
            print(f"\n🎯 {timeframe.upper()} DATA:")
            print(f"   Successful: {successful}/{len(all_results)} ({success_rate:.1f}%)")
            print(f"   Total bars: {total_bars:,}")
            print(f"   Avg bars/symbol: {avg_bars:.0f}")
        
        # Top performers by data availability
        symbol_scores = {}
        for symbol, results in all_results.items():
            score = 0
            for timeframe in self.timeframe_limits.keys():
                if timeframe in results and results[timeframe]['status'] == 'success':
                    score += 1
            symbol_scores[symbol] = score
        
        # Show top symbols with complete data
        top_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP SYMBOLS (Complete Data):")
        complete_data_symbols = [s for s, score in top_symbols if score == 4]
        
        if complete_data_symbols:
            print(f"   Complete (all 4 timeframes): {len(complete_data_symbols)} symbols")
            for symbol in complete_data_symbols[:10]:
                print(f"      {symbol}")
            if len(complete_data_symbols) > 10:
                print(f"      ... and {len(complete_data_symbols) - 10} more")
        
        # Show symbols with at least 1m and 15m data
        intraday_symbols = []
        for symbol, results in all_results.items():
            has_1m = '1m' in results and results['1m']['status'] == 'success'
            has_15m = '15m' in results and results['15m']['status'] == 'success'
            if has_1m or has_15m:
                intraday_symbols.append(symbol)
        
        print(f"\n📊 INTRADAY DATA SUMMARY:")
        print(f"   Symbols with 1m or 15m data: {len(intraday_symbols)}")
        print(f"   Ready for AEGS backtesting: {len([s for s in intraday_symbols if '15m' in all_results[s] and all_results[s]['15m']['status'] == 'success'])}")
        
        return timeframe_stats, complete_data_symbols, intraday_symbols
    
    def save_fetch_summary(self, all_results, fetch_time, timeframe_stats):
        """Save comprehensive fetch summary"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"comprehensive_fetch_summary_{timestamp}.json"
        
        summary = {
            'fetch_date': datetime.now().isoformat(),
            'fetch_duration_minutes': fetch_time / 60,
            'total_symbols_processed': len(all_results),
            'timeframe_statistics': timeframe_stats,
            'symbol_results': all_results,
            'cache_directory': self.cache_dir,
            'timeframe_limits': self.timeframe_limits
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Fetch summary saved: {summary_file}")
        return summary_file

def main():
    """Execute comprehensive intraday data fetch"""
    
    print("📊🚀 COMPREHENSIVE INTRADAY DATA FETCHER 🚀📊")
    print("=" * 80)
    print("🎯 Fetching 1m, 15m, 1h, daily data for all goldmine symbols")
    print("⚡ Smart caching with rate limiting")
    print("💎 Maximum lookback periods per timeframe")
    
    # Initialize fetcher
    fetcher = ComprehensiveIntraDayFetcher()
    
    # Load goldmine symbols (limit to 30 for initial test)
    symbols = fetcher.load_goldmine_symbols(limit=30)  # Remove limit for full run
    
    if not symbols:
        print("❌ No goldmine symbols found!")
        return
    
    # Run comprehensive fetch
    all_results, fetch_time = fetcher.run_comprehensive_fetch(
        symbols, 
        rate_limit_seconds=4  # Conservative rate limiting
    )
    
    # Analyze results
    timeframe_stats, complete_symbols, intraday_symbols = fetcher.analyze_fetch_results(
        all_results, fetch_time
    )
    
    # Save summary
    summary_file = fetcher.save_fetch_summary(all_results, fetch_time, timeframe_stats)
    
    print(f"\n🎯 COMPREHENSIVE FETCH COMPLETE!")
    print(f"   📊 Processed {len(all_results)} symbols in {fetch_time/60:.1f} minutes")
    print(f"   💾 Cache: {fetcher.cache_dir}")
    print(f"   📄 Summary: {summary_file}")
    print(f"   🚀 Ready for multi-timeframe AEGS backtesting!")

if __name__ == "__main__":
    main()