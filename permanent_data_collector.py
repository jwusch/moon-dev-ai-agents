"""
📊 Permanent Historical Data Collector
Builds and maintains a comprehensive 1-minute dataset for backtesting
NO EXPIRATION - Data stored permanently for unlimited backtesting

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import os
import json
import hashlib
from datetime import datetime, timedelta
import pickle
import time
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PermanentDataCollector:
    """
    Permanent data collection system with no expiration
    Systematically builds comprehensive historical datasets
    """
    
    def __init__(self, data_dir="permanent_data"):
        self.data_dir = data_dir
        self.setup_directories()
        
        # Core symbols for systematic collection
        self.core_symbols = [
            # Volatility Products (our bread and butter)
            "VXX", "UVXY", "VIXY", "SVXY", "VXZ", "VIXM",
            # Major ETFs
            "SPY", "QQQ", "IWM", "DIA", 
            # Leveraged ETFs  
            "TQQQ", "SQQQ", "SPXL", "SPXS", "TNA", "TZA",
            # Sector ETFs
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLP",
            # Individual Stocks (high volume, good for backtesting)
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
            # Commodities
            "GLD", "SLV", "USO", "UNG", "DBA",
            # Bonds
            "TLT", "IEF", "SHY", "TBT", "AGG",
            # Crypto ETFs
            "BITO", "BITQ"
        ]
        
        # Data collection schedule
        self.collection_intervals = {
            "1m": {"max_period": "7d", "collect_frequency_hours": 6},    # Collect every 6 hours
            "5m": {"max_period": "60d", "collect_frequency_hours": 24},  # Daily collection
            "15m": {"max_period": "60d", "collect_frequency_hours": 24}, # Daily collection  
            "1h": {"max_period": "730d", "collect_frequency_hours": 168}, # Weekly collection
            "1d": {"max_period": "max", "collect_frequency_hours": 168}   # Weekly collection
        }
        
        self.load_collection_metadata()
    
    def setup_directories(self):
        """Create directory structure for permanent data storage"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        subdirs = ["1m", "5m", "15m", "1h", "1d", "metadata", "logs"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)
        
        print(f"📁 Data storage initialized at: {os.path.abspath(self.data_dir)}")
    
    def load_collection_metadata(self):
        """Load metadata about data collection status"""
        self.metadata_file = os.path.join(self.data_dir, "metadata", "collection_status.json")
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "symbols": {},
                "last_collection": {},
                "total_records": 0,
                "collection_history": []
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save collection metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_data_filename(self, symbol: str, interval: str) -> str:
        """Generate filename for permanent data storage"""
        return os.path.join(self.data_dir, interval, f"{symbol}_{interval}.pkl")
    
    def load_existing_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load existing data for symbol/interval if available"""
        filename = self.get_data_filename(symbol, interval)
        
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    df = pickle.load(f)
                return df
            except Exception as e:
                print(f"⚠️ Error loading {symbol} {interval}: {e}")
                return None
        
        return None
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save data permanently"""
        filename = self.get_data_filename(symbol, interval)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(df, f)
            
            # Update metadata
            symbol_key = f"{symbol}_{interval}"
            self.metadata["symbols"][symbol_key] = {
                "symbol": symbol,
                "interval": interval,
                "records": len(df),
                "start_date": str(df.index[0]) if len(df) > 0 else None,
                "end_date": str(df.index[-1]) if len(df) > 0 else None,
                "last_updated": datetime.now().isoformat(),
                "file_size_mb": os.path.getsize(filename) / 1024 / 1024
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving {symbol} {interval}: {e}")
            return False
    
    def collect_fresh_data(self, symbol: str, interval: str, period: str = None) -> Optional[pd.DataFrame]:
        """Collect fresh data from Yahoo Finance"""
        
        if period is None:
            period = self.collection_intervals[interval]["max_period"]
        
        try:
            print(f"🌐 Collecting {symbol} {interval} data (period: {period})...")
            
            # Add retry logic for reliability
            for attempt in range(3):
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if len(df) > 0:
                        # Clean up column names
                        if df.columns.nlevels > 1:
                            df.columns = [col[0] for col in df.columns]
                        
                        print(f"   ✅ Downloaded {len(df)} records for {symbol} {interval}")
                        return df
                    else:
                        print(f"   ⚠️ No data returned for {symbol} {interval}")
                        return None
                        
                except Exception as e:
                    if attempt < 2:
                        print(f"   ⚠️ Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)
                    else:
                        print(f"   ❌ Failed after 3 attempts: {e}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error collecting {symbol} {interval}: {e}")
            return None
    
    def merge_with_existing(self, new_data: pd.DataFrame, existing_data: pd.DataFrame) -> pd.DataFrame:
        """Intelligently merge new data with existing data"""
        
        if existing_data is None or len(existing_data) == 0:
            return new_data
        
        # Combine datasets
        combined = pd.concat([existing_data, new_data])
        
        # Remove duplicates (keep last occurrence for each timestamp)
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # Sort by timestamp
        combined = combined.sort_index()
        
        # Data quality checks
        initial_count = len(existing_data) + len(new_data)
        final_count = len(combined)
        duplicates_removed = initial_count - final_count
        
        print(f"   📊 Merged: {len(existing_data)} existing + {len(new_data)} new = {final_count} total")
        if duplicates_removed > 0:
            print(f"   🔄 Removed {duplicates_removed} duplicates")
        
        return combined
    
    def collect_symbol_data(self, symbol: str, intervals: List[str] = None) -> Dict[str, bool]:
        """Collect data for a single symbol across intervals"""
        
        if intervals is None:
            intervals = ["1m", "5m", "15m", "1h", "1d"]
        
        results = {}
        
        for interval in intervals:
            try:
                # Load existing data
                existing_data = self.load_existing_data(symbol, interval)
                
                # Collect fresh data
                new_data = self.collect_fresh_data(symbol, interval)
                
                if new_data is not None and len(new_data) > 0:
                    # Merge with existing
                    final_data = self.merge_with_existing(new_data, existing_data)
                    
                    # Save permanently  
                    success = self.save_data(final_data, symbol, interval)
                    results[interval] = success
                    
                    if success:
                        self.metadata["total_records"] += len(new_data)
                else:
                    results[interval] = False
                    
            except Exception as e:
                print(f"❌ Error collecting {symbol} {interval}: {e}")
                results[interval] = False
        
        return results
    
    def run_systematic_collection(self, symbols: List[str] = None, intervals: List[str] = None):
        """Run systematic data collection across symbols and intervals"""
        
        if symbols is None:
            symbols = self.core_symbols
        
        if intervals is None:
            intervals = ["1m", "5m", "15m", "1h", "1d"]
        
        start_time = datetime.now()
        
        print(f"🚀 STARTING SYSTEMATIC DATA COLLECTION")
        print(f"{'='*60}")
        print(f"Symbols: {len(symbols)} ({', '.join(symbols[:10])}{', ...' if len(symbols) > 10 else ''})")
        print(f"Intervals: {', '.join(intervals)}")
        print(f"Start time: {start_time}")
        
        collection_results = {
            "success": 0,
            "failed": 0,
            "details": {}
        }
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 [{i}/{len(symbols)}] Collecting {symbol}...")
            
            try:
                results = self.collect_symbol_data(symbol, intervals)
                collection_results["details"][symbol] = results
                
                # Count successes/failures
                for interval, success in results.items():
                    if success:
                        collection_results["success"] += 1
                    else:
                        collection_results["failed"] += 1
                
                # Brief pause between symbols to be respectful to Yahoo Finance
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error with {symbol}: {e}")
                collection_results["failed"] += len(intervals)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Log collection run
        self.metadata["collection_history"].append({
            "timestamp": start_time.isoformat(),
            "duration_minutes": duration.total_seconds() / 60,
            "symbols_attempted": len(symbols),
            "intervals_attempted": len(intervals),
            "successes": collection_results["success"],
            "failures": collection_results["failed"]
        })
        
        self.metadata["last_collection"] = {
            "timestamp": start_time.isoformat(),
            "symbols": symbols,
            "intervals": intervals
        }
        
        self.save_metadata()
        
        # Print summary
        print(f"\n🎯 COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {duration.total_seconds():.0f} seconds")
        print(f"Successful collections: {collection_results['success']}")
        print(f"Failed collections: {collection_results['failed']}")
        print(f"Success rate: {collection_results['success']/(collection_results['success']+collection_results['failed'])*100:.1f}%")
        
        return collection_results
    
    def get_data(self, symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Get data for a symbol (for backtesting use)"""
        return self.load_existing_data(symbol, interval)
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about collected data"""
        stats = {
            "total_symbols": len(set([info["symbol"] for info in self.metadata["symbols"].values()])),
            "total_files": len(self.metadata["symbols"]),
            "total_records": sum([info["records"] for info in self.metadata["symbols"].values()]),
            "total_size_mb": sum([info["file_size_mb"] for info in self.metadata["symbols"].values()]),
            "by_interval": {},
            "by_symbol": {},
            "date_range": {},
            "collection_history": len(self.metadata.get("collection_history", []))
        }
        
        for symbol_key, info in self.metadata["symbols"].items():
            # By interval
            interval = info["interval"]
            if interval not in stats["by_interval"]:
                stats["by_interval"][interval] = {"count": 0, "records": 0, "size_mb": 0}
            
            stats["by_interval"][interval]["count"] += 1
            stats["by_interval"][interval]["records"] += info["records"]
            stats["by_interval"][interval]["size_mb"] += info["file_size_mb"]
            
            # By symbol
            symbol = info["symbol"]
            if symbol not in stats["by_symbol"]:
                stats["by_symbol"][symbol] = {"intervals": 0, "records": 0, "size_mb": 0}
            
            stats["by_symbol"][symbol]["intervals"] += 1
            stats["by_symbol"][symbol]["records"] += info["records"]
            stats["by_symbol"][symbol]["size_mb"] += info["file_size_mb"]
            
            # Date range
            if info["start_date"] and info["end_date"]:
                key = f"{symbol}_{interval}"
                stats["date_range"][key] = {
                    "start": info["start_date"],
                    "end": info["end_date"],
                    "records": info["records"]
                }
        
        return stats
    
    def detect_data_gaps(self, symbol: str, interval: str) -> List[Dict]:
        """Detect gaps in time series data"""
        df = self.load_existing_data(symbol, interval)
        
        if df is None or len(df) < 2:
            return []
        
        # Calculate expected frequency
        freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "1d": "1D"}
        expected_freq = freq_map.get(interval, "1min")
        
        # Generate expected date range
        start_date = df.index[0]
        end_date = df.index[-1]
        expected_index = pd.date_range(start=start_date, end=end_date, freq=expected_freq)
        
        # Find missing timestamps (gaps)
        missing = expected_index.difference(df.index)
        
        # Filter out weekends and holidays for market hours
        if interval in ["1m", "5m", "15m", "1h"]:
            # Remove weekend gaps
            missing = missing[missing.dayofweek < 5]  # Monday=0, Friday=4
            
            # Remove after-hours gaps (before 9:30 AM or after 4:00 PM ET)
            missing = missing[
                (missing.hour >= 9) & 
                ((missing.hour < 16) | ((missing.hour == 9) & (missing.minute >= 30)))
            ]
        
        # Group consecutive gaps
        gaps = []
        if len(missing) > 0:
            missing_sorted = missing.sort_values()
            gap_start = missing_sorted[0]
            gap_end = missing_sorted[0]
            
            for i in range(1, len(missing_sorted)):
                current = missing_sorted[i]
                # Check if this timestamp continues the current gap
                if interval == "1m":
                    expected_next = gap_end + pd.Timedelta(minutes=1)
                elif interval == "5m":
                    expected_next = gap_end + pd.Timedelta(minutes=5)
                elif interval == "15m":
                    expected_next = gap_end + pd.Timedelta(minutes=15)
                elif interval == "1h":
                    expected_next = gap_end + pd.Timedelta(hours=1)
                else:  # 1d
                    expected_next = gap_end + pd.Timedelta(days=1)
                
                if current == expected_next:
                    gap_end = current
                else:
                    # End current gap, start new one
                    gaps.append({
                        "start": str(gap_start),
                        "end": str(gap_end),
                        "missing_periods": len(pd.date_range(gap_start, gap_end, freq=expected_freq))
                    })
                    gap_start = current
                    gap_end = current
            
            # Add final gap
            gaps.append({
                "start": str(gap_start),
                "end": str(gap_end), 
                "missing_periods": len(pd.date_range(gap_start, gap_end, freq=expected_freq))
            })
        
        return gaps
    
    def create_data_dashboard(self):
        """Create a comprehensive data collection dashboard"""
        stats = self.get_collection_stats()
        
        print(f"\n📊 PERMANENT DATA COLLECTION DASHBOARD")
        print(f"{'='*70}")
        
        # Overview
        print(f"\n📈 OVERVIEW:")
        print(f"  Total symbols: {stats['total_symbols']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Total storage: {stats['total_size_mb']:.1f} MB")
        print(f"  Collection runs: {stats['collection_history']}")
        
        # By interval breakdown
        print(f"\n⏱️ BY INTERVAL:")
        for interval, data in stats["by_interval"].items():
            print(f"  {interval:>3}: {data['count']:>3} files, {data['records']:>8,} records, {data['size_mb']:>6.1f} MB")
        
        # Top symbols by data volume
        print(f"\n🏆 TOP SYMBOLS BY RECORDS:")
        top_symbols = sorted(stats["by_symbol"].items(), key=lambda x: x[1]["records"], reverse=True)[:10]
        for symbol, data in top_symbols:
            print(f"  {symbol:>6}: {data['records']:>8,} records, {data['intervals']} intervals, {data['size_mb']:>6.1f} MB")
        
        # Recent collection info
        if "last_collection" in self.metadata and self.metadata["last_collection"]:
            last = self.metadata["last_collection"]
            print(f"\n🕐 LAST COLLECTION:")
            print(f"  Time: {last['timestamp']}")
            print(f"  Symbols: {len(last['symbols'])}")
            print(f"  Intervals: {', '.join(last['intervals'])}")
        
        # Data quality check (gaps)
        print(f"\n🔍 DATA QUALITY CHECK (Sample):")
        sample_symbols = ["VXX", "SPY", "QQQ"][:3]  # Check first 3 core symbols
        for symbol in sample_symbols:
            for interval in ["1m", "5m"]:
                gaps = self.detect_data_gaps(symbol, interval)
                if gaps:
                    total_missing = sum([gap["missing_periods"] for gap in gaps])
                    print(f"  {symbol} {interval}: {len(gaps)} gaps, {total_missing} missing periods")
                else:
                    print(f"  {symbol} {interval}: ✅ No gaps detected")

def main():
    """Demonstrate permanent data collection system"""
    print("🚀 PERMANENT DATA COLLECTOR")
    print("=" * 60)
    print("Building unlimited historical dataset for backtesting")
    print("NO EXPIRATION - Data stored permanently")
    
    # Initialize collector
    collector = PermanentDataCollector()
    
    # Show current state
    collector.create_data_dashboard()
    
    # Run systematic collection for a subset (for demo)
    test_symbols = ["VXX", "SPY", "QQQ", "NVDA", "TSLA"]  # Start small for demo
    test_intervals = ["1m", "5m", "15m"]  # Focus on higher frequency data
    
    print(f"\n🎯 RUNNING TEST COLLECTION")
    print(f"Symbols: {', '.join(test_symbols)}")
    print(f"Intervals: {', '.join(test_intervals)}")
    
    results = collector.run_systematic_collection(test_symbols, test_intervals)
    
    # Show updated dashboard
    print(f"\n📊 UPDATED DASHBOARD:")
    collector.create_data_dashboard()
    
    print(f"\n💡 USAGE EXAMPLES:")
    print(f"# Get 1-minute VXX data for backtesting")
    print(f"vxx_1m = collector.get_data('VXX', '1m')")
    print(f"")
    print(f"# Build comprehensive dataset")
    print(f"collector.run_systematic_collection()  # All core symbols")
    print(f"")
    print(f"# Check data quality")
    print(f"gaps = collector.detect_data_gaps('VXX', '1m')")
    
    print(f"\n✅ Permanent data collection system ready!")
    print(f"💾 Data stored in: {os.path.abspath(collector.data_dir)}")
    print(f"🔄 Run collector regularly to build comprehensive dataset")
    
    return collector

if __name__ == "__main__":
    collector = main()