"""
📈 Comprehensive QQQ Ensemble Strategy Backtest with Caching
Enhanced version that uses YFinanceCache for efficient data management
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced backtest module
from comprehensive_qqq_backtest_enhanced import *
from aegs_permanent_cache import AEGSPermanentCache

class CachedComprehensiveBacktester(EnhancedComprehensiveBacktester):
    """
    Enhanced backtester that uses caching for data downloads
    """
    
    def __init__(self, symbol="QQQ", use_cache=True):
        super().__init__(symbol)
        self.use_cache = use_cache
        if use_cache:
            # Use AEGS-specific cache directory
            self.cache = AEGSPermanentCache(cache_dir="aegs_permanent_cache")
    
    def download_maximum_data(self) -> pd.DataFrame:
        """
        Download historical data using cache when available
        """
        print(f"\n{'='*60}")
        print(f"📊 Data Download for {self.symbol}")
        print(f"{'='*60}")
        
        try:
            if self.use_cache:
                # Use cache for data retrieval
                print("📦 Using AEGS cache system...")
                df = self.cache.get_data(
                    self.symbol, 
                    period="max",  # Maximum available data
                    interval="1d"
                )
            else:
                # Fall back to direct download
                print("🌐 Direct download (no cache)...")
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period="max", interval="1d")
            
            if df.empty:
                print(f"❌ No data returned for {self.symbol}")
                return pd.DataFrame()
            
            # Clean up data
            if df.columns.nlevels > 1:
                df.columns = [col[0] for col in df.columns]
            
            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print("❌ Missing required columns in data")
                return pd.DataFrame()
            
            # Calculate date range and stats
            start_date = df.index[0]
            end_date = df.index[-1]
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25
            
            print(f"✅ Downloaded {len(df):,} days of data")
            print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"📊 Total period: {total_years:.1f} years")
            
            if self.use_cache:
                print("💾 Data cached for faster future access")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return pd.DataFrame()


def update_aegs_agents_to_use_cache():
    """
    Update AEGS agents to use the cached backtester
    """
    
    print("🔧 Updating AEGS agents to use caching...")
    
    # Read the current aegs_backtest_agent.py
    agent_file = "src/agents/aegs_backtest_agent.py"
    
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Check if already using cache
    if "CachedComprehensiveBacktester" in content:
        print("✅ AEGS agents already using cache!")
        return
    
    # Update the import
    old_import = "from comprehensive_qqq_backtest import ComprehensiveBacktester"
    new_import = "from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester as ComprehensiveBacktester"
    
    content = content.replace(old_import, new_import)
    
    # Write back
    with open(agent_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated AEGS backtest agent to use caching!")


def demo_cached_backtest():
    """
    Demonstrate the cached backtesting
    """
    
    print("🔥💎 CACHED AEGS BACKTESTING DEMO 💎🔥")
    print("=" * 60)
    
    # Test with cache
    print("\n1️⃣ First run (downloads and caches):")
    import time
    start = time.time()
    
    backtester = CachedComprehensiveBacktester("BB", use_cache=True)
    df = backtester.download_maximum_data()
    
    first_time = time.time() - start
    print(f"⏱️  Time taken: {first_time:.1f} seconds")
    
    # Test cache hit
    print("\n2️⃣ Second run (uses cache):")
    start = time.time()
    
    backtester2 = CachedComprehensiveBacktester("BB", use_cache=True)
    df2 = backtester2.download_maximum_data()
    
    second_time = time.time() - start
    print(f"⏱️  Time taken: {second_time:.1f} seconds")
    print(f"🚀 Speed improvement: {first_time/second_time:.1f}x faster!")
    
    # Show cache stats
    print("\n📊 Cache Statistics:")
    cache = YFinanceCache(cache_dir="aegs_yfinance_cache")
    stats = cache.get_cache_stats()
    print(f"   Total cached items: {stats['total_items']}")
    print(f"   Cache size: {stats['total_size_mb']:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-agents', action='store_true', help='Update AEGS agents to use cache')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    
    args = parser.parse_args()
    
    if args.update_agents:
        update_aegs_agents_to_use_cache()
    elif args.demo:
        demo_cached_backtest()
    else:
        print("Use --update-agents to update AEGS agents or --demo to see caching in action")