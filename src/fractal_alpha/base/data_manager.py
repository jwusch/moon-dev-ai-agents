"""
📊 Fractal Data Manager
Unified data access layer with multi-source support and synthetic tick generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import sqlite3
import json
import os
from collections import defaultdict, deque

from ..utils.synthetic_ticks import SyntheticTickGenerator
from .types import TickData, BarData, TimeFrame


class FractalDataManager:
    """
    Manages data acquisition from multiple sources with intelligent fallback
    """
    
    def __init__(self, cache_dir: str = "fractal_cache"):
        """
        Initialize data manager
        
        Args:
            cache_dir: Directory for local cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.tick_generator = SyntheticTickGenerator(method='adaptive')
        self.sources = self._initialize_sources()
        self.cache = DataCache(cache_dir)
        
        # In-memory buffers
        self.tick_buffers = defaultdict(lambda: deque(maxlen=100000))  # ~100k ticks per symbol
        self.bar_cache = {}  # Symbol -> TimeFrame -> DataFrame
        
    def _initialize_sources(self) -> Dict:
        """Initialize available data sources"""
        sources = {}
        
        # Try to import each source
        try:
            import yfinance as yf
            sources['yfinance'] = YFinanceSource()
            print("✅ YFinance data source initialized")
        except ImportError:
            print("⚠️ YFinance not available")
            
        # Add more sources as needed (Polygon, Binance, etc.)
        
        return sources
    
    async def get_tick_data(self, 
                          symbol: str, 
                          start: datetime, 
                          end: datetime,
                          source: Optional[str] = None) -> List[TickData]:
        """
        Get tick data from best available source
        
        Args:
            symbol: Symbol to fetch
            start: Start datetime
            end: End datetime
            source: Specific source to use (optional)
            
        Returns:
            List of tick data
        """
        # Check memory buffer first
        buffered_ticks = self._get_buffered_ticks(symbol, start, end)
        if buffered_ticks:
            return buffered_ticks
            
        # Check cache
        cached_ticks = self.cache.get_ticks(symbol, start, end)
        if cached_ticks:
            return cached_ticks
            
        # Generate synthetic ticks from bars
        bars = await self.get_bar_data(symbol, TimeFrame.ONE_MIN, start, end)
        
        if bars.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Generate synthetic ticks
        all_ticks = []
        for _, bar in bars.iterrows():
            ticks = self.tick_generator.generate_ticks(bar, n_ticks=10)
            all_ticks.extend(ticks)
            
        # Cache for future use
        self.cache.store_ticks(symbol, all_ticks)
        
        # Add to memory buffer
        for tick in all_ticks:
            self.tick_buffers[symbol].append(tick)
            
        return all_ticks
    
    async def get_bar_data(self,
                         symbol: str,
                         timeframe: TimeFrame,
                         start: datetime,
                         end: datetime,
                         source: Optional[str] = None) -> pd.DataFrame:
        """
        Get OHLCV bar data
        
        Args:
            symbol: Symbol to fetch
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            source: Specific source to use
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe.value}"
        if cache_key in self.bar_cache:
            df = self.bar_cache[cache_key]
            # Filter to requested range
            mask = (df.index >= start) & (df.index <= end)
            filtered = df[mask]
            if not filtered.empty:
                return filtered
                
        # Fetch from source
        if source:
            if source in self.sources:
                df = await self.sources[source].fetch_bars(symbol, timeframe, start, end)
            else:
                raise ValueError(f"Unknown source: {source}")
        else:
            # Try sources in priority order
            for source_name, source_obj in self.sources.items():
                try:
                    df = await source_obj.fetch_bars(symbol, timeframe, start, end)
                    if not df.empty:
                        break
                except Exception as e:
                    print(f"⚠️ {source_name} failed for {symbol}: {e}")
                    continue
            else:
                raise ValueError(f"No data available for {symbol}")
                
        # Cache the data
        self.bar_cache[cache_key] = df
        
        return df
    
    def _get_buffered_ticks(self, 
                          symbol: str, 
                          start: datetime, 
                          end: datetime) -> List[TickData]:
        """Get ticks from memory buffer"""
        
        if symbol not in self.tick_buffers:
            return []
            
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        
        # Filter ticks in range
        ticks = []
        for tick in self.tick_buffers[symbol]:
            if start_ts <= tick.timestamp <= end_ts:
                ticks.append(tick)
                
        return ticks
    
    async def stream_ticks(self, symbols: List[str], callback):
        """
        Stream real-time tick data
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with new ticks
        """
        # For now, generate synthetic ticks from 1-min bars
        # In production, would connect to WebSocket feeds
        
        while True:
            for symbol in symbols:
                try:
                    # Get latest bar
                    end = datetime.now()
                    start = end - timedelta(minutes=1)
                    
                    bars = await self.get_bar_data(
                        symbol, 
                        TimeFrame.ONE_MIN, 
                        start, 
                        end
                    )
                    
                    if not bars.empty:
                        # Generate ticks for latest bar
                        latest_bar = bars.iloc[-1]
                        ticks = self.tick_generator.generate_ticks(latest_bar, n_ticks=5)
                        
                        # Stream ticks with delays
                        for tick in ticks:
                            await callback(symbol, tick)
                            await asyncio.sleep(0.1)  # 100ms between ticks
                            
                except Exception as e:
                    print(f"❌ Stream error for {symbol}: {e}")
                    
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute


class YFinanceSource:
    """YFinance data source adapter"""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        
    async def fetch_bars(self, 
                        symbol: str, 
                        timeframe: TimeFrame,
                        start: datetime,
                        end: datetime) -> pd.DataFrame:
        """Fetch OHLCV bars from YFinance"""
        
        # Map timeframe to yfinance interval
        interval_map = {
            TimeFrame.ONE_MIN: "1m",
            TimeFrame.FIVE_MIN: "5m",
            TimeFrame.FIFTEEN_MIN: "15m",
            TimeFrame.ONE_HOUR: "1h",
            TimeFrame.FOUR_HOUR: "4h",
            TimeFrame.DAILY: "1d"
        }
        
        interval = interval_map.get(timeframe, "1h")
        
        # Download data
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=interval
        )
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
            
        return df[required]


class DataCache:
    """Simple cache implementation using SQLite"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "fractal_cache.db")
        self._init_db()
        
    def _init_db(self):
        """Initialize cache database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Tick cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_cache (
                    symbol TEXT,
                    timestamp INTEGER,
                    price REAL,
                    volume INTEGER,
                    side INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Bar cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bar_cache (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tick_time ON tick_cache(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bar_time ON bar_cache(symbol, timeframe, timestamp)")
            
    def get_ticks(self, symbol: str, start: datetime, end: datetime) -> List[TickData]:
        """Get cached tick data"""
        
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, price, volume, side
                FROM tick_cache
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (symbol, start_ts, end_ts))
            
            ticks = []
            for row in cursor:
                ticks.append(TickData(
                    timestamp=row[0],
                    price=row[1],
                    volume=row[2],
                    side=row[3]
                ))
                
        return ticks
    
    def store_ticks(self, symbol: str, ticks: List[TickData]):
        """Store ticks in cache"""
        
        if not ticks:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data
            data = [(symbol, t.timestamp, t.price, t.volume, t.side) for t in ticks]
            
            # Insert or replace
            conn.executemany("""
                INSERT OR REPLACE INTO tick_cache 
                (symbol, timestamp, price, volume, side)
                VALUES (?, ?, ?, ?, ?)
            """, data)
            
            conn.commit()


def demonstrate_data_manager():
    """Quick demonstration of the data manager"""
    
    import asyncio
    
    async def demo():
        # Initialize manager
        manager = FractalDataManager()
        
        print("📊 Fractal Data Manager Demo\n")
        
        # Test bar data fetching
        symbol = "AAPL"
        end = datetime.now()
        start = end - timedelta(days=5)
        
        print(f"Fetching {symbol} 5-min bars...")
        bars = await manager.get_bar_data(
            symbol, 
            TimeFrame.FIVE_MIN,
            start,
            end
        )
        
        print(f"✅ Got {len(bars)} bars")
        print(f"Latest bar: {bars.iloc[-1].to_dict()}")
        
        # Test synthetic tick generation
        print(f"\nGenerating synthetic ticks...")
        tick_start = end - timedelta(hours=1)
        ticks = await manager.get_tick_data(symbol, tick_start, end)
        
        print(f"✅ Generated {len(ticks)} ticks")
        print(f"First tick: Price=${ticks[0].price:.2f}, Volume={ticks[0].volume}")
        print(f"Last tick: Price=${ticks[-1].price:.2f}, Volume={ticks[-1].volume}")
        
        # Calculate tick statistics
        buy_volume = sum(t.volume for t in ticks if t.side == 1)
        sell_volume = sum(t.volume for t in ticks if t.side == -1)
        
        print(f"\nTick Statistics:")
        print(f"Buy Volume: {buy_volume:,}")
        print(f"Sell Volume: {sell_volume:,}")
        print(f"Buy/Sell Ratio: {buy_volume/sell_volume:.2f}")
    
    # Run demo
    asyncio.run(demo())


if __name__ == "__main__":
    demonstrate_data_manager()