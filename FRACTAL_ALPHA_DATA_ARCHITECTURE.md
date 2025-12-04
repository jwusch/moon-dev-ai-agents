# 📊 Fractal Alpha Data Architecture Specification

## Overview

This document defines the data architecture for handling multi-timeframe tick data in the AEGS Fractal Alpha system. The architecture prioritizes performance, scalability, and integration with existing AEGS infrastructure.

## 🗄️ Storage Architecture

### Three-Tier Storage System

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier 1: Hot Cache (Redis)                 │
│  - Last 2 hours of tick data                               │
│  - Real-time indicator values                               │
│  - Order book snapshots (100ms intervals)                  │
│  - Size: ~2GB per symbol                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Tier 2: Warm Storage (SQLite)               │
│  - 30 days of aggregated data (1min, 5min, 15min bars)    │
│  - Daily indicator snapshots                                │
│  - Backtest results cache                                  │
│  - Size: ~50MB per symbol                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Tier 3: Cold Storage (Parquet Files)          │
│  - Historical tick data (> 30 days)                        │
│  - Compressed with zstd                                    │
│  - Partitioned by symbol/date                              │
│  - Size: ~100MB per symbol per month                       │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Pipeline

### Real-Time Processing Flow

```python
# Pseudo-code for data pipeline
class FractalDataPipeline:
    def __init__(self):
        self.redis_client = Redis(decode_responses=False)  # Binary for speed
        self.sqlite_conn = sqlite3.connect('fractal_alpha.db')
        self.tick_buffer = deque(maxlen=10000)  # In-memory buffer
        
    async def process_tick(self, symbol: str, tick: dict):
        """Process incoming tick data"""
        # 1. Add to memory buffer
        self.tick_buffer.append(tick)
        
        # 2. Update Redis cache (non-blocking)
        await self.update_redis_cache(symbol, tick)
        
        # 3. Calculate real-time indicators
        if len(self.tick_buffer) % 100 == 0:  # Every 100 ticks
            await self.calculate_microstructure_indicators(symbol)
            
        # 4. Aggregate to bars if needed
        if self.should_create_bar(symbol, tick):
            bar = self.create_bar(symbol)
            await self.store_bar(bar)
```

### Data Schema

#### Redis Schema (Binary Protocol)
```python
# Tick data key structure
TICK_KEY = "tick:{symbol}:{timestamp}"  # tick:AAPL:1701234567890
TICK_VALUE = msgpack.packb({
    'p': float,     # price
    'v': int,       # volume
    's': int,       # side (1=buy, -1=sell)
    'ts': int       # timestamp (ms)
})

# Indicator cache
INDICATOR_KEY = "ind:{symbol}:{indicator}:{timeframe}"
INDICATOR_VALUE = msgpack.packb({
    'value': float,
    'ts': int,
    'meta': dict
})

# Order book snapshot
ORDERBOOK_KEY = "ob:{symbol}:{timestamp}"
ORDERBOOK_VALUE = msgpack.packb({
    'bids': [[price, size], ...],  # Top 10
    'asks': [[price, size], ...],  # Top 10
    'ts': int
})
```

#### SQLite Schema
```sql
-- Aggregated bars table
CREATE TABLE bars (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    buy_volume INTEGER,
    sell_volume INTEGER,
    tick_count INTEGER,
    vwap REAL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX idx_bars_symbol_time ON bars(symbol, timestamp);

-- Indicator values table
CREATE TABLE indicator_values (
    symbol TEXT NOT NULL,
    indicator TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    value REAL NOT NULL,
    metadata TEXT,  -- JSON
    PRIMARY KEY (symbol, indicator, timeframe, timestamp)
);

-- Microstructure metrics table  
CREATE TABLE microstructure (
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    tick_imbalance REAL,
    order_flow_ratio REAL,
    bid_ask_spread REAL,
    kyle_lambda REAL,
    vpin REAL,
    PRIMARY KEY (symbol, timestamp)
);
```

## 🚀 Performance Optimizations

### Memory Management Strategy

```python
class MemoryEfficientDataManager:
    def __init__(self):
        self.window_size = 7200  # 2 hours in seconds
        self.tick_cache = {}     # Symbol -> CircularBuffer
        
    def add_tick(self, symbol: str, tick: dict):
        """Add tick with automatic memory cleanup"""
        if symbol not in self.tick_cache:
            self.tick_cache[symbol] = CircularBuffer(
                capacity=100_000,  # ~10MB per symbol
                dtype=np.dtype([
                    ('timestamp', 'i8'),
                    ('price', 'f4'),
                    ('volume', 'i4'),
                    ('side', 'i1')
                ])
            )
        
        # Add to circular buffer (automatically drops old data)
        self.tick_cache[symbol].append(tick)
        
    def get_recent_ticks(self, symbol: str, seconds: int):
        """Get recent ticks without memory copy"""
        if symbol not in self.tick_cache:
            return np.array([])
            
        buffer = self.tick_cache[symbol]
        cutoff = time.time() - seconds
        
        # Use view instead of copy
        mask = buffer.data['timestamp'] > cutoff
        return buffer.data[mask]
```

### Batch Processing for Efficiency

```python
class BatchProcessor:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.pending_ticks = defaultdict(list)
        
    async def process_tick_batch(self, symbol: str):
        """Process ticks in batches for efficiency"""
        ticks = self.pending_ticks[symbol]
        if len(ticks) < self.batch_size:
            return
            
        # Vectorized calculations
        prices = np.array([t['price'] for t in ticks])
        volumes = np.array([t['volume'] for t in ticks])
        sides = np.array([t['side'] for t in ticks])
        
        # Calculate microstructure metrics in batch
        metrics = {
            'tick_imbalance': self.calc_tick_imbalance(sides, volumes),
            'volume_weighted_price': np.average(prices, weights=volumes),
            'order_flow': np.sum(volumes * sides)
        }
        
        # Store results
        await self.store_metrics(symbol, metrics)
        
        # Clear batch
        self.pending_ticks[symbol] = []
```

## 📈 Data Access Patterns

### Indicator Calculation Access
```python
# Optimized for indicator calculations
class IndicatorDataAccess:
    def get_data_for_indicator(self, symbol: str, indicator_type: str):
        """Get data optimized for specific indicator"""
        
        if indicator_type == "tick_microstructure":
            # Get from Redis (last 2 hours)
            return self.redis_client.get_tick_stream(symbol)
            
        elif indicator_type == "williams_fractal":
            # Get from SQLite (5min bars, 30 days)
            return self.sqlite_conn.execute("""
                SELECT timestamp, high, low, close
                FROM bars
                WHERE symbol = ? AND timeframe = '5m'
                ORDER BY timestamp DESC
                LIMIT 8640  -- 30 days of 5min bars
            """, (symbol,))
            
        elif indicator_type == "hurst_exponent":
            # Get from SQLite (daily bars, 1 year)
            return self.sqlite_conn.execute("""
                SELECT timestamp, close
                FROM bars
                WHERE symbol = ? AND timeframe = '1d'
                ORDER BY timestamp DESC
                LIMIT 252
            """, (symbol,))
```

### Backtesting Access
```python
class BacktestDataProvider:
    def get_backtest_data(self, symbol: str, start_date: str, end_date: str):
        """Get historical data for backtesting"""
        
        # Check cache first
        cache_key = f"backtest:{symbol}:{start_date}:{end_date}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return msgpack.unpackb(cached)
            
        # Load from SQLite/Parquet
        data = self.load_historical_data(symbol, start_date, end_date)
        
        # Cache for future use (24 hour TTL)
        self.redis_client.setex(
            cache_key, 
            86400, 
            msgpack.packb(data)
        )
        
        return data
```

## 🔧 Integration with AEGS

### Seamless Integration Points

1. **Shared SQLite Database**
   - AEGS tables remain unchanged
   - New tables added with `fractal_` prefix
   - Foreign key relationships where applicable

2. **Unified Configuration**
   ```python
   # In config.py
   FRACTAL_ALPHA_CONFIG = {
       'redis_host': 'localhost',
       'redis_port': 6379,
       'redis_db': 1,  # Separate DB from main Redis
       'tick_retention_hours': 2,
       'max_memory_per_symbol_mb': 10,
       'batch_size': 1000,
       'enable_compression': True
   }
   ```

3. **Data Access Layer**
   ```python
   # Extend existing nice_funcs.py
   def get_fractal_indicators(symbol: str, timeframe: str = '5m'):
       """Get all fractal indicators for symbol"""
       # Implementation here
       pass
   ```

## 📊 Monitoring & Metrics

### Performance Metrics to Track
- Tick processing latency (p50, p95, p99)
- Memory usage per symbol
- Cache hit rates
- Indicator calculation times
- Data pipeline throughput

### Health Checks
```python
class DataHealthMonitor:
    async def check_data_health(self):
        """Regular health checks"""
        return {
            'redis_connected': await self.check_redis(),
            'sqlite_accessible': self.check_sqlite(),
            'memory_usage_mb': self.get_memory_usage(),
            'tick_lag_ms': self.get_tick_processing_lag(),
            'cache_hit_rate': self.get_cache_hit_rate()
        }
```

## 🚦 Implementation Checklist

- [ ] Set up Redis with proper configuration
- [ ] Create SQLite schema and indexes  
- [ ] Implement circular buffer for tick data
- [ ] Build batch processing pipeline
- [ ] Create data access layer
- [ ] Add performance monitoring
- [ ] Write integration tests
- [ ] Document API endpoints
- [ ] Create migration scripts
- [ ] Set up backup strategy

---

*"Efficient data architecture is the foundation of high-frequency alpha capture!"* - Moon Dev 🌙