# 🌊 Fractal Alpha Data Pipeline - Complete Architecture

## Executive Summary

The Fractal Alpha data pipeline is designed to handle multiple data sources, from free APIs to premium feeds, with a focus on tick-level microstructure analysis while maintaining compatibility with existing AEGS infrastructure.

## 📡 Data Sources Architecture

### Primary Data Sources (In Order of Priority)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│ 1. YFinance (FREE)                                                  │
│    ├─ Real-time: 1-min delayed quotes                              │
│    ├─ Historical: 1m, 5m, 15m, 1h, 1d bars                        │
│    ├─ Coverage: Stocks, ETFs, Crypto (limited)                     │
│    └─ Rate Limit: ~2000 requests/hour                              │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Polygon.io (FREEMIUM)                                           │
│    ├─ Real-time: WebSocket tick data (paid)                        │
│    ├─ Historical: Full tick history                                │
│    ├─ Coverage: US Stocks, Options, Crypto                         │
│    └─ Free Tier: 5 API calls/minute                                │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Alpha Vantage (FREE with limits)                                │
│    ├─ Real-time: 1-min bars                                        │
│    ├─ Historical: Intraday back to 1 month                         │
│    ├─ Technical Indicators: Pre-calculated                         │
│    └─ Rate Limit: 5 calls/min (free)                               │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Binance API (FREE for crypto)                                   │
│    ├─ Real-time: WebSocket tick stream                             │
│    ├─ Historical: Full tick/trade data                             │
│    ├─ Order Book: L2 depth snapshots                               │
│    └─ Rate Limit: 1200 requests/min                                │
├─────────────────────────────────────────────────────────────────────┤
│ 5. IEX Cloud (FREEMIUM)                                            │
│    ├─ Real-time: 15-min delayed (free)                             │
│    ├─ Historical: 5 years of data                                  │
│    ├─ Coverage: US Stocks                                          │
│    └─ Free Tier: 50,000 messages/month                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Exchange-Specific Sources (Already Integrated)

- **Solana**: BirdEye API (via existing nice_funcs.py)
- **HyperLiquid**: Native WebSocket API
- **Robinhood**: robin_stocks library

## 🔄 Data Pipeline Implementation

### 1. Multi-Source Data Fetcher

```python
class UnifiedDataFetcher:
    """Intelligent data fetcher that uses best available source"""
    
    def __init__(self):
        self.sources = {
            'yfinance': YFinanceAdapter(),
            'polygon': PolygonAdapter(),
            'alphavantage': AlphaVantageAdapter(),
            'binance': BinanceAdapter(),
            'iex': IEXAdapter()
        }
        self.cache = RedisCache()
        
    async def get_tick_data(self, symbol: str, start: datetime, end: datetime):
        """Get tick data from best available source"""
        
        # Check cache first
        cached = self.cache.get_ticks(symbol, start, end)
        if cached and len(cached) > 0:
            return cached
            
        # Priority order for tick data
        if self._is_crypto(symbol):
            # Crypto: Try Binance first (free tick data!)
            data = await self._try_source('binance', symbol, start, end)
            if data:
                return data
                
        # Stocks: Try sources in order
        for source in ['polygon', 'yfinance', 'alphavantage', 'iex']:
            data = await self._try_source(source, symbol, start, end)
            if data:
                # Cache for future use
                self.cache.store_ticks(symbol, data)
                return data
                
        raise ValueError(f"No data available for {symbol}")
```

### 2. Real-Time Data Streaming

```python
class FractalDataStream:
    """Real-time data streaming with fallback"""
    
    def __init__(self):
        self.streams = {}
        self.tick_buffer = defaultdict(deque)
        
    async def subscribe(self, symbols: List[str]):
        """Subscribe to real-time data"""
        
        for symbol in symbols:
            if self._is_crypto(symbol):
                # Use Binance WebSocket for crypto
                stream = BinanceWebSocket(symbol)
                stream.on_tick = lambda tick: self._process_tick(symbol, tick)
            else:
                # Use YFinance 1-min bars as "pseudo-ticks"
                stream = YFinanceMinuteStream(symbol)
                stream.on_bar = lambda bar: self._process_bar(symbol, bar)
                
            self.streams[symbol] = stream
            await stream.connect()
            
    def _process_tick(self, symbol: str, tick: dict):
        """Process incoming tick"""
        # Convert to standard format
        std_tick = TickData(
            timestamp=tick['timestamp'],
            price=tick['price'],
            volume=tick['quantity'],
            side=1 if tick['is_buyer_maker'] else -1
        )
        
        # Add to buffer
        self.tick_buffer[symbol].append(std_tick)
        
        # Trigger microstructure calculations every 100 ticks
        if len(self.tick_buffer[symbol]) >= 100:
            self._calculate_microstructure(symbol)
```

### 3. Historical Data Backfill

```python
class HistoricalDataManager:
    """Manages historical data acquisition and storage"""
    
    async def backfill_symbol(self, symbol: str, days_back: int = 30):
        """Backfill historical data for symbol"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get different timeframes
        tasks = []
        
        # 1-minute bars (most granular available for free)
        tasks.append(self._fetch_bars(symbol, '1m', start_date, end_date))
        
        # 5-minute bars (for fractal patterns)
        tasks.append(self._fetch_bars(symbol, '5m', start_date, end_date))
        
        # 15-minute bars (for regime detection)
        tasks.append(self._fetch_bars(symbol, '15m', start_date, end_date))
        
        # 1-hour bars (for trend analysis)
        tasks.append(self._fetch_bars(symbol, '1h', start_date, end_date))
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks)
        
        # Store in SQLite
        self._store_historical_data(symbol, results)
```

### 4. Tick Reconstruction from Bars

Since true tick data is expensive, we can reconstruct "synthetic ticks" from 1-minute bars:

```python
class SyntheticTickGenerator:
    """Generate synthetic tick data from minute bars"""
    
    def generate_ticks(self, bars: pd.DataFrame) -> List[TickData]:
        """Convert OHLCV bars to synthetic ticks"""
        
        ticks = []
        
        for _, bar in bars.iterrows():
            timestamp = int(bar['timestamp'] * 1000)
            
            # Generate ticks based on price movement
            # This is a simplified model - can be enhanced
            
            # Opening tick
            ticks.append(TickData(
                timestamp=timestamp,
                price=bar['open'],
                volume=int(bar['volume'] * 0.2),  # 20% at open
                side=1
            ))
            
            # High tick (if different from open)
            if bar['high'] > bar['open']:
                ticks.append(TickData(
                    timestamp=timestamp + 15000,  # +15 seconds
                    price=bar['high'],
                    volume=int(bar['volume'] * 0.3),  # 30% at high
                    side=1  # Buying pressure to reach high
                ))
            
            # Low tick (if different from open)
            if bar['low'] < bar['open']:
                ticks.append(TickData(
                    timestamp=timestamp + 30000,  # +30 seconds
                    price=bar['low'],
                    volume=int(bar['volume'] * 0.3),  # 30% at low
                    side=-1  # Selling pressure to reach low
                ))
            
            # Closing tick
            close_side = 1 if bar['close'] > bar['open'] else -1
            ticks.append(TickData(
                timestamp=timestamp + 59000,  # +59 seconds
                price=bar['close'],
                volume=int(bar['volume'] * 0.2),  # 20% at close
                side=close_side
            ))
            
        return ticks
```

### 5. Order Book Reconstruction

For microstructure analysis without L2 data:

```python
class OrderBookEstimator:
    """Estimate order book dynamics from price/volume"""
    
    def estimate_spread(self, recent_ticks: List[TickData]) -> float:
        """Estimate bid-ask spread from tick data"""
        
        if len(recent_ticks) < 10:
            return 0.0
            
        # Group by direction
        buys = [t.price for t in recent_ticks if t.side == 1]
        sells = [t.price for t in recent_ticks if t.side == -1]
        
        if not buys or not sells:
            # Use high-low of recent ticks as proxy
            prices = [t.price for t in recent_ticks]
            return (max(prices) - min(prices)) / np.mean(prices)
            
        # Estimated spread
        estimated_ask = np.percentile(sells, 25)  # Lower quartile of sells
        estimated_bid = np.percentile(buys, 75)   # Upper quartile of buys
        
        return (estimated_ask - estimated_bid) / ((estimated_ask + estimated_bid) / 2)
```

## 🏗️ Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         1. DATA ACQUISITION                          │
├─────────────────────────────────────────────────────────────────────┤
│  YFinance ──┐                                                       │
│  Polygon ───┤                                                       │
│  Binance ───┼──► UnifiedDataFetcher ──► Rate Limiter ──► Cache    │
│  IEX ───────┤                                                       │
│  Alpha V ───┘                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         2. DATA PROCESSING                           │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Data ──► Validation ──► Normalization ──► Tick Generation     │
│                                │                                     │
│                                ▼                                     │
│                         Quality Checks                               │
│                         • Gaps Detection                             │
│                         • Outlier Removal                            │
│                         • Time Alignment                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          3. STORAGE LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  Hot (Redis)                                                        │
│  • Last 2 hours ticks          SQLite Tables:                      │
│  • Real-time indicators        • bars                               │
│  • Order book snapshots        • microstructure                     │
│                                • indicators                          │
│  Warm (SQLite) ◄──────────────• tick_samples                       │
│  • 30 days aggregated                                               │
│  • Indicator history           Parquet Files:                       │
│                                • Historical ticks                    │
│  Cold (Parquet) ◄─────────────• Compressed daily                   │
│  • Long-term storage           • S3 compatible                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      4. INDICATOR CALCULATION                        │
├─────────────────────────────────────────────────────────────────────┤
│  Microstructure Engine         Fractal Engine                      │
│  • Tick Imbalance              • Williams Fractals                  │
│  • Order Flow                  • Hurst Exponent                     │
│  • VPIN                        • DFA Analysis                       │
│                                                                      │
│  Market Structure              ML Features                          │
│  • Kyle's Lambda               • Entropy Indicators                  │
│  • Amihud Ratio                • Wavelet Decomposition              │
│  • Spread Dynamics             • HMM Regimes                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        5. SIGNAL GENERATION                          │
├─────────────────────────────────────────────────────────────────────┤
│  Individual Signals ──► Ensemble Voting ──► Risk Filters           │
│                                │                                     │
│                                ▼                                     │
│                         AEGS Integration                             │
│                         • Weighted Combination                       │
│                         • Confidence Scoring                         │
│                         • Final Signal                               │
└─────────────────────────────────────────────────────────────────────┘
```

## 💰 Cost Analysis

### Free Tier Capabilities
- **YFinance**: Unlimited (rate limited)
- **Binance**: Unlimited crypto tick data
- **Polygon**: 5 requests/min = ~150 symbols/hour updates
- **Alpha Vantage**: 5 requests/min with API key
- **IEX**: 50,000 messages/month = ~1,600/day

### Estimated Costs for Premium
- **Polygon**: $79/month for unlimited tick data
- **IEX Cloud**: $9/month for real-time data
- **Alpha Vantage**: $50/month for 30 requests/min

## 🚀 Implementation Priority

### Phase 1: Free Data Sources (Week 1)
1. Integrate YFinance for all symbols ✓
2. Add Binance WebSocket for crypto ticks
3. Implement synthetic tick generation
4. Build caching layer

### Phase 2: Enhanced Data (Week 2)
1. Add Polygon free tier
2. Implement order book estimation
3. Build tick aggregation engine
4. Create data quality monitoring

### Phase 3: Premium Features (Optional)
1. Polygon tick data subscription
2. Real-time WebSocket feeds
3. Full order book depth
4. Cross-exchange arbitrage data

## 📊 Data Quality Assurance

```python
class DataQualityMonitor:
    """Monitor and ensure data quality"""
    
    def validate_tick_data(self, ticks: List[TickData]) -> bool:
        """Validate tick data quality"""
        
        checks = {
            'has_data': len(ticks) > 0,
            'timestamps_sequential': self._check_timestamps(ticks),
            'prices_reasonable': self._check_prices(ticks),
            'volumes_positive': all(t.volume > 0 for t in ticks),
            'no_large_gaps': self._check_gaps(ticks),
            'spread_reasonable': self._check_spreads(ticks)
        }
        
        return all(checks.values())
```

## 🔧 Configuration

```python
# config.py additions
FRACTAL_DATA_CONFIG = {
    'sources': {
        'yfinance': {'enabled': True, 'priority': 1},
        'polygon': {'enabled': True, 'priority': 2, 'api_key': os.getenv('POLYGON_API_KEY')},
        'binance': {'enabled': True, 'priority': 1},  # For crypto
        'alphavantage': {'enabled': False, 'priority': 3},
        'iex': {'enabled': False, 'priority': 4}
    },
    'synthetic_ticks': {
        'enabled': True,
        'model': 'uniform',  # or 'vwap_weighted'
        'ticks_per_bar': 4
    },
    'caching': {
        'redis_ttl_seconds': 7200,  # 2 hours
        'sqlite_retention_days': 30,
        'parquet_compression': 'zstd'
    }
}
```

---

*"The best data pipeline uses multiple sources intelligently - free when possible, premium when necessary!"* - Moon Dev 🌙