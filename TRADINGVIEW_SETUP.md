# 🌙 TradingView Integration Guide

This guide explains how to use TradingView as an alternative data source for market data and technical analysis.

## Overview

TradingView provides free access to:
- Real-time price data
- 200+ technical indicators
- Buy/Sell/Neutral recommendations
- Multi-timeframe analysis
- Support for crypto, stocks, forex, and more

## Setup

### 1. Install the Python Library

```bash
pip install tradingview-ta
```

### 2. No API Key Required!

Unlike Binance or other APIs, TradingView doesn't require any API keys or authentication.

## Usage

### Basic Example

```python
from src.agents.tradingview_api import TradingViewAPI

# Initialize
tv = TradingViewAPI()

# Get analysis
analysis = tv.get_analysis('BTCUSDT', 'BINANCE', '1h')
if analysis:
    print(f"Price: ${analysis['indicators']['close']:,.2f}")
    print(f"Recommendation: {analysis['summary']['RECOMMENDATION']}")
    print(f"RSI: {analysis['indicators']['RSI']}")
```

### Using with APIAdapter

To use TradingView as your primary data source:

```bash
# Set environment variable
export DATA_SOURCE=tradingview

# Or in your .env file
DATA_SOURCE=tradingview
```

Then use APIAdapter normally:

```python
from src.agents.api_adapter import APIAdapter

api = APIAdapter()
# Will automatically use TradingView

funding = api.get_funding_data()
```

## Available Features

### 1. Technical Analysis
- Get comprehensive technical analysis for any symbol
- Includes all major indicators (RSI, MACD, Moving Averages, etc.)
- Buy/Sell recommendations based on multiple indicators

### 2. Multi-Timeframe Analysis
Supported intervals:
- 1m, 5m, 15m, 30m (minute intervals)
- 1h, 2h, 4h (hour intervals)
- 1d, 1w, 1M (daily, weekly, monthly)

### 3. Multiple Markets
- **Crypto**: Binance, Coinbase, Kraken, etc.
- **Stocks**: NASDAQ, NYSE, etc.
- **Forex**: Major currency pairs
- **Futures**: Various futures markets

### 4. Indicator Categories

**Oscillators** (14 indicators):
- RSI, MACD, Stochastic, CCI, Williams %R, etc.

**Moving Averages** (17 indicators):
- SMA, EMA, WMA, Hull MA, VWMA, etc.

**Other Indicators**:
- Bollinger Bands, Ichimoku Cloud, Pivot Points, etc.

## Rate Limits

⚠️ **Important**: TradingView has rate limits to prevent abuse:
- HTTP 429 errors indicate you're being rate limited
- Recommended: 1 request per second maximum
- The library includes built-in rate limiting

## Integration Options

### Option 1: Direct Usage
Use `TradingViewAPI` directly for specific needs:

```python
from src.agents.tradingview_api import TradingViewAPI

tv = TradingViewAPI()
indicators = tv.get_indicators('BTCUSDT')
momentum = tv.get_momentum_indicators('BTCUSDT')
trend = tv.get_trend_indicators('BTCUSDT')
```

### Option 2: Via Adapter
Use `TradingViewAdapter` for Binance-compatible interface:

```python
from src.agents.tradingview_adapter import TradingViewAdapter

adapter = TradingViewAdapter()
# Methods mimic Binance API
liq_data = adapter.get_liquidation_data()  # Estimated
funding = adapter.get_funding_data()        # Calculated from indicators
```

### Option 3: Unified API
Use `UnifiedDataAPI` for automatic source selection:

```python
from src.agents.unified_data_api import UnifiedDataAPI

api = UnifiedDataAPI()
# Automatically uses best available source
price = api.get_price('BTCUSDT')
```

## Limitations

Since TradingView provides technical analysis rather than raw market data:

1. **No Historical Candles**: Only current values are available
2. **No Order Book**: No depth data or order flow
3. **No Liquidations**: These are estimated based on volatility
4. **No Real Funding Rates**: Calculated from momentum indicators

## Best Use Cases

TradingView is ideal for:
- ✅ Technical analysis signals
- ✅ Multi-timeframe screening
- ✅ Indicator-based strategies
- ✅ Market sentiment (Buy/Sell recommendations)
- ✅ Cross-market analysis (stocks + crypto)

Not ideal for:
- ❌ High-frequency trading (rate limits)
- ❌ Historical backtesting (no candle history)
- ❌ Order flow analysis
- ❌ Liquidation hunting

## Example: Multi-Asset Screener

```python
from src.agents.tradingview_api import TradingViewAPI
import time

# Screen multiple assets
symbols = ['BTCUSDT', 'ETHUSDT', 'AAPL', 'EURUSD']
exchanges = ['BINANCE', 'BINANCE', 'NASDAQ', 'FX_IDC']
screeners = ['crypto', 'crypto', 'america', 'forex']

for symbol, exchange, screener in zip(symbols, exchanges, screeners):
    tv = TradingViewAPI(screener=screener)
    rec = tv.get_recommendation(symbol, exchange, '1h')
    print(f"{symbol}: {rec}")
    time.sleep(1)  # Respect rate limit
```

## Troubleshooting

### HTTP 429 Errors
- You're hitting rate limits
- Add delays between requests (1+ seconds)
- Reduce request frequency

### Invalid Symbol Errors
- Check symbol format (e.g., 'BTCUSDT' not 'BTC/USDT')
- Verify exchange name (e.g., 'BINANCE' not 'binance')
- Some symbols may not be available

### No Data Returned
- Symbol might not exist on that exchange
- Try different exchange or screener
- Check internet connection

## Summary

TradingView is a great free alternative when:
- Binance API is geo-restricted
- You need technical analysis signals
- You want multi-market coverage
- You don't need historical data

For production trading bots, consider using it alongside other data sources for redundancy.