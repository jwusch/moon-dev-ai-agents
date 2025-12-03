# 🌙 TradingView Historical Data Guide

## Overview

Yes! TradingView API can retrieve historical data for any symbol. This is done through the authenticated TradingView WebSocket API which provides access to real historical OHLCV (Open, High, Low, Close, Volume) data.

## Two Approaches

### 1. Basic TradingView (tradingview-ta)
- **Current data only** - No historical data
- No authentication required
- Good for real-time technical analysis
- Limited by rate limits

### 2. Authenticated TradingView API ✅
- **Full historical data** available
- Requires TradingView account credentials
- Better rate limits
- Access to all timeframes

## Setup

### Prerequisites
1. TradingView account (free or paid)
2. Node.js server running
3. Credentials in `.env` file:
```env
TRADINGVIEW_USERNAME=your_email@example.com
TRADINGVIEW_PASSWORD=your_password
```

### Start the Server
```bash
cd tradingview-server
npm install  # First time only
npm start
```

## Usage Examples

### Get Historical Data

```python
from src.agents.tradingview_adapter import TradingViewAdapter

# Initialize adapter (automatically uses auth if available)
adapter = TradingViewAdapter()

# Get 100 hourly bars for BTC
df = adapter.get_ohlcv_data(
    symbol='BTCUSDT',
    interval='1h',  # 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
    limit=100       # Number of bars
)

# DataFrame contains:
# - timestamp, open, high, low, close, volume
# - Optional: rsi, macd, recommendation
```

### Direct API Usage

```python
from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI

# Create authenticated client
api = TradingViewAuthenticatedAPI()

# Get historical data with specific timeframe
df = api.get_historical_data(
    symbol='ETHUSDT',
    timeframe='60',  # Minutes: 1, 5, 15, 30, 60, 240 or '1D', '1W', '1M'
    bars=200,        # Number of bars to fetch
    exchange='BINANCE'
)

# Clean up
api.close()
```

## Available Timeframes

| Code | Timeframe |
|------|-----------|
| 1    | 1 minute  |
| 5    | 5 minutes |
| 15   | 15 minutes|
| 30   | 30 minutes|
| 60   | 1 hour    |
| 240  | 4 hours   |
| 1D   | 1 day     |
| 1W   | 1 week    |
| 1M   | 1 month   |

## Data Format

Historical data is returned as a pandas DataFrame:

```python
# Example output
      timestamp      open      high       low     close    volume
0    2024-01-01  42000.0   42500.0   41800.0   42300.0   1234.56
1    2024-01-01  42300.0   42600.0   42100.0   42400.0   1345.67
...
```

## Benefits Over Binance API

1. **No geo-restrictions** - Works globally
2. **No proxy needed** - Direct access
3. **Multiple markets** - Crypto, stocks, forex, etc.
4. **Built-in indicators** - RSI, MACD, etc included
5. **Better rate limits** with authentication

## Example: Simple Backtest

```python
# Get historical data
df = adapter.get_ohlcv_data('BTCUSDT', '1h', 500)

# Calculate indicators
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()

# Generate signals
df['signal'] = 0
df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1

# Backtest logic here...
```

## Rate Limits

- **Unauthenticated**: ~1 request/second
- **Authenticated**: Much higher limits
- **Recommendation**: Always use authentication for historical data

## Troubleshooting

### "Server not running" error
```bash
cd tradingview-server
npm start
```

### "Invalid credentials" error
- Check username/password in `.env`
- Try manual login first
- Special characters in password may need escaping

### No historical data
- Ensure authenticated API is initialized
- Check server logs for errors
- Verify symbol exists on exchange

## Integration with Moon Dev Bots

The TradingView adapter is fully compatible with existing Moon Dev infrastructure:

```python
# In api_adapter.py - automatically uses TradingView
from src.agents.api_adapter import APIAdapter

api = APIAdapter()  # Will use TradingView as primary source
df = api.get_ohlcv_data('SOLUSDT', '15m', 100)
```

---

Built with 🌙 by Moon Dev