# 🌙 TradingView Authenticated API Guide

This guide explains how to use TradingView with authentication for better rate limits and more features.

## Why Use Authentication?

### Without Authentication (Current Python Library):
- ❌ Rate limited to ~1 request/second
- ❌ HTTP 429 errors frequently
- ❌ Limited to public indicators
- ❌ No real-time streaming

### With Authentication (Node.js + Python):
- ✅ Much higher rate limits
- ✅ Access to premium indicators
- ✅ Real-time streaming data
- ✅ Batch operations
- ✅ Your custom Pine Script indicators

## Setup Instructions

### 1. Prerequisites

- Node.js installed (v14+)
- TradingView account (free or premium)
- Python with requests library

### 2. Quick Setup

```bash
# Run the setup script
./setup_tradingview_auth.sh
```

Or manually:

```bash
# Install Node.js dependencies
cd tradingview-server
npm install

# Copy environment template
cp .env.example .env
```

### 3. Configure Credentials

Edit `tradingview-server/.env`:

```env
# Your TradingView login credentials
TV_USERNAME=your_tradingview_username
TV_PASSWORD=your_tradingview_password

# Server port (optional)
TV_SERVER_PORT=8888
```

### 4. Start the Server

In one terminal:

```bash
cd tradingview-server
npm start
```

You should see:
```
🌙 TradingView API Server
📡 Running on http://localhost:8888
```

### 5. Use from Python

```python
from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI

# Initialize (auto-login with env credentials)
api = TradingViewAuthenticatedAPI()

# Get real-time data
price = api.get_price('BTCUSDT')
print(f"BTC: ${price:,.2f}")

# Get OHLCV
ohlcv = api.get_ohlcv('ETHUSDT', timeframe='60')

# Batch operations (fast!)
data = api.get_multiple_symbols(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

# Real-time monitoring
api.monitor_symbol('BTCUSDT', duration=60)
```

## API Features

### 1. Real-Time Data

Unlike the public API, authenticated access provides real-time streaming:

```python
# Monitor price changes
def on_price_update(data):
    print(f"Price update: ${data['close']:,.2f}")

api.monitor_symbol('BTCUSDT', duration=300, callback=on_price_update)
```

### 2. Technical Indicators

Access all TradingView indicators:

```python
indicators = api.get_technical_indicators(
    'BTCUSDT',
    [
        'RSI@tv-basicstudies',
        'MACD@tv-basicstudies',
        'BB@tv-basicstudies',
        'EMA@tv-basicstudies',
        'Volume@tv-basicstudies',
    ]
)
```

### 3. Market Scanner

Scan multiple symbols efficiently:

```python
from src.agents.tradingview_authenticated_api import TradingViewAnalyzer

analyzer = TradingViewAnalyzer(api)

# Get market overview
overview = analyzer.get_market_overview([
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'
])

# Scan for opportunities
opportunities = analyzer.scan_for_opportunities(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    criteria={'min_volume': 1000000}
)
```

### 4. Custom Timeframes

Access any timeframe:
- Minutes: 1, 3, 5, 15, 30, 45
- Hours: 60, 120, 180, 240
- Days: D, 2D, 3D
- Weeks: W, 2W
- Months: M, 2M, 3M, 6M, 12M

### 5. Multiple Markets

```python
# Crypto
crypto_api = TradingViewAuthenticatedAPI()
btc = crypto_api.get_price('BTCUSDT', exchange='BINANCE')

# Stocks (change exchange)
aapl = crypto_api.get_price('AAPL', exchange='NASDAQ')

# Forex
eur_usd = crypto_api.get_price('EURUSD', exchange='FX_IDC')
```

## Integration with Moon Dev

### Option 1: Replace TradingView adapter

```python
# In your agents, use authenticated API
from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI

class YourAgent:
    def __init__(self):
        self.tv_api = TradingViewAuthenticatedAPI()
    
    def get_market_data(self):
        return self.tv_api.get_ohlcv('BTCUSDT')
```

### Option 2: Environment variable

Set in `.env`:
```env
USE_TRADINGVIEW_AUTH=true
```

Then the adapters can check and use authenticated version.

## Troubleshooting

### Server won't start
- Check Node.js is installed: `node --version`
- Ensure you're in the right directory
- Check port 8888 isn't already used

### Authentication fails
- Verify your TradingView username/password
- Try logging into TradingView website first
- Check for 2FA issues

### Connection refused
- Ensure server is running
- Check firewall isn't blocking port 8888
- Try `http://localhost:8888/health`

### Rate limits still occurring
- Authenticated API has limits too (just much higher)
- Add small delays between requests
- Use batch operations when possible

## Server Endpoints

The Node.js server provides these endpoints:

- `POST /login` - Authenticate with TradingView
- `POST /chart` - Get real-time chart data
- `POST /indicator` - Get indicator values
- `POST /batch` - Get multiple symbols at once
- `GET /search` - Search for symbols
- `GET /health` - Check server status
- `POST /logout` - Close session

## Production Deployment

For production use:

1. **Use PM2** for process management:
   ```bash
   npm install -g pm2
   pm2 start server.js --name tradingview-api
   ```

2. **Secure credentials** with environment variables or secrets manager

3. **Add error handling** and reconnection logic

4. **Monitor server health** and auto-restart on failure

5. **Use HTTPS** if exposing to network

## Summary

The authenticated TradingView API provides:

- ✅ **No rate limit issues** - Make hundreds of requests
- ✅ **Real-time data** - True streaming updates
- ✅ **Premium features** - Access all indicators
- ✅ **Better reliability** - No 429 errors
- ✅ **Batch operations** - Get multiple symbols at once

Perfect for production trading bots that need reliable market data!