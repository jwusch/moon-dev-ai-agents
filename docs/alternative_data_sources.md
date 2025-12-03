# 🌙 Alternative Data Sources Guide

Since you don't have access to the Moon Dev API, here's a comprehensive guide to free alternatives:

## Quick Start

Replace `MoonDevAPI` imports in your agents:

```python
# Old way:
from src.agents.api import MoonDevAPI
api = MoonDevAPI()

# New way (automatic fallback):
from src.agents.api_adapter import APIAdapter
api = APIAdapter()
```

## Free Data Sources

### 1. Liquidation Data

#### Binance (Free, No API Key)
```python
import requests

def get_binance_liquidations(symbol='BTCUSDT'):
    url = f"https://fapi.binance.com/fapi/v1/allForceOrders"
    params = {'symbol': symbol, 'limit': 1000}
    response = requests.get(url, params=params)
    return response.json()
```

#### CoinGlass (Free Tier)
- Website: https://coinglass.com/
- API Docs: https://coinglass.github.io/API-Reference/
- Free tier: 30 requests/minute

#### Alternative: WebSocket for Real-time
```python
# Binance WebSocket for real-time liquidations
ws_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
```

### 2. Funding Rates

#### Multiple Exchanges (All Free)
```python
# Binance
binance_funding = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Bybit
bybit_funding = "https://api.bybit.com/v5/market/funding/history"

# OKX
okx_funding = "https://www.okx.com/api/v5/public/funding-rate"
```

### 3. Open Interest

#### Using CoinGecko (You already have this key!)
```python
headers = {'x-cg-pro-api-key': 'YOUR_COINGECKO_KEY'}
url = "https://api.coingecko.com/api/v3/derivatives/tickers"
```

#### DeFiLlama (Completely Free)
```python
# No API key needed!
defi_derivatives = "https://api.llama.fi/overview/derivatives"
```

### 4. Whale Tracking

#### Free Options:
1. **WhaleAlert.io** - 10 requests/min free
2. **Etherscan API** - 5 calls/second free
3. **BSCScan API** - Similar to Etherscan

### 5. Additional Free Resources

#### CryptoQuant (Limited Free)
- On-chain metrics
- Exchange flows
- Website: https://cryptoquant.com/

#### Santiment (Free Tier)
- Social metrics
- Development activity
- Website: https://santiment.net/

## Implementation Examples

### Modified Liquidation Agent
```python
# In liquidation_agent.py, replace:
self.api = MoonDevAPI()

# With:
from src.agents.api_adapter import APIAdapter
self.api = APIAdapter()
```

### Modified Funding Agent
```python
# Same replacement - the adapter handles the differences
self.api = APIAdapter()
```

## Rate Limits & Best Practices

1. **Binance**: 
   - 2400 requests/minute
   - 100 requests/10 seconds

2. **CoinGecko**:
   - Pro: 500 calls/minute
   - Use your existing key!

3. **DeFiLlama**:
   - No explicit limits
   - Be respectful

## Advanced: Building Your Own Data Pipeline

```python
import asyncio
import aiohttp

class DataAggregator:
    """Aggregate data from multiple free sources"""
    
    async def fetch_all_liquidations(self):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_binance_liqs(session),
                self.fetch_bybit_liqs(session),
                self.fetch_okx_liqs(session)
            ]
            results = await asyncio.gather(*tasks)
            return self.merge_results(results)
```

## Webhooks & Alerts

Set up your own alerts using:
- **IFTTT** - Free webhooks
- **Zapier** - Limited free tier
- **Discord Webhooks** - Completely free

## Summary

You don't need the Moon Dev API! The combination of:
- Binance public endpoints (no key needed)
- Your existing CoinGecko API key
- DeFiLlama (completely free)
- Exchange WebSockets

...provides all the data you need for:
✅ Liquidation monitoring
✅ Funding rate arbitrage
✅ Open interest tracking
✅ Basic whale monitoring

The `APIAdapter` class makes switching seamless - your agents will work with or without the Moon Dev API!