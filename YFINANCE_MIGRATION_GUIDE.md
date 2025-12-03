# 🚀 YFinance Migration Guide

## Quick Start

YFinance is now integrated as your primary data source! No more authentication issues with TradingView.

### Force YFinance Usage

Add to your `.env` file:
```bash
DATA_SOURCE=yfinance
```

Or set in your scripts:
```python
import os
os.environ['DATA_SOURCE'] = 'yfinance'
```

### What Works

✅ **Price Data**
- Current prices for stocks and crypto
- OHLCV historical data (1m, 5m, 15m, 1h, 1d)
- 24h statistics

✅ **Timeframe Limits**
- 1m, 5m: Max 7 days
- 15m, 30m: Max 60 days
- 1h: Max 730 days (2 years)
- 1d: Unlimited

✅ **Symbol Support**
- US Stocks: TSLA, AAPL, SPY, etc.
- Crypto: BTC-USD, ETH-USD (auto-converts BTCUSDT → BTC-USD)
- ETFs: SPY, QQQ, etc.

### What's Limited

⚠️ **Crypto-Specific Features**
- No funding rates (spot market only)
- No liquidation data
- No perpetual futures data

### Code Examples

#### Basic Usage
```python
from src.agents.api_adapter import APIAdapter

# YFinance will be used automatically
api = APIAdapter()

# Get price
price = api.api.get_price('TSLA')
print(f"TSLA: ${price}")
```

#### Get OHLCV Data
```python
from src.agents.yfinance_adapter import YFinanceAdapter

yf = YFinanceAdapter()

# Get 5-minute bars
data = yf.get_ohlcv_data('TSLA', '5m', 100)
print(f"Got {len(data)} bars")
```

#### Bulk Price Fetch
```python
# Get multiple prices efficiently
symbols = ['TSLA', 'AAPL', 'NVDA', 'BTCUSDT']
prices = yf.get_multiple_prices(symbols)
```

### For Your Trading Agents

Your existing agents will automatically use YFinance when:
1. No TradingView session tokens are available
2. `DATA_SOURCE=yfinance` is set
3. TradingView fails to connect

### Backtesting Benefits

YFinance is **perfect** for backtesting:
- No rate limits
- Reliable historical data
- Works with backtesting.py library
- Free and unlimited

### Migration Checklist

- [x] YFinance adapter created (`src/agents/yfinance_api.py`)
- [x] API adapter updated to prioritize YFinance
- [x] Symbol conversion for crypto (BTCUSDT → BTC-USD)
- [x] OHLCV data formatting for backtesting.py
- [x] Bulk download support for efficiency

### Need Help?

- Run `python test_yfinance_integration.py` to verify setup
- Run `python demo_yfinance_trading.py` for usage examples
- Check `explore_yfinance_capabilities.py` for full feature list

---

**Author**: Claude (Anthropic)  
**Created**: December 2025  
**Purpose**: Reliable, free market data for Moon Dev AI Agents