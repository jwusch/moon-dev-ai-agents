# Robinhood Exchange Setup Guide

## Overview

This guide will help you set up and use Robinhood as an exchange option in the Moon Dev AI Trading System. Robinhood integration allows you to trade US stocks, options, and cryptocurrencies through the unofficial robin_stocks API.

⚠️ **IMPORTANT DISCLAIMER**: This integration uses an **unofficial API** that could break at any time. Use at your own risk. This is for educational purposes only.

## Prerequisites

1. **Robinhood Account**: You need an active Robinhood account
2. **Python Environment**: Ensure you have the conda environment activated
3. **2FA Setup**: Robinhood requires Two-Factor Authentication

## Installation

### Step 1: Install Required Libraries

```bash
conda activate tflow
pip install robin-stocks pyotp
pip freeze > requirements.txt
```

### Step 2: Environment Variables

Add the following to your `.env` file:

```bash
# Robinhood Authentication
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password

# Optional: TOTP Secret for automatic MFA (see MFA Setup below)
ROBINHOOD_MFA_TOKEN=your_totp_secret
```

⚠️ **Security Note**: Never commit your `.env` file! It should be in `.gitignore`.

### Step 3: Configure Exchange

In `src/config.py`, set the exchange to Robinhood:

```python
# 🔄 Exchange Selection
EXCHANGE = 'robinhood'  # Switch to Robinhood
```

## Multi-Factor Authentication (MFA) Setup

### Option 1: Manual SMS Code (Default)

If you don't provide `ROBINHOOD_MFA_TOKEN`, the system will prompt for SMS code during login.

### Option 2: Automatic TOTP Authentication

To enable automatic login with TOTP:

1. **Get your TOTP Secret** from Robinhood:
   - This is the secret key shown when you first set up authenticator app
   - It looks like: `ABCD1234EFGH5678IJKL9012MNOP3456`

2. **Add to .env**:
   ```bash
   ROBINHOOD_MFA_TOKEN=ABCD1234EFGH5678IJKL9012MNOP3456
   ```

## Usage Examples

### Basic Trading

```python
from src.exchange_manager import ExchangeManager

# Initialize for Robinhood
em = ExchangeManager('robinhood')

# Get current price
price = em.get_current_price('AAPL')
print(f"AAPL Price: ${price}")

# Buy $100 worth of Apple stock
order = em.market_buy('AAPL', 100)

# Check position
position = em.get_position('AAPL')
print(f"Position: {position}")

# Sell 50% of position
sell_order = em.market_sell('AAPL', 50)
```

### Working with Crypto

```python
# Buy $500 worth of Bitcoin
btc_order = em.market_buy('BTC', 500)

# Check crypto position
btc_position = em.get_position('BTC')
print(f"BTC Position: {btc_position}")
```

### Portfolio Management

```python
# Get account value
account_value = em.get_account_value()
print(f"Total Account Value: ${account_value:,.2f}")

# Get buying power
buying_power = em.get_balance()
print(f"Available to Trade: ${buying_power:,.2f}")

# Get all positions
positions = em.get_all_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} @ ${pos['current_price']}")
```

### Market Hours Check

```python
# Check if market is open
if em.is_market_open():
    print("Market is OPEN")
else:
    print("Market is CLOSED")

# Get market hours
hours = em.get_market_hours()
print(f"Market hours: {hours}")
```

## Configuring Trading Parameters

### Stock Trading Settings

In `src/config.py`:

```python
# 🏦 Robinhood Configuration
ROBINHOOD_SYMBOLS = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'NVDA']  # Your watchlist
ROBINHOOD_EXTENDED_HOURS = False  # Set True for extended hours trading

# 📊 Stock-Specific Configuration
STOCK_USD_SIZE = 100  # Default position size for stocks
MAX_STOCK_POSITIONS = 10  # Maximum number of positions
STOCK_STOP_LOSS_PERCENTAGE = -5  # Stop loss at -5%
STOCK_TAKE_PROFIT_PERCENTAGE = 10  # Take profit at 10%
```

### Crypto Trading Settings

```python
# Crypto available on Robinhood
ROBINHOOD_CRYPTO = ['BTC', 'ETH', 'DOGE', 'SOL', 'MATIC']
ROBINHOOD_CRYPTO_ENABLED = True  # Enable/disable crypto trading
```

### Options Trading Settings

```python
# 📈 Options Configuration
ROBINHOOD_OPTIONS_ENABLED = True  # Enable options trading
OPTIONS_MAX_CONTRACTS = 5  # Max contracts per trade
OPTIONS_DEFAULT_EXPIRY_DAYS = 30  # Default expiration
OPTIONS_MAX_RISK_PER_TRADE = 500  # Max risk per trade
```

## Running Agents with Robinhood

### Update Agent Configuration

Most agents will work with Robinhood automatically when you set `EXCHANGE = 'robinhood'`.

### Example: Trading Agent

```python
# The trading agent will use Robinhood automatically
python src/agents/trading_agent.py
```

### Example: Risk Agent

The risk agent monitors your Robinhood portfolio:

```python
python src/agents/risk_agent.py
```

## Important Differences from Crypto Exchanges

### 1. Market Hours
- US stocks trade Monday-Friday, 9:30 AM - 4:00 PM ET
- Extended hours: 4:00 AM - 8:00 PM ET (if enabled)
- Crypto trades 24/7 on Robinhood

### 2. Pattern Day Trading (PDT) Rule
- Accounts under $25,000 limited to 3 day trades per 5 trading days
- The system does NOT automatically track this - monitor manually

### 3. Settlement Times
- Stocks/ETFs: T+2 settlement (2 business days)
- Options: T+1 settlement
- Crypto: Instant settlement

### 4. Order Types
- Market orders execute immediately
- Limit orders supported for stocks (not crypto via API)
- Stop loss orders available for stocks

## Troubleshooting

### Authentication Issues

1. **"Authentication failed"**
   - Verify username/password in `.env`
   - Check if Robinhood requires CAPTCHA (login manually once)
   - Ensure 2FA is properly configured

2. **"MFA code invalid"**
   - For SMS: Enter code when prompted
   - For TOTP: Verify your secret key is correct
   - Check system time is synced

### Trading Issues

1. **"Market is closed"**
   - Check market hours
   - Enable extended hours if needed

2. **"Insufficient funds"**
   - Check buying power with `get_balance()`
   - Ensure no pending settlements

3. **"Pattern day trader restricted"**
   - You've exceeded 3 day trades in 5 days
   - Need $25,000+ or wait for restriction to clear

### API Limitations

1. **Rate Limiting**
   - The adapter includes 500ms delays between requests
   - If you get rate limited, increase delays

2. **Functionality Gaps**
   - Some features may not be available via unofficial API
   - Options trading has limited support

## Best Practices

### Security
1. **Use strong passwords** and enable 2FA
2. **Never share API credentials**
3. **Monitor account activity** regularly
4. **Use paper trading** first to test strategies

### Risk Management
1. **Start small** - Test with minimal capital
2. **Set stop losses** to limit downside
3. **Monitor PDT rules** if account < $25k
4. **Diversify positions** across symbols

### Development
1. **Handle errors gracefully** - API can fail
2. **Log all trades** for analysis
3. **Test during market hours** for stocks
4. **Use limit orders** for better fills

## Example Strategy Configuration

Here's a sample configuration for a conservative stock trading strategy:

```python
# In config.py
EXCHANGE = 'robinhood'

# Conservative stock portfolio
ROBINHOOD_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']  # Index ETFs

# Position sizing
STOCK_USD_SIZE = 500  # $500 per position
MAX_STOCK_POSITIONS = 5  # Max 5 positions

# Risk management
STOCK_STOP_LOSS_PERCENTAGE = -2  # Tight 2% stop loss
STOCK_TAKE_PROFIT_PERCENTAGE = 5  # 5% profit target

# Disable risky features
ROBINHOOD_OPTIONS_ENABLED = False  # No options
ROBINHOOD_EXTENDED_HOURS = False  # Regular hours only
```

## Monitoring and Logging

### View Positions
```bash
python -c "from src.exchange_manager import ExchangeManager; em = ExchangeManager('robinhood'); print(em.get_all_positions())"
```

### Check Account Status
```bash
python -c "from src.exchange_manager import ExchangeManager; em = ExchangeManager('robinhood'); print(em.get_account_info())"
```

### Export Trades
The system automatically logs trades. Check:
- `src/data/trading_agent/trades.csv`
- `src/data/risk_agent/positions.csv`

## Legal and Compliance

1. **Unofficial API**: Robin Stocks is NOT affiliated with Robinhood
2. **Terms of Service**: Ensure compliance with Robinhood ToS
3. **Tax Reporting**: You're responsible for tax reporting
4. **No Warranty**: This integration comes with NO guarantees

## Getting Help

1. **Robin Stocks Documentation**: https://robin-stocks.readthedocs.io/
2. **Moon Dev Discord**: Join our community for support
3. **GitHub Issues**: Report bugs in the repository

---

⚠️ **Final Warning**: Trading involves risk of loss. This integration uses an unofficial API that could stop working at any time. Always trade responsibly and never risk more than you can afford to lose.

Built with love by Moon Dev 🌙 - Happy Trading!