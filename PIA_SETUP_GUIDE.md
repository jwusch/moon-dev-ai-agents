# 🌙 PIA VPN Setup Guide for Moon Dev Trading Bots

This guide explains how to configure Private Internet Access (PIA) VPN to bypass Binance API geo-restrictions (451 errors).

## Option 1: PIA Dedicated Proxy Credentials (Recommended)

This is the most reliable method that doesn't require the PIA desktop app to be running.

### Step 1: Get PIA Proxy Credentials
1. Log in to your PIA account at https://www.privateinternetaccess.com
2. Go to the Control Panel
3. Find the "Proxy" section
4. Generate SOCKS5/HTTP proxy credentials (these are different from your main PIA login)
5. Note down the username and password

### Step 2: Configure .env File

Add to your `.env` file:

```bash
# PIA Proxy Credentials (from PIA Control Panel)
PIA_USERNAME=your_proxy_username
PIA_PASSWORD=your_proxy_password

# Optional: Choose proxy type and region
# For HTTP proxy (default, recommended):
PIA_HTTP_PROXY=http://proxy-nl.privateinternetaccess.com:8080

# For SOCKS5 proxy (requires pip install requests[socks]):
# PIA_USE_SOCKS5=true
# PIA_SOCKS5_HOST=proxy-nl.privateinternetaccess.com
# PIA_SOCKS5_PORT=1080
```

### Available PIA Proxy Servers:
- Netherlands: `proxy-nl.privateinternetaccess.com`
- UK: `proxy-uk.privateinternetaccess.com`
- Canada: `proxy-ca.privateinternetaccess.com`
- Japan: `proxy-jp.privateinternetaccess.com`
- Singapore: `proxy-sg.privateinternetaccess.com`
- Australia: `proxy-au.privateinternetaccess.com`

## Option 2: Local PIA Desktop App SOCKS5

If you have PIA desktop app with SOCKS5 enabled:

### Step 1: Enable SOCKS5 in PIA Desktop
1. Open PIA desktop application
2. Go to Settings → Proxy
3. Enable "SOCKS5 Proxy"
4. Set port to 1080
5. No authentication needed for localhost

### Step 2: Configure .env File

```bash
# Use local PIA SOCKS5 proxy
SOCKS5_PROXY=socks5://127.0.0.1:1080
```

## Option 3: Direct Proxy Configuration

For other proxy services or manual configuration:

```bash
# HTTP/HTTPS proxy
HTTP_PROXY=http://username:password@proxy-server:port
HTTPS_PROXY=http://username:password@proxy-server:port

# Or Binance-specific proxy
BINANCE_PROXY=http://username:password@proxy-server:port
```

## Testing Your Setup

Run the test script:
```bash
python test_binance_proxy.py
```

## Troubleshooting

### SOCKS5 Issues
If you get SOCKS5 errors, install the required package:
```bash
pip install requests[socks]
# or
pip install PySocks
```

### 451 Geo-restriction Errors
- Ensure proxy credentials are correct
- Try a different proxy region
- Verify PIA subscription is active

### Connection Timeouts
- Some proxy regions may be slower than others
- Try Netherlands or UK servers for best performance

### "No fake data" Errors
This is by design! The system will never generate synthetic data. You must have a working proxy to get real Binance data.

## Security Notes

1. **Never commit credentials**: Keep your `.env` file out of git
2. **Use dedicated proxy credentials**: Don't use your main PIA login
3. **Rotate credentials regularly**: Change proxy credentials periodically
4. **Monitor usage**: Check PIA dashboard for unusual activity

## How It Works

1. When you run any agent that needs Binance data:
   - The system checks for proxy configuration
   - Connects to Binance through the proxy
   - Fails immediately if no proxy is configured (no fake data)

2. The proxy is automatically used for:
   - Liquidation data (`get_liquidation_data()`)
   - Funding rates (`get_funding_data()`)
   - Open interest (`get_oi_data()`)
   - Any other Binance API calls

## Example Usage

```python
from src.agents.api_adapter import APIAdapter

# This automatically uses proxy if configured
api = APIAdapter()

# Get real liquidation data through proxy
liq_data = api.get_liquidation_data(limit=100)

# Get funding rates
funding = api.get_funding_data()

# Get open interest
oi = api.get_oi_data()
```

All calls will use the proxy automatically - no code changes needed!