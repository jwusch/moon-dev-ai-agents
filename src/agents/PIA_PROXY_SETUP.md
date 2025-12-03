# Using Private Internet Access (PIA) VPN with Moon Dev Trading Bots

## Overview
The Public Data API now supports routing requests through PIA VPN to bypass geographic restrictions.

## Setup Methods

### Method 1: Environment Variables (Recommended)

Add to your `.env` file:
```bash
# PIA SOCKS5 Proxy Configuration
PIA_USERNAME=your_pia_username
PIA_PASSWORD=your_pia_password
PIA_PROXY_HOST=proxy-nl.privateinternetaccess.com  # Netherlands proxy
PIA_PROXY_PORT=1080

# Alternative regional proxies:
# proxy-uk.privateinternetaccess.com (UK)
# proxy-japan.privateinternetaccess.com (Japan)
# proxy-singapore.privateinternetaccess.com (Singapore)
```

### Method 2: System Proxy Variables

```bash
# For SOCKS5 proxy
export SOCKS5_PROXY=socks5://username:password@proxy-nl.privateinternetaccess.com:1080

# For HTTP proxy (if PIA provides HTTP proxy in your region)
export HTTP_PROXY=http://username:password@proxy.privateinternetaccess.com:8080
export HTTPS_PROXY=http://username:password@proxy.privateinternetaccess.com:8080
```

### Method 3: PIA Desktop App + Local SOCKS5

If using PIA desktop app with local SOCKS5 proxy enabled:
```bash
export SOCKS5_PROXY=socks5://127.0.0.1:1080
```

## Installing Required Dependencies

For SOCKS5 support, install:
```bash
pip install requests[socks]
# or
pip install PySocks
```

## Testing Your Setup

```python
from src.agents.public_data_api import PublicDataAPI

# Initialize with proxy
api = PublicDataAPI()

# Test liquidation data (should now work from restricted regions)
liq_data = api.get_liquidation_data()
```

## Troubleshooting

1. **Connection Errors**: Ensure PIA is connected and SOCKS5 proxy is enabled in PIA settings
2. **Authentication Errors**: Check username/password (use PIA SOCKS5 credentials, not main account)
3. **Slow Performance**: Try different regional proxies for better speed

## Security Notes

- Never commit credentials to git
- Use environment variables or `.env` file
- PIA SOCKS5 credentials are different from your main PIA login

## Available PIA Proxy Servers

Common servers that support SOCKS5:
- `proxy-nl.privateinternetaccess.com:1080` (Netherlands)
- `proxy-uk.privateinternetaccess.com:1080` (UK)
- `proxy-ca.privateinternetaccess.com:1080` (Canada)
- `proxy-jp.privateinternetaccess.com:1080` (Japan)
- `proxy-sg.privateinternetaccess.com:1080` (Singapore)
- `proxy-au.privateinternetaccess.com:1080` (Australia)