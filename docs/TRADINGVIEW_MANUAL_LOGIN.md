# 🌙 TradingView Manual Login with Captcha Support

## Overview

When TradingView detects automated login attempts, it may require captcha verification. This manual login UI allows you to complete the captcha and authenticate successfully.

## Setup

### 1. Update Environment Variables

Add these to your `.env` file:
```env
TV_USERNAME=your_email@example.com
TV_PASSWORD=your_password
TV_SERVER_URL=http://localhost:8888
TV_SERVER_PORT=8888
```

### 2. Start the Server

```bash
cd tradingview-server
npm start
```

### 3. Access the Login UI

Open your browser to: http://localhost:8888

## Using the Login UI

### One-Click Login
1. Credentials are loaded from .env file automatically
2. Click "🔐 Login to TradingView" button
3. If successful, you'll see a success message
4. No need to enter password - it's always used from .env

### Manual Login (with Captcha)
1. Enter your credentials
2. Click "Login to TradingView"
3. If a captcha is required:
   - A new window/tab will open with TradingView login page
   - Complete the captcha in the new window
   - The system will automatically detect successful login
   - The window will close automatically once login is detected

**Note**: Due to security restrictions (X-Frame-Options), TradingView login cannot be embedded in an iframe and must open in a new window.

## API Endpoints

### Web UI
- `GET /` - Login interface

### Authentication
- `POST /login` - Standard login
- `POST /manual-login` - Trigger manual login with iframe
- `GET /login-status` - Check authentication status

### Data Endpoints (require auth)
- `POST /chart` - Get real-time data
- `POST /history` - Get historical OHLCV data
- `POST /indicator` - Get indicator values
- `POST /batch` - Get multiple symbols

## Troubleshooting

### "Captcha Required" Error
1. Use the web UI at http://localhost:8888
2. Complete the captcha in the iframe
3. Or login to TradingView in your browser first

### Server Not Found
- Check port 8888 is free: `lsof -i :8888`
- Verify server is running: `npm start`
- Check firewall settings

### Authentication Failed
- Verify credentials in `.env`
- Try logging in to TradingView website directly
- Check for 2FA requirements

## Using with Python

After successful login via the UI:

```python
from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI

# Create client (will use existing session)
api = TradingViewAuthenticatedAPI()

# Get historical data
df = api.get_historical_data(
    symbol='TSLA',
    timeframe='1D',
    bars=100,
    exchange='NASDAQ'
)

print(f"Got {len(df)} days of TSLA data")
```

## Security Notes

- Credentials are stored in environment variables only
- Session tokens are maintained by the Node.js server
- The UI uses localStorage for username only (not password)
- Always use HTTPS in production

## Alternative: Headless Mode

If you can't use the UI, try starting the server in development mode:

```bash
NODE_ENV=development npm start
```

This may allow manual login via console prompts.

---

Built with 🌙 by Moon Dev