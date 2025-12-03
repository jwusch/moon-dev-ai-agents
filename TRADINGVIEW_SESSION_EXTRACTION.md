# 🍪 Extract TradingView Session from Browser

## Quick Steps (Chrome/Firefox)

1. **Login to TradingView**
   - Go to https://www.tradingview.com
   - Sign in with your credentials
   - Complete any captcha/2FA

2. **Open Developer Tools**
   - Press `F12` or right-click → "Inspect"
   - Go to the **Application** tab (Chrome) or **Storage** tab (Firefox)

3. **Find Cookies**
   - In left sidebar: Cookies → https://www.tradingview.com
   - Look for these cookies:
     - `sessionid` - Your session token
     - `sessionid_sign` - Your signature token

4. **Copy Values**
   - Click on each cookie
   - Copy the "Value" field (long string)

## Alternative Method: Network Tab

1. After logging in, go to Network tab in DevTools
2. Look for any TradingView API request
3. Check Request Headers for:
   ```
   Cookie: sessionid=YOUR_SESSION_ID; sessionid_sign=YOUR_SIGNATURE
   ```

## Update Your .env File

Add these lines to your `.env`:

```env
# TradingView Session (extracted from browser)
TV_SESSION_ID=your_sessionid_value_here
TV_SESSION_SIGNATURE=your_sessionid_sign_value_here
```

## Quick JavaScript Console Method

While on tradingview.com, open console (F12) and run:

```javascript
// Extract session cookies
const cookies = document.cookie.split('; ');
const sessionid = cookies.find(c => c.startsWith('sessionid='))?.split('=')[1];
const sessionid_sign = cookies.find(c => c.startsWith('sessionid_sign='))?.split('=')[1];

console.log('TV_SESSION_ID=' + sessionid);
console.log('TV_SESSION_SIGNATURE=' + sessionid_sign);
```

## Using Session Tokens in Code

Create a new file to use these tokens:

```javascript
// tradingview-server/server-with-session.js
const TradingView = require('../tradingview-api/main');

// Use session tokens directly
const tvClient = new TradingView.Client({
  token: process.env.TV_SESSION_ID,
  signature: process.env.TV_SESSION_SIGNATURE,
});

// Now you can use all authenticated features!
```

## Python Usage

```python
# src/agents/tradingview_session_api.py
import os
from typing import Dict, Any

class TradingViewSessionAPI:
    def __init__(self):
        self.session_id = os.getenv('TV_SESSION_ID')
        self.signature = os.getenv('TV_SESSION_SIGNATURE')
        
        if not self.session_id or not self.signature:
            raise ValueError("TV_SESSION_ID and TV_SESSION_SIGNATURE must be set in .env")
        
        # Initialize with existing session
        self.headers = {
            'Cookie': f'sessionid={self.session_id}; sessionid_sign={self.signature}'
        }
    
    def get_chart_data(self, symbol: str) -> Dict[str, Any]:
        # Use headers in your requests
        pass
```

## Important Notes

- 🕐 Sessions expire (usually after a few hours/days)
- 🔄 You'll need to re-extract when they expire
- 🔒 Keep these tokens as secure as passwords
- 📱 If you use 2FA, the session will last longer

## Test Your Session

After updating .env, test with:

```bash
curl -H "Cookie: sessionid=YOUR_SESSION; sessionid_sign=YOUR_SIGNATURE" \
  https://www.tradingview.com/api/v1/symbols_list
```

If it returns data, your session is valid!