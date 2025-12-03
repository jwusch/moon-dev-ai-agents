# 🔄 Fix TradingView Duplicate Session Issue

## Problem
TradingView detected multiple connections using the same session ID:
- Browser session (where you extracted tokens)
- API server session (using the same tokens)

## Solution Steps

### 1. **Close Browser TradingView Session**
- Go to https://www.tradingview.com in your browser
- **Logout** from TradingView completely
- Or close all TradingView browser tabs

### 2. **Clear Browser TradingView Cookies** (Optional)
- Press F12 in browser
- Go to Application → Storage → Cookies → tradingview.com
- Delete `sessionid` and `sessionid_sign` cookies
- This forces a clean logout

### 3. **Restart API Server**
```bash
cd tradingview-server
# Kill any existing session servers
pkill -f "server-session"
# Start fresh
TV_SESSION_PORT=8891 node server-session.js
```

### 4. **Test TSLA Data Retrieval**
```bash
python get_tsla_data.py
```

## Alternative: Get Fresh Session

If the above doesn't work:

### 1. **Login Fresh in Browser**
- Go to https://www.tradingview.com
- Login with your credentials
- **Keep this tab open**

### 2. **Extract New Tokens**
```javascript
(function() {
    const c = document.cookie.split('; ');
    const sid = c.find(x => x.startsWith('sessionid='))?.split('=')[1];
    const sig = c.find(x => x.startsWith('sessionid_sign='))?.split('=')[1];
    console.log(`TV_SESSION_ID=${sid}`);
    console.log(`TV_SESSION_SIGNATURE=${sig}`);
})();
```

### 3. **Update .env Immediately**
Replace the tokens in .env with fresh ones

### 4. **Restart Server Within 30 seconds**
```bash
cd tradingview-server
TV_SESSION_PORT=8891 node server-session.js
```

### 5. **Close Browser Tab**
Once server starts successfully, close the TradingView browser tab

## Why This Happens

TradingView's security system prevents:
- Session hijacking
- Multiple concurrent connections
- API abuse

When it detects the same session from multiple sources, it blocks data access.

## Best Practice

For API access:
1. Extract tokens from browser
2. Immediately start API server
3. Close browser session
4. Use API server exclusively

Don't mix browser + API usage with same session tokens.