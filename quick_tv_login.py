"""
🌙 Quick TradingView Login
Uses credentials from .env file automatically
"""

import requests
import os
import time

server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')

print("🌙 Quick TradingView Login")
print("=" * 60)

# Check current status
try:
    status = requests.get(f'{server_url}/login-status')
    if status.json().get('authenticated'):
        print("✅ Already authenticated!")
        exit(0)
except:
    pass

print("🔐 Attempting login with .env credentials...")

# Try login without sending credentials (server will use env vars)
try:
    response = requests.post(f'{server_url}/login', json={})
    
    if response.status_code == 200:
        print("✅ Login successful!")
        data = response.json()
        print(f"User: {data.get('user', 'Unknown')}")
        
        # Test authentication
        print("\n📊 Testing authentication...")
        test = requests.post(
            f'{server_url}/chart',
            json={'symbol': 'BTCUSDT', 'timeframe': '60', 'exchange': 'BINANCE'}
        )
        
        if test.status_code == 200:
            chart_data = test.json()
            print(f"✅ BTC Price: ${chart_data.get('close', 0):,.2f}")
        else:
            print("⚠️  Authentication test failed")
            
    else:
        data = response.json()
        error = data.get('error', 'Unknown error')
        
        if 'captcha' in error.lower():
            print("\n⚠️  Captcha required!")
            print("\nOptions:")
            print("1. Open http://localhost:8888 in your browser")
            print("2. The form will be prepopulated with your credentials")
            print("3. Click Login and complete the captcha")
            print("\nOr run: python open_tradingview_login.py")
        else:
            print(f"❌ Login failed: {error}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure:")
    print("1. Server is running (cd tradingview-server && npm start)")
    print("2. Credentials are in .env file:")
    print("   TV_USERNAME=your_email")
    print("   TV_PASSWORD=your_password")