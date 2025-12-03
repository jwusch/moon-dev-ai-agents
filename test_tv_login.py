"""
Test TradingView login with credentials from .env
"""

import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv('tradingview-server/.env')

# Get credentials
username = os.getenv('TV_USERNAME')
password = os.getenv('TV_PASSWORD')

print("🌙 Testing TradingView Login")
print("="*50)
print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'Not set'}")

# Server URL
server_url = "http://localhost:8888"

# Check if server is running
try:
    health = requests.get(f"{server_url}/health", timeout=2)
    print(f"\n✅ Server is running: {health.json()}")
except:
    print(f"\n❌ Server not responding at {server_url}")
    print("Please ensure the server is running:")
    print("  cd tradingview-server")
    print("  npm start")
    exit(1)

# Test login
print("\n🔐 Testing login...")
try:
    response = requests.post(f"{server_url}/login", json={
        'username': username,
        'password': password
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful!")
        print(f"User: {data.get('user', 'Unknown')}")
        print(f"Message: {data.get('message', '')}")
        
        # Test getting chart data
        print("\n📊 Testing chart data fetch...")
        chart_response = requests.post(f"{server_url}/chart", json={
            'symbol': 'BTCUSDT',
            'timeframe': '60',
            'exchange': 'BINANCE'
        })
        
        if chart_response.status_code == 200:
            chart_data = chart_response.json()
            print(f"✅ Got BTC data!")
            print(f"Price: ${chart_data.get('close', 0):,.2f}")
            print(f"Volume: {chart_data.get('volume', 0):,.0f}")
        else:
            print(f"❌ Chart fetch failed: {chart_response.status_code}")
            print(chart_response.text)
            
    elif response.status_code == 401:
        print(f"❌ Login failed - Invalid credentials")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Login failed with status: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error during login: {e}")

print("\n📝 If login failed:")
print("1. Check your TradingView username/password")
print("2. Try logging into tradingview.com to verify")
print("3. Check if you have 2FA enabled")
print("4. Make sure the credentials in .env are correct")