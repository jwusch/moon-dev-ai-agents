"""
Test TradingView server endpoints on port 8888
"""

import requests
import json

base_url = 'http://localhost:8888'

print("🔍 Testing TradingView Server Endpoints on port 8888")
print("=" * 60)

# Test various endpoints that the TradingView server should have
endpoints = [
    '/',
    '/health',
    '/api',
    '/login',
    '/chart',
    '/history',
    '/indicator',
    '/batch'
]

for endpoint in endpoints:
    try:
        url = f'{base_url}{endpoint}'
        print(f"\n📡 Testing {endpoint}...")
        
        # GET request first
        response = requests.get(url, timeout=2)
        print(f"   GET: Status {response.status_code}")
        
        # Try POST for some endpoints
        if endpoint in ['/login', '/chart', '/history']:
            response = requests.post(url, json={}, timeout=2)
            print(f"   POST: Status {response.status_code}")
            if response.status_code < 500:
                try:
                    data = response.json()
                    print(f"   Response: {data}")
                except:
                    print(f"   Response: {response.text[:100]}")
                    
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")

# Check if it's actually the TradingView server by looking for specific responses
print("\n\n📊 Checking server type...")

# The TradingView server should respond with specific error messages
try:
    # Try an authenticated endpoint without auth
    response = requests.post(f'{base_url}/chart', json={'symbol': 'TEST'})
    
    if 'Not authenticated' in response.text or response.status_code == 401:
        print("✅ This appears to be the TradingView server (requires auth)")
    elif 'error' in response.text.lower():
        print(f"🤔 Possible TradingView server. Response: {response.text}")
    else:
        print(f"❌ This doesn't look like the TradingView server")
        print(f"   Response: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ Server check failed: {e}")