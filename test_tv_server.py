"""
Test TradingView server connection
"""

import requests
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing TradingView Server Connection")
print("=" * 60)

# Test different ports
ports = [8888, 5000, 3000, 8080]

for port in ports:
    try:
        url = f'http://localhost:{port}'
        print(f"\n📡 Testing {url}...")
        response = requests.get(url, timeout=2)
        print(f"✅ Port {port}: Status {response.status_code}")
        print(f"   Response: {response.text[:100]}...")
    except requests.ConnectionError:
        print(f"❌ Port {port}: Connection refused")
    except requests.Timeout:
        print(f"❌ Port {port}: Timeout")
    except Exception as e:
        print(f"❌ Port {port}: {type(e).__name__}: {e}")

# Test with environment variable
print("\n📡 Testing with environment variable...")
tv_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')
print(f"TV_SERVER_URL: {tv_url}")

try:
    response = requests.get(tv_url, timeout=5)
    print(f"✅ Status: {response.status_code}")
    
    # Try to get more info
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"✅ JSON Response: {data}")
        except:
            print(f"📄 Text Response: {response.text[:200]}")
            
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

# Test authentication endpoint
print("\n📡 Testing authentication...")
try:
    # Get credentials
    username = os.getenv('TV_USERNAME') or os.getenv('TRADINGVIEW_USERNAME')
    password = os.getenv('TV_PASSWORD') or os.getenv('TRADINGVIEW_PASSWORD')
    
    print(f"Username: {username[:3]}...{username[-3:] if username else 'Not set'}")
    print(f"Password: {'*' * 8 if password else 'Not set'}")
    
    if username and password:
        from src.agents.tradingview_auth_client import TradingViewAuthClient
        
        client = TradingViewAuthClient(auto_login=False)
        print(f"Client server URL: {client.server_url}")
        
        # Test health check
        try:
            health = client.check_health()
            print(f"✅ Health check: {health}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            
except Exception as e:
    print(f"❌ Auth test error: {e}")
    import traceback
    traceback.print_exc()