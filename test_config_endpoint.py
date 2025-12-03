"""
Test the /config endpoint
"""

import requests
import os

server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')

print("🔍 Testing TradingView Config Endpoint")
print("=" * 60)

# Test /config endpoint
try:
    response = requests.get(f'{server_url}/config')
    
    if response.status_code == 200:
        config = response.json()
        print("✅ Config endpoint working")
        print(f"\nConfiguration:")
        print(f"  Username configured: {config.get('hasUsername', False)}")
        print(f"  Username: {config.get('username', 'Not set')}")
        print(f"  Password configured: {config.get('passwordConfigured', False)}")
        
        if config.get('username'):
            print(f"\n📝 The login form will prepopulate with:")
            print(f"  Username: {config['username']}")
            if config.get('passwordConfigured'):
                print(f"  Password: Will use .env password automatically")
        else:
            print("\n⚠️  No credentials in .env file")
            print("Add to your .env:")
            print("  TV_USERNAME=your_email@example.com")
            print("  TV_PASSWORD=your_password")
            
    else:
        print(f"❌ Config endpoint returned: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure the server is running:")
    print("  cd tradingview-server")
    print("  npm start")