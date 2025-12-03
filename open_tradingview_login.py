"""
🌙 Open TradingView Login UI
Opens the manual login interface in your default browser
"""

import webbrowser
import time
import requests
import os

print("🌙 TradingView Manual Login Helper")
print("=" * 60)

server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')

# Check if server is running
try:
    response = requests.get(f'{server_url}/health', timeout=2)
    if response.status_code == 200:
        print(f"✅ Server is running at {server_url}")
        
        # Open browser
        print(f"\n📱 Opening login UI in browser...")
        webbrowser.open(server_url)
        
        print("\n📝 Instructions:")
        print("1. Enter your TradingView email and password")
        print("2. Click 'Login to TradingView'")
        print("3. If you see a captcha, complete it in the iframe")
        print("4. The system will detect successful login")
        
        # Monitor login status
        print("\n⏳ Monitoring login status...")
        authenticated = False
        attempts = 0
        max_attempts = 60  # 2 minutes
        
        while not authenticated and attempts < max_attempts:
            try:
                status = requests.get(f'{server_url}/login-status')
                data = status.json()
                
                if data.get('authenticated'):
                    print("\n✅ Login successful!")
                    print(f"User: {data.get('user', 'Unknown')}")
                    authenticated = True
                    break
                    
            except:
                pass
            
            time.sleep(2)
            attempts += 1
            
            if attempts % 10 == 0:
                print(f"⏳ Still waiting... ({attempts * 2} seconds)")
        
        if not authenticated:
            print("\n⏱️ Login timeout. Please check the browser window.")
            
    else:
        print(f"❌ Server returned status {response.status_code}")
        
except requests.ConnectionError:
    print(f"❌ Server is not running at {server_url}")
    print("\n📝 To start the server:")
    print("1. cd tradingview-server")
    print("2. npm start")
    print("\nThen run this script again.")
    
except Exception as e:
    print(f"❌ Error: {e}")