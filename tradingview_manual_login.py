"""
🌙 TradingView Manual Login Helper
Handles login with captcha support
"""

import requests
import os
import time
import webbrowser
from datetime import datetime

print("🌙 TradingView Manual Login Helper")
print("=" * 60)

# Configuration
server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')
username = os.getenv('TV_USERNAME', '')
password = os.getenv('TV_PASSWORD', '')

print(f"Server: {server_url}")

# Check server health
try:
    health = requests.get(f'{server_url}/health', timeout=2)
    if health.status_code == 200:
        data = health.json()
        print(f"✅ Server is online")
        print(f"   Authenticated: {data.get('authenticated', False)}")
        
        if data.get('authenticated'):
            print("\n✅ Already authenticated!")
            exit(0)
    else:
        print(f"❌ Server returned status {health.status_code}")
        exit(1)
        
except requests.ConnectionError:
    print(f"❌ Cannot connect to server at {server_url}")
    print("\nMake sure the server is running:")
    print("cd tradingview-server && npm start")
    exit(1)

# Get credentials if not in environment
if not username:
    username = input("\nTradingView username (email): ")
if not password:
    import getpass
    password = getpass.getpass("TradingView password: ")

print(f"\n🔐 Attempting login for {username}...")

# Try standard login first
try:
    response = requests.post(
        f'{server_url}/login',
        json={'username': username, 'password': password},
        timeout=10
    )
    
    data = response.json()
    
    if response.status_code == 200:
        print("✅ Login successful!")
        print(f"   User: {data.get('user', 'Unknown')}")
        exit(0)
        
    elif 'captcha' in data.get('error', '').lower():
        print("\n⚠️  Captcha detected!")
        print("TradingView requires manual verification.\n")
        
        print("Options:")
        print("1. Open the standalone login page: tradingview-login.html")
        print("2. Complete captcha at: https://www.tradingview.com/accounts/signin/")
        print("3. Wait for the server to detect successful login\n")
        
        # Open the login page
        choice = input("Open TradingView login page in browser? (y/n): ")
        if choice.lower() == 'y':
            webbrowser.open('https://www.tradingview.com/accounts/signin/')
            
        # Monitor login status
        print("\n⏳ Monitoring login status (2 minutes timeout)...")
        start_time = time.time()
        authenticated = False
        
        while time.time() - start_time < 120:  # 2 minute timeout
            try:
                status = requests.get(f'{server_url}/login-status')
                if status.json().get('authenticated'):
                    print("\n✅ Login detected!")
                    authenticated = True
                    break
            except:
                pass
                
            time.sleep(2)
            print(".", end="", flush=True)
            
        if not authenticated:
            print("\n\n⏱️ Timeout. Please complete login manually and try again.")
            
    else:
        print(f"❌ Login failed: {data.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n📝 Tips:")
print("- If you keep getting captcha, try logging in via browser first")
print("- Make sure 2FA is disabled or handled")
print("- Check server console for detailed error messages")