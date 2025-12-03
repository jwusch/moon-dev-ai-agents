"""
🌙 One-Click TradingView Login
Uses credentials from .env file only
Opens browser if captcha is needed
"""

import requests
import webbrowser
import os
import time
from datetime import datetime

server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')

print("🌙 One-Click TradingView Login")
print("=" * 60)

# Check server
try:
    health = requests.get(f'{server_url}/health', timeout=2)
    if health.status_code != 200:
        print("❌ Server not responding")
        exit(1)
        
    health_data = health.json()
    print(f"✅ Server is running")
    
    # Check if already authenticated
    if health_data.get('authenticated'):
        print("✅ Already authenticated!")
        
        # Get user info
        status = requests.get(f'{server_url}/login-status').json()
        if status.get('user'):
            print(f"📧 Logged in as: {status['user']}")
        exit(0)
        
except Exception as e:
    print(f"❌ Server error: {e}")
    print("\nStart the server with:")
    print("  cd tradingview-server && npm start")
    exit(1)

# Check config
try:
    config = requests.get(f'{server_url}/config').json()
    
    if not config.get('username') or not config.get('passwordConfigured'):
        print("\n❌ Missing credentials in .env file!")
        print("Add these to your .env:")
        print("  TV_USERNAME=your_email@example.com")
        print("  TV_PASSWORD=your_password")
        exit(1)
        
    print(f"📧 Username: {config['username']}")
    print("🔐 Password: Configured in .env")
    
except Exception as e:
    print(f"❌ Config error: {e}")
    exit(1)

# Attempt login
print(f"\n🔐 Attempting login...")

try:
    # Send only username, server will use env password
    response = requests.post(
        f'{server_url}/login',
        json={'username': config['username']},
        timeout=10
    )
    
    if response.status_code == 200:
        print("✅ Login successful!")
        data = response.json()
        if data.get('user'):
            print(f"📧 Logged in as: {data['user']}")
        exit(0)
        
    # Check for captcha
    error_msg = response.json().get('error', '')
    if 'captcha' in error_msg.lower():
        print("\n⚠️  Captcha required!")
        print("Opening TradingView login in browser...")
        
        # Open browser
        webbrowser.open('https://www.tradingview.com/accounts/signin/')
        
        print("\n📝 Instructions:")
        print("1. Complete the captcha in your browser")
        print("2. This script will detect when login succeeds")
        print("\n⏳ Waiting for login (2 minute timeout)...")
        
        # Poll for success
        start_time = time.time()
        while time.time() - start_time < 120:
            try:
                status = requests.get(f'{server_url}/login-status').json()
                if status.get('authenticated'):
                    print("\n✅ Login detected!")
                    if status.get('user'):
                        print(f"📧 Logged in as: {status['user']}")
                    exit(0)
            except:
                pass
                
            time.sleep(2)
            print(".", end="", flush=True)
            
        print("\n\n⏱️ Timeout waiting for login")
        print("Complete the login manually in your browser")
        
    else:
        print(f"\n❌ Login failed: {error_msg}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    
print("\n💡 Tip: You can also use the web UI at:")
print(f"   {server_url}")