"""
🔄 Switch TradingView from Browser to API Session
Helps resolve duplicate session conflicts

Author: Claude (Anthropic)
"""

import requests
import time
import subprocess
import os

def check_api_server_status():
    """Check if the session server is running"""
    try:
        response = requests.get("http://localhost:8891/session", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("📊 API Server Status:")
            print(f"  Port 8891: ✅ Running")
            print(f"  Session ID: {'✅' if data.get('hasSessionId') else '❌'}")
            print(f"  Signature: {'✅' if data.get('hasSignature') else '❌'}")
            print(f"  Client Active: {'✅' if data.get('clientActive') else '❌'}")
            return data.get('clientActive', False)
        else:
            print("❌ API server responded but not ready")
            return False
    except:
        print("❌ API server not running on port 8891")
        return False

def test_api_data_access():
    """Test if API can get data"""
    try:
        payload = {"symbol": "AAPL", "timeframe": "D", "exchange": "NASDAQ"}
        response = requests.post(
            "http://localhost:8891/chart",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'close' in data:
                print(f"✅ API Test: AAPL = ${data['close']:.2f}")
                return True
            else:
                print("❌ API Test: No data returned (likely session conflict)")
                return False
        else:
            print(f"❌ API Test: Error {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return False

def restart_api_server():
    """Restart the API server"""
    print("\n🔄 Restarting API server...")
    
    # Kill existing servers
    try:
        subprocess.run(["pkill", "-f", "server-session"], capture_output=True)
        print("  Killed existing session servers")
    except:
        pass
    
    time.sleep(2)
    
    # Start new server
    print("  Starting fresh session server on port 8891...")
    try:
        # Start in background
        env = os.environ.copy()
        env['TV_SESSION_PORT'] = '8891'
        
        process = subprocess.Popen([
            "node", "server-session.js"
        ], 
        cwd="/mnt/c/Users/jwusc/moon-dev-ai-agents/tradingview-server",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )
        
        # Give it time to start
        time.sleep(5)
        
        if process.poll() is None:
            print("  ✅ Server started successfully")
            return True
        else:
            print("  ❌ Server failed to start")
            return False
    except Exception as e:
        print(f"  ❌ Error starting server: {e}")
        return False

def main():
    print("🔄 TradingView Session Conflict Resolver")
    print("=" * 50)
    
    print("\n1️⃣ Checking current API server status...")
    api_working = check_api_server_status()
    
    if api_working:
        print("\n2️⃣ Testing data access...")
        data_working = test_api_data_access()
        
        if data_working:
            print("\n✅ Everything is working! No action needed.")
            return
        else:
            print("\n⚠️ Session conflict detected!")
    
    print("\n🔧 Fixing session conflict:")
    print("\nSTEP 1: Please close all TradingView browser tabs")
    print("  - Go to any open tradingview.com tabs")
    print("  - Close them or logout from TradingView")
    print("  - This releases the browser session")
    
    input("\nPress Enter after closing browser tabs...")
    
    print("\nSTEP 2: Restarting API server...")
    if restart_api_server():
        print("\nSTEP 3: Testing API access...")
        time.sleep(3)
        
        if check_api_server_status():
            if test_api_data_access():
                print("\n✅ SUCCESS! Session conflict resolved.")
                print("\nYou can now use the API:")
                print("  python get_tsla_data.py")
            else:
                print("\n❌ Still having issues. Try these steps:")
                print("1. Make sure you're completely logged out of TradingView")
                print("2. Clear browser cookies for tradingview.com")
                print("3. Get fresh session tokens")
        else:
            print("\n❌ API server not responding correctly")
    else:
        print("\n❌ Could not restart API server")

if __name__ == "__main__":
    main()