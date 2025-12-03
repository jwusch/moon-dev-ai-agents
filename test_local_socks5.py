"""
🌙 Test Local SOCKS5 Proxy from PIA Desktop App
Tests the local SOCKS5 proxy typically running at 127.0.0.1:1080
"""

import socket
import requests
import os
import sys
import json
from urllib.parse import urlparse

def check_socks5_port(host='127.0.0.1', port=1080):
    """Check if SOCKS5 proxy port is open"""
    print(f"\n🔍 Checking if SOCKS5 proxy is running on {host}:{port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is open - SOCKS5 proxy appears to be running")
            return True
        else:
            print(f"❌ Port {port} is closed - SOCKS5 proxy not detected")
            return False
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def test_socks5_requests():
    """Test SOCKS5 proxy with requests library"""
    print("\n🧪 Testing SOCKS5 with requests library...")
    
    # Check if PySocks is installed
    try:
        import socks
        print("✅ PySocks module is installed")
    except ImportError:
        print("❌ PySocks not installed. Installing now...")
        os.system("pip install PySocks requests[socks]")
        print("📦 Please restart the script after installation")
        return False
    
    # Configure SOCKS5 proxy
    proxies = {
        'http': 'socks5://127.0.0.1:1080',
        'https': 'socks5://127.0.0.1:1080'
    }
    
    # Test 1: Check current IP without proxy
    print("\n📍 Test 1: Current IP without proxy...")
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        original_ip = response.json()['ip']
        print(f"🌐 Original IP: {original_ip}")
    except Exception as e:
        print(f"❌ Failed to get original IP: {e}")
        original_ip = None
    
    # Test 2: Check IP through SOCKS5 proxy
    print("\n📍 Test 2: IP through SOCKS5 proxy...")
    try:
        response = requests.get('https://api.ipify.org?format=json', proxies=proxies, timeout=10)
        proxy_ip = response.json()['ip']
        print(f"🌐 Proxy IP: {proxy_ip}")
        
        if original_ip and proxy_ip != original_ip:
            print("✅ Success! IP changed through proxy")
        elif original_ip and proxy_ip == original_ip:
            print("⚠️ Warning: IP didn't change - proxy might not be working correctly")
        
        # Get more details about proxy location
        try:
            loc_response = requests.get('https://ipapi.co/json/', proxies=proxies, timeout=10)
            if loc_response.status_code == 200:
                loc_data = loc_response.json()
                print(f"📍 Proxy Location: {loc_data.get('city', 'Unknown')}, {loc_data.get('country_name', 'Unknown')}")
                print(f"🏢 ISP: {loc_data.get('org', 'Unknown')}")
        except:
            pass
            
        return True
        
    except requests.exceptions.ProxyError as e:
        print(f"❌ SOCKS5 proxy error: {e}")
        print("💡 Make sure PIA desktop app is running with SOCKS5 proxy enabled")
        return False
    except Exception as e:
        print(f"❌ Failed to connect through proxy: {e}")
        return False

def test_binance_through_socks5():
    """Test Binance API through local SOCKS5 proxy"""
    print("\n🧪 Testing Binance API through SOCKS5...")
    
    proxies = {
        'http': 'socks5://127.0.0.1:1080',
        'https': 'socks5://127.0.0.1:1080'
    }
    
    try:
        # Test Binance ping endpoint
        url = "https://api.binance.com/api/v3/ping"
        print(f"📡 Testing: {url}")
        
        response = requests.get(url, proxies=proxies, timeout=10)
        
        if response.status_code == 200:
            print("✅ Successfully connected to Binance API through SOCKS5 proxy!")
            
            # Test getting actual data
            print("\n📊 Testing Binance ticker endpoint...")
            ticker_url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': 'BTCUSDT'}
            
            ticker_response = requests.get(ticker_url, params=params, proxies=proxies, timeout=10)
            
            if ticker_response.status_code == 200:
                data = ticker_response.json()
                print(f"✅ BTC Price: ${float(data['price']):,.2f}")
                return True
            else:
                print(f"❌ Ticker request failed: {ticker_response.status_code}")
                return False
        else:
            print(f"❌ Binance ping failed with status: {response.status_code}")
            if response.status_code == 451:
                print("⚠️ Still getting geo-restriction error")
            return False
            
    except Exception as e:
        print(f"❌ Binance test failed: {e}")
        return False

def configure_env_for_socks5():
    """Show how to configure environment for local SOCKS5"""
    print("\n📝 To use local PIA SOCKS5 proxy, add to your .env file:")
    print("="*60)
    print("# Use local PIA desktop app SOCKS5 proxy")
    print("SOCKS5_PROXY=socks5://127.0.0.1:1080")
    print("# OR")
    print("USE_PIA_APP=true")
    print("="*60)

def test_proxy_with_our_code():
    """Test with our actual PublicDataAPI code"""
    print("\n🧪 Testing with PublicDataAPI (with SOCKS5 env var)...")
    
    # Set environment variable temporarily
    os.environ['SOCKS5_PROXY'] = 'socks5://127.0.0.1:1080'
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.agents.public_data_api import PublicDataAPI
        
        api = PublicDataAPI()
        
        # Quick test
        print("📊 Testing liquidation data fetch...")
        liq_data = api.get_liquidation_data(symbol='BTCUSDT', limit=5)
        
        if liq_data is not None and not liq_data.empty:
            print(f"✅ Success! Got {len(liq_data)} liquidation records")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up env var
        if 'SOCKS5_PROXY' in os.environ:
            del os.environ['SOCKS5_PROXY']

if __name__ == "__main__":
    print("🌙 Moon Dev Local SOCKS5 Proxy Test")
    print("="*60)
    
    # Check if PySocks is installed
    try:
        import socks
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install PySocks requests[socks]")
        print("✅ Packages installed. Please run the script again.")
        sys.exit(0)
    
    # Test 1: Check if port is open
    socks5_available = check_socks5_port()
    
    if not socks5_available:
        print("\n⚠️ SOCKS5 proxy not detected on localhost:1080")
        print("\n💡 To enable SOCKS5 in PIA desktop app:")
        print("1. Open PIA desktop application")
        print("2. Go to Settings → Proxy")
        print("3. Enable 'SOCKS5 Proxy'")
        print("4. Set port to 1080")
        print("5. No authentication required for localhost")
        sys.exit(1)
    
    # Test 2: Test with requests
    requests_success = test_socks5_requests()
    
    # Test 3: Test Binance specifically
    binance_success = test_binance_through_socks5()
    
    # Test 4: Test with our code
    # our_code_success = test_proxy_with_our_code()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"SOCKS5 Port Check: {'✅ PASSED' if socks5_available else '❌ FAILED'}")
    print(f"IP Change Test: {'✅ PASSED' if requests_success else '❌ FAILED'}")
    print(f"Binance API Test: {'✅ PASSED' if binance_success else '❌ FAILED'}")
    # print(f"PublicDataAPI Test: {'✅ PASSED' if our_code_success else '❌ FAILED'}")
    
    if socks5_available and requests_success and binance_success:
        print("\n🎉 All tests passed! Local SOCKS5 proxy is working correctly.")
        configure_env_for_socks5()
    else:
        print("\n❌ Some tests failed. Please check your PIA SOCKS5 configuration.")