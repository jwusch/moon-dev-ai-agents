"""
🌙 Test PIA Remote Proxy Servers
Tests connection to PIA's proxy servers (not local)
"""

import os
import requests
import sys
from datetime import datetime

def test_pia_credentials():
    """Test if PIA credentials are set"""
    username = os.getenv('PIA_USERNAME')
    password = os.getenv('PIA_PASSWORD')
    
    print("🔐 Checking PIA credentials...")
    if username and password:
        print(f"✅ PIA Username: {username}")
        print(f"✅ PIA Password: {'*' * 8}")
        return True
    else:
        print("❌ PIA credentials not found in environment")
        print("\n📝 Add to your .env file:")
        print("PIA_USERNAME=your_pia_proxy_username")
        print("PIA_PASSWORD=your_pia_proxy_password")
        return False

def test_proxy_connection(proxy_type="http", host=None, port=None):
    """Test connection through PIA proxy"""
    username = os.getenv('PIA_USERNAME')
    password = os.getenv('PIA_PASSWORD')
    
    if not (username and password):
        return False
    
    # Default hosts and ports
    if proxy_type == "socks5":
        default_host = "proxy-nl.privateinternetaccess.com"
        default_port = "1080"
        protocol = "socks5"
    else:
        default_host = "proxy-nl.privateinternetaccess.com"
        default_port = "8080"
        protocol = "http"
    
    host = host or os.getenv(f'PIA_{proxy_type.upper()}_HOST', default_host)
    port = port or os.getenv(f'PIA_{proxy_type.upper()}_PORT', default_port)
    
    proxy_url = f"{protocol}://{username}:{password}@{host}:{port}"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    print(f"\n🧪 Testing {proxy_type.upper()} proxy: {host}:{port}")
    
    try:
        # Test 1: Get current IP
        print("📍 Fetching IP through proxy...")
        response = requests.get('https://api.ipify.org?format=json', 
                              proxies=proxies, 
                              timeout=10)
        
        if response.status_code == 200:
            ip_data = response.json()
            print(f"✅ Success! Proxy IP: {ip_data['ip']}")
            
            # Test 2: Get location info
            try:
                loc_response = requests.get('https://ipapi.co/json/', 
                                          proxies=proxies, 
                                          timeout=10)
                if loc_response.status_code == 200:
                    loc_data = loc_response.json()
                    print(f"📍 Location: {loc_data.get('city', 'Unknown')}, {loc_data.get('country_name', 'Unknown')}")
                    print(f"🏢 ISP: {loc_data.get('org', 'Unknown')}")
            except:
                pass
            
            # Test 3: Binance API
            print("\n📊 Testing Binance API...")
            binance_response = requests.get('https://api.binance.com/api/v3/ping',
                                          proxies=proxies,
                                          timeout=10)
            
            if binance_response.status_code == 200:
                print("✅ Binance API accessible through proxy!")
                
                # Get BTC price
                ticker_response = requests.get('https://api.binance.com/api/v3/ticker/price',
                                             params={'symbol': 'BTCUSDT'},
                                             proxies=proxies,
                                             timeout=10)
                if ticker_response.status_code == 200:
                    btc_data = ticker_response.json()
                    print(f"💰 BTC Price: ${float(btc_data['price']):,.2f}")
            else:
                print(f"❌ Binance returned status: {binance_response.status_code}")
                
            return True
            
        else:
            print(f"❌ Proxy request failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.ProxyError as e:
        print(f"❌ Proxy connection failed: {str(e)[:100]}...")
        print("\n💡 Common issues:")
        print("1. Wrong proxy credentials (get from PIA Control Panel → Proxy)")
        print("2. PIA subscription expired")
        print("3. Proxy server is down - try a different region")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}...")
        return False

def test_multiple_regions():
    """Test different PIA proxy regions"""
    print("\n🌍 Testing multiple PIA regions...")
    
    regions = {
        'Netherlands': 'proxy-nl.privateinternetaccess.com',
        'UK': 'proxy-uk.privateinternetaccess.com',
        'Canada': 'proxy-ca.privateinternetaccess.com',
        'Japan': 'proxy-jp.privateinternetaccess.com',
        'Singapore': 'proxy-sg.privateinternetaccess.com',
    }
    
    working_regions = []
    
    for region, host in regions.items():
        print(f"\n🔄 Testing {region}...")
        if test_proxy_connection("http", host, "8080"):
            working_regions.append(region)
            print(f"✅ {region} proxy is working!")
        else:
            print(f"❌ {region} proxy failed")
    
    return working_regions

if __name__ == "__main__":
    print("🌙 Moon Dev PIA Remote Proxy Test")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if credentials exist
    if not test_pia_credentials():
        print("\n❌ Please configure PIA credentials first!")
        print("\n📝 Instructions:")
        print("1. Log in to PIA website")
        print("2. Go to Control Panel → Proxy")
        print("3. Generate proxy credentials")
        print("4. Add to .env file")
        sys.exit(1)
    
    # Test HTTP proxy
    http_success = test_proxy_connection("http")
    
    # Test SOCKS5 proxy
    print("\n" + "-"*60)
    socks5_success = test_proxy_connection("socks5")
    
    # Quick region test
    if http_success or socks5_success:
        response = input("\n🌍 Test multiple regions? (y/n): ")
        if response.lower() == 'y':
            working_regions = test_multiple_regions()
            
            print("\n" + "="*60)
            print("📊 Regional Proxy Summary")
            print("="*60)
            if working_regions:
                print(f"✅ Working regions: {', '.join(working_regions)}")
            else:
                print("❌ No regions tested successfully")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"HTTP Proxy: {'✅ PASSED' if http_success else '❌ FAILED'}")
    print(f"SOCKS5 Proxy: {'✅ PASSED' if socks5_success else '❌ FAILED'}")
    
    if http_success or socks5_success:
        print("\n🎉 PIA proxy is working! You can now access Binance API.")
        print("\n📝 Your .env configuration is correct!")
    else:
        print("\n❌ Proxy tests failed.")
        print("\n💡 Next steps:")
        print("1. Verify credentials at privateinternetaccess.com")
        print("2. Check if PIA subscription is active")
        print("3. Try different proxy regions")