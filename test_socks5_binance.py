"""
🌙 Test SOCKS5 Proxy with Binance
Specifically tests SOCKS5 on port 1080 which we know is open
"""

import os
import requests
import sys

def test_socks5_binance():
    """Test Binance through SOCKS5 proxy"""
    
    # Check for PySocks
    try:
        import socks
        print("✅ PySocks is installed")
    except ImportError:
        print("❌ PySocks not installed. Installing...")
        os.system("pip install PySocks requests[socks]")
        print("Please restart the script")
        return False
    
    # Get credentials
    username = os.getenv('PIA_USERNAME')
    password = os.getenv('PIA_PASSWORD')
    
    if not username or not password:
        print("\n❌ PIA credentials not found!")
        print("\n📝 Add to your .env file:")
        print("PIA_USERNAME=your_pia_proxy_username")
        print("PIA_PASSWORD=your_pia_proxy_password")
        print("PIA_USE_SOCKS5=true")
        print("\nGet these from: https://www.privateinternetaccess.com/")
        print("Control Panel → Proxy → Generate Credentials")
        return False
    
    print(f"\n🔐 Using credentials:")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")
    
    # SOCKS5 proxy configuration
    proxy_host = "proxy-nl.privateinternetaccess.com"
    proxy_port = 1080
    
    proxy_url = f"socks5://{username}:{password}@{proxy_host}:{proxy_port}"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    print(f"\n🔌 Connecting through SOCKS5 proxy:")
    print(f"Host: {proxy_host}")
    print(f"Port: {proxy_port}")
    
    # Test 1: Check IP
    print("\n📍 Test 1: Checking IP address...")
    try:
        response = requests.get('https://api.ipify.org?format=json', 
                              proxies=proxies, 
                              timeout=15)
        if response.status_code == 200:
            ip_data = response.json()
            print(f"✅ Connected! Proxy IP: {ip_data['ip']}")
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)[:100]}...")
        return False
    
    # Test 2: Binance API
    print("\n📊 Test 2: Binance API...")
    try:
        # Test ping endpoint
        response = requests.get('https://api.binance.com/api/v3/ping',
                              proxies=proxies,
                              timeout=15)
        
        if response.status_code == 200:
            print("✅ Binance API is accessible!")
            
            # Get BTC price
            ticker_response = requests.get('https://api.binance.com/api/v3/ticker/price',
                                         params={'symbol': 'BTCUSDT'},
                                         proxies=proxies,
                                         timeout=15)
            
            if ticker_response.status_code == 200:
                btc_data = ticker_response.json()
                print(f"💰 BTC Price: ${float(btc_data['price']):,.2f}")
                
                # Test futures API too
                print("\n📊 Test 3: Binance Futures API...")
                futures_response = requests.get('https://fapi.binance.com/fapi/v1/allForceOrders',
                                              params={'symbol': 'BTCUSDT', 'limit': 5},
                                              proxies=proxies,
                                              timeout=15)
                
                if futures_response.status_code == 200:
                    print("✅ Futures API is accessible!")
                    liqs = futures_response.json()
                    if liqs:
                        print(f"📈 Found {len(liqs)} recent liquidations")
                    return True
                else:
                    print(f"⚠️ Futures API returned: {futures_response.status_code}")
                    
        elif response.status_code == 451:
            print("❌ Still getting 451 geo-restriction error!")
            print("💡 The proxy might not be working correctly")
        else:
            print(f"❌ Binance returned status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Binance test failed: {str(e)[:100]}...")
        
    return False

def test_with_our_code():
    """Test with actual PublicDataAPI"""
    print("\n\n🧪 Testing with PublicDataAPI...")
    
    # Ensure environment is set
    os.environ['PIA_USE_SOCKS5'] = 'true'
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.agents.public_data_api import PublicDataAPI
        
        api = PublicDataAPI()
        
        # Quick test
        print("📊 Fetching liquidation data...")
        liq_data = api.get_liquidation_data(symbol='BTCUSDT', limit=5)
        
        if liq_data is not None and not liq_data.empty:
            print(f"✅ Success! Got {len(liq_data)} liquidation records")
            print(f"💰 Total liquidation value: ${liq_data['total_size'].sum():,.2f}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌙 Moon Dev SOCKS5 Proxy Test for Binance")
    print("="*60)
    
    # Test SOCKS5
    success = test_socks5_binance()
    
    if success:
        print("\n🎉 SOCKS5 proxy is working with Binance!")
        
        # Test with our code
        code_success = test_with_our_code()
        
        if code_success:
            print("\n✅ Everything is working! You can now use Binance API.")
        else:
            print("\n⚠️ Manual tests work but PublicDataAPI needs adjustment")
    else:
        print("\n❌ SOCKS5 proxy test failed")
        print("\n💡 Make sure you have:")
        print("1. Valid PIA proxy credentials (not your main login)")
        print("2. Active PIA subscription")
        print("3. Correct settings in .env file")