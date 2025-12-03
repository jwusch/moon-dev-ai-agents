"""
🌙 Debug Proxy URL Construction
Verifies that proxy URLs are being constructed correctly
"""

import os
import requests

def test_proxy_url_formats():
    """Test different proxy URL formats to find what works"""
    
    # Get credentials from environment
    username = os.getenv('PIA_USERNAME', 'test_user')
    password = os.getenv('PIA_PASSWORD', 'test_pass')
    
    print("🔍 Testing proxy URL formats...")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")
    print("="*60)
    
    # Different URL formats to test
    proxy_formats = [
        {
            'name': 'Standard HTTP with auth',
            'url': f'http://{username}:{password}@proxy-nl.privateinternetaccess.com:8080',
            'description': 'Standard format with embedded credentials'
        },
        {
            'name': 'HTTPS variant',
            'url': f'https://{username}:{password}@proxy-nl.privateinternetaccess.com:8080',
            'description': 'HTTPS protocol (may not be supported)'
        },
        {
            'name': 'Without protocol prefix',
            'url': f'{username}:{password}@proxy-nl.privateinternetaccess.com:8080',
            'description': 'No protocol specified'
        },
        {
            'name': 'SOCKS5 format',
            'url': f'socks5://{username}:{password}@proxy-nl.privateinternetaccess.com:1080',
            'description': 'SOCKS5 proxy format'
        }
    ]
    
    for format_info in proxy_formats:
        print(f"\n📝 Testing: {format_info['name']}")
        print(f"   URL: {format_info['url'].replace(password, '*'*8)}")
        print(f"   Description: {format_info['description']}")
        
        proxies = {
            'http': format_info['url'],
            'https': format_info['url']
        }
        
        try:
            # Simple connection test
            response = requests.get('https://httpbin.org/ip', 
                                  proxies=proxies, 
                                  timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ Success! Response: {response.json()}")
            else:
                print(f"   ❌ Failed with status: {response.status_code}")
                
        except requests.exceptions.ProxyError as e:
            error_msg = str(e)
            if 'Connection refused' in error_msg:
                print(f"   ❌ Connection refused - proxy not accepting connections")
            elif '407' in error_msg:
                print(f"   ❌ Proxy authentication failed - check credentials")
            else:
                print(f"   ❌ Proxy error: {error_msg[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}...")

def test_direct_connection():
    """Test direct connection without proxy"""
    print("\n🔍 Testing direct connection (no proxy)...")
    
    try:
        response = requests.get('https://httpbin.org/ip', timeout=5)
        if response.status_code == 200:
            print(f"✅ Direct connection works! Your IP: {response.json()['origin']}")
        else:
            print(f"❌ Direct connection failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_binance_endpoints():
    """Test Binance-specific endpoints"""
    print("\n🔍 Testing Binance endpoints directly...")
    
    endpoints = [
        ('https://api.binance.com/api/v3/ping', 'Binance Spot API'),
        ('https://fapi.binance.com/fapi/v1/ping', 'Binance Futures API'),
        ('https://api.binance.us/api/v3/ping', 'Binance US API'),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Accessible")
            elif response.status_code == 451:
                print(f"❌ {name}: Geo-restricted (451)")
            else:
                print(f"⚠️ {name}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}...")

if __name__ == "__main__":
    print("🌙 Moon Dev Proxy URL Debug Tool")
    print("="*60)
    
    # Test direct connection first
    test_direct_connection()
    
    # Test Binance endpoints
    test_binance_endpoints()
    
    # Test proxy formats
    print("\n" + "="*60)
    test_proxy_url_formats()
    
    print("\n📝 Recommendations:")
    print("1. If you see '451 Geo-restricted' for Binance, you need a proxy")
    print("2. If proxy tests fail with 'Connection refused', check:")
    print("   - PIA proxy server is correct (proxy-nl.privateinternetaccess.com)")
    print("   - Port is correct (8080 for HTTP, 1080 for SOCKS5)")
    print("   - Your PIA subscription is active")
    print("3. If you see '407 Authentication failed', check:")
    print("   - You're using proxy credentials (not main PIA login)")
    print("   - Credentials are correctly set in .env file")