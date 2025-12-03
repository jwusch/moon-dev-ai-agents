"""
🌙 Binance Proxy Configuration - Always use real data through proxy
Ensures Binance API calls work from any location using PIA VPN
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# Disable SSL warnings when using proxy (optional)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BinanceProxySession:
    """
    Session configured specifically for Binance API with proxy support
    """
    
    def __init__(self):
        self.session = requests.Session()
        self._configure_proxy()
        self._configure_retry_strategy()
        
    def _configure_proxy(self):
        """
        Configure proxy from environment variables
        Priority order:
        1. BINANCE_PROXY - Specific proxy for Binance
        2. HTTPS_PROXY/HTTP_PROXY - General proxy settings
        3. PIA proxy auto-configuration
        """
        # Check for Binance-specific proxy first
        binance_proxy = os.getenv('BINANCE_PROXY')
        if binance_proxy:
            self.proxies = {
                'http': binance_proxy,
                'https': binance_proxy
            }
            print(f"🔐 Using Binance-specific proxy: {self._mask_credentials(binance_proxy)}")
            return
            
        # Check for general proxy settings
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        
        if https_proxy or http_proxy:
            self.proxies = {}
            if https_proxy:
                self.proxies['https'] = https_proxy
            if http_proxy:
                self.proxies['http'] = http_proxy
            print(f"🔐 Using system proxy configuration")
            return
            
        # Check for PIA configuration
        pia_username = os.getenv('PIA_USERNAME')
        pia_password = os.getenv('PIA_PASSWORD')
        
        if pia_username and pia_password:
            # Check if user prefers SOCKS5
            use_socks5 = os.getenv('PIA_USE_SOCKS5', '').lower() == 'true'
            
            if use_socks5:
                # First check if PySocks is installed for SOCKS5 support
                try:
                    import socks
                    # Use SOCKS5 proxy
                    pia_socks5_host = os.getenv('PIA_SOCKS5_HOST', 'proxy-nl.privateinternetaccess.com')
                    pia_socks5_port = os.getenv('PIA_SOCKS5_PORT', '1080')
                    proxy_url = f"socks5://{pia_username}:{pia_password}@{pia_socks5_host}:{pia_socks5_port}"
                    
                    self.proxies = {
                        'http': proxy_url,
                        'https': proxy_url
                    }
                    print(f"🔐 Using PIA SOCKS5 proxy: {pia_socks5_host}:{pia_socks5_port}")
                except ImportError:
                    print("⚠️ PySocks not installed for SOCKS5. Falling back to HTTP proxy.")
                    print("💡 Install with: pip install requests[socks]")
                    use_socks5 = False
            
            if not use_socks5:
                # Use HTTP proxy (better compatibility)
                pia_http_host = os.getenv('PIA_HTTP_HOST', 'proxy-nl.privateinternetaccess.com')
                pia_http_port = os.getenv('PIA_HTTP_PORT', '8080')
                
                # Properly construct proxy URL with authentication
                proxy_url = f'http://{pia_username}:{pia_password}@{pia_http_host}:{pia_http_port}'
                
                self.proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                print(f"🔐 Using PIA HTTP proxy: {pia_http_host}:{pia_http_port}")
                print(f"🔐 Proxy URL: {self._mask_credentials(proxy_url)}")
            return
            
        # No proxy configured - will fail for geo-restricted regions
        self.proxies = None
        print("⚠️ No proxy configured - Binance API may fail in restricted regions")
        print("💡 Set PIA_USERNAME and PIA_PASSWORD in .env file to use PIA proxy")
        
    def _mask_credentials(self, proxy_url):
        """Hide credentials in proxy URL for logging"""
        if '@' in proxy_url:
            parts = proxy_url.split('@')
            return f"{parts[0].split('://')[0]}://***:***@{parts[1]}"
        return proxy_url
        
    def _configure_retry_strategy(self):
        """Configure retry strategy for resilience"""
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504, 408, 429]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def test_connection(self):
        """Test if we can reach Binance through proxy"""
        try:
            # First test the proxy itself
            print("🔍 Testing proxy connection...")
            test_url = "https://api.ipify.org?format=json"
            
            if self.proxies:
                # Test with proxy
                proxy_response = self.session.get(test_url, proxies=self.proxies, timeout=10)
                if proxy_response.status_code == 200:
                    proxy_ip = proxy_response.json()['ip']
                    print(f"✅ Proxy connected! IP: {proxy_ip}")
                else:
                    print(f"❌ Proxy test failed: {proxy_response.status_code}")
                    return False
            
            # Now test Binance
            print("🔍 Testing Binance API...")
            url = "https://api.binance.com/api/v3/ping"
            response = self.session.get(url, proxies=self.proxies, timeout=10)
            
            if response.status_code == 200:
                print("✅ Successfully connected to Binance API")
                
                # Test IP location
                try:
                    ip_response = self.session.get('https://ipapi.co/json/', 
                                                 proxies=self.proxies, 
                                                 timeout=5)
                    if ip_response.status_code == 200:
                        ip_data = ip_response.json()
                        print(f"📍 Current location: {ip_data.get('city', 'Unknown')}, {ip_data.get('country_name', 'Unknown')}")
                        print(f"🌐 IP: {ip_data.get('ip', 'Unknown')}")
                except:
                    pass
                    
                return True
            else:
                print(f"❌ Binance API returned status: {response.status_code}")
                return False
                
        except requests.exceptions.ProxyError as e:
            print(f"❌ Proxy connection failed: {e}")
            print("💡 Check your proxy credentials and connection")
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
            
    def get(self, url, **kwargs):
        """GET request with automatic proxy"""
        if self.proxies:
            kwargs['proxies'] = self.proxies
        return self.session.get(url, **kwargs)
        
    def post(self, url, **kwargs):
        """POST request with automatic proxy"""
        if self.proxies:
            kwargs['proxies'] = self.proxies
        return self.session.post(url, **kwargs)


# Enhanced PublicDataAPI that enforces proxy usage for Binance
def create_binance_session():
    """
    Create a properly configured session for Binance API calls
    """
    session = BinanceProxySession()
    
    # Test connection
    if not session.test_connection():
        raise Exception("❌ Cannot connect to Binance API. Please configure proxy in .env file")
        
    return session


if __name__ == "__main__":
    print("🌙 Testing Binance Proxy Configuration...")
    
    try:
        session = create_binance_session()
        
        # Test getting ticker price
        print("\n📊 Testing market data endpoint...")
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {'symbol': 'BTCUSDT'}
        
        response = session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ BTC Price: ${float(data['price']):,.2f}")
        else:
            print(f"❌ Failed to get price: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n📝 To fix this, add to your .env file:")
        print("PIA_USERNAME=your_pia_username")
        print("PIA_PASSWORD=your_pia_password")
        print("# Optional: Use specific proxy")
        print("# BINANCE_PROXY=http://username:password@proxy:port")