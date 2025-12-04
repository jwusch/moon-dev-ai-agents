#!/usr/bin/env python3
"""
🌐 YFINANCE PIA PROXY WRAPPER 🌐
Use Private Internet Access SOCKS5 proxy with yfinance to bypass geo-restrictions
"""

import os
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
import time
from datetime import datetime, timedelta

class YFinanceProxyWrapper:
    """Wrapper for yfinance with PIA SOCKS5 proxy support"""
    
    def __init__(self):
        self.setup_proxy_session()
    
    def setup_proxy_session(self):
        """Setup requests session with PIA SOCKS5 proxy"""
        
        # Load PIA configuration from environment
        self.pia_username = os.getenv('PIA_USERNAME')
        self.pia_password = os.getenv('PIA_PASSWORD')
        self.use_socks5 = os.getenv('PIA_USE_SOCKS5', 'true').lower() == 'true'
        self.socks5_host = os.getenv('PIA_SOCKS5_HOST', 'proxy-nl.privateinternetaccess.com')
        self.socks5_port = int(os.getenv('PIA_SOCKS5_PORT', '1080'))
        
        print(f"🌐 YFINANCE PIA PROXY CONFIGURATION:")
        
        if self.pia_username:
            print(f"   Username: {self.pia_username[:4]}***")
        else:
            print(f"   Username: Not configured")
            
        print(f"   Use SOCKS5: {self.use_socks5}")
        print(f"   Host: {self.socks5_host}")
        print(f"   Port: {self.socks5_port}")
        
        if not self.pia_username or not self.pia_password:
            print("⚠️  No PIA credentials found - yfinance will run without proxy")
            print("💡 To enable proxy: Set PIA_USERNAME and PIA_PASSWORD in .env file")
            self.proxy_session = None
            return
        
        # Create proxy session
        try:
            if self.use_socks5:
                # Import required for SOCKS5
                try:
                    import socks
                    import socket
                    from urllib3.contrib.socks import SOCKSProxyManager
                except ImportError:
                    print("❌ SOCKS5 dependencies not installed. Run: pip install requests[socks] PySocks")
                    self.proxy_session = None
                    return
                
                # Create SOCKS5 proxy URL
                proxy_url = f"socks5://{self.pia_username}:{self.pia_password}@{self.socks5_host}:{self.socks5_port}"
                
                # Configure session with SOCKS5
                self.proxy_session = requests.Session()
                self.proxy_session.proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                
                print(f"✅ SOCKS5 proxy session configured: {self.socks5_host}:{self.socks5_port}")
                
            else:
                # HTTP proxy fallback
                http_proxy = f"http://{self.pia_username}:{self.pia_password}@{self.socks5_host}:8080"
                
                self.proxy_session = requests.Session()
                self.proxy_session.proxies = {
                    'http': http_proxy,
                    'https': http_proxy
                }
                
                print(f"✅ HTTP proxy session configured: {self.socks5_host}:8080")
            
            # Test proxy connection
            self.test_proxy_connection()
            
        except Exception as e:
            print(f"❌ Proxy setup error: {e}")
            self.proxy_session = None
    
    def test_proxy_connection(self):
        """Test proxy connection with a simple request"""
        
        if not self.proxy_session:
            return False
        
        try:
            print("🔍 Testing proxy connection...")
            
            # Test with a simple HTTP request
            response = self.proxy_session.get('http://httpbin.org/ip', timeout=10)
            
            if response.status_code == 200:
                proxy_ip = response.json().get('origin', 'unknown')
                print(f"✅ Proxy connection successful! External IP: {proxy_ip}")
                return True
            else:
                print(f"⚠️  Proxy test returned status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Proxy test failed: {e}")
            return False
    
    def patch_yfinance_session(self):
        """Patch yfinance to use our proxy session"""
        
        if not self.proxy_session:
            print("⚠️  No proxy session available - yfinance will run without proxy")
            return
        
        # Store original yfinance session creation
        original_session_func = getattr(yf.utils, '_get_session', None)
        
        def proxy_session_func():
            """Return our proxy-enabled session"""
            return self.proxy_session
        
        # Patch yfinance to use our proxy session
        if hasattr(yf.utils, '_get_session'):
            yf.utils._get_session = proxy_session_func
            print("✅ yfinance patched to use PIA proxy")
        elif hasattr(yf, '_get_session'):
            yf._get_session = proxy_session_func
            print("✅ yfinance patched to use PIA proxy (alt method)")
        else:
            # Try to patch the requests module that yfinance uses
            import yfinance.base as yf_base
            if hasattr(yf_base, 'requests'):
                yf_base.requests = self.proxy_session
                print("✅ yfinance base requests patched to use PIA proxy")
            else:
                print("⚠️  Unable to patch yfinance - may not use proxy")
    
    def fetch_with_proxy(self, symbol, **kwargs):
        """Fetch data using proxy-enabled yfinance"""
        
        # Patch yfinance to use proxy (if available)
        if self.proxy_session:
            self.patch_yfinance_session()
            proxy_status = "via PIA proxy"
        else:
            proxy_status = "without proxy"
        
        try:
            # Use yfinance normally - it will now use our proxy
            ticker = yf.Ticker(symbol)
            
            # Default parameters for history call
            default_params = {
                'period': '1mo',
                'interval': '1d',
                'auto_adjust': True,
                'prepost': True,
                'timeout': 30
            }
            
            # Update with user parameters
            params = {**default_params, **kwargs}
            
            print(f"📊 Fetching {symbol} {proxy_status}...")
            df = ticker.history(**params)
            
            if df is not None and not df.empty:
                print(f"✅ {symbol}: {len(df)} bars retrieved {proxy_status}")
                return df
            else:
                print(f"❌ {symbol}: No data returned")
                return None
                
        except Exception as e:
            print(f"❌ {symbol} proxy fetch error: {e}")
            return None
    
    def bulk_fetch_with_proxy(self, symbols, rate_limit_seconds=2, **kwargs):
        """Fetch multiple symbols with rate limiting and proxy"""
        
        print(f"\n🚀 BULK FETCH WITH PIA PROXY")
        print("=" * 50)
        print(f"📊 Symbols: {len(symbols)}")
        print(f"⏱️  Rate limit: {rate_limit_seconds}s between requests")
        
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i:3}/{len(symbols)}] Fetching {symbol}...")
            
            # Fetch data
            df = self.fetch_with_proxy(symbol, **kwargs)
            results[symbol] = df
            
            # Rate limiting (except for last symbol)
            if i < len(symbols):
                print(f"    ⏳ Rate limit delay: {rate_limit_seconds}s")
                time.sleep(rate_limit_seconds)
        
        # Summary
        successful = len([s for s, df in results.items() if df is not None and not df.empty])
        failed = len(symbols) - successful
        
        print(f"\n📊 BULK FETCH SUMMARY:")
        print(f"   ✅ Successful: {successful}/{len(symbols)}")
        print(f"   ❌ Failed: {failed}")
        
        return results

def create_proxy_enabled_yfinance():
    """Factory function to create proxy-enabled yfinance wrapper"""
    
    wrapper = YFinanceProxyWrapper()
    return wrapper

def fetch_symbol_with_pia_proxy(symbol, **kwargs):
    """Convenience function to fetch single symbol with PIA proxy"""
    
    wrapper = create_proxy_enabled_yfinance()
    return wrapper.fetch_with_proxy(symbol, **kwargs)

def fetch_symbols_with_pia_proxy(symbols, **kwargs):
    """Convenience function to fetch multiple symbols with PIA proxy"""
    
    wrapper = create_proxy_enabled_yfinance()
    return wrapper.bulk_fetch_with_proxy(symbols, **kwargs)

# Example usage and testing
def main():
    """Test PIA proxy integration with yfinance"""
    
    print("🌐🚀 YFINANCE PIA PROXY INTEGRATION TEST 🚀🌐")
    print("=" * 70)
    
    # Create proxy wrapper
    wrapper = create_proxy_enabled_yfinance()
    
    # Test symbols (mix of known working symbols)
    test_symbols = ["AAPL", "TSLA", "SPY", "QQQ", "MSFT"]
    
    print(f"🎯 Testing {len(test_symbols)} symbols with PIA proxy")
    
    # Test single symbol fetch
    print(f"\n📊 SINGLE SYMBOL TEST:")
    df = wrapper.fetch_with_proxy("AAPL", period="5d", interval="1h")
    
    if df is not None and not df.empty:
        print(f"✅ Single fetch successful: AAPL ({len(df)} bars)")
    else:
        print("❌ Single fetch failed")
    
    # Test bulk fetch
    print(f"\n📊 BULK FETCH TEST:")
    results = wrapper.bulk_fetch_with_proxy(
        test_symbols[:3],  # Test first 3 symbols
        period="3d",
        interval="15m",
        rate_limit_seconds=3
    )
    
    successful_symbols = [s for s, df in results.items() if df is not None and not df.empty]
    
    print(f"\n🎯 INTEGRATION TEST COMPLETE!")
    print(f"   📊 Symbols tested: {len(test_symbols[:3])}")
    print(f"   ✅ Successful: {len(successful_symbols)}")
    print(f"   🌐 Proxy: {'Enabled' if wrapper.proxy_session else 'Disabled'}")
    
    if successful_symbols:
        print(f"   🏆 Working symbols: {', '.join(successful_symbols)}")

if __name__ == "__main__":
    main()