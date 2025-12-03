"""
🌙 TradingView Authenticated Client
Python client for the Node.js TradingView server with authentication
"""

import requests
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
import time

class TradingViewAuthClient:
    """
    Client for authenticated TradingView access via Node.js server
    """
    
    def __init__(self, server_url: str = None, auto_login: bool = True):
        """
        Initialize client
        
        Args:
            server_url: URL of the Node.js server (default: http://localhost:8888)
            auto_login: Automatically login using env credentials
        """
        self.server_url = server_url or os.getenv('TV_SERVER_URL', 'http://localhost:8888')
        self.session = requests.Session()
        self.authenticated = False
        
        print(f"🌙 TradingView Auth Client initialized")
        print(f"📡 Server: {self.server_url}")
        
        # Auto-login if credentials available
        if auto_login:
            username = os.getenv('TV_USERNAME')
            password = os.getenv('TV_PASSWORD')
            if username and password:
                self.login(username, password)
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make request to server"""
        url = f"{self.server_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to TradingView server at {self.server_url}. Is the server running?")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise Exception("Not authenticated. Please login first.")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {e}")
    
    def check_health(self) -> Dict:
        """Check server health"""
        return self._request('GET', '/health')
    
    def login(self, username: str, password: str) -> Dict:
        """
        Login to TradingView
        
        Args:
            username: TradingView username
            password: TradingView password
            
        Returns:
            Login response with user info
        """
        print(f"🔐 Logging in as {username}...")
        
        result = self._request('POST', '/login', json={
            'username': username,
            'password': password
        })
        
        if result.get('success'):
            self.authenticated = True
            print("✅ Login successful!")
        
        return result
    
    def logout(self) -> Dict:
        """Logout and close session"""
        result = self._request('POST', '/logout')
        self.authenticated = False
        return result
    
    def get_chart_data(self, symbol: str, timeframe: str = 'D', 
                      exchange: str = 'BINANCE') -> Dict:
        """
        Get real-time chart data
        
        Args:
            symbol: Symbol to fetch (e.g., 'BTCUSDT')
            timeframe: Timeframe (1, 5, 15, 30, 60, D, W, M)
            exchange: Exchange name
            
        Returns:
            Chart data with OHLCV
        """
        return self._request('POST', '/chart', json={
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange
        })
    
    def get_indicator(self, symbol: str, indicator: str,
                     timeframe: str = '60', exchange: str = 'BINANCE',
                     options: Dict = None) -> Dict:
        """
        Get indicator values
        
        Args:
            symbol: Symbol to analyze
            indicator: Indicator name (e.g., 'RSI@tv-basicstudies')
            timeframe: Timeframe
            exchange: Exchange
            options: Indicator-specific options
            
        Returns:
            Indicator data
        """
        return self._request('POST', '/indicator', json={
            'symbol': symbol,
            'indicator': indicator,
            'timeframe': timeframe,
            'exchange': exchange,
            'options': options or {}
        })
    
    def get_batch_data(self, symbols: List[str], 
                      timeframe: str = '60', 
                      exchange: str = 'BINANCE') -> Dict:
        """
        Get data for multiple symbols at once
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for all symbols
            exchange: Exchange for all symbols
            
        Returns:
            Batch results
        """
        return self._request('POST', '/batch', json={
            'symbols': symbols,
            'timeframe': timeframe,
            'exchange': exchange
        })
    
    def search_symbols(self, query: str, type: str = 'crypto') -> List[Dict]:
        """
        Search for symbols
        
        Args:
            query: Search query
            type: Market type (crypto, stock, forex, etc.)
            
        Returns:
            Search results
        """
        return self._request('GET', '/search', params={
            'query': query,
            'type': type
        })
    
    def get_realtime_data(self, symbol: str, exchange: str = 'BINANCE',
                         callback=None, duration: int = 60):
        """
        Get real-time streaming-like data
        
        Args:
            symbol: Symbol to monitor
            exchange: Exchange
            callback: Function to call with each update
            duration: How long to run (seconds)
        """
        print(f"📊 Starting real-time monitoring of {symbol}...")
        
        start_time = time.time()
        last_price = None
        
        while time.time() - start_time < duration:
            try:
                data = self.get_chart_data(symbol, '1', exchange)
                
                if 'close' in data:
                    current_price = data['close']
                    
                    if current_price != last_price:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"{symbol}: ${current_price:,.2f}")
                        
                        if callback:
                            callback(data)
                        
                        last_price = current_price
                
                time.sleep(1)  # Poll every second
                
            except KeyboardInterrupt:
                print("\n⏹️  Stopped monitoring")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(5)  # Wait longer on error
    
    def get_historical_data(self, symbol: str, timeframe: str = '60',
                          bars: int = 100, exchange: str = 'BINANCE') -> Dict:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1, 5, 15, 30, 60, 240, 1D, 1W, 1M)
            bars: Number of bars to fetch
            exchange: Exchange name
            
        Returns:
            Dictionary with historical data
        """
        payload = {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': bars,
            'exchange': exchange
        }
        
        response = requests.post(f'{self.server_url}/history', json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_multiple_indicators(self, symbol: str, 
                               indicators: List[str],
                               timeframe: str = '60',
                               exchange: str = 'BINANCE') -> Dict[str, Any]:
        """
        Get multiple indicators for a symbol
        
        Args:
            symbol: Symbol to analyze
            indicators: List of indicator names
            timeframe: Timeframe
            exchange: Exchange
            
        Returns:
            Dictionary of indicator results
        """
        results = {}
        
        for indicator in indicators:
            try:
                data = self.get_indicator(symbol, indicator, timeframe, exchange)
                results[indicator] = data
                time.sleep(0.5)  # Be nice to the server
            except Exception as e:
                print(f"⚠️ Failed to get {indicator}: {e}")
                results[indicator] = None
        
        return results

# Convenience functions
def create_authenticated_client(username: str = None, password: str = None) -> TradingViewAuthClient:
    """
    Create and authenticate a client
    
    Args:
        username: TradingView username (or from env)
        password: TradingView password (or from env)
        
    Returns:
        Authenticated client
    """
    client = TradingViewAuthClient(auto_login=False)
    
    # Check server health first
    try:
        health = client.check_health()
        print(f"✅ Server is running: {health}")
    except Exception as e:
        raise Exception(f"Server not available: {e}")
    
    # Login
    username = username or os.getenv('TV_USERNAME')
    password = password or os.getenv('TV_PASSWORD')
    
    if not username or not password:
        raise Exception("TradingView credentials required")
    
    client.login(username, password)
    return client

# Example usage
if __name__ == "__main__":
    print("🌙 TradingView Authenticated Client Test")
    print("="*60)
    
    # Check if server is running
    try:
        client = TradingViewAuthClient(auto_login=False)
        health = client.check_health()
        print(f"Server status: {health}")
    except Exception as e:
        print(f"❌ Server not running: {e}")
        print("\n📝 To start the server:")
        print("1. cd tradingview-server")
        print("2. npm install")
        print("3. Create .env with TV_USERNAME and TV_PASSWORD")
        print("4. npm start")
        exit(1)
    
    # Login
    username = os.getenv('TV_USERNAME')
    password = os.getenv('TV_PASSWORD')
    
    if not username or not password:
        print("\n⚠️ No credentials found in environment")
        print("Set TV_USERNAME and TV_PASSWORD in your .env file")
        exit(1)
    
    try:
        client.login(username, password)
        
        # Test 1: Get chart data
        print("\n📊 Test 1: Chart Data")
        data = client.get_chart_data('BTCUSDT', '60')
        print(f"BTC Price: ${data.get('close', 0):,.2f}")
        
        # Test 2: Batch data
        print("\n📊 Test 2: Batch Data")
        batch = client.get_batch_data(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        for result in batch.get('results', []):
            print(f"{result['symbol']}: ${result['close']:,.2f}")
        
        # Test 3: Search
        print("\n📊 Test 3: Symbol Search")
        results = client.search_symbols('BTC')
        print(f"Found {len(results)} results for 'BTC'")
        
        # Test 4: Real-time monitoring (10 seconds)
        print("\n📊 Test 4: Real-time Monitoring (10 seconds)")
        client.get_realtime_data('BTCUSDT', duration=10)
        
        # Logout
        client.logout()
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")