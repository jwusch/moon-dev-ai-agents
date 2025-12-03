"""
🌙 TradingView Authenticated API
Enhanced API that uses authenticated access for better rate limits and features
"""

import os
import pandas as pd
from typing import Optional, Dict, List, Union
from datetime import datetime
import time

from src.agents.tradingview_auth_client import TradingViewAuthClient, create_authenticated_client

class TradingViewAuthenticatedAPI:
    """
    TradingView API with authentication support
    Provides better rate limits and access to more features
    """
    
    def __init__(self, username: str = None, password: str = None):
        """
        Initialize authenticated API
        
        Args:
            username: TradingView username (or from env)
            password: TradingView password (or from env)
        """
        self.client = None
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # Much faster with auth!
        
        # Initialize client
        try:
            self.client = create_authenticated_client(username, password)
            print("✅ Authenticated TradingView API ready")
            print("🚀 Higher rate limits enabled!")
        except Exception as e:
            print(f"❌ Failed to initialize authenticated API: {e}")
            raise
    
    def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_price(self, symbol: str, exchange: str = 'BINANCE') -> Optional[float]:
        """Get current price"""
        self._rate_limit()
        
        try:
            data = self.client.get_chart_data(symbol, '1', exchange)
            return data.get('close')
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = '60', 
                  exchange: str = 'BINANCE') -> pd.DataFrame:
        """
        Get OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: 1, 5, 15, 30, 60, D, W, M
            exchange: Exchange name
            
        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()
        
        try:
            data = self.client.get_chart_data(symbol, timeframe, exchange)
            
            if data:
                df = pd.DataFrame([{
                    'timestamp': datetime.fromtimestamp(data.get('time', 0)),
                    'open': data.get('open', 0),
                    'high': data.get('max', data.get('high', 0)),
                    'low': data.get('min', data.get('low', 0)),
                    'close': data.get('close', 0),
                    'volume': data.get('volume', 0)
                }])
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting OHLCV: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], 
                           timeframe: str = '60',
                           exchange: str = 'BINANCE') -> pd.DataFrame:
        """Get data for multiple symbols efficiently"""
        self._rate_limit()
        
        try:
            batch_data = self.client.get_batch_data(symbols, timeframe, exchange)
            results = batch_data.get('results', [])
            
            if results:
                df = pd.DataFrame(results)
                df['timestamp'] = datetime.now()
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting batch data: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str,
                               indicators: List[str],
                               timeframe: str = '60',
                               exchange: str = 'BINANCE') -> Dict[str, any]:
        """
        Get multiple technical indicators
        
        Common indicators:
        - 'RSI@tv-basicstudies'
        - 'MACD@tv-basicstudies'
        - 'BB@tv-basicstudies'
        - 'EMA@tv-basicstudies'
        - 'SMA@tv-basicstudies'
        """
        self._rate_limit()
        
        try:
            return self.client.get_multiple_indicators(
                symbol, indicators, timeframe, exchange
            )
        except Exception as e:
            print(f"❌ Error getting indicators: {e}")
            return {}
    
    def monitor_symbol(self, symbol: str, duration: int = 60,
                      callback=None, exchange: str = 'BINANCE'):
        """
        Monitor a symbol in real-time
        
        Args:
            symbol: Symbol to monitor
            duration: How long to monitor (seconds)
            callback: Function to call with updates
            exchange: Exchange
        """
        print(f"📊 Monitoring {symbol} for {duration} seconds...")
        
        self.client.get_realtime_data(
            symbol, exchange, callback, duration
        )
    
    def get_historical_data(self, symbol: str, timeframe: str = '60', 
                          bars: int = 100, exchange: str = 'BINANCE') -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe - '1', '5', '15', '30', '60', '240', '1D', '1W', '1M'
            bars: Number of historical bars to fetch (default 100)
            exchange: Exchange name (default 'BINANCE')
            
        Returns:
            DataFrame with time, open, high, low, close, volume columns
        """
        self._rate_limit()
        
        try:
            data = self.client.get_historical_data(symbol, timeframe, bars, exchange)
            
            if data and 'data' in data:
                df = pd.DataFrame(data['data'])
                if not df.empty:
                    # Convert timestamp to datetime
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df = df.sort_values('time')
                    print(f"✅ Got {len(df)} historical bars for {symbol}")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting historical data: {e}")
            return pd.DataFrame()
    
    def search_symbols(self, query: str, type: str = 'crypto') -> List[Dict]:
        """Search for symbols"""
        self._rate_limit()
        
        try:
            return self.client.search_symbols(query, type)
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def close(self):
        """Close the connection"""
        if self.client:
            self.client.logout()
            print("👋 Logged out from TradingView")

# High-level analysis functions
class TradingViewAnalyzer:
    """
    High-level analysis using authenticated TradingView
    """
    
    def __init__(self, api: TradingViewAuthenticatedAPI):
        self.api = api
    
    def get_market_overview(self, symbols: List[str]) -> pd.DataFrame:
        """Get market overview for multiple symbols"""
        data = self.api.get_multiple_symbols(symbols)
        
        if not data.empty:
            # Calculate changes
            data['change_pct'] = ((data['close'] - data['open']) / data['open'] * 100)
            data['spread'] = data['high'] - data['low']
            data['spread_pct'] = (data['spread'] / data['low'] * 100)
            
            return data.sort_values('change_pct', ascending=False)
        
        return pd.DataFrame()
    
    def get_momentum_signals(self, symbol: str) -> Dict:
        """Get momentum-based trading signals"""
        
        # Get key momentum indicators
        indicators = self.api.get_technical_indicators(
            symbol,
            [
                'RSI@tv-basicstudies',
                'MACD@tv-basicstudies',
                'Mom@tv-basicstudies',
                'StochRSI@tv-basicstudies'
            ]
        )
        
        signals = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'indicators': indicators,
            'signals': []
        }
        
        # Analyze RSI
        rsi_data = indicators.get('RSI@tv-basicstudies', {})
        if rsi_data and 'data' in rsi_data:
            # Implementation would analyze the indicator data
            pass
        
        return signals
    
    def scan_for_opportunities(self, symbols: List[str],
                             criteria: Dict = None) -> pd.DataFrame:
        """
        Scan multiple symbols for trading opportunities
        
        Args:
            symbols: List of symbols to scan
            criteria: Dict with scan criteria (e.g., {'min_volume': 1000000})
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                # Get data
                ohlcv = self.api.get_ohlcv(symbol)
                
                if not ohlcv.empty:
                    row = ohlcv.iloc[-1]
                    
                    # Apply criteria
                    if criteria:
                        if 'min_volume' in criteria and row['volume'] < criteria['min_volume']:
                            continue
                        if 'min_price' in criteria and row['close'] < criteria['min_price']:
                            continue
                    
                    # Calculate metrics
                    change_pct = ((row['close'] - row['open']) / row['open'] * 100)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'price': row['close'],
                        'volume': row['volume'],
                        'change_pct': change_pct,
                        'timestamp': row['timestamp']
                    })
                
                time.sleep(0.1)  # Rate limit
                
            except Exception as e:
                print(f"⚠️ Error scanning {symbol}: {e}")
        
        if opportunities:
            df = pd.DataFrame(opportunities)
            return df.sort_values('change_pct', ascending=False)
        
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    print("🌙 TradingView Authenticated API Demo")
    print("="*60)
    
    # Initialize API
    try:
        api = TradingViewAuthenticatedAPI()
        analyzer = TradingViewAnalyzer(api)
        
        # Test 1: Single price
        print("\n📊 Test 1: Get BTC Price")
        price = api.get_price('BTCUSDT')
        if price:
            print(f"BTC Price: ${price:,.2f}")
        
        # Test 2: Multiple symbols
        print("\n📊 Test 2: Market Overview")
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        overview = analyzer.get_market_overview(symbols)
        if not overview.empty:
            print(overview[['symbol', 'close', 'change_pct', 'volume']])
        
        # Test 3: OHLCV data
        print("\n📊 Test 3: ETH OHLCV Data")
        ohlcv = api.get_ohlcv('ETHUSDT', '60')
        if not ohlcv.empty:
            print(ohlcv)
        
        # Test 4: Search
        print("\n📊 Test 4: Search for SOL")
        results = api.search_symbols('SOL')
        print(f"Found {len(results)} results")
        
        # Close
        api.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n📝 Make sure:")
        print("1. TradingView server is running")
        print("2. TV_USERNAME and TV_PASSWORD are set in .env")
        print("3. You have a valid TradingView account")