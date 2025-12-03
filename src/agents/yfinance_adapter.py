"""
🚀 YFinance Adapter for Moon Dev Trading System
Free, reliable market data - no API keys required!

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List

class YFinanceAdapter:
    """
    YFinance adapter that matches the Moon Dev API interface
    Perfect replacement for TradingView with better reliability
    """
    
    def __init__(self):
        self.name = "YFinance"
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        print(f"🚀 {self.name} Adapter initialized - no auth needed!")
        
    def _rate_limit(self):
        """Simple rate limiting to be respectful"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
            
        self.last_request_time = time.time()
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            self._rate_limit()
            
            # Convert symbol format if needed
            symbol = self._convert_symbol(symbol)
            
            ticker = yf.Ticker(symbol)
            # Get last 1 minute bar for most recent price
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"❌ YFinance price error for {symbol}: {e}")
            return None
    
    def get_funding_data(self) -> pd.DataFrame:
        """
        Get funding rates for major cryptos
        Note: YFinance doesn't have funding rates, so we'll return a placeholder
        """
        # For crypto trading, funding rates are important but not available in yfinance
        # Return empty DataFrame to maintain compatibility
        print("⚠️ Funding rates not available in YFinance (spot market only)")
        return pd.DataFrame()
    
    def get_liquidation_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Get liquidation data
        Note: YFinance doesn't have liquidation data (spot market)
        """
        print("⚠️ Liquidation data not available in YFinance (spot market only)")
        return pd.DataFrame()
    
    def get_oi_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get Open Interest data
        For stocks, we can get options OI
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'TSLA', 'AAPL']  # Default popular symbols
            
        oi_data = []
        
        for symbol in symbols:
            try:
                self._rate_limit()
                symbol = self._convert_symbol(symbol)
                
                ticker = yf.Ticker(symbol)
                
                # Get current price
                current_price = self.get_price(symbol)
                
                if current_price:
                    # Get options data if available
                    try:
                        options_dates = ticker.options
                        if options_dates:
                            # Get nearest expiry
                            nearest_date = options_dates[0]
                            opt_chain = ticker.option_chain(nearest_date)
                            
                            # Calculate total OI
                            call_oi = opt_chain.calls['openInterest'].sum()
                            put_oi = opt_chain.puts['openInterest'].sum()
                            
                            oi_data.append({
                                'symbol': symbol,
                                'price': current_price,
                                'call_oi': call_oi,
                                'put_oi': put_oi,
                                'total_oi': call_oi + put_oi,
                                'put_call_ratio': put_oi / call_oi if call_oi > 0 else 0,
                                'timestamp': datetime.now()
                            })
                    except:
                        # If no options data, just price
                        oi_data.append({
                            'symbol': symbol,
                            'price': current_price,
                            'timestamp': datetime.now()
                        })
                        
            except Exception as e:
                print(f"⚠️ OI data error for {symbol}: {e}")
                
        return pd.DataFrame(oi_data)
    
    def get_ohlcv_data(self, symbol: str, interval: str = '1h', 
                      limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data matching Moon Dev format"""
        try:
            self._rate_limit()
            
            symbol = self._convert_symbol(symbol)
            ticker = yf.Ticker(symbol)
            
            # Convert interval
            yf_interval = self._convert_interval(interval)
            
            # Calculate period based on interval and limit
            period = self._calculate_period(yf_interval, limit)
            
            # Fetch data
            data = ticker.history(period=period, interval=yf_interval)
            
            if data.empty:
                return pd.DataFrame()
                
            # Limit to requested bars
            data = data.tail(limit)
            
            # Format to match expected structure
            data = data.reset_index()
            # YFinance returns Date/Datetime as index, then Open, High, Low, Close, Volume
            # We only need these columns
            data = data[['Date' if 'Date' in data.columns else 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            return data
            
        except Exception as e:
            print(f"❌ OHLCV error for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_24h_stats(self, symbol: str) -> Dict:
        """Get 24h statistics for a symbol"""
        try:
            self._rate_limit()
            
            symbol = self._convert_symbol(symbol)
            ticker = yf.Ticker(symbol)
            
            # Get 1-day data with 5m intervals
            data = ticker.history(period="1d", interval="5m")
            
            if data.empty:
                return {}
                
            stats = {
                'symbol': symbol,
                'price': float(data['Close'].iloc[-1]),
                'open_24h': float(data['Open'].iloc[0]),
                'high_24h': float(data['High'].max()),
                'low_24h': float(data['Low'].min()),
                'volume_24h': float(data['Volume'].sum()),
                'change_24h': float((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0] * 100)
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ 24h stats error for {symbol}: {e}")
            return {}
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols efficiently"""
        try:
            # Convert symbols
            converted_symbols = [self._convert_symbol(s) for s in symbols]
            
            # Bulk download - much more efficient
            data = yf.download(
                tickers=' '.join(converted_symbols),
                period="1d",
                interval="1m",
                progress=False,
                threads=True,
                auto_adjust=True,
                prepost=True
            )
            
            prices = {}
            
            if len(converted_symbols) == 1:
                # Single symbol
                if not data.empty:
                    prices[symbols[0]] = float(data['Close'].iloc[-1])
            else:
                # Multiple symbols
                for i, symbol in enumerate(symbols):
                    try:
                        conv_symbol = converted_symbols[i]
                        # Handle MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            close_data = data['Close'][conv_symbol]
                        else:
                            close_data = data[conv_symbol]
                        
                        if not close_data.isna().all():
                            prices[symbol] = float(close_data.iloc[-1])
                    except:
                        pass
                        
            return prices
            
        except Exception as e:
            print(f"❌ Bulk price error: {e}")
            # Fallback to individual requests
            prices = {}
            for symbol in symbols:
                price = self.get_price(symbol)
                if price:
                    prices[symbol] = price
            return prices
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert crypto symbols to Yahoo format"""
        # Common crypto conversions
        crypto_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'BNBUSDT': 'BNB-USD',
            'SOLUSDT': 'SOL-USD',
            'ADAUSDT': 'ADA-USD',
            'DOGEUSDT': 'DOGE-USD',
            'MATICUSDT': 'MATIC-USD',
            'DOTUSDT': 'DOT-USD',
            'AVAXUSDT': 'AVAX-USD',
            'LINKUSDT': 'LINK-USD',
            'LTCUSDT': 'LTC-USD',
            'XRPUSDT': 'XRP-USD'
        }
        
        # Check if it's a crypto symbol
        if symbol in crypto_map:
            return crypto_map[symbol]
        elif symbol.endswith('USDT'):
            # Generic USDT conversion
            base = symbol[:-4]
            return f"{base}-USD"
        else:
            # Assume it's a stock symbol
            return symbol.upper()
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to YFinance format"""
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '60m', '60m': '60m', '1H': '60m',
            '4h': '60m', '4H': '60m',  # No 4h in yfinance
            '1d': '1d', '1D': '1d',
            '1w': '1wk', '1W': '1wk'
        }
        
        return interval_map.get(interval, '1d')
    
    def _calculate_period(self, interval: str, limit: int) -> str:
        """Calculate the period string based on interval and limit"""
        # YFinance periods
        if interval in ['1m', '2m', '5m']:
            return "7d"  # Max for these intervals
        elif interval in ['15m', '30m']:
            days = min(limit * 15 // (60 * 24), 60)
            return f"{days}d"
        elif interval in ['60m', '90m']:
            days = min(limit // 24, 730)
            return f"{days}d"
        else:  # Daily or larger
            days = min(limit * 2, 10000)
            return f"{days}d"

# Test the adapter
if __name__ == "__main__":
    adapter = YFinanceAdapter()
    
    print("\n🧪 Testing YFinance Adapter")
    print("=" * 50)
    
    # Test price
    btc_price = adapter.get_price('BTCUSDT')
    print(f"BTC Price: ${btc_price:,.2f}" if btc_price else "BTC price failed")
    
    # Test OHLCV
    tsla_data = adapter.get_ohlcv_data('TSLA', '5m', 20)
    print(f"TSLA 5m bars: {len(tsla_data)} records")
    
    # Test bulk prices
    prices = adapter.get_multiple_prices(['TSLA', 'AAPL', 'BTCUSDT', 'ETHUSDT'])
    print(f"Bulk prices: {prices}")