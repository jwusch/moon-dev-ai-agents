"""
🚀 YFinance API Adapter
Free, reliable market data without authentication hassles

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np

class YFinanceAPI:
    """YFinance adapter matching TradingView API interface"""
    
    def __init__(self):
        """Initialize YFinance API adapter"""
        self.name = "YFinance"
        print(f"✅ {self.name} API initialized (no auth required!)")
        
    def get_price(self, symbol: str, exchange: Optional[str] = None) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            # Get last 1 minute bar for most recent price
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"❌ Price fetch error for {symbol}: {e}")
            return None
    
    def get_indicators(self, symbol: str, exchange: Optional[str] = None,
                      interval: str = '1h') -> Optional[Dict]:
        """Get current OHLCV + basic indicators"""
        try:
            # Convert interval format (1h -> 60m for yfinance)
            yf_interval = self._convert_interval(interval)
            
            ticker = yf.Ticker(symbol)
            
            # Get appropriate period based on interval
            if yf_interval in ['1m', '2m', '5m', '15m', '30m']:
                period = "5d"
            elif yf_interval in ['60m', '90m', '1h']:
                period = "60d"
            else:
                period = "6mo"
                
            data = ticker.history(period=period, interval=yf_interval)
            
            if data.empty:
                return None
                
            # Latest values
            latest = data.iloc[-1]
            
            # Calculate basic indicators
            close_series = data['Close']
            
            # Simple moving averages
            sma_20 = close_series.rolling(window=20).mean().iloc[-1] if len(close_series) >= 20 else latest['Close']
            sma_50 = close_series.rolling(window=50).mean().iloc[-1] if len(close_series) >= 50 else latest['Close']
            
            # RSI calculation
            rsi = self._calculate_rsi(close_series, 14)
            
            # VWAP for the day
            vwap = self._calculate_vwap(data)
            
            return {
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': float(latest['Volume']),
                'change': float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'rsi': float(rsi),
                'vwap': float(vwap),
                'timestamp': latest.name.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Indicators error for {symbol}: {e}")
            return None
    
    def get_history(self, symbol: str, interval: str = '1D', 
                   bars: int = 100, exchange: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        try:
            # Convert interval
            yf_interval = self._convert_interval(interval)
            
            ticker = yf.Ticker(symbol)
            
            # Calculate period based on bars and interval
            if yf_interval in ['1m', '2m', '5m']:
                # Limited to 7 days for these intervals
                max_bars = 7 * 390  # 390 bars per day (6.5 hours * 60 min)
                bars = min(bars, max_bars)
                period = "7d"
            elif yf_interval in ['15m', '30m']:
                period = "60d"
            elif yf_interval in ['60m', '1h', '90m']:
                period = "730d"
            else:
                # Daily or larger
                days = bars * (5 if interval == '1d' else 30)
                period = f"{min(days, 10000)}d"
                
            # Fetch data
            data = ticker.history(period=period, interval=yf_interval)
            
            if data.empty:
                return None
                
            # Limit to requested bars
            data = data.tail(bars)
            
            # Format to match expected structure
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            return data
            
        except Exception as e:
            print(f"❌ History error for {symbol}: {e}")
            return None
    
    def get_multiple_symbols(self, symbols: List[str], interval: str = '5m',
                           period: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
        try:
            # Bulk download
            yf_interval = self._convert_interval(interval)
            
            data = yf.download(
                tickers=' '.join(symbols),
                period=period,
                interval=yf_interval,
                group_by='ticker',
                progress=False,
                threads=True
            )
            
            result = {}
            
            if len(symbols) == 1:
                # Single symbol returns different structure
                if not data.empty:
                    df = data.reset_index()
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    result[symbols[0]] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            else:
                # Multiple symbols
                for symbol in symbols:
                    try:
                        symbol_data = data[symbol]
                        if not symbol_data['Close'].isna().all():
                            df = symbol_data.reset_index()
                            df.columns = ['timestamp', 'adj_close', 'close', 'high', 'low', 'open', 'volume']
                            result[symbol] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    except:
                        print(f"⚠️ No data for {symbol}")
                        
            return result
            
        except Exception as e:
            print(f"❌ Bulk download error: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get detailed quote information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current data
            current = ticker.history(period="1d", interval="1m")
            
            if current.empty:
                return None
                
            latest = current.iloc[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(info.get('regularMarketOpen', latest['Open'])),
                'high': float(info.get('regularMarketDayHigh', 0)),
                'low': float(info.get('regularMarketDayLow', 0)),
                'volume': int(info.get('regularMarketVolume', 0)),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'name': info.get('longName', symbol)
            }
            
        except Exception as e:
            print(f"❌ Quote error for {symbol}: {e}")
            return None
    
    def search_symbol(self, query: str) -> List[Dict]:
        """Search for symbols (basic implementation)"""
        # YFinance doesn't have native search, but we can validate symbols
        test_symbols = []
        
        # Common variations
        if query.upper() != query:
            test_symbols.append(query.upper())
        
        # Add common suffixes for international markets
        for suffix in ['', '.L', '.TO', '.AX', '.HK']:
            test_symbols.append(f"{query.upper()}{suffix}")
            
        results = []
        
        for symbol in test_symbols[:5]:  # Limit to 5 attempts
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if 'symbol' in info or 'longName' in info:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'exchange': info.get('exchange', 'Unknown'),
                        'type': info.get('quoteType', 'Unknown')
                    })
                    
            except:
                pass
                
        return results
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to yfinance format"""
        interval_map = {
            '1': '1m', '1m': '1m',
            '5': '5m', '5m': '5m',
            '15': '15m', '15m': '15m',
            '30': '30m', '30m': '30m',
            '60': '60m', '1h': '60m', '1H': '60m',
            '240': '60m', '4h': '60m', '4H': '60m',  # No 4h, use 60m
            '1D': '1d', '1d': '1d', 'D': '1d',
            '1W': '1wk', '1w': '1wk', 'W': '1wk',
            '1M': '1mo', '1mo': '1mo', 'M': '1mo'
        }
        
        return interval_map.get(interval, '1d')
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate VWAP for the latest day"""
        try:
            # Get today's data
            today = data.index.date[-1]
            today_data = data[data.index.date == today]
            
            if today_data.empty:
                return float(data['Close'].iloc[-1])
                
            # VWAP = sum(price * volume) / sum(volume)
            typical_price = (today_data['High'] + today_data['Low'] + today_data['Close']) / 3
            vwap = (typical_price * today_data['Volume']).sum() / today_data['Volume'].sum()
            
            return float(vwap)
            
        except:
            return float(data['Close'].iloc[-1])