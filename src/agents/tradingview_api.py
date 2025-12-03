"""
🌙 TradingView Data API
Alternative data source using TradingView's technical analysis
Built with love by Moon Dev 🚀
"""

import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange
from typing import Optional, Dict, List, Union
from datetime import datetime
import time
import json
import traceback

class TradingViewAPI:
    """
    TradingView API wrapper for getting market data and technical indicators
    Uses tradingview-ta library for technical analysis
    """
    
    # Map common exchanges
    EXCHANGE_MAP = {
        'BINANCE': 'BINANCE',
        'COINBASE': 'COINBASE',
        'KRAKEN': 'KRAKEN',
        'BITFINEX': 'BITFINEX',
        'BYBIT': 'BYBIT',
        'OKX': 'OKX',
        'KUCOIN': 'KUCOIN',
        'GATEIO': 'GATEIO',
        'BINGX': 'BINGX',
        'NASDAQ': 'NASDAQ',
        'NYSE': 'NYSE',
        'FOREX': 'FX_IDC',
        'FX': 'FX_IDC',
    }
    
    # Map intervals
    INTERVAL_MAP = {
        '1m': Interval.INTERVAL_1_MINUTE,
        '5m': Interval.INTERVAL_5_MINUTES,
        '15m': Interval.INTERVAL_15_MINUTES,
        '30m': Interval.INTERVAL_30_MINUTES,
        '1h': Interval.INTERVAL_1_HOUR,
        '2h': Interval.INTERVAL_2_HOURS,
        '4h': Interval.INTERVAL_4_HOURS,
        '1d': Interval.INTERVAL_1_DAY,
        '1w': Interval.INTERVAL_1_WEEK,
        '1M': Interval.INTERVAL_1_MONTH,
    }
    
    def __init__(self, screener="crypto", default_exchange="BINANCE"):
        """
        Initialize TradingView API
        
        Args:
            screener: Market screener (crypto, forex, stocks, cfd)
            default_exchange: Default exchange to use
        """
        self.screener = screener
        self.default_exchange = default_exchange
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests
        print(f"🌙 TradingView API initialized!")
        print(f"📊 Screener: {screener}, Default Exchange: {default_exchange}")
    
    def get_analysis(self, symbol: str, exchange: Optional[str] = None, 
                    interval: str = '1h') -> Optional[Dict]:
        """
        Get complete technical analysis for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT", "AAPL")
            exchange: Exchange name (optional, uses default if not specified)
            interval: Time interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1M)
            
        Returns:
            Dictionary with price data, indicators, and analysis
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                time.sleep(sleep_time)
            
            # Use default exchange if not specified
            if not exchange:
                exchange = self.default_exchange
            
            # Convert exchange name to TradingView format
            tv_exchange = self.EXCHANGE_MAP.get(exchange.upper(), exchange)
            
            # Convert interval to TradingView format
            tv_interval = self.INTERVAL_MAP.get(interval, Interval.INTERVAL_1_HOUR)
            
            print(f"🔍 Fetching analysis for {symbol} on {tv_exchange} ({interval})...")
            
            # Create handler
            handler = TA_Handler(
                symbol=symbol,
                screener=self.screener,
                exchange=tv_exchange,
                interval=tv_interval
            )
            
            # Get analysis
            analysis = handler.get_analysis()
            self.last_request_time = time.time()
            
            # Extract all data
            result = {
                'symbol': symbol,
                'exchange': exchange,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                
                # Summary
                'summary': {
                    'RECOMMENDATION': analysis.summary.get('RECOMMENDATION'),
                    'BUY': analysis.summary.get('BUY', 0),
                    'SELL': analysis.summary.get('SELL', 0),
                    'NEUTRAL': analysis.summary.get('NEUTRAL', 0),
                },
                
                # Moving Averages
                'moving_averages': {
                    'RECOMMENDATION': analysis.moving_averages.get('RECOMMENDATION'),
                    'BUY': analysis.moving_averages.get('BUY', 0),
                    'SELL': analysis.moving_averages.get('SELL', 0),
                    'NEUTRAL': analysis.moving_averages.get('NEUTRAL', 0),
                },
                
                # Oscillators
                'oscillators': {
                    'RECOMMENDATION': analysis.oscillators.get('RECOMMENDATION'),
                    'BUY': analysis.oscillators.get('BUY', 0),
                    'SELL': analysis.oscillators.get('SELL', 0),
                    'NEUTRAL': analysis.oscillators.get('NEUTRAL', 0),
                },
                
                # Indicators (all available)
                'indicators': analysis.indicators,
            }
            
            print(f"✅ Analysis complete: {result['summary']['RECOMMENDATION']}")
            return result
            
        except Exception as e:
            print(f"❌ Error getting analysis: {e}")
            traceback.print_exc()
            return None
    
    def get_indicators(self, symbol: str, exchange: Optional[str] = None,
                      interval: str = '1h') -> Optional[Dict]:
        """
        Get only indicator values for a symbol
        
        Returns:
            Dictionary of indicator values
        """
        analysis = self.get_analysis(symbol, exchange, interval)
        if analysis:
            return analysis['indicators']
        return None
    
    def get_multiple_analyses(self, symbols: List[str], exchange: Optional[str] = None,
                            interval: str = '1h') -> Dict[str, Dict]:
        """
        Get analysis for multiple symbols
        
        Args:
            symbols: List of symbols
            exchange: Exchange name
            interval: Time interval
            
        Returns:
            Dictionary mapping symbol to analysis
        """
        results = {}
        
        for symbol in symbols:
            try:
                analysis = self.get_analysis(symbol, exchange, interval)
                if analysis:
                    results[symbol] = analysis
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_recommendation(self, symbol: str, exchange: Optional[str] = None,
                          interval: str = '1h') -> Optional[str]:
        """
        Get simple buy/sell/neutral recommendation
        
        Returns:
            'BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL', or 'NEUTRAL'
        """
        analysis = self.get_analysis(symbol, exchange, interval)
        if analysis:
            return analysis['summary']['RECOMMENDATION']
        return None
    
    def get_price(self, symbol: str, exchange: Optional[str] = None,
                  interval: str = '1h') -> Optional[float]:
        """
        Get current price (close price)
        
        Returns:
            Current close price as float
        """
        indicators = self.get_indicators(symbol, exchange, interval)
        if indicators:
            return indicators.get('close')
        return None
    
    def get_price_data(self, symbol: str, exchange: Optional[str] = None,
                      interval: str = '1h') -> Optional[Dict]:
        """
        Get current price data from indicators
        
        Returns:
            Dictionary with open, high, low, close, volume
        """
        indicators = self.get_indicators(symbol, exchange, interval)
        if indicators:
            return {
                'open': indicators.get('open'),
                'high': indicators.get('high'),
                'low': indicators.get('low'),
                'close': indicators.get('close'),
                'volume': indicators.get('volume'),
                'change': indicators.get('change'),
                'change_percent': indicators.get('change%'),
            }
        return None
    
    def get_momentum_indicators(self, symbol: str, exchange: Optional[str] = None,
                               interval: str = '1h') -> Optional[Dict]:
        """
        Get momentum indicators (RSI, MACD, etc.)
        """
        indicators = self.get_indicators(symbol, exchange, interval)
        if indicators:
            return {
                'RSI': indicators.get('RSI'),
                'RSI[1]': indicators.get('RSI[1]'),
                'MACD.macd': indicators.get('MACD.macd'),
                'MACD.signal': indicators.get('MACD.signal'),
                'Mom': indicators.get('Mom'),
                'CCI20': indicators.get('CCI20'),
                'Stoch.K': indicators.get('Stoch.K'),
                'Stoch.D': indicators.get('Stoch.D'),
                'Stoch.RSI.K': indicators.get('Stoch.RSI.K'),
                'W%R': indicators.get('W%R'),
                'UO': indicators.get('UO'),
            }
        return None
    
    def get_trend_indicators(self, symbol: str, exchange: Optional[str] = None,
                            interval: str = '1h') -> Optional[Dict]:
        """
        Get trend indicators (Moving Averages, Bollinger Bands, etc.)
        """
        indicators = self.get_indicators(symbol, exchange, interval)
        if indicators:
            return {
                # Simple Moving Averages
                'SMA5': indicators.get('SMA5'),
                'SMA10': indicators.get('SMA10'),
                'SMA20': indicators.get('SMA20'),
                'SMA50': indicators.get('SMA50'),
                'SMA100': indicators.get('SMA100'),
                'SMA200': indicators.get('SMA200'),
                
                # Exponential Moving Averages
                'EMA5': indicators.get('EMA5'),
                'EMA10': indicators.get('EMA10'),
                'EMA20': indicators.get('EMA20'),
                'EMA50': indicators.get('EMA50'),
                'EMA100': indicators.get('EMA100'),
                'EMA200': indicators.get('EMA200'),
                
                # Other trend indicators
                'ADX': indicators.get('ADX'),
                'ADX+DI': indicators.get('ADX+DI'),
                'ADX-DI': indicators.get('ADX-DI'),
                'AO': indicators.get('AO'),
                'P.SAR': indicators.get('P.SAR'),
                
                # Bollinger Bands
                'BB.upper': indicators.get('BB.upper'),
                'BB.lower': indicators.get('BB.lower'),
                
                # Ichimoku
                'Ichimoku.BLine': indicators.get('Ichimoku.BLine'),
                'Ichimoku.CLine': indicators.get('Ichimoku.CLine'),
            }
        return None
    
    def get_volatility_indicators(self, symbol: str, exchange: Optional[str] = None,
                                 interval: str = '1h') -> Optional[Dict]:
        """
        Get volatility indicators (ATR, Bollinger Bands width, etc.)
        """
        indicators = self.get_indicators(symbol, exchange, interval)
        if indicators:
            bb_upper = indicators.get('BB.upper')
            bb_lower = indicators.get('BB.lower')
            bb_width = None
            if bb_upper and bb_lower:
                bb_width = bb_upper - bb_lower
                
            return {
                'ATR': indicators.get('ATR'),
                'BB.width': bb_width,
                'high': indicators.get('high'),
                'low': indicators.get('low'),
                'close': indicators.get('close'),
            }
        return None

# Create a function similar to Binance API for compatibility
def get_tradingview_data(symbol: str, interval: str = '1h', 
                        exchange: str = 'BINANCE') -> pd.DataFrame:
    """
    Get TradingView data in a format similar to Binance OHLCV
    
    Returns:
        DataFrame with technical analysis data
    """
    api = TradingViewAPI()
    analysis = api.get_analysis(symbol, exchange, interval)
    
    if analysis:
        # Create DataFrame from analysis
        df = pd.DataFrame([{
            'timestamp': analysis['timestamp'],
            'open': analysis['indicators'].get('open'),
            'high': analysis['indicators'].get('high'),
            'low': analysis['indicators'].get('low'),
            'close': analysis['indicators'].get('close'),
            'volume': analysis['indicators'].get('volume'),
            'recommendation': analysis['summary']['RECOMMENDATION'],
            'rsi': analysis['indicators'].get('RSI'),
            'macd': analysis['indicators'].get('MACD.macd'),
            'signal': analysis['indicators'].get('MACD.signal'),
        }])
        return df
    
    return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Initialize API
    tv = TradingViewAPI()
    
    # Test 1: Get BTC analysis
    print("\n📊 Test 1: BTC Technical Analysis")
    btc_analysis = tv.get_analysis('BTCUSDT', 'BINANCE', '1h')
    if btc_analysis:
        print(f"Recommendation: {btc_analysis['summary']['RECOMMENDATION']}")
        print(f"Buy signals: {btc_analysis['summary']['BUY']}")
        print(f"Sell signals: {btc_analysis['summary']['SELL']}")
        print(f"Current price: ${btc_analysis['indicators'].get('close', 'N/A'):,.2f}")
        print(f"RSI: {btc_analysis['indicators'].get('RSI', 'N/A')}")
    
    # Test 2: Multiple symbols
    print("\n📊 Test 2: Multiple Symbols")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    results = tv.get_multiple_analyses(symbols, 'BINANCE', '1h')
    
    for symbol, analysis in results.items():
        if analysis:
            print(f"\n{symbol}:")
            print(f"  Recommendation: {analysis['summary']['RECOMMENDATION']}")
            print(f"  Price: ${analysis['indicators'].get('close', 0):,.2f}")
            print(f"  RSI: {analysis['indicators'].get('RSI', 'N/A')}")
    
    # Test 3: Different intervals
    print("\n📊 Test 3: Different Intervals for BTC")
    intervals = ['5m', '1h', '1d']
    for interval in intervals:
        rec = tv.get_recommendation('BTCUSDT', 'BINANCE', interval)
        print(f"{interval}: {rec}")
    
    # Test 4: Stock market
    print("\n📊 Test 4: Stock Market (AAPL)")
    tv_stocks = TradingViewAPI(screener="america", default_exchange="NASDAQ")
    aapl = tv_stocks.get_analysis('AAPL', 'NASDAQ', '1d')
    if aapl:
        print(f"AAPL Recommendation: {aapl['summary']['RECOMMENDATION']}")
        print(f"Price: ${aapl['indicators'].get('close', 0):,.2f}")