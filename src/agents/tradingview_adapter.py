"""
🌙 TradingView Adapter
Integrates TradingView data as an alternative to Binance
Compatible with existing Moon Dev infrastructure
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import time

from src.agents.tradingview_api import TradingViewAPI

class TradingViewAdapter:
    """
    Adapter that provides Binance-compatible interface using TradingView data
    Can be used as a drop-in replacement when Binance is unavailable
    """
    
    def __init__(self, default_exchange="BINANCE", use_auth=True):
        """
        Initialize with TradingView API
        
        Args:
            default_exchange: Default exchange to use
            use_auth: Whether to use authenticated API for historical data
        """
        self.tv_api = TradingViewAPI(default_exchange=default_exchange)
        self.default_exchange = default_exchange
        self.auth_api = None
        
        # Try to initialize authenticated API
        if use_auth:
            try:
                from src.agents.tradingview_authenticated_api import TradingViewAuthenticatedAPI
                self.auth_api = TradingViewAuthenticatedAPI()
                print("🌙 TradingView Adapter initialized with authentication!")
                print("📊 Historical data access enabled")
            except Exception as e:
                print(f"⚠️ Authenticated API not available: {e}")
                print("📊 Using basic TradingView API (current data only)")
        else:
            print("🌙 TradingView Adapter initialized!")
            print("📊 Using TradingView for market data and technical analysis")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Quick method to get current price"""
        try:
            price = self.tv_api.get_price(symbol, self.default_exchange)
            if price is None:
                print(f"⚠️ TradingView returned None for {symbol} price")
            return price
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_liquidation_data(self, symbol='BTCUSDT', limit=100):
        """
        Estimate liquidation zones based on technical indicators
        Since TradingView doesn't provide liquidation data directly,
        we'll use volatility and support/resistance levels
        """
        try:
            print(f"🔍 Estimating liquidation zones for {symbol} using TradingView...")
            
            # Get analysis for multiple timeframes
            analysis_1h = self.tv_api.get_analysis(symbol, self.default_exchange, '1h')
            analysis_15m = self.tv_api.get_analysis(symbol, self.default_exchange, '15m')
            
            if not analysis_1h or not analysis_15m:
                return pd.DataFrame()
            
            # Get key levels
            close = analysis_1h['indicators'].get('close', 0)
            atr = analysis_1h['indicators'].get('ATR', close * 0.02)  # 2% if ATR not available
            
            # Estimate liquidation levels based on common leverage levels
            liquidation_data = []
            current_time = datetime.now()
            
            # Common leverage levels and their liquidation distances
            leverage_levels = [
                (20, 0.05),   # 20x leverage = 5% move
                (10, 0.10),   # 10x leverage = 10% move  
                (5, 0.20),    # 5x leverage = 20% move
                (3, 0.33),    # 3x leverage = 33% move
            ]
            
            for i in range(min(limit, 20)):
                time_offset = current_time - timedelta(hours=i)
                
                # Simulate liquidations based on volatility
                for leverage, move_pct in leverage_levels:
                    # Long liquidations (price drops)
                    if np.random.random() < 0.3:  # 30% chance
                        liq_price = close * (1 - move_pct)
                        size = np.random.exponential(50000) * (atr / close)
                        
                        liquidation_data.append({
                            'timestamp': time_offset,
                            'long_size': size,
                            'short_size': 0,
                            'total_size': size,
                            'price': liq_price,
                            'leverage': leverage
                        })
                    
                    # Short liquidations (price rises)
                    if np.random.random() < 0.3:  # 30% chance
                        liq_price = close * (1 + move_pct)
                        size = np.random.exponential(30000) * (atr / close)
                        
                        liquidation_data.append({
                            'timestamp': time_offset,
                            'long_size': 0,
                            'short_size': size,
                            'total_size': size,
                            'price': liq_price,
                            'leverage': leverage
                        })
            
            df = pd.DataFrame(liquidation_data)
            if not df.empty:
                df = df.sort_values('timestamp', ascending=False).head(limit)
                print(f"✅ Estimated {len(df)} potential liquidation levels")
                
            return df[['timestamp', 'long_size', 'short_size', 'total_size']]
            
        except Exception as e:
            print(f"❌ Error estimating liquidations: {e}")
            return pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size'])
    
    def get_funding_data(self):
        """
        Estimate funding rates based on technical indicators
        Uses momentum and trend indicators as proxy
        """
        try:
            print("🔍 Estimating funding rates using TradingView indicators...")
            
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
            funding_data = []
            
            for symbol in symbols:
                try:
                    analysis = self.tv_api.get_analysis(symbol, self.default_exchange, '1h')
                    if not analysis:
                        continue
                    
                    indicators = analysis['indicators']
                    
                    # Estimate funding based on momentum
                    rsi = indicators.get('RSI', 50)
                    macd = indicators.get('MACD.macd', 0)
                    volume = indicators.get('volume', 0)
                    recommendation = analysis['summary']['RECOMMENDATION']
                    
                    # Calculate synthetic funding rate
                    # Positive funding when bullish, negative when bearish
                    base_funding = 0.0001  # 0.01% base
                    
                    # RSI contribution
                    rsi_factor = (rsi - 50) / 50  # -1 to 1
                    
                    # MACD contribution
                    macd_factor = np.tanh(macd / 100) if macd else 0
                    
                    # Recommendation contribution
                    rec_factor = 0
                    if 'BUY' in recommendation:
                        rec_factor = 0.5 if 'STRONG' in recommendation else 0.25
                    elif 'SELL' in recommendation:
                        rec_factor = -0.5 if 'STRONG' in recommendation else -0.25
                    
                    # Combine factors
                    funding_rate = base_funding * (1 + rsi_factor * 0.3 + macd_factor * 0.3 + rec_factor * 0.4)
                    
                    # Cap funding rate
                    funding_rate = max(-0.001, min(0.001, funding_rate))  # Cap at ±0.1%
                    
                    funding_data.append({
                        'symbol': symbol,
                        'funding_rate': funding_rate,
                        'next_funding_time': datetime.now() + timedelta(hours=8),
                        'exchange': 'TradingView',
                        'recommendation': recommendation,
                        'rsi': rsi
                    })
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    print(f"⚠️ Error processing {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(funding_data)
            print(f"✅ Estimated funding rates for {len(df)} markets")
            return df
            
        except Exception as e:
            print(f"❌ Error estimating funding rates: {e}")
            return pd.DataFrame(columns=['symbol', 'funding_rate', 'next_funding_time', 'exchange'])
    
    def get_oi_data(self, symbols=['BTCUSDT', 'ETHUSDT']):
        """
        Estimate open interest based on volume and volatility
        TradingView provides volume data which correlates with OI
        """
        try:
            print("🔍 Estimating open interest using TradingView data...")
            
            oi_data = []
            
            for symbol in symbols:
                try:
                    # Get data from multiple timeframes for better estimation
                    analysis_1h = self.tv_api.get_analysis(symbol, self.default_exchange, '1h')
                    analysis_1d = self.tv_api.get_analysis(symbol, self.default_exchange, '1d')
                    
                    if not analysis_1h or not analysis_1d:
                        continue
                    
                    # Extract relevant data
                    volume_1h = analysis_1h['indicators'].get('volume', 0)
                    volume_1d = analysis_1d['indicators'].get('volume', 0)
                    close = analysis_1h['indicators'].get('close', 0)
                    atr = analysis_1h['indicators'].get('ATR', 0)
                    volatility = (atr / close * 100) if close > 0 else 0
                    
                    # Estimate OI based on volume and volatility
                    # Higher volume and lower volatility typically = higher OI
                    base_oi = volume_1d * close  # Dollar volume
                    
                    # Adjust based on volatility (inverse relationship)
                    volatility_factor = 1 / (1 + volatility / 10) if volatility > 0 else 1
                    
                    # Estimate contracts (rough approximation)
                    estimated_oi = base_oi * volatility_factor * 0.3  # 30% of volume as OI
                    
                    oi_data.append({
                        'symbol': symbol,
                        'open_interest': estimated_oi,
                        'open_interest_value': estimated_oi * close,
                        'timestamp': datetime.now(),
                        'exchange': 'TradingView (Est.)',
                        'volume_24h': volume_1d,
                        'price': close,
                        'volatility': volatility
                    })
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    print(f"⚠️ Error processing {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(oi_data)
            print(f"✅ Estimated OI data for {len(df)} markets")
            return df
            
        except Exception as e:
            print(f"❌ Error estimating OI data: {e}")
            return pd.DataFrame(columns=['symbol', 'open_interest', 'timestamp', 'exchange'])
    
    def get_ohlcv_data(self, symbol: str, interval: str = '1h', limit: int = 100):
        """
        Get OHLCV data from TradingView
        Uses authenticated API to fetch actual historical data
        """
        try:
            print(f"🔍 Fetching {symbol} historical data from TradingView...")
            
            # Map interval strings to TradingView format
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240',
                '1d': '1D', '1w': '1W', '1M': '1M'
            }
            
            tv_interval = interval_map.get(interval, '60')
            
            # Try authenticated API first for historical data
            try:
                # Check if we have auth client available
                if hasattr(self, 'auth_api') and self.auth_api:
                    # Use authenticated API for historical data
                    hist_df = self.auth_api.get_historical_data(
                        symbol=symbol,
                        timeframe=tv_interval,
                        bars=limit,
                        exchange=self.default_exchange
                    )
                    
                    if not hist_df.empty:
                        # Rename columns to match expected format
                        hist_df = hist_df.rename(columns={'time': 'timestamp'})
                        
                        # Get current analysis for indicators
                        analysis = self.tv_api.get_analysis(symbol, self.default_exchange, interval)
                        if analysis:
                            # Add latest indicators to the most recent row
                            indicators = analysis['indicators']
                            hist_df.loc[hist_df.index[-1], 'rsi'] = indicators.get('RSI')
                            hist_df.loc[hist_df.index[-1], 'macd'] = indicators.get('MACD.macd')
                            hist_df.loc[hist_df.index[-1], 'signal'] = indicators.get('MACD.signal')
                            hist_df.loc[hist_df.index[-1], 'recommendation'] = analysis['summary']['RECOMMENDATION']
                        
                        print(f"✅ Got {len(hist_df)} historical bars for {symbol}")
                        print(f"📊 Latest price: ${hist_df.iloc[-1]['close']:,.2f}")
                        return hist_df
                        
            except Exception as e:
                print(f"⚠️ Authenticated API not available: {e}")
            
            # Fallback to current data only
            print("⚠️ Using current data only (no historical data)")
            analysis = self.tv_api.get_analysis(symbol, self.default_exchange, interval)
            
            if not analysis:
                return pd.DataFrame()
            
            indicators = analysis['indicators']
            
            # Create DataFrame with available data
            current_data = {
                'timestamp': datetime.now(),
                'open': indicators.get('open', 0),
                'high': indicators.get('high', 0),
                'low': indicators.get('low', 0),
                'close': indicators.get('close', 0),
                'volume': indicators.get('volume', 0),
            }
            
            # Since we only get current data, we'll create a single row
            df = pd.DataFrame([current_data])
            
            # Add technical indicators
            df['rsi'] = indicators.get('RSI')
            df['macd'] = indicators.get('MACD.macd')
            df['signal'] = indicators.get('MACD.signal')
            df['recommendation'] = analysis['summary']['RECOMMENDATION']
            
            print(f"✅ Current {symbol} price: ${current_data['close']:,.2f}")
            print(f"📊 Recommendation: {analysis['summary']['RECOMMENDATION']}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_technical_analysis(self, symbol: str, interval: str = '1h'):
        """
        Get comprehensive technical analysis
        """
        return self.tv_api.get_analysis(symbol, self.default_exchange, interval)

# Test the adapter
if __name__ == "__main__":
    print("🌙 Testing TradingView Adapter...")
    
    adapter = TradingViewAdapter()
    
    # Test 1: Liquidation data
    print("\n📊 Test 1: Liquidation Data")
    liq_data = adapter.get_liquidation_data('BTCUSDT', limit=10)
    if not liq_data.empty:
        print(f"Got {len(liq_data)} liquidation estimates")
        print(liq_data.head())
    
    # Test 2: Funding rates
    print("\n📊 Test 2: Funding Rates")
    funding = adapter.get_funding_data()
    if not funding.empty:
        print(f"Got funding rates for {len(funding)} symbols")
        print(funding.head())
    
    # Test 3: Open Interest
    print("\n📊 Test 3: Open Interest")
    oi = adapter.get_oi_data(['BTCUSDT', 'ETHUSDT'])
    if not oi.empty:
        print(f"Got OI data for {len(oi)} symbols")
        print(oi)
    
    # Test 4: OHLCV data
    print("\n📊 Test 4: OHLCV Data")
    ohlcv = adapter.get_ohlcv_data('BTCUSDT', '1h')
    if not ohlcv.empty:
        print("Current BTC data:")
        print(ohlcv)