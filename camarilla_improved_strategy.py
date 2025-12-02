"""
🎯 Improved Camarilla Strategy with Trend Filters
Learning from failures - adding market regime detection

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
import talib

class ImprovedCamarillaStrategy(Strategy):
    """
    Camarilla strategy with improvements:
    1. Trend filter - only trade in sideways markets
    2. VWAP confirmation
    3. Better position sizing
    4. Tighter stops
    """
    
    # Parameters
    trend_threshold = 0.15  # Don't trade if trend > 15%
    rsi_period = 14
    atr_period = 14
    position_size = 0.5  # Start conservative
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # Camarilla levels
        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_range = prev_high - prev_low
        
        self.pivot = self.I(lambda: ((prev_high + prev_low + prev_close) / 3).values)
        self.r1 = self.I(lambda: (prev_close + prev_range * 1.1 / 12).values)
        self.r2 = self.I(lambda: (prev_close + prev_range * 1.1 / 6).values)
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.r4 = self.I(lambda: (prev_close + prev_range * 1.1 / 2).values)
        self.s1 = self.I(lambda: (prev_close - prev_range * 1.1 / 12).values)
        self.s2 = self.I(lambda: (prev_close - prev_range * 1.1 / 6).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        self.s4 = self.I(lambda: (prev_close - prev_range * 1.1 / 2).values)
        
        # Trend indicators
        self.sma20 = self.I(talib.SMA, close, 20)
        self.sma50 = self.I(talib.SMA, close, 50)
        self.sma200 = self.I(talib.SMA, close, 200)
        
        # VWAP approximation (daily reset)
        typical_price = (high + low + close) / 3
        
        # Since we can't properly reset VWAP daily in backtesting.py,
        # we'll use a 20-period volume-weighted moving average
        def vwma(period=20):
            tp = typical_price
            vol = volume
            
            # Volume weighted moving average
            vwma_values = []
            for i in range(len(tp)):
                if i < period - 1:
                    vwma_values.append(tp.iloc[i])
                else:
                    weights = vol.iloc[i-period+1:i+1]
                    prices = tp.iloc[i-period+1:i+1]
                    if weights.sum() > 0:
                        vwma_val = (prices * weights).sum() / weights.sum()
                    else:
                        vwma_val = prices.mean()
                    vwma_values.append(vwma_val)
            
            return np.array(vwma_values)
        
        self.vwap = self.I(vwma)
        
        # Momentum indicators
        self.rsi = self.I(talib.RSI, close, self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        
        # Trend strength - use 50-day price change
        def trend_strength():
            returns_50d = (close / close.shift(50) - 1).fillna(0)
            return returns_50d.values
        
        self.trend_strength = self.I(trend_strength)
        
        # Market regime detection - use numeric codes
        def market_regime():
            # 0 = unknown, 1 = sideways, 2 = weak_trend, 3 = strong_bullish, 4 = strong_bearish, 5 = volatile
            sma20_vals = pd.Series(self.sma20)
            sma50_vals = pd.Series(self.sma50)
            sma200_vals = pd.Series(self.sma200)
            
            regime = []
            for i in range(len(close)):
                if i < 200:
                    regime.append(0)  # unknown
                else:
                    # Check trend alignment
                    bullish = sma20_vals[i] > sma50_vals[i] > sma200_vals[i]
                    bearish = sma20_vals[i] < sma50_vals[i] < sma200_vals[i]
                    
                    # Check price movement
                    price_change_50d = abs(self.trend_strength[i])
                    
                    if price_change_50d < self.trend_threshold:
                        if not bullish and not bearish:
                            regime.append(1)  # sideways
                        else:
                            regime.append(2)  # weak_trend
                    else:
                        if bullish:
                            regime.append(3)  # strong_bullish
                        elif bearish:
                            regime.append(4)  # strong_bearish
                        else:
                            regime.append(5)  # volatile
            
            return np.array(regime)
        
        self.regime = self.I(market_regime)
        
    def next(self):
        # Wait for indicators
        if len(self.data) < 201 or self.position:
            return
        
        # Get current values
        price = self.data.Close[-1]
        regime = self.regime[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        vwap = self.vwap[-1]
        trend = self.trend_strength[-1]
        
        # Skip if indicators are invalid
        if pd.isna(rsi) or pd.isna(atr) or pd.isna(vwap):
            return
        
        # Camarilla levels
        r3 = self.r3[-1]
        r2 = self.r2[-1]
        s2 = self.s2[-1]
        s3 = self.s3[-1]
        
        # Only trade in sideways (1) or weak trend (2) markets
        if regime not in [1, 2]:
            return
        
        # Additional filter: Skip if trend too strong
        if abs(trend) > self.trend_threshold:
            return
        
        # Position sizing based on ATR
        atr_pct = atr / price
        size = self.position_size * (1 - min(atr_pct * 10, 0.5))  # Reduce size in high volatility
        
        # ENTRY RULES
        
        # Buy at S3 in sideways market
        if regime == 1 and price <= s3 * 1.002:
            if rsi < 35 and price > vwap * 0.98:  # Not too far below VWAP
                # Tighter stop - use S2 or 1.5 ATR
                sl = max(s2, price - 1.5 * atr)
                tp = r2  # More conservative target
                
                if price > sl and price < tp and (tp - price) > atr:
                    self.buy(size=size, sl=sl, tp=tp)
        
        # Sell at R3 in sideways market  
        elif regime == 1 and price >= r3 * 0.998:
            if rsi > 65 and price < vwap * 1.02:  # Not too far above VWAP
                # Tighter stop
                sl = min(r2, price + 1.5 * atr)
                tp = s2  # More conservative target
                
                if price < sl and price > tp and (price - tp) > atr:
                    self.sell(size=size, sl=sl, tp=tp)
        
        # Mean reversion from extremes in weak trends
        elif regime == 2:
            # Only take trades toward the mean
            if price < s3 and rsi < 25 and price < vwap * 0.97:
                # Oversold bounce
                sl = price - atr
                tp = vwap
                if price > sl and price < tp:
                    self.buy(size=size * 0.5, sl=sl, tp=tp)  # Smaller size
                    
            elif price > r3 and rsi > 75 and price > vwap * 1.03:
                # Overbought fade
                sl = price + atr
                tp = vwap
                if price < sl and price > tp:
                    self.sell(size=size * 0.5, sl=sl, tp=tp)  # Smaller size