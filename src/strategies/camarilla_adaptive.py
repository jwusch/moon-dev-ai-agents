"""
🚀 Adaptive Camarilla Strategy
Automatically adapts between range-trading and trend-following based on market conditions

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
import talib

class AdaptiveCamarillaStrategy(Strategy):
    """
    Adaptive strategy that switches between Camarilla range trading 
    and trend following based on market conditions
    """
    
    # Core parameters
    atr_period = 14
    trend_period = 50
    rsi_period = 14
    
    # Risk management
    risk_per_trade = 0.02  # Risk 2% per trade
    
    def init(self):
        """Initialize all indicators"""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Calculate Camarilla levels (using previous day's range)
        def camarilla_levels(h, l, c):
            """Calculate Camarilla pivot points"""
            # Create arrays for calculations
            h_arr = pd.Series(h)
            l_arr = pd.Series(l)
            c_arr = pd.Series(c)
            
            # Previous day values
            prev_h = h_arr.shift(1).fillna(h_arr.iloc[0])
            prev_l = l_arr.shift(1).fillna(l_arr.iloc[0])
            prev_c = c_arr.shift(1).fillna(c_arr.iloc[0])
            
            # Range
            range_val = prev_h - prev_l
            
            # Pivot
            pivot = (prev_h + prev_l + prev_c) / 3
            
            # Resistance levels
            r1 = prev_c + range_val * 1.1 / 12
            r2 = prev_c + range_val * 1.1 / 6
            r3 = prev_c + range_val * 1.1 / 4
            r4 = prev_c + range_val * 1.1 / 2
            
            # Support levels  
            s1 = prev_c - range_val * 1.1 / 12
            s2 = prev_c - range_val * 1.1 / 6
            s3 = prev_c - range_val * 1.1 / 4
            s4 = prev_c - range_val * 1.1 / 2
            
            return pivot.values, r3.values, r4.values, s3.values, s4.values
        
        # Calculate levels
        pivot, r3, r4, s3, s4 = camarilla_levels(high, low, close)
        
        self.pivot = self.I(lambda: pivot)
        self.r3 = self.I(lambda: r3)
        self.r4 = self.I(lambda: r4)
        self.s3 = self.I(lambda: s3)
        self.s4 = self.I(lambda: s4)
        
        # Trend and momentum indicators
        self.sma = self.I(talib.SMA, close, self.trend_period)
        self.rsi = self.I(talib.RSI, close, self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        
        # Market regime detection
        self.adx = self.I(talib.ADX, high, low, close, 14)
        
    def next(self):
        """Trading logic"""
        # Skip if not enough data
        if len(self.data) < self.trend_period:
            return
            
        # Current values
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        sma = self.sma[-1]
        adx = self.adx[-1] if len(self.adx) > 0 else 20
        
        # Skip if invalid values
        if pd.isna(atr) or pd.isna(sma) or pd.isna(rsi):
            return
            
        # Current Camarilla levels
        r3 = self.r3[-1]
        r4 = self.r4[-1]
        s3 = self.s3[-1]
        s4 = self.s4[-1]
        pivot = self.pivot[-1]
        
        # Market regime (trending vs ranging)
        is_trending = adx > 25  # ADX > 25 indicates trend
        is_uptrend = price > sma
        is_downtrend = price < sma
        
        # Position sizing based on ATR
        stop_distance = atr * 2  # 2 ATR stop
        position_size = (self.equity * self.risk_per_trade) / stop_distance
        position_size = min(position_size, self.equity * 0.9 / price)
        
        # Exit if we have a position (simple exit logic)
        if self.position:
            return
            
        # TRENDING MARKET STRATEGY
        if is_trending:
            if is_uptrend and rsi < 70:
                # Trend continuation buy
                if price > r4:  # Breakout above R4
                    self.buy(
                        size=position_size,
                        sl=r3,  # Stop at R3
                        tp=price + (price - r3) * 2  # 2:1 RR
                    )
                elif s3 < price < pivot and rsi < 50:  # Pullback buy
                    self.buy(
                        size=position_size,
                        sl=s4,
                        tp=r3
                    )
                    
            elif is_downtrend and rsi > 30:
                # Trend continuation sell
                if price < s4:  # Breakdown below S4
                    self.sell(
                        size=position_size,
                        sl=s3,  # Stop at S3
                        tp=price - (s3 - price) * 2  # 2:1 RR
                    )
                elif pivot < price < r3 and rsi > 50:  # Pullback sell
                    self.sell(
                        size=position_size,
                        sl=r4,
                        tp=s3
                    )
                    
        # RANGING MARKET STRATEGY
        else:
            # Buy at support
            if price <= s3 and price > s4 and rsi < 40:
                self.buy(
                    size=position_size * 0.5,  # Half size for range trades
                    sl=s4 - atr,
                    tp=pivot
                )
                
            # Sell at resistance
            elif price >= r3 and price < r4 and rsi > 60:
                self.sell(
                    size=position_size * 0.5,
                    sl=r4 + atr,
                    tp=pivot
                )
                
            # Fade breakout attempts in ranging market
            elif price > r4 and rsi > 75 and not is_trending:
                # False breakout - fade it
                self.sell(
                    size=position_size * 0.3,  # Small size
                    sl=price + atr,
                    tp=r3
                )
                
            elif price < s4 and rsi < 25 and not is_trending:
                # False breakdown - fade it
                self.buy(
                    size=position_size * 0.3,
                    sl=price - atr,
                    tp=s3
                )