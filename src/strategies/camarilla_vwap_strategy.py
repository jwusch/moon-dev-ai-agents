"""
🚀 Camarilla + VWAP Strategy
Enhanced with Volume Weighted Average Price for better entries

VWAP Benefits:
1. Shows institutional buying/selling levels
2. Acts as dynamic support/resistance
3. Helps confirm trend direction
4. Filters out false breakouts

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

class CamarillaVWAPStrategy(Strategy):
    """
    Camarilla strategy enhanced with VWAP indicator
    """
    
    # Parameters
    trend_period = 50
    atr_period = 14
    risk_per_trade = 0.02
    
    # Position sizing
    with_trend_size = 0.5      # 50% when trading with trend
    counter_trend_size = 0.1   # 10% when trading against trend
    
    def init(self):
        """Initialize indicators including VWAP"""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Calculate VWAP
        def calculate_vwap():
            """Calculate Volume Weighted Average Price"""
            # Typical price (High + Low + Close) / 3
            typical_price = (high + low + close) / 3
            
            # Create DataFrame for easier calculation
            df = pd.DataFrame({
                'typical_price': typical_price,
                'volume': volume,
                'date': pd.Series(self.data.index).dt.date
            })
            
            # Calculate VWAP for each day
            vwap_values = []
            
            for date in df['date'].unique():
                day_data = df[df['date'] == date]
                if len(day_data) > 0 and day_data['volume'].sum() > 0:
                    # VWAP = sum(price * volume) / sum(volume)
                    day_vwap = (day_data['typical_price'] * day_data['volume']).sum() / day_data['volume'].sum()
                    # Fill all values for that day with the VWAP
                    vwap_day = [day_vwap] * len(day_data)
                    vwap_values.extend(vwap_day)
                else:
                    # Use typical price if no volume
                    vwap_values.extend(day_data['typical_price'].tolist())
            
            return np.array(vwap_values)
        
        # Initialize VWAP
        self.vwap = self.I(calculate_vwap)
        
        # Camarilla levels
        prev_high = pd.Series(high).shift(1).fillna(high[0])
        prev_low = pd.Series(low).shift(1).fillna(low[0])
        prev_close = pd.Series(close).shift(1).fillna(close[0])
        prev_range = prev_high - prev_low
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.r4 = self.I(lambda: (prev_close + prev_range * 1.1 / 2).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        self.s4 = self.I(lambda: (prev_close - prev_range * 1.1 / 2).values)
        self.pivot = self.I(lambda: ((prev_high + prev_low + prev_close) / 3).values)
        
        # Trend indicators
        self.sma = self.I(talib.SMA, close, self.trend_period)
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        
        # RSI for momentum
        self.rsi = self.I(talib.RSI, close, 14)
        
    def next(self):
        """Trading logic with VWAP confirmation"""
        # Skip if not enough data
        if len(self.data) < self.trend_period:
            return
            
        # Current values
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        sma = self.sma[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        volume = self.data.Volume[-1]
        
        # Skip if invalid values
        if pd.isna(vwap) or pd.isna(sma) or pd.isna(atr):
            return
            
        # Camarilla levels
        r3 = self.r3[-1]
        r4 = self.r4[-1]
        s3 = self.s3[-1]
        s4 = self.s4[-1]
        pivot = self.pivot[-1]
        
        # Trend determination with VWAP
        is_uptrend = price > sma and price > vwap
        is_downtrend = price < sma and price < vwap
        
        # VWAP position relative to price
        above_vwap = price > vwap
        vwap_distance = abs(price - vwap) / price
        
        # Volume confirmation (above average = stronger signal)
        avg_volume = pd.Series(self.data.Volume).rolling(20).mean().iloc[-1]
        high_volume = volume > avg_volume if not pd.isna(avg_volume) else False
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        position_size = min(risk_amount / (atr * 2), self.equity * 0.9 / price)
        
        # Exit if we have a position
        if self.position:
            return
            
        # TRADING LOGIC WITH VWAP
        
        # 1. STRONG UPTREND (price > VWAP > SMA)
        if is_uptrend and vwap > sma:
            # Buy pullback to VWAP or S3
            if s3 < price <= vwap * 1.01 and rsi < 50:
                # Strong buy signal - pullback to VWAP in uptrend
                sl = min(vwap - atr, s4)
                tp = r3
                
                if sl < price < tp:
                    self.buy(size=position_size * self.with_trend_size, sl=sl, tp=tp)
                    
            # Breakout above R4 with VWAP support
            elif price > r4 and above_vwap and high_volume:
                sl = max(vwap, r3)
                tp = price + (price - sl) * 2
                
                if sl < price < tp:
                    self.buy(size=position_size * self.with_trend_size, sl=sl, tp=tp)
        
        # 2. STRONG DOWNTREND (price < VWAP < SMA)
        elif is_downtrend and vwap < sma:
            # Sell rally to VWAP or R3
            if r3 > price >= vwap * 0.99 and rsi > 50:
                sl = max(vwap + atr, r4)
                tp = s3
                
                if sl > price > tp:
                    self.sell(size=position_size * self.with_trend_size, sl=sl, tp=tp)
                    
            # Breakdown below S4 with VWAP resistance
            elif price < s4 and not above_vwap and high_volume:
                sl = min(vwap, s3)
                tp = price - (sl - price) * 2
                
                if sl > price > tp:
                    self.sell(size=position_size * self.with_trend_size, sl=sl, tp=tp)
        
        # 3. NEUTRAL/RANGE (VWAP acts as pivot)
        elif abs(price - sma) / sma < 0.02:  # Within 2% of SMA
            # Use VWAP as additional confirmation for range trades
            
            # Buy at S3 if above VWAP (bullish bias)
            if price <= s3 * 1.01 and above_vwap and rsi < 40:
                sl = s4
                tp = vwap  # Target VWAP as first target
                
                if sl < price < tp:
                    self.buy(size=position_size * 0.3, sl=sl, tp=tp)
                    
            # Sell at R3 if below VWAP (bearish bias)
            elif price >= r3 * 0.99 and not above_vwap and rsi > 60:
                sl = r4
                tp = vwap  # Target VWAP as first target
                
                if sl > price > tp:
                    self.sell(size=position_size * 0.3, sl=sl, tp=tp)
        
        # 4. VWAP MEAN REVERSION TRADES
        # Extreme distance from VWAP often reverts
        if vwap_distance > 0.03:  # More than 3% from VWAP
            if price > vwap and rsi > 70:
                # Overbought, far from VWAP - short opportunity
                sl = price + atr
                tp = vwap
                
                if sl > price > tp:
                    self.sell(size=position_size * self.counter_trend_size, sl=sl, tp=tp)
                    
            elif price < vwap and rsi < 30:
                # Oversold, far from VWAP - long opportunity
                sl = price - atr
                tp = vwap
                
                if sl < price < tp:
                    self.buy(size=position_size * self.counter_trend_size, sl=sl, tp=tp)