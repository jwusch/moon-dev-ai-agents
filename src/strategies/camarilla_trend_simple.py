"""
🚀 Simplified Camarilla Trend Strategy for TSLA
Focus on key improvements that work for trending stocks

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
import talib

class CamarillaTrendSimple(Strategy):
    """
    Simplified trend-aware Camarilla strategy
    """
    
    # Camarilla parameters
    range_threshold = 0.002
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.04  # 4% take profit
    
    # Trend parameters  
    trend_ma = 50  # Moving average for trend
    
    # Risk management
    position_size_pct = 0.25  # Use 25% of capital per trade
    
    def init(self):
        """Initialize indicators"""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Calculate daily range
        daily_range = high - low
        
        # Camarilla levels (simplified)
        # Using rolling calculations for previous day's values
        self.pivot = self.I(lambda: (high + low + close) / 3)
        
        # Key levels based on range
        self.r3 = self.I(lambda: close + daily_range * 1.1 / 4)
        self.r4 = self.I(lambda: close + daily_range * 1.1 / 2)
        
        self.s3 = self.I(lambda: close - daily_range * 1.1 / 4)
        self.s4 = self.I(lambda: close - daily_range * 1.1 / 2)
        
        # Trend indicator
        self.trend_sma = self.I(talib.SMA, close, self.trend_ma)
        
        # ATR for volatility-based stops
        self.atr = self.I(talib.ATR, high, low, close, 14)
        
        # RSI for momentum
        self.rsi = self.I(talib.RSI, close, 14)
        
    def next(self):
        """Execute trading logic"""
        # Skip if not enough data
        if len(self.data) < self.trend_ma:
            return
            
        # Current values
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_rsi = self.rsi[-1]
        trend_sma = self.trend_sma[-1]
        
        # Skip if invalid values
        if pd.isna(current_atr) or pd.isna(trend_sma):
            return
            
        # Determine trend
        is_uptrend = current_price > trend_sma
        is_downtrend = current_price < trend_sma
        
        # Dynamic stop loss based on ATR
        atr_stop = min(current_atr * 2, current_price * 0.05)  # Max 5% stop
        
        # No position - look for entries
        if not self.position:
            
            # TREND FOLLOWING ENTRIES
            if is_uptrend and current_rsi < 70:
                # Buy on pullback to support in uptrend
                if current_price <= self.s3[-1] and current_price > self.s4[-1]:
                    # Calculate position size
                    size = (self.equity * self.position_size_pct) / current_price
                    
                    # Set stops
                    stop_loss = current_price - atr_stop
                    take_profit = current_price + (atr_stop * 2)  # 2:1 reward/risk
                    
                    self.buy(size=size, sl=stop_loss, tp=take_profit)
                    
                # Buy on breakout above R4 in strong uptrend
                elif current_price > self.r4[-1] and current_rsi < 80:
                    size = (self.equity * self.position_size_pct) / current_price
                    
                    stop_loss = self.r3[-1]  # Use R3 as stop
                    take_profit = current_price * 1.05  # 5% target
                    
                    if stop_loss < current_price:  # Ensure valid stop
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        
            elif is_downtrend and current_rsi > 30:
                # Sell on rally to resistance in downtrend
                if current_price >= self.r3[-1] and current_price < self.r4[-1]:
                    size = (self.equity * self.position_size_pct) / current_price
                    
                    stop_loss = current_price + atr_stop
                    take_profit = current_price - (atr_stop * 2)
                    
                    self.sell(size=size, sl=stop_loss, tp=take_profit)
                    
                # Sell on breakdown below S4 in strong downtrend
                elif current_price < self.s4[-1] and current_rsi > 20:
                    size = (self.equity * self.position_size_pct) / current_price
                    
                    stop_loss = self.s3[-1]  # Use S3 as stop
                    take_profit = current_price * 0.95  # 5% target
                    
                    if stop_loss > current_price:  # Ensure valid stop
                        self.sell(size=size, sl=stop_loss, tp=take_profit)
                        
            # RANGE TRADING (only when trend is weak)
            elif abs(current_price - trend_sma) / trend_sma < 0.02:  # Within 2% of MA
                # Fade extreme moves
                if current_price >= self.r3[-1] and current_rsi > 70:
                    # Overbought at resistance
                    size = (self.equity * self.position_size_pct * 0.5) / current_price  # Half size
                    
                    stop_loss = self.r4[-1] * 1.01
                    take_profit = self.pivot[-1]
                    
                    if stop_loss > current_price and take_profit < current_price:
                        self.sell(size=size, sl=stop_loss, tp=take_profit)
                        
                elif current_price <= self.s3[-1] and current_rsi < 30:
                    # Oversold at support
                    size = (self.equity * self.position_size_pct * 0.5) / current_price
                    
                    stop_loss = self.s4[-1] * 0.99
                    take_profit = self.pivot[-1]
                    
                    if stop_loss < current_price and take_profit > current_price:
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        
        # POSITION MANAGEMENT - Trailing stops in strong trends
        elif self.position:
            if self.position.is_long and is_uptrend:
                # Trail stop to latest swing low or ATR-based stop
                new_stop = max(
                    self.data.Low[-3:].min(),  # Recent swing low
                    current_price - atr_stop
                )
                if new_stop > self.position.sl:
                    self.position.sl = new_stop
                    
            elif self.position.is_short and is_downtrend:
                # Trail stop to latest swing high or ATR-based stop
                new_stop = min(
                    self.data.High[-3:].max(),  # Recent swing high
                    current_price + atr_stop
                )
                if new_stop < self.position.sl:
                    self.position.sl = new_stop