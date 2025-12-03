"""
🚀 Camarilla Trend Strategy
Enhanced Camarilla strategy adapted for trending markets like TSLA

Key improvements:
1. Trend detection using moving averages
2. Dynamic parameter adjustment based on volatility
3. Trend-following mode for strong trends
4. Better stop loss placement for volatile stocks

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

class CamarillaTrendStrategy(Strategy):
    """
    Enhanced Camarilla strategy for trending markets
    """
    
    # Strategy parameters
    range_threshold = 0.002  # Minimum range for range-trading
    stop_loss_buffer = 0.001  # Buffer beyond S4/R4 for stop loss
    take_profit_multiplier = 2.0  # Risk/reward ratio
    
    # Trend parameters
    fast_ma = 20  # Fast moving average
    slow_ma = 50  # Slow moving average
    trend_strength_threshold = 0.02  # 2% difference for strong trend
    
    # Volatility adjustment
    atr_period = 14  # ATR period for volatility
    volatility_adjustment = True  # Enable dynamic adjustment
    
    # Position sizing
    risk_per_trade = 0.02  # Risk 2% per trade
    
    def init(self):
        """Initialize indicators"""
        # Camarilla calculations
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.Close
        self.open = self.data.Open
        
        # Previous day's data using pandas shift
        def shift_series(series, periods=1):
            """Shift a series by n periods"""
            s = pd.Series(series).shift(periods)
            # Use forward fill for first value
            if periods > 0:
                s.iloc[:periods] = s.iloc[periods]
            return s.values
        
        self.prev_high = self.I(lambda: shift_series(self.high, 1))
        self.prev_low = self.I(lambda: shift_series(self.low, 1))
        self.prev_close = self.I(lambda: shift_series(self.close, 1))
        
        # Range calculation
        self.range = self.I(lambda: self.prev_high - self.prev_low)
        
        # Camarilla levels
        self.pivot = self.I(lambda: (self.prev_high + self.prev_low + self.prev_close) / 3)
        
        # Resistance levels
        self.r1 = self.I(lambda: self.prev_close + self.range * 1.1 / 12)
        self.r2 = self.I(lambda: self.prev_close + self.range * 1.1 / 6)
        self.r3 = self.I(lambda: self.prev_close + self.range * 1.1 / 4)
        self.r4 = self.I(lambda: self.prev_close + self.range * 1.1 / 2)
        self.r5 = self.I(lambda: (self.prev_high / self.prev_low) * self.prev_close)
        
        # Support levels
        self.s1 = self.I(lambda: self.prev_close - self.range * 1.1 / 12)
        self.s2 = self.I(lambda: self.prev_close - self.range * 1.1 / 6)
        self.s3 = self.I(lambda: self.prev_close - self.range * 1.1 / 4)
        self.s4 = self.I(lambda: self.prev_close - self.range * 1.1 / 2)
        self.s5 = self.I(lambda: self.prev_close - (self.r5 - self.prev_close))
        
        # Trend indicators
        self.fast_sma = self.I(talib.SMA, self.close, self.fast_ma)
        self.slow_sma = self.I(talib.SMA, self.close, self.slow_ma)
        
        # Volatility indicator
        self.atr = self.I(talib.ATR, self.high, self.low, self.close, self.atr_period)
        
        # RSI for overbought/oversold
        self.rsi = self.I(talib.RSI, self.close, 14)
        
    def calculate_position_size(self, stop_distance):
        """Calculate position size based on risk management"""
        risk_amount = self.equity * self.risk_per_trade
        shares = risk_amount / (stop_distance * self.close[-1])
        return min(shares, self.equity * 0.95 / self.close[-1])  # Max 95% of equity
        
    def detect_trend(self):
        """Detect market trend and strength"""
        if len(self.fast_sma) < 2 or len(self.slow_sma) < 2:
            return 'neutral', 0
            
        # Calculate trend
        fast_ma_current = self.fast_sma[-1]
        slow_ma_current = self.slow_sma[-1]
        
        if pd.isna(fast_ma_current) or pd.isna(slow_ma_current):
            return 'neutral', 0
            
        # Trend direction
        if fast_ma_current > slow_ma_current:
            trend = 'bullish'
        elif fast_ma_current < slow_ma_current:
            trend = 'bearish'
        else:
            trend = 'neutral'
            
        # Trend strength (percentage difference)
        trend_strength = abs(fast_ma_current - slow_ma_current) / slow_ma_current
        
        return trend, trend_strength
        
    def get_volatility_multiplier(self):
        """Adjust parameters based on volatility"""
        if not self.volatility_adjustment or len(self.atr) < 2:
            return 1.0
            
        current_atr = self.atr[-1]
        avg_price = self.close[-1]
        
        if pd.isna(current_atr) or pd.isna(avg_price) or avg_price == 0:
            return 1.0
            
        # Volatility as percentage of price
        volatility_pct = current_atr / avg_price
        
        # Higher volatility = wider stops and targets
        if volatility_pct > 0.03:  # High volatility (>3%)
            return 1.5
        elif volatility_pct > 0.02:  # Medium volatility (2-3%)
            return 1.2
        else:  # Low volatility (<2%)
            return 1.0
            
    def next(self):
        """Execute trading logic"""
        # Skip if we don't have enough data
        if len(self.data) < max(self.slow_ma, self.atr_period) + 1:
            return
            
        # Get current values
        current_price = self.close[-1]
        current_range = self.range[-1]
        
        # Skip if invalid data
        if pd.isna(current_range) or current_range <= 0:
            return
            
        # Detect trend
        trend, trend_strength = self.detect_trend()
        
        # Get volatility adjustment
        vol_multiplier = self.get_volatility_multiplier()
        
        # Adjust thresholds based on volatility
        adjusted_threshold = self.range_threshold * vol_multiplier
        adjusted_stop_buffer = self.stop_loss_buffer * vol_multiplier
        
        # Current levels
        r3 = self.r3[-1]
        r4 = self.r4[-1]
        r5 = self.r5[-1]
        s3 = self.s3[-1]
        s4 = self.s4[-1]
        s5 = self.s5[-1]
        
        # Skip if invalid levels
        if any(pd.isna(x) for x in [r3, r4, s3, s4]):
            return
            
        # RSI for additional confirmation
        rsi_value = self.rsi[-1] if len(self.rsi) > 0 else 50
        
        # TREND FOLLOWING MODE (for strong trends)
        if trend_strength > self.trend_strength_threshold:
            if not self.position:
                if trend == 'bullish' and current_price > self.slow_sma[-1]:
                    # Buy on pullback to fast MA in uptrend
                    if current_price <= self.fast_sma[-1] * 1.01:  # Near fast MA
                        stop_loss = self.fast_sma[-1] * (1 - adjusted_stop_buffer * 2)
                        take_profit = current_price * (1 + adjusted_stop_buffer * 4)
                        
                        size = self.calculate_position_size(current_price - stop_loss)
                        if size > 0:
                            self.buy(size=size, sl=stop_loss, tp=take_profit)
                            
                elif trend == 'bearish' and current_price < self.slow_sma[-1]:
                    # Sell on rally to fast MA in downtrend
                    if current_price >= self.fast_sma[-1] * 0.99:  # Near fast MA
                        stop_loss = self.fast_sma[-1] * (1 + adjusted_stop_buffer * 2)
                        take_profit = current_price * (1 - adjusted_stop_buffer * 4)
                        
                        size = self.calculate_position_size(stop_loss - current_price)
                        if size > 0:
                            self.sell(size=size, sl=stop_loss, tp=take_profit)
        
        # CAMARILLA MODE (for ranging/moderate trends)
        elif current_range / current_price > adjusted_threshold:
            if not self.position:
                # BREAKOUT STRATEGY (enhanced for trends)
                if trend == 'bullish' or trend == 'neutral':
                    # Long breakout above R4
                    if current_price > r4 and rsi_value < 70:
                        stop_loss = r3
                        take_profit = r5
                        
                        # Adjust for trend
                        if trend == 'bullish':
                            take_profit = max(r5, current_price * (1 + adjusted_stop_buffer * 3))
                        
                        size = self.calculate_position_size(current_price - stop_loss)
                        if size > 0:
                            self.buy(size=size, sl=stop_loss, tp=take_profit)
                
                if trend == 'bearish' or trend == 'neutral':
                    # Short breakout below S4
                    if current_price < s4 and rsi_value > 30:
                        stop_loss = s3
                        take_profit = s5
                        
                        # Adjust for trend
                        if trend == 'bearish':
                            take_profit = min(s5, current_price * (1 - adjusted_stop_buffer * 3))
                        
                        size = self.calculate_position_size(stop_loss - current_price)
                        if size > 0:
                            self.sell(size=size, sl=stop_loss, tp=take_profit)
                
                # RANGE TRADING (only in neutral/weak trends)
                if trend_strength < self.trend_strength_threshold * 0.5:
                    # Fade move to R3 (sell)
                    if current_price >= r3 * 0.999 and current_price < r4 and rsi_value > 60:
                        stop_loss = r4 * (1 + adjusted_stop_buffer)
                        risk = stop_loss - current_price
                        take_profit = current_price - (risk * self.take_profit_multiplier)
                        
                        size = self.calculate_position_size(risk)
                        if size > 0 and take_profit > s3:
                            self.sell(size=size, sl=stop_loss, tp=take_profit)
                    
                    # Fade move to S3 (buy)
                    elif current_price <= s3 * 1.001 and current_price > s4 and rsi_value < 40:
                        stop_loss = s4 * (1 - adjusted_stop_buffer)
                        risk = current_price - stop_loss
                        take_profit = current_price + (risk * self.take_profit_multiplier)
                        
                        size = self.calculate_position_size(risk)
                        if size > 0 and take_profit < r3:
                            self.buy(size=size, sl=stop_loss, tp=take_profit)
        
        # TRAILING STOP for trend following positions
        if self.position:
            if self.position.is_long and trend == 'bullish':
                # Trail stop to fast MA in strong uptrend
                new_stop = self.fast_sma[-1] * (1 - adjusted_stop_buffer)
                if new_stop > self.position.sl:
                    self.position.sl = new_stop
                    
            elif self.position.is_short and trend == 'bearish':
                # Trail stop to fast MA in strong downtrend
                new_stop = self.fast_sma[-1] * (1 + adjusted_stop_buffer)
                if new_stop < self.position.sl:
                    self.position.sl = new_stop