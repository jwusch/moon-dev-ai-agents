"""
🌙 Camarilla Pivot Strategy
A sophisticated trading strategy based on Camarilla pivot levels with both
range-bound and breakout trading approaches.

Strategy Rules:
1. Range-bound trading (S3/R3):
   - Buy near S3 expecting bounce, sell near R3 expecting rejection
   - Stop losses beyond S4/R4
   
2. Breakout trading (S4/R4):
   - Long on R4 breakout, short on S4 breakdown
   - Confirms trend day potential

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import crossover


class CamarillaLevels:
    """Calculate Camarilla pivot levels"""
    
    @staticmethod
    def calculate(high: float, low: float, close: float) -> dict:
        """
        Calculate all Camarilla levels for a given day
        
        Formula:
        - Pivot = (High + Low + Close) / 3
        - R4 = Close + ((High - Low) * 1.1 / 2)
        - R3 = Close + ((High - Low) * 1.1 / 4)
        - R2 = Close + ((High - Low) * 1.1 / 6)
        - R1 = Close + ((High - Low) * 1.1 / 12)
        - S1 = Close - ((High - Low) * 1.1 / 12)
        - S2 = Close - ((High - Low) * 1.1 / 6)
        - S3 = Close - ((High - Low) * 1.1 / 4)
        - S4 = Close - ((High - Low) * 1.1 / 2)
        """
        range_hl = high - low
        
        levels = {
            'pivot': (high + low + close) / 3,
            'r4': close + (range_hl * 1.1 / 2),
            'r3': close + (range_hl * 1.1 / 4),
            'r2': close + (range_hl * 1.1 / 6),
            'r1': close + (range_hl * 1.1 / 12),
            's1': close - (range_hl * 1.1 / 12),
            's2': close - (range_hl * 1.1 / 6),
            's3': close - (range_hl * 1.1 / 4),
            's4': close - (range_hl * 1.1 / 2)
        }
        
        return levels
    
    @staticmethod
    def calculate_series(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Camarilla levels for entire DataFrame"""
        # Shift by 1 to use previous day's HLC for today's levels
        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        prev_close = df['Close'].shift(1)
        
        range_hl = prev_high - prev_low
        
        df['Pivot'] = (prev_high + prev_low + prev_close) / 3
        df['R4'] = prev_close + (range_hl * 1.1 / 2)
        df['R3'] = prev_close + (range_hl * 1.1 / 4)
        df['R2'] = prev_close + (range_hl * 1.1 / 6)
        df['R1'] = prev_close + (range_hl * 1.1 / 12)
        df['S1'] = prev_close - (range_hl * 1.1 / 12)
        df['S2'] = prev_close - (range_hl * 1.1 / 6)
        df['S3'] = prev_close - (range_hl * 1.1 / 4)
        df['S4'] = prev_close - (range_hl * 1.1 / 2)
        
        return df


class CamarillaStrategy(Strategy):
    """
    Camarilla Trading Strategy combining range-bound and breakout approaches
    
    Parameters:
    - range_threshold: How close price must be to S3/R3 to trigger fade trade (default 0.1%)
    - breakout_confirmation: Bars to confirm breakout (default 1)
    - stop_loss_buffer: Extra buffer beyond S4/R4 for stop loss (default 0.1%)
    - take_profit_ratio: Risk/reward ratio for targets (default 2.0)
    - use_range_trading: Enable range-bound S3/R3 trading (default True)
    - use_breakout_trading: Enable S4/R4 breakout trading (default True)
    """
    
    # Strategy parameters
    range_threshold = 0.001  # 0.1% proximity to S3/R3
    breakout_confirmation = 1  # Bars to confirm breakout
    stop_loss_buffer = 0.001  # 0.1% beyond S4/R4
    take_profit_ratio = 2.0  # Risk/reward ratio
    use_range_trading = True
    use_breakout_trading = True
    
    def init(self):
        """Initialize strategy indicators"""
        # Calculate Camarilla levels
        self.df = pd.DataFrame({
            'High': self.data.High,
            'Low': self.data.Low,
            'Close': self.data.Close
        })
        
        levels_df = CamarillaLevels.calculate_series(self.df)
        
        # Store levels as indicators
        self.r4 = self.I(lambda: levels_df['R4'].values)
        self.r3 = self.I(lambda: levels_df['R3'].values)
        self.r2 = self.I(lambda: levels_df['R2'].values)
        self.r1 = self.I(lambda: levels_df['R1'].values)
        self.pivot = self.I(lambda: levels_df['Pivot'].values)
        self.s1 = self.I(lambda: levels_df['S1'].values)
        self.s2 = self.I(lambda: levels_df['S2'].values)
        self.s3 = self.I(lambda: levels_df['S3'].values)
        self.s4 = self.I(lambda: levels_df['S4'].values)
        
        # Track trade type
        self.trade_type = None
        
    def next(self):
        """Execute strategy logic on each bar"""
        # Skip if we don't have levels yet (first day)
        if pd.isna(self.r4[-1]) or pd.isna(self.s4[-1]):
            return
        
        current_price = self.data.Close[-1]
        
        # Exit logic for existing positions
        if self.position:
            self._check_exits(current_price)
            return
        
        # Entry logic
        if self.use_range_trading:
            self._check_range_entries(current_price)
            
        if self.use_breakout_trading and not self.position:
            self._check_breakout_entries(current_price)
    
    def _check_range_entries(self, price):
        """Check for range-bound trading opportunities (S3/R3 fade)"""
        # Long entry near S3
        if self._near_level(price, self.s3[-1], self.range_threshold):
            if price > self.s4[-1]:  # Ensure we're above S4
                stop_loss = self.s4[-1] * (1 - self.stop_loss_buffer)
                # Make sure take profit is above entry
                take_profit = max(self.r1[-1], price * 1.005)  # At least 0.5% profit
                
                # Validate order levels
                if stop_loss < price < take_profit:
                    size = self._calculate_position_size(price, stop_loss)
                    if size > 0:
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_type = 'range_long'
        
        # Short entry near R3
        elif self._near_level(price, self.r3[-1], self.range_threshold):
            if price < self.r4[-1]:  # Ensure we're below R4
                stop_loss = self.r4[-1] * (1 + self.stop_loss_buffer)
                # Make sure take profit is below entry
                take_profit = min(self.s1[-1], price * 0.995)  # At least 0.5% profit
                
                # Validate order levels
                if take_profit < price < stop_loss:
                    size = self._calculate_position_size(price, stop_loss)
                    if size > 0:
                        self.sell(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_type = 'range_short'
    
    def _check_breakout_entries(self, price):
        """Check for breakout trading opportunities (S4/R4 breakout)"""
        # Bullish breakout above R4
        if price > self.r4[-1] * (1 + self.range_threshold):
            if self._confirm_breakout('bull'):
                stop_loss = max(self.r3[-1], price * 0.98)  # Stop at R3 or 2% below
                risk = price - stop_loss
                take_profit = price + (risk * self.take_profit_ratio)
                
                # Validate order levels
                if stop_loss < price < take_profit:
                    size = self._calculate_position_size(price, stop_loss)
                    if size > 0:
                        self.buy(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_type = 'breakout_long'
        
        # Bearish breakout below S4
        elif price < self.s4[-1] * (1 - self.range_threshold):
            if self._confirm_breakout('bear'):
                stop_loss = min(self.s3[-1], price * 1.02)  # Stop at S3 or 2% above
                risk = stop_loss - price
                take_profit = price - (risk * self.take_profit_ratio)
                
                # Validate order levels
                if take_profit < price < stop_loss:
                    size = self._calculate_position_size(price, stop_loss)
                    if size > 0:
                        self.sell(size=size, sl=stop_loss, tp=take_profit)
                        self.trade_type = 'breakout_short'
    
    def _check_exits(self, price):
        """Check for manual exit conditions beyond stop/target"""
        # Range trades: Exit at opposite level
        if self.trade_type == 'range_long' and price >= self.r3[-1]:
            self.position.close()
        elif self.trade_type == 'range_short' and price <= self.s3[-1]:
            self.position.close()
        
        # Breakout trades: Trail stop to next level
        if self.trade_type == 'breakout_long' and price > self.r4[-1] * 1.02:
            self.position.sl = self.r4[-1]
        elif self.trade_type == 'breakout_short' and price < self.s4[-1] * 0.98:
            self.position.sl = self.s4[-1]
    
    def _near_level(self, price, level, threshold):
        """Check if price is within threshold of a level"""
        return abs(price - level) / level <= threshold
    
    def _confirm_breakout(self, direction):
        """Confirm breakout with volume or momentum"""
        if self.breakout_confirmation == 0:
            return True
        
        # Simple confirmation: Check if previous bars also broke level
        if direction == 'bull':
            for i in range(1, min(self.breakout_confirmation + 1, len(self.data))):
                if self.data.Close[-i] <= self.r4[-1]:
                    return False
        else:
            for i in range(1, min(self.breakout_confirmation + 1, len(self.data))):
                if self.data.Close[-i] >= self.s4[-1]:
                    return False
        
        return True
    
    def _calculate_position_size(self, entry, stop_loss):
        """Calculate position size based on risk"""
        # For backtesting.py, we need to return a fraction of equity
        # not number of shares
        
        # Risk 1% of equity per trade
        risk_fraction = 0.01
        
        # Calculate the price movement risk as a fraction
        price_risk = abs(entry - stop_loss) / entry
        
        if price_risk == 0:
            return 0
        
        # Position size as fraction of equity
        # If we risk 1% and price moves 2%, we can use 50% of equity
        size_fraction = risk_fraction / price_risk
        
        # Cap at 95% of equity
        size_fraction = min(size_fraction, 0.95)
        
        # Ensure it's positive and not too small
        if size_fraction < 0.01:
            return 0
        
        return size_fraction