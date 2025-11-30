"""
🌙 Moon Dev's SqueezeSurge Strategy
Identifies explosive price moves from low volatility conditions
"""

import pandas as pd
import talib
import pandas_ta
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class SqueezeSurge(Strategy):
    def init(self):
        # Bollinger Bands
        self.middle_band = self.I(talib.SMA, self.data.Close, 20)
        std = self.I(talib.STDDEV, self.data.Close, 20)
        self.upper_band = self.I(lambda: self.middle_band + 2 * std)
        self.lower_band = self.I(lambda: self.middle_band - 2 * std)
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 30)
        
        # VWAP
        # Note: pandas_ta.vwap requires proper implementation with backtesting
        # Using simple approximation for now
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.vwap = self.I(lambda: typical_price)
        
        # Stochastic RSI
        rsi = self.I(talib.RSI, self.data.Close, 14)
        self.stoch_k = self.I(lambda: (rsi - talib.MIN(rsi, 14)) / (talib.MAX(rsi, 14) - talib.MIN(rsi, 14)) * 100)
        
        # Bandwidth calculations
        self.bandwidth = self.I(lambda: self.upper_band - self.lower_band)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, 20)

    def next(self):
        if not self.position:
            # Long entry conditions
            long_cond = (
                (self.bandwidth[-1] == self.min_bandwidth[-1]) and
                (self.data.Close[-1] > self.upper_band[-1]) and
                (self.data.Volume[-1] >= 2 * self.volume_sma[-1]) and
                (self.data.Close[-1] > self.vwap[-1] and self.data.Close[-2] <= self.vwap[-2])
            )
            
            # Short entry conditions
            short_cond = (
                (self.bandwidth[-1] == self.min_bandwidth[-1]) and
                (self.data.Close[-1] < self.lower_band[-1]) and
                (self.data.Volume[-1] >= 2 * self.volume_sma[-1]) and
                (self.data.Close[-1] < self.vwap[-1] and self.data.Close[-2] >= self.vwap[-2])
            )
            
            if long_cond:
                risk_amount = self.equity * 0.01
                sl_price = self.middle_band[-1]
                risk_per_unit = self.data.Close[-1] - sl_price
                if risk_per_unit > 0:
                    size = min(0.95, risk_amount / risk_per_unit / self.data.Close[-1])
                    self.buy(size=size, sl=sl_price)
                    print(f"🚀 MOON DEV LONG ENTRY | Size: {size:.4f} | Price: {self.data.Close[-1]} | SL: {sl_price}")
            
            elif short_cond:
                risk_amount = self.equity * 0.01
                sl_price = self.middle_band[-1]
                risk_per_unit = sl_price - self.data.Close[-1]
                if risk_per_unit > 0:
                    size = min(0.95, risk_amount / risk_per_unit / self.data.Close[-1])
                    self.sell(size=size, sl=sl_price)
                    print(f"🌑 MOON DEV SHORT ENTRY | Size: {size:.4f} | Price: {self.data.Close[-1]} | SL: {sl_price}")
        
        else:
            if self.position.is_long:
                # Long exit conditions
                exit_cond = (
                    (self.stoch_k[-1] < 80 and self.stoch_k[-2] >= 80) or
                    (self.data.Close[-1] < self.upper_band[-1])
                )
                if exit_cond:
                    self.position.close()
                    print(f"✅ MOON DEV LONG EXIT | Price: {self.data.Close[-1]} | StochRSI: {self.stoch_k[-1]}")
            
            elif self.position.is_short:
                # Short exit conditions
                exit_cond = (
                    (self.stoch_k[-1] > 20 and self.stoch_k[-2] <= 20) or
                    (self.data.Close[-1] > self.lower_band[-1])
                )
                if exit_cond:
                    self.position.close()
                    print(f"✅ MOON DEV SHORT EXIT | Price: {self.data.Close[-1]} | StochRSI: {self.stoch_k[-1]}")

# Backtest setup
bt = Backtest(data, SqueezeSurge, commission=.002, exclusive_orders=True)

# Run backtest
stats = bt.run()
print("🌙✨ MOON DEV BACKTEST COMPLETE ✨🌙")
print(stats)

# Plot results
bt.plot(filename='moon_dev_squeeze_surge')