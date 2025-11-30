from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

class LiquidationVolt(Strategy):
    risk_pct = 0.01  # 1% risk per trade 
    atr_period = 14
    atr_ma_period = 20
    cluster_lookback = 20
    atr_threshold_period = 50
    atr_percentile = 0.3
    cluster_buffer = 0.005  # 0.5% buffer
    
    def init(self):
        # 🌊 Volatility indicators
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 
                         self.atr_period, name='ATR_14')
        
        self.atr_ma = self.I(talib.SMA, self.atr, 
                            self.atr_ma_period, name='ATR_MA')
        
        # 🏔️ Liquidation zones
        self.swing_high = self.I(talib.MAX, self.data.High, 
                                self.cluster_lookback, name='SWING_HIGH')
        
        self.swing_low = self.I(talib.MIN, self.data.Low, 
                               self.cluster_lookback, name='SWING_LOW')
        
        # 📉 Low volatility filter
        self.atr_threshold = self.I(lambda s: s.rolling(self.atr_threshold_period).quantile(self.atr_percentile),
                                   self.atr, name='ATR_THRESHOLD')

    def next(self):
        if len(self.data) < self.atr_threshold_period:
            return  # ⏳ Wait for sufficient data
        
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_threshold = self.atr_threshold[-1]
        
        # 🌙 MOON DEV ENTRY CONDITIONS
        near_swing_low = current_close <= (self.swing_low[-1] * (1 + self.cluster_buffer))
        low_volatility = current_atr < current_threshold
        
        if not self.position and near_swing_low and low_volatility:
            # 🎯 Calculate risk parameters
            entry_price = self.data.Open[-1]  # Next candle's open'
            stop_loss = self.swing_low[-1] - 2 * current_atr
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share > 0:  # Avoid zero division
                position_size = (self.risk_pct * self.equity) / risk_per_share
                position_size = int(round(position_size))  #  Critical sizing rule
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, 
                            tag=" MOON VOLTAGE ENTRY")
                    print(f" ENTERED LONG: {position_size} units at {entry_price:.2f}")
                    print(f"   SL: {stop_loss:.2f} | ATR: {current_atr:.2f} ")

        # 🚀 VOLTAGE EXPANSION EXIT (replaced crossover with manual check)
        elif self.position and (self.atr[-2] < self.atr_ma[-2] and self.atr[-1] > self.atr_ma[-1]):
            self.position.close()
            print(f" EXIT: ATR({current_atr:.2f}) > MA({self.atr_ma[-1]:.2f}) | Profit: {self.position.pl:.2f} ")

# MOON DEV DATA PREPARATION 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# 🌙✨ LAUNCH BACKTEST
bt = Backtest(data, LiquidationVolt, cash=1_000_000, commission=.002)