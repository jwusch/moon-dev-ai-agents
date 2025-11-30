import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class LiquidationRetrace(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 VWAP Calculation using pandas_ta
        def calculate_vwap(high, low, close, volume):
            return ta.vwap(high=high, low=low, close=close, volume=volume)
        self.vwap = self.I(calculate_vwap, 
                          self.data.High, self.data.Low, 
                          self.data.Close, self.data.Volume, 
                          name='VWAP')

        # ✨ 15-period ATR using TA-Lib
        self.atr = self.I(talib.ATR,
                         self.data.High, self.data.Low, self.data.Close,
                         timeperiod=15, name='ATR')

    def next(self):
        # 🌙 Skip early bars without indicator values
        if len(self.data) < 15:
            return

        current_close = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        current_atr = self.atr[-1]

        # 🚀 Moon-themed debug prints
        print(f"\n Bar: {self.data.index[-1]} | Close: {current_close:.2f}")
        print(f"   VWAP: {current_vwap:.2f} | ATR: {current_atr:.2f}")

        if not self.position:
            # 💎 Long entry: Price < VWAP - 3*ATR
            if current_close < (current_vwap - 3*current_atr):
                self.calculate_position_size('long', current_close, current_atr)

            # 💎 Short entry: Price > VWAP + 3*ATR
            elif current_close > (current_vwap + 3*current_atr):
                self.calculate_position_size('short', current_close, current_atr)
        else:
            # ⏳ Check hourly timeout (4 bars = 1 hour)
            if (len(self.data) - self.position.entry_bar) >= 4:
                print(f"⏰ Timeout exit at {current_close:.2f}")
                self.position.close()

    def calculate_position_size(self, direction, price, atr):
        # 🛡 Risk management calculations
        risk_amount = self.equity * self.risk_percent
        stop_distance = 0.5 * atr
        size = risk_amount / (stop_distance * price)
        size = int(round(size))  #  Ensuring whole number position size

        # 🎯 Set order parameters
        if direction == 'long':
            sl = price - stop_distance
            tp = price + 1.5*atr
            print(f" LONG SIGNAL | Size: {size}")
            print(f"   Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            self.buy(size=size, sl=sl, tp=tp)
        else:
            sl = price + stop_distance
            tp = price - 1.5*atr
            print(f" SHORT SIGNAL | Size: {size}")
            print(f"   Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            self.sell(size=size, sl=sl, tp=tp)

# 🗃 Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# 🌙 Run backtest with $1M capital
bt = Backtest(data, LiquidationRetrace, 
             cash=1_000_000, commission=.002)
stats = bt.run()

# ✨ Print full performance stats