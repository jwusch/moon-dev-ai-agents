import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib  # cspell:ignore talib timeperiod
import os

class KeltnerDivergence(Strategy):
    KC_Multiplier = 2.0
    BB_Multiplier = 2.0
    BB_Width_Factor = 1.2
    Atr_Multiplier = 2
    Risk_Percent = 0.005

    def init(self):
        # Indicators
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma200 = self.I(talib.SMA, close, timeperiod=200)
        self.ema20 = self.I(talib.EMA, close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, high, low, close, timeperiod=20)

        self.bb_mid = self.I(talib.SMA, close, timeperiod=20)
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=self.BB_Multiplier, nbdevdn=self.BB_Multiplier)
        self.kc_upper = self.ema20 + self.KC_Multiplier * self.atr20
        self.kc_lower = self.ema20 - self.KC_Multiplier * self.atr20
        
        self.rsi14 = self.I(talib.RSI, close, timeperiod=14)
        bb_width = (bb_upper - bb_lower) / self.bb_mid
        self.bb_width = self.I(lambda: bb_width)

        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=50)

    def next(self):
        if self.position:
            # Exit logic
            if self.data.Close[-1] < self.ema20[-1]:
                self.position.close()
                print("🛑 Exit position as price closes below KC basis 🚀")

        else:
            # Entry logic
            close = self.data.Close[-1]
            kc_upper = self.kc_upper[-1]

            trend_ok = (close > self.sma200[-1]) and (self.sma200[-1] > self.sma200[-5])
            vol_ok = (self.bb_width[-1] > self.BB_Width_Factor * self.bb_width_sma[-1])
            breakout_ok = close > kc_upper

            # Previous Low and RSI check for divergence
            recent_lows = list(self.data.Low[-20:])
            recent_rsi = list(self.rsi14[-20:])
            
            if len(recent_lows) >= 20 and len(recent_rsi) >= 20:
                # Find lowest low
                LL2_index = recent_lows.index(min(recent_lows))
                # Find previous low before LL2
                if LL2_index > 0:
                    LL1_index = recent_lows[:LL2_index].index(min(recent_lows[:LL2_index]))
                else:
                    LL1_index = 0
                
                RSI_LL2 = recent_rsi[LL2_index] if LL2_index < len(recent_rsi) else 50
                RSI_LL1 = recent_rsi[LL1_index] if LL1_index < len(recent_rsi) else 50

                div_ok = (LL2_index > LL1_index) and (recent_lows[LL2_index] < recent_lows[LL1_index]) and \
                         (RSI_LL2 > RSI_LL1) and (self.rsi14[-1] > 50)
            else:
                div_ok = False

            if trend_ok and div_ok and vol_ok and breakout_ok:
                atr20_val = self.atr20[-1]
                entry_price = self.data.Close[-1]
                stop_loss = min(self.ema20[-1], entry_price - self.Atr_Multiplier * atr20_val)
                risk_amount = self.equity * self.Risk_Percent

                position_size = risk_amount / abs(entry_price - stop_loss)
                position_size = int(round(position_size))
                self.buy(size=position_size)

                print("🌙 Enter long position as all conditions satisfied 🚀")

# Get the correct path to the data file
# The script is at: /mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/11_30_2025/backtests/
# Go up two directories to get to /mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BTC-USD-15m.csv')

# Check if file exists in rbi directory
if not os.path.exists(data_path):
    # Try the parent data directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'BTC-USD-15m.csv')

data = pd.read_csv(data_path)
# Fix column names to match backtesting requirements
data.columns = [col.strip().title() for col in data.columns]
# Set datetime as index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

bt = Backtest(data, KeltnerDivergence, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)