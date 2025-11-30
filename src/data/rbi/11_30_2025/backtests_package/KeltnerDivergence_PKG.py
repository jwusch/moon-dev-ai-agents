import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import os

class KeltnerDivergence(Strategy):
    KC_Multiplier = 2.0
    BB_Multiplier = 2.0
    BB_Width_Factor = 1.2
    Atr_Multiplier = 2
    Risk_Percent = 0.005

    def init(self):
        # Clean and prepare data
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data = self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col.lower()])

        # Indicators
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']

        self.sma200 = self.I(talib.SMA, close, timeperiod=200)
        self.ema20 = self.I(talib.EMA, close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, high, low, close, timeperiod=20)

        self.bb_mid = self.I(talib.SMA, close, timeperiod=20)
        bb_upper, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=self.BB_Multiplier, nbdevdn=self.BB_Multiplier)
        self.bb_upper = self.I(lambda: bb_upper)
        self.bb_lower = self.I(lambda: bb_lower)
        self.kc_upper = self.I(lambda: self.ema20 + self.KC_Multiplier * self.atr20)
        self.kc_lower = self.I(lambda: self.ema20 - self.KC_Multiplier * self.atr20)
        
        self.rsi14 = self.I(talib.RSI, close, timeperiod=14)
        bb_width = (bb_upper - bb_lower) / self.bb_mid
        self.bb_width = self.I(lambda: bb_width)

        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=50)

    def next(self):
        if self.position:
            # Exit logic
            if self.data.close[-1] < self.ema20[-1]:
                self.position.close()
                print("🛑 Exit position as price closes below KC basis 🚀")

        else:
            # Entry logic
            close = self.data.close[-1]
            kc_upper = self.kc_upper[-1]

            trend_ok = (close > self.sma200[-1]) and (self.sma200[-1] > self.sma200[-5])
            vol_ok = (self.bb_width[-1] > self.BB_Width_Factor * self.bb_width_sma[-1])
            breakout_ok = close > kc_upper

            # Previous Low and RSI check for divergence
            recent_lows = self.data.low[-20:]
            recent_rsi = self.rsi14[-20:]
            LL2_index = recent_lows.idxmin()
            LL1_index = (recent_lows[:LL2_index].idxmin() if LL2_index > 0 else 0)
            RSI_LL2 = recent_rsi[LL2_index]
            RSI_LL1 = recent_rsi[LL1_index]

            div_ok = (LL2_index > LL1_index) and (recent_lows[LL2_index] < recent_lows[LL1_index]) and \
                     (RSI_LL2 > RSI_LL1) and (self.rsi14[-1] > 50)

            if trend_ok and div_ok and vol_ok and breakout_ok:
                atr20_val = self.atr20[-1]
                entry_price = self.data.close[-1]
                stop_loss = min(self.ema20[-1], entry_price - self.Atr_Multiplier * atr20_val)
                risk_amount = self.equity * self.Risk_Percent

                position_size = risk_amount / abs(entry_price - stop_loss)
                position_size = int(round(position_size))
                self.buy(size=position_size)

                print("🌙 Enter long position as all conditions satisfied 🚀")

data_path = os.path.join(os.getenv('PROJECT_ROOT', '.'), 'src/data/rbi/BTC-USD-15m.csv')
data = pd.read_csv(data_path)

bt = Backtest(data, KeltnerDivergence, cash=1_000_000, commission=.002)
stats = bt.run()
print(stats)