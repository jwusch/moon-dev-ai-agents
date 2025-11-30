import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumeDivergenceReversal(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    rsi_period = 14
    sma_short = 50
    sma_long = 200
    swing_period = 20

    def init(self):
        # Clean and prepare data
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # Calculate indicators using TA-Lib with self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_short, name='SMA50')
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_long, name='SMA200')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SwingHigh')
        
        # Calculate Up/Down Volume
        close = self.data.Close
        prev_close = pd.Series(close).shift(1).values
        up_day = (close > prev_close).astype(int)
        down_day = (close < prev_close).astype(int)
        
        upside_vol = self.data.Volume * up_day
        downside_vol = self.data.Volume * down_day
        
        self.sum_up_vol = self.I(talib.SMA, upside_vol, timeperiod=self.rsi_period, name='SumUpVol')
        self.sum_down_vol = self.I(talib.SMA, downside_vol, timeperiod=self.rsi_period, name='SumDownVol')

    def next(self):
        # Skip initial bars without indicator values
        if len(self.data) < max(self.sma_long, self.swing_period, self.rsi_period) + 1:
            return
        
        # Moon Dev Debug Prints 🌙✨
#         print(f"\n🌙 MOON DEV DEBUG 🌙")
        print(f" Current Close: {self.data.Close[-1]:0.2f}")
        print(f" RSI: {self.rsi[-1]:0.2f}")
        print(f" SMA50: {self.sma50[-1]:0.2f} | SMA200: {self.sma200[-1]:0.2f}")
        print(f" Volume: Up {self.sum_up_vol[-1]:0.2f} vs Down {self.sum_down_vol[-1]:0.2f}")
        
        # Check if we're NOT in position and conditions are met'
        if not self.position:
            # Entry Conditions
            uptrend = self.sma50[-1] > self.sma200[-1]
            rsi_overbought = self.rsi[-1] >= 70 and self.rsi[-2] < 70
            volume_divergence = self.sum_up_vol[-1] > self.sum_down_vol[-1]
            
            if uptrend and rsi_overbought and volume_divergence:
                # Moon Dev Entry Signal 🌙🚀
                entry_price = self.data.Close[-1]
                stop_loss = self.swing_high[-1] * 1.001  # 0.1% buffer
                risk_per_share = abs(entry_price - stop_loss)
                
                if risk_per_share > 0:
                    risk_amount = self.risk_per_trade * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    # Moon Dev Position Sizing 🌙💸
#                     print(f"\n🚀 MOON DEV POSITION SIZING 🚀")
                    print(f" Equity: {self.equity:0.2f}")
                    print(f" Risk per Trade: {risk_amount:0.2f}")
                    print(f" Position Size: {position_size} units")
                    
                    # Execute short position
                    self.sell(size=position_size, 
                            sl=stop_loss,
                            tp=entry_price - 2*risk_per_share)
                    
                    # Debug Print
                    print(f"\n SHORT ENTRY SIGNAL ")
                    print(f" Entry Price: {entry_price:0.2f}")
                    print(f" Stop Loss: {stop_loss:0.2f}")
                    print(f" RSI: {self.rsi[-1]:0.2f}")