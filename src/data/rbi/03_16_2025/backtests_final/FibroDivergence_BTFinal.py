import pandas as pd
import talib
from backtesting import Strategy, Backtest

# Data Preparation 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class FibroDivergence(Strategy):
    # Strategy Parameters ✨
    rsi_period = 14
    swing_window = 20
    risk_pct = 0.01
    fib_levels = [0.382, 0.5, 0.618]
    divergence_lookback = 5
    trailing_stop_pct = 0.02
    
    def init(self):
        # Core Indicators 🌙
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        
    def next(self):
        # Moon Dev's Trading Logic ✨'
        price = self.data.Close[-1]
        equity = self.equity
        
        # Trend Detection 🌊
        trend = 'bullish' if price > self.sma50[-1] else 'bearish'
        
        # Fibonacci Calculations 📐
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        fib_diff = sh - sl
        
        # Entry Conditions 🔍
        if not self.position:
            if trend == 'bullish':
                # Bearish Divergence Short Setup 🐻
                for level in self.fib_levels:
                    fib_price = sh - (fib_diff * level)
                    if abs(price - fib_price)/fib_price <= 0.005:
                        # Verify RSI Divergence
                        rsi_div = (self.data.High[-1] > self.data.High[-self.divergence_lookback] and 
                                  self.rsi[-1] < self.rsi[-self.divergence_lookback])
                        if rsi_div:
                            # Risk Management 🛡️
                            stop_loss = sh * 1.01
                            risk_amount = equity * self.risk_pct
                            position_size = int(round(risk_amount / (stop_loss - price)))
                            
                            print(f" BEARISH DIVERGENCE! Shorting at {price:.2f} ")
                            print(f" Entry: {price:.2f} | SL: {stop_loss:.2f} | Size: {position_size}")
                            self.sell(size=position_size, sl=stop_loss)
                            break
                            
            else:  # Bearish trend
                # Bullish Divergence Long Setup 🐂
                for level in self.fib_levels:
                    fib_price = sl + (fib_diff * level)
                    if abs(price - fib_price)/fib_price <= 0.005:
                        # Verify RSI Divergence
                        rsi_div = (self.data.Low[-1] < self.data.Low[-self.divergence_lookback] and 
                                  self.rsi[-1] > self.rsi[-self.divergence_lookback])
                        if rsi_div:
                            # Risk Management 🛡️
                            stop_loss = sl * 0.99
                            risk_amount = equity * self.risk_pct
                            position_size = int(round(risk_amount / (price - stop_loss)))
                            
                            print(f" BULLISH DIVERGENCE! Buying at {price:.2f} ")
                            print(f" Entry: {price:.2f} | SL: {stop_loss:.2f} | Size: {position_size}")
                            self.buy(size=position_size, sl=stop_loss)
                            break
        
        # Trailing Stop Management 🚂
        for trade in self.trades:
            if trade.is_long:
                trail_price = self.data.High