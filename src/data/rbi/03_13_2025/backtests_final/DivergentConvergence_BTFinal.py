# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR DIVERGENTCONVERGENCE STRATEGY ✨

import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utils for dynamic path resolution
try:
    from utils import get_data_file_path, prepare_backtest_data
except ImportError:
    def get_data_file_path(filename='BTC-USD-15m.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths = [
            os.path.join(script_dir, '..', '..', filename),
            os.path.join(script_dir, '..', '..', 'rbi', filename),
            '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv',
            '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/BTC-USD-15m.csv'
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {filename}")

class DivergentConvergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    stoch_fastk = 14
    stoch_slowk = 3
    stoch_slowd = 3
    swing_period = 20
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2
    
    def init(self):
        # 🌙 Calculate RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # 🌙 Calculate Stochastic
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH,
            self.data.High,
            self.data.Low,
            self.data.Close,
            fastk_period=self.stoch_fastk,
            slowk_period=self.stoch_slowk,
            slowk_matype=0,
            slowd_period=self.stoch_slowd,
            slowd_matype=0
        )
        
        # 🌙 Calculate swing highs and lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # 🌙 ATR for position sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # Skip if not enough data
        if len(self.data) < self.swing_period + 5:
            return
            
        print(f"🌙 Moon Dev | Price: {current_price:0.2f} | RSI: {self.rsi[-1]:0.2f} | Stoch K: {self.stoch_k[-1]:0.2f}")
        
        if not self.position:
            # 🚀 Bullish Divergence Detection
            # Price making lower lows but RSI/Stoch making higher lows
            if (self.data.Low[-1] < self.swing_low[-5] and
                self.rsi[-1] > self.rsi[-5] and
                self.stoch_k[-1] < 30):
                
                # Position sizing
                equity = self.equity
                risk_amount = equity * self.risk_pct
                stop_loss = self.swing_low[-1] - (0.5 * self.atr[-1])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price + (self.rr_ratio * risk_per_share)
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 BULLISH DIVERGENCE! Long Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
            
            # 📉 Bearish Divergence Detection
            # Price making higher highs but RSI/Stoch making lower highs
            elif (self.data.High[-1] > self.swing_high[-5] and
                  self.rsi[-1] < self.rsi[-5] and
                  self.stoch_k[-1] > 70):
                
                # Position sizing
                equity = self.equity
                risk_amount = equity * self.risk_pct
                stop_loss = self.swing_high[-1] + (0.5 * self.atr[-1])
                risk_per_share = stop_loss - current_price
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price - (self.rr_ratio * risk_per_share)
                    
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"📉 BEARISH DIVERGENCE! Short Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")

# 🌙 Load data
try:
    data_path = get_data_file_path('BTC-USD-15m.csv')
    data = pd.read_csv(data_path)
    print(f"✅ Found data file at: {data_path}")
except FileNotFoundError:
    print("⚠️ No data file found, generating sample data")
    dates = pd.date_range(start='2023-01-01', end='2023-12-01', freq='15min')
    n = len(dates)
    np.random.seed(42)
    price = 30000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'datetime': dates,
        'Open': price + np.random.randn(n) * 50,
        'High': price + np.abs(np.random.randn(n) * 100),
        'Low': price - np.abs(np.random.randn(n) * 100),
        'Close': price,
        'Volume': np.random.randint(100, 10000, n)
    })

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# 🚀 Run backtest
bt = Backtest(data, DivergentConvergence, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌕 Print results
print("\n🌕 MOON DEV DIVERGENT CONVERGENCE RESULTS 🌕")
print("="*50)
print(f"Return [%]: {stats['Return [%]']:0.2f}")
print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:0.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:0.2f}")
print(f"Win Rate [%]: {stats['Win Rate [%]']:0.2f}" if 'Win Rate [%]' in stats else "Win Rate: N/A")
print(f"Total Trades: {stats['# Trades']}")
print("="*50)