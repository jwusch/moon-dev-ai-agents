# 🌙 Moon Dev MomentumDivergence.py Strategy
# MomentumDivergence_BT trading strategy

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
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

class MomentumDivergence(Strategy):
    # Strategy parameters
    risk_per_trade = 0.02  # 2% risk per trade
    
    def init(self):
        # 🌙 Initialize indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.macd, self.signal, self.histogram = self.I(talib.MACD, self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.momentum = self.I(talib.MOM, self.data.Close, timeperiod=10)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # Skip if not enough data
        if len(self.data) < 200:
            return
        
        print(f"🌙 Moon Dev | Price: {current_price:.2f}")
        
        if not self.position:
            # 🚀 Entry Logic: Momentum-based signals
            # Simplified entry conditions - implement actual strategy logic
            if self.rsi[-1] < 30 and current_price > self.sma200[-1]:
                # Calculate position size
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                atr_value = current_price * 0.02
                stop_loss = current_price - (2 * atr_value)
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price + (3 * atr_value)
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 LONG Entry | Size: {position_size} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        
        else:
            # 🛑 Exit conditions
            if self.rsi[-1] > 70:
                self.position.close()
                print(f"🛑 Exit Position | Price: {current_price:.2f}")

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
bt = Backtest(data, MomentumDivergence, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌕 Print results
print("\n🌕 MOON DEV MOMENTUMDIVERGENCE RESULTS 🌕")
print("="*50)
print(f"Return [%]: {stats['Return [%]']:.2f}")
print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
print(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}" if 'Win Rate [%]' in stats else "Win Rate: N/A")
print(f"Total Trades: {stats['# Trades']}")
print("="*50)
