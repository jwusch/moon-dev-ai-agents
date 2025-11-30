# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import os
import sys
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utils for dynamic path resolution
try:
    from utils import get_data_file_path, prepare_backtest_data
except ImportError:
    # Fallback if utils not found
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

class BandStochBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙✨ Calculate indicators using TA-Lib with self.I()
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        # Calculate Stochastic Oscillator
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH,
            self.data.High,
            self.data.Low,
            self.data.Close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        # Calculate ATR for position sizing and stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def next(self):
        # Get current values
        current_close = self.data.Close[-1]
        current_upper = self.upper_band[-1]
        current_lower = self.lower_band[-1]
        current_k = self.stoch_k[-1]
        current_d = self.stoch_d[-1]
        current_atr = self.atr[-1]
        
        print(f"🌙 Moon Dev | Close: {current_close:0.2f} | K: {current_k:0.2f} | D: {current_d:0.2f}")
        
        if not self.position:
            # Long entry conditions
            if (current_close > current_upper and 
                current_k < 80 and 
                current_k > current_d):
                
                # Risk management
                stop_loss = current_close - (2 * current_atr)
                take_profit = current_close + (3 * current_atr)
                
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 Long Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
            
            # Short entry conditions
            elif (current_close < current_lower and 
                  current_k > 20 and 
                  current_k < current_d):
                
                # Risk management
                stop_loss = current_close + (2 * current_atr)
                take_profit = current_close - (3 * current_atr)
                
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = stop_loss - current_close
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🔻 Short Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")

# Load data with dynamic path resolution
try:
    data_path = get_data_file_path('BTC-USD-15m.csv')
    data = pd.read_csv(data_path)
    print(f"✅ Found data file at: {data_path}")
except FileNotFoundError:
    print("⚠️ No data file found, generating sample data")
    # Generate sample data
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
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Run backtest
bt = Backtest(data, BandStochBreakout, cash=1_000_000, commission=0.002)
stats = bt.run()

# Print results
print("🌙✨ Moon Dev Backtest Results:")
print(f"Return: {stats['Return [%]']:0.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:0.2f}")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:0.2f}%")
print(f"Number of Trades: {stats['# Trades']}")