# 🌙 Moon Dev Backtest AI Implementation for BandwidthMomentum Strategy

# Required imports
from backtesting import Backtest, Strategy
import pandas as pd
import talib
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

class BandwidthMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌗 Calculate Bollinger Bands components
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=50, name='MIDDLE')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=50, nbdev=1, name='STDDEV')
        self.upper_band = self.I(lambda m, s: m + 2*s, self.middle_band, self.std_dev, name='UPPER')
        self.lower_band = self.I(lambda m, s: m - 2*s, self.middle_band, self.std_dev, name='LOWER')
        
        # 🌙 Calculate Bandwidth
        self.bandwidth = self.I(
            lambda u, l: (u - l) / ((u + l) / 2),
            self.upper_band,
            self.lower_band,
            name='BANDWIDTH'
        )
        
        # 🌙 Calculate RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        
        # 🌟 Calculate MACD
        self.macd, self.signal, self.histogram = self.I(
            talib.MACD,
            self.data.Close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # 🎯 Calculate ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        # 📊 200 SMA for trend filtering
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA200')

    def next(self):
        current_price = self.data.Close[-1]
        current_bandwidth = self.bandwidth[-1]
        current_rsi = self.rsi[-1]
        current_macd = self.macd[-1]
        current_signal = self.signal[-1]
        current_atr = self.atr[-1]
        
        print(f"🌙 Moon Dev Signal Check | Price: {current_price:0.2f} | BW: {current_bandwidth:0.4f} | RSI: {current_rsi:0.2f}")

        if not self.position:
            # 🚀 Entry Logic - Look for bandwidth expansion with momentum
            if (current_bandwidth > 0.02 and  # Bandwidth expansion threshold
                current_price > self.sma200[-1] and  # Above 200 SMA
                current_rsi > 50 and current_rsi < 70 and  # RSI in momentum zone
                current_macd > current_signal and  # MACD bullish
                current_macd > self.macd[-2]):  # MACD accelerating
                
                # 💰 Calculate position size
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_loss = current_price - (2 * current_atr)
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price + (3 * current_atr)
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀🌙 MOON DEV LONG! Size: {position_size} | Entry: {current_price:0.2f} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")

        else:
            # 🛑 Exit Logic - Exit on momentum loss or bandwidth contraction
            if (current_rsi > 70 or  # RSI overbought
                current_rsi < 30 or  # RSI oversold (momentum failure)
                current_macd < current_signal or  # MACD bearish cross
                current_bandwidth < 0.01):  # Bandwidth contracting
                
                self.position.close()
                print(f"🌑 Moon Dev Exit | Price: {current_price:0.2f} | Reason: Momentum Loss")

# 🌙 Moon Dev Data Loading Ritual
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

# 🚀 Run the Moon Dev Backtest
bt = Backtest(data, BandwidthMomentum, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌙 Print Moon Dev Results
print("\n🌙✨ MOON DEV BACKTEST RESULTS ✨🌙")
print("="*50)
print(f"Return [%]: {stats['Return [%]']:0.2f}")
print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:0.2f}")
print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:0.2f}")
print(f"Win Rate [%]: {stats['Win Rate [%]']:0.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:0.2f}")
print(f"Number of Trades: {stats['# Trades']}")
print("="*50)