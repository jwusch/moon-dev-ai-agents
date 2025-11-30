# 🌙 Moon Dev's OpenInterestPut Backtest Implementation ✨
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import sys
import os

# Add parent directories to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from utils import get_data_file_path, prepare_backtest_data
except ImportError:
    # If utils not found, define inline
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

# Data Preparation 🌐
try:
    data_path = get_data_file_path('BTC-USD-15m.csv')
    data = pd.read_csv(data_path)
except FileNotFoundError as e:
    print(f"⚠️ No data file found for BTC, generating synthetic data")
    # Generate synthetic data for testing
    import numpy as np
    dates = pd.date_range(start='2023-01-01', end='2023-12-01', freq='15min')
    n = len(dates)
    np.random.seed(42)
    price_series = 30000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'Datetime': dates,
        'Open': price_series + np.random.randn(n) * 50,
        'High': price_series + np.abs(np.random.randn(n) * 100),
        'Low': price_series - np.abs(np.random.randn(n) * 100),
        'Close': price_series,
        'Volume': np.random.randint(100, 10000, n),
        'open_interest': np.random.randint(1000000, 5000000, n)
    })
    data.set_index('Datetime', inplace=True)

# Clean and prepare columns
if 'Datetime' not in data.index.names:
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

# Ensure required columns exist
if 'open_interest' not in data.columns:
    print("⚠️ Warning: 'open_interest' column not found in data. Simulating OI data based on volume...")
    # Simulate open interest based on volume patterns
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        # Create synthetic open interest that correlates with volume
        import numpy as np
        volume_ma = data['Volume'].rolling(window=20).mean().fillna(data['Volume'].mean())
        volume_std = data['Volume'].rolling(window=20).std().fillna(data['Volume'].std())
        
        # Generate OI that trends with volume but has its own patterns
        base_oi = 2000000  # Base open interest
        oi_multiplier = 100  # How much volume affects OI
        noise = np.random.randn(len(data)) * 50000  # Random noise
        
        data['open_interest'] = base_oi + (volume_ma * oi_multiplier) + noise
        data['open_interest'] = data['open_interest'].clip(lower=100000)  # Minimum OI
        data['open_interest'] = data['open_interest'].astype(int)
    else:
        # Fallback: completely random OI
        import numpy as np
        data['open_interest'] = np.random.randint(1000000, 5000000, len(data))

class OpenInterestPut(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌡️
    stop_loss_pct = 0.02  # 2% stop loss 🛑
    take_profit_pct = 0.03  # 3% take profit 🎯
    
    def init(self):
        # Price Trend Indicators 📉
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        
        # Open Interest Analysis 🔍
        if 'open_interest' not in self.data.df.columns:
            raise ValueError("🌙✨ MOON DEV ERROR: 'open_interest' data not available!")
            
        oi_series = self.data.df['open_interest']
        self.oi_max = self.I(talib.MAX, oi_series, timeperiod=20)
        self.oi_sma5 = self.I(talib.SMA, oi_series, timeperiod=5)
        self.oi = self.I(lambda x: x, oi_series)  # Raw OI values
        
    def next(self):
        current_close = self.data.Close[-1]
        current_oi = self.oi[-1]
        oi_max = self.oi_max[-1]
        oi_sma5 = self.oi_sma5[-1]

        # Moon Dev Debug Prints 🌙
        print(f"\n🌕 Price: {current_close:0.2f} | SMA20: {self.sma20[-1]:0.2f}")
        print(f"📊 OI: {current_oi} | OI Max: {oi_max} | OI SMA5: {oi_sma5:0.2f}")

        if not self.position:
            # Entry Logic �
            bearish_trend = current_close < self.sma20[-1]
            high_oi = current_oi >= 0.95 * oi_max
            
            if bearish_trend and high_oi:
                risk_amount = self.equity * self.risk_percent
                entry_price = current_close
                sl_price = entry_price * (1 + self.stop_loss_pct)
                tp_price = entry_price * (1 - self.take_profit_pct)
                risk_per_share = abs(sl_price - entry_price)
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        print(f"🚀 SELL SIGNAL! Size: {position_size}")
                        self.sell(size=position_size, 
                                 sl=sl_price,
                                 tp=tp_price)
        else:
            # Exit Logic 🌧️
            if current_oi > oi_sma5:
                print("🔔 OI Increasing - Closing Position!")
                self.position.close()

# Run Backtest 📊
bt = Backtest(data, OpenInterestPut, cash=1_000_000, commission=0.002)
stats = bt.run()

# Moon Dev Results Print 🌙
print("\n" + "="*55)
print("🌙✨ MOON DEV BACKTEST RESULTS ✨🌙")
print("="*55)
print(stats)
print(stats._strategy)