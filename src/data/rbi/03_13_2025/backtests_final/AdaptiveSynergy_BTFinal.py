import pandas as pd
import talib
import pandas_ta as ta
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

class AdaptiveSynergy(Strategy):
    def init(self):
        # Calculate indicators with Moon Dev precision 🌙
        self.macd_line, self.macd_signal_line, _ = self.I(talib.MACD, self.data.Close, 12, 26, 9)
        self.rsi_series = self.I(talib.RSI, self.data.Close, 14)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)
        self.atr_series = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Calculate VWAP using pandas_ta
        # Create a DataFrame for pandas_ta
        df = pd.DataFrame({
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume
        })
        
        vwap_result = ta.vwap(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        
        # If VWAP returns None or empty, use typical price as fallback
        if vwap_result is None or len(vwap_result) == 0:
            vwap_values = (self.data.High + self.data.Low + self.data.Close) / 3
        else:
            vwap_values = vwap_result.ffill().fillna((self.data.High + self.data.Low + self.data.Close) / 3).values
        
        self.vwap = self.I(lambda: vwap_values, name='VWAP')

    def next(self):
        current_close = self.data.Close[-1]
        print(f"🌙 Moon Dev Pulse | Close: {current_close:0.2f} | MACD: {self.macd_line[-1]:0.2f} | RSI: {self.rsi_series[-1]:0.2f}")

        if not self.position:
            # Entry conditions with Moon Dev precision 🌙
            entry_conditions = [
                (self.macd_line[-2] < self.macd_signal_line[-2] and self.macd_line[-1] > self.macd_signal_line[-1]),  # Bullish crossover
                50 < self.rsi_series[-1] < 70,
                current_close > self.upper_bb[-1],
                current_close > self.vwap[-1]
            ]
            
            if all(entry_conditions):
                risk_percent = 0.01
                equity = self.equity
                risk_amount = equity * risk_percent
                entry_price = self.data.Open[-1]  # Next candle's open
                atr_value = self.atr_series[-1]
                
                stop_loss = entry_price - 1.5 * atr_value
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    take_profit = entry_price + 2 * atr_value
                    
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit
                    )
                    print(f"🚀🌙 Moon Dev LONG Launch | Size: {position_size} | Entry: {entry_price:0.2f} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")

        else:
            # Exit conditions with Moon Dev vigilance 🌙
            exit_conditions = [
                (self.macd_signal_line[-2] < self.macd_line[-2] and self.macd_signal_line[-1] > self.macd_line[-1]),  # Bearish crossover
                self.rsi_series[-1] >= 70,
                current_close < self.middle_bb[-1],
                current_close < self.vwap[-1]
            ]
            
            if any(exit_conditions):
                self.position.close()
                print(f"🌑🌙 Moon Dev Exit Signal | Price: {current_close:0.2f} | Equity: {self.equity:0.2f}")

# Moon Dev Data Preparation Ritual 🌙✨
try:
    data_path = get_data_file_path('BTC-USD-15m.csv')
    data = pd.read_csv(data_path)
    print(f"✅ Found data file at: {data_path}")
except FileNotFoundError:
    print("⚠️ No data file found, using sample data")
    # Create sample data for testing
    import numpy as np
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

# Clean columns
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

# Execute Moon Dev Backtest Ritual 🌙💫
bt = Backtest(data, AdaptiveSynergy, cash=1_000_000, commission=0.002)
stats = bt.run()
print(f"🌙✨ Moon Dev Backtest Complete! Return: {stats['Return [%]']:0.2f}% | Sharpe: {stats['Sharpe Ratio']:0.2f}")