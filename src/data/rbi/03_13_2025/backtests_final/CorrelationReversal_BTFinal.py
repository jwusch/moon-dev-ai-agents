# 🌙 Moon Dev Correlation Reversal Strategy
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

class CorrelationReversal(Strategy):
    # Parameters
    vix_period = 14
    corr_period = 20
    risk_per_trade = 0.02  # 2% risk per trade
    
    def init(self):
        # 🌙 Calculate ATR (VIX proxy)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.vix_period)
        
        # 🌙 Calculate price rate of change for correlation
        self.roc = self.I(talib.ROC, self.data.Close, timeperiod=1)
        
        # 🌙 Calculate moving correlation of returns
        # Using rolling correlation between price and volume as proxy
        def rolling_correlation(close, volume, period):
            # Convert to pandas for correlation calculation
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)
            
            # Calculate rolling correlation
            corr = close_series.rolling(window=period).corr(volume_series)
            return corr.fillna(0).values
        
        self.correlation = self.I(
            rolling_correlation,
            self.data.Close,
            self.data.Volume,
            self.corr_period,
            name='CORRELATION'
        )
        
        # 🌙 Calculate RSI for additional confirmation
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # 🌙 Moving averages for trend
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
    def next(self):
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_corr = self.correlation[-1]
        current_rsi = self.rsi[-1]
        
        print(f"🌙 Moon Dev | Price: {current_price:0.2f} | ATR: {current_atr:0.2f} | Corr: {current_corr:0.3f}")
        
        if not self.position:
            # 🚀 Entry Logic - Look for correlation reversals
            # Low correlation suggests decorrelation and potential reversal
            if (abs(current_corr) < 0.2 and  # Low correlation
                current_rsi < 30 and  # Oversold
                current_price > self.sma50[-1] and  # Above short-term MA
                current_atr > self.atr[-20:].mean()):  # Volatility expansion
                
                # Calculate position size
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_loss = current_price - (2 * current_atr)
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price + (3 * current_atr)
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 LONG Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
                    
            # 📉 Short entry conditions
            elif (abs(current_corr) < 0.2 and  # Low correlation
                  current_rsi > 70 and  # Overbought
                  current_price < self.sma50[-1] and  # Below short-term MA
                  current_atr > self.atr[-20:].mean()):  # Volatility expansion
                
                # Calculate position size
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_loss = current_price + (2 * current_atr)
                risk_per_share = stop_loss - current_price
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price - (3 * current_atr)
                    
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"📉 SHORT Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
        
        else:
            # 🛑 Exit conditions - High correlation suggests trend continuation
            if abs(current_corr) > 0.7:  # High correlation, trend following
                self.position.close()
                print(f"🛑 Exit Position | High Correlation: {current_corr:0.3f}")

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
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
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
bt = Backtest(data, CorrelationReversal, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌕 Print results
print("\n🌕 MOON DEV BACKTEST RESULTS 🌕")
print("="*50)
print(f"Return [%]: {stats['Return [%]']:0.2f}")
print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:0.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:0.2f}")
print(f"Win Rate [%]: {stats['Win Rate [%]']:0.2f}")
print(f"Profit Factor: {stats['Profit Factor']:0.2f}")
print(f"Total Trades: {stats['# Trades']}")
print("="*50)