# 🌙 Moon Dev Divergent Bands Strategy
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Strategy, Backtest
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

class DivergentBands(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2
    kc_period = 20
    kc_mult = 1.5
    rsi_period = 14
    risk_per_trade = 0.02
    
    def init(self):
        # 🌙 Calculate Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS,
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        
        # 🌙 Calculate Keltner Channels
        self.kc_middle = self.I(talib.EMA, self.data.Close, timeperiod=self.kc_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.kc_period)
        
        # Manual calculation for Keltner bands
        self.kc_upper = self.I(lambda: self.kc_middle + (self.kc_mult * self.atr))
        self.kc_lower = self.I(lambda: self.kc_middle - (self.kc_mult * self.atr))
        
        # 🌙 Calculate RSI for divergence detection
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # 🌙 Calculate MACD for additional confirmation
        self.macd, self.signal, self.histogram = self.I(
            talib.MACD,
            self.data.Close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # Band divergence detection
        bb_width = self.bb_upper[-1] - self.bb_lower[-1]
        kc_width = self.kc_upper[-1] - self.kc_lower[-1]
        band_divergence = bb_width - kc_width
        
        print(f"🌙 Moon Dev | Price: {current_price:0.2f} | Band Div: {band_divergence:0.2f} | RSI: {self.rsi[-1]:0.2f}")
        
        if not self.position:
            # 🚀 Long Entry - Bands converging with oversold RSI
            if (band_divergence < 0 and  # BB inside KC (squeeze)
                self.rsi[-1] < 30 and  # Oversold
                self.macd[-1] > self.signal[-1] and  # MACD bullish
                current_price > self.bb_lower[-1]):  # Above lower BB
                
                # Position sizing
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_loss = min(self.bb_lower[-1], self.kc_lower[-1])
                risk_per_share = current_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price + (2 * (current_price - stop_loss))
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 LONG Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
            
            # 📉 Short Entry - Bands diverging with overbought RSI
            elif (band_divergence > bb_width * 0.5 and  # BB expanding outside KC
                  self.rsi[-1] > 70 and  # Overbought
                  self.macd[-1] < self.signal[-1] and  # MACD bearish
                  current_price < self.bb_upper[-1]):  # Below upper BB
                
                # Position sizing
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_loss = max(self.bb_upper[-1], self.kc_upper[-1])
                risk_per_share = stop_loss - current_price
                
                if risk_per_share > 0:
                    position_size = int(risk_amount / risk_per_share)
                    take_profit = current_price - (2 * (stop_loss - current_price))
                    
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"📉 SHORT Entry | Size: {position_size} | SL: {stop_loss:0.2f} | TP: {take_profit:0.2f}")
        
        else:
            # 🛑 Exit conditions
            if self.position.is_long:
                if self.rsi[-1] > 70 or current_price < self.kc_middle[-1]:
                    self.position.close()
                    print(f"🛑 Exit LONG | RSI: {self.rsi[-1]:0.2f}")
            
            elif self.position.is_short:
                if self.rsi[-1] < 30 or current_price > self.kc_middle[-1]:
                    self.position.close()
                    print(f"🛑 Exit SHORT | RSI: {self.rsi[-1]:0.2f}")

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
bt = Backtest(data, DivergentBands, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌕 Print results
print("\n🌕 MOON DEV DIVERGENT BANDS RESULTS 🌕")
print("="*50)
print(f"Return [%]: {stats['Return [%]']:0.2f}")
print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:0.2f}")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:0.2f}")
print(f"Win Rate [%]: {stats['Win Rate [%]']:0.2f}")
print(f"Total Trades: {stats['# Trades']}")
print("="*50)