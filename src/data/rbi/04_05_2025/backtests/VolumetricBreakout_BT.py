import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricBreakout(Strategy):
    ema50_period = 50
    ema200_period = 200
    atr_period = 14
    recent_high_window = 20
    volume_sma_period = 20
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        # 🌙 Indicators using TA-Lib via self.I()
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=self.recent_high_window)
        self.avg_volume = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_sma_period)

    def next(self):
        # Wait for indicators to warm up
        if len(self.data.Close) < self.ema200_period:
            return

        # 🌙 Moon Dev Debug Prints
        print(f" Current Close: {self.data.Close[-1]:.2f} | EMA50: {self.ema50[-1]:.2f} vs EMA200: {self.ema200[-1]:.2f}")

        # Entry Logic
        if not self.position:
            # Golden Cross Check
            golden_cross = crossover(self.ema50, self.ema200)[-1]
            if golden_cross:
                print(f" GOLDEN CROSS! EMA50 ({self.ema50[-1]:.2f}) > EMA200 ({self.ema200[-1]:.2f})")

            # Volume Filter
            volume_ok = self.data.Volume[-1] > self.avg_volume[-1]
            if volume_ok:
                print(f" VOLUME SPIKE! {self.data.Volume[-1]:.2f} > Avg {self.avg_volume[-1]:.2f}")

            # Breakout Check
            price_breakout = self.data.Close[-1] > self.recent_high[-1]
            if price_breakout:
                print(f" PRICE BREAKOUT! Close {self.data.Close[-1]:.2f} > Recent High {self.recent_high[-1]:.2f}")

            # All conditions