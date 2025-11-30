import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ========================
# 🛠 DATA PREPARATION
# ========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping with Moon Dev precision 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})[['Open', 'High', 'Low', 'Close', 'Volume']]

# Simulate order book imbalance (placeholder for real data)
data['imbalance'] = 0.72  # Mock bullish imbalance

# ========================
# 🚀 LIQUIDATION FADE STRATEGY
# ========================
class LiquidationFade(Strategy):
    vwap_period = 15
    risk_percent = 0.01
    max_daily_trades = 5
    trades_today = 0
    
    def init(self):
        # 🌙 Core Indicators
        # Calculate VWAP with fallback
        vwap_result = ta.vwap(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume
        )
        
        if vwap_result is None or (hasattr(vwap_result, '__len__') and len(vwap_result) == 0):
            vwap_values = (self.data.High + self.data.Low + self.data.Close) / 3
        else:
            vwap_values = vwap_result.ffill().fillna((self.data.High + self.data.Low + self.data.Close) / 3).values
            
        self.vwap = self.I(lambda: vwap_values, name='VWAP')
        
        self.price_spike = self.I(talib.MAX,
                                self.data.High,
                                timeperiod=5)  # 5-bar spike window

    def next(self):
        # 🌙 New Day Reset
        if self.data.index[-1].day != self.data.index[-2].day:
            self.trades_today = 0

        if self.trades_today >= self.max_daily_trades:
            return

        # 🌙 Current Market Conditions
        current_close = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        price_deviation = abs(current_close - current_vwap) / current_vwap
        
        # 🌙 Liquidation Spike Detection
        if price_deviation > 0.02 and self.data.imbalance[-1] > 0.7:
            if current_close > current_vwap:  # Bullish liquidation spike
                self.enter_short(current_close, current_vwap)
            else:  # Bearish liquidation spike
                self.enter_long(current_close, current_vwap)

        # 🌙 Time-based Exit
        for trade in self.trades:
            if len(self.data) - trade.entry_bar > 2:  # 30min = 2x15m bars
                trade.close()
                print(f" MOON ALERT: Time Exit at {current_close} ")

    def enter_short(self, price, vwap):
        # 🌙 Risk Calculation
        risk_amount = self.equity * self.risk_percent
        stop_loss = price * 1.03  # 1.5x spike magnitude (2% * 1.5)
        position_size = int(round(risk_amount / (stop_loss - price)))
        
        if position_size > 0:
            self.sell(size=position_size, 
                     sl=stop_loss,
                     tp=vwap)
            self.trades_today += 1
            print(f" SHORT ENTRY  Price: {price} | Size: {position_size} ")

    def enter_long(self, price, vwap):
        # 🌙 Risk Calculation
        risk_amount = self.equity * self.risk_percent
        stop_loss = price * 0.97  # 1.5x spike magnitude
        position_size = int(round(risk_amount / (price - stop_loss)))
        
        if position_size > 0:
            self.buy(size=position_size,
                    sl=stop_loss,
                    tp=vwap)
            self.trades_today += 1
            print(f" LONG ENTRY  Price: {price} | Size: {position_size} ")

# ========================
# 🌙 BACKTEST EXECUTION
# ========================
bt = Backtest(data, LiquidationFade, cash=1_000_000, commission=.002)
stats = bt.run()
print(stats)