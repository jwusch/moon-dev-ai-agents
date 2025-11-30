import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Clean and prepare the data
def prepare_data(filepath):
    data = pd.read_csv(filepath)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map columns to required format
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    # Convert datetime column
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

# Strategy Class
class VengeanceTrender(Strategy):
    # Parameters for optimization
    trailing_stop_pct = 2.0  # Trailing stop percentage
    risk_per_trade = 0.01  # Risk 1% of capital per trade
    atr_period = 14  # ATR period for volatility-based stop loss

    def init(self):
        # Calculate ATR for volatility-based stop loss
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        print(" VengeanceTrender initialized! Ready to ride the trends with a vengeance! ")

    def next(self):
        # Skip if ATR is not calculated yet
        if len(self.atr) < self.atr_period:
            return

        # Calculate position size based on risk percentage
        risk_amount = self.equity * self.risk_per_trade
        atr_value = self.atr[-1]
        position_size = risk_amount / atr_value

        # Entry Logic: Wait for pullbacks in a strong trend
        if not self.position:
            # Long entry: Price above 200 SMA (trend confirmation) and pullback to support
            if self.data.Close[-1] > self.I(talib.SMA, self.data.Close, timeperiod=200)[-1]:
                if self.data.Close[-1] < self.data.Close[-2]:  # Pullback condition
                    print(" Long entry signal detected! Entering with a vengeance! ")
                    self.buy(size=position_size)

            # Short entry: Price below 200 SMA (trend confirmation) and pullback to resistance
            elif self.data.Close[-1] < self.I(talib.SMA, self.data.Close, timeperiod=200)[-1]:
                if self.data.Close[-1] > self.data.Close[-2]:  # Pullback condition
                    print(" Short entry signal detected! Entering with a vengeance! ")
                    self.sell(size=position_size)

        # Exit Logic: Trailing stops and break of support/resistance
        if self.position:
            # Trailing stop for long positions
            if self.position.is_long:
                trailing_stop = self.data.Close[-1] * (1 - self.trailing_stop_pct / 100)
                self.position.sl = max(self.position.sl or 0, trailing_stop)
                print(f" Long position trailing stop updated to {trailing_stop:.2f} ")

            # Trailing stop for short positions
            elif self.position.is_short:
                trailing_stop = self.data.Close[-1] * (1 + self.trailing_stop_pct / 100)
                self.position.sl = min(self.position.sl or float('inf'), trailing_stop)
                print(f" Short position trailing stop updated to {trailing_stop:.2f} ")

            # Exit if price breaks key support/resistance
            if self.position.is_long and self.data.Close[-1] < self.I(talib.MIN, self.data.Low, timeperiod=20)[-1]:
                print(" Long position exited due to support break! ")
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] > self.I(talib.MAX, self.data.High, timeperiod=20)[-1]:
                print(" Short position exited due to resistance break! ")
                self.position.close()

# Load and prepare data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = prepare_data(data_path)

# Run initial backtest
bt = Backtest(data, VengeanceTrender, cash=1_000_000, commission=0.002)
stats = bt.run()
print(" Initial Backtest Results ")
print(stats)
print(stats._strategy)

# Plot initial performance
bt.plot()

# Optimize parameters
optimization_results = bt.optimize(
    trailing_stop_pct=range(1, 5, 1),  # Trailing stop percentage
    risk_per_trade=[0.01, 0.02, 0.03],  # Risk per trade
    atr_period=range(10, 20, 2),  # ATR period
    maximize='Return [%]'
)
print(" Optimization Results ")
print(optimization_results)

# Run backtest with optimized parameters
optimized_stats = bt.run(**optimization_results._params)
print(" Optimized Backtest Results ")
print(optimized_stats)
print(optimized_stats._strategy)

# Plot optimized performance
bt.plot()