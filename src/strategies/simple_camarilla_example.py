"""
🌙 Simple Camarilla Backtest Example
The easiest way to run the Camarilla strategy

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy

# Create simple test data
print("Creating test data...")
dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
np.random.seed(42)

# Simple price series
prices = 100
price_list = []
for _ in range(len(dates)):
    # Random walk with slight upward bias
    change = np.random.normal(0.0002, 0.015)  # 0.02% drift, 1.5% volatility
    prices = prices * (1 + change)
    price_list.append(prices)

# Create DataFrame
df = pd.DataFrame({
    'Open': price_list,
    'High': [p * 1.01 for p in price_list],
    'Low': [p * 0.99 for p in price_list],
    'Close': price_list,
    'Volume': 1000000
}, index=dates)

print(f"Data created: {len(df)} days")
print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# Run backtest
print("\nRunning backtest...")
bt = Backtest(
    data=df,
    strategy=CamarillaStrategy,
    cash=10000,
    commission=0.002,  # 0.2% commission
    exclusive_orders=True
)

# Execute backtest
results = bt.run()

# Print results
print("\n📊 BACKTEST RESULTS:")
print("=" * 40)
print(f"Return: {results['Return [%]']:.2f}%")
print(f"Buy & Hold Return: {results['Buy & Hold Return [%]']:.2f}%")
print(f"Number of Trades: {results['# Trades']}")
print(f"Win Rate: {results.get('Win Rate [%]', 0):.1f}%")
print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
print(f"Exposure Time: {results['Exposure Time [%]']:.1f}%")

# Run with different parameters
print("\n🔧 Testing different configurations...")

# Conservative settings
conservative = bt.run(
    range_threshold=0.0005,
    stop_loss_buffer=0.002,
    take_profit_ratio=1.5
)
print(f"\nConservative: {conservative['Return [%]']:.2f}% ({conservative['# Trades']} trades)")

# Aggressive settings
aggressive = bt.run(
    range_threshold=0.002,
    stop_loss_buffer=0.0005,
    take_profit_ratio=3.0
)
print(f"Aggressive: {aggressive['Return [%]']:.2f}% ({aggressive['# Trades']} trades)")

# Range trading only
range_only = bt.run(
    use_range_trading=True,
    use_breakout_trading=False
)
print(f"Range Only: {range_only['Return [%]']:.2f}% ({range_only['# Trades']} trades)")

print("\n✅ Done! To use with your own data:")
print("1. Load your CSV: df = pd.read_csv('your_file.csv', index_col='Date', parse_dates=True)")
print("2. Ensure columns: Open, High, Low, Close, Volume")
print("3. Run: bt = Backtest(df, CamarillaStrategy, cash=10000)")
print("4. Execute: results = bt.run()")
print("5. Plot: bt.plot()")