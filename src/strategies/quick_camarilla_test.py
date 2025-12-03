"""
🌙 Quick Camarilla Backtest
Simple example showing how to run the Camarilla strategy

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from datetime import datetime
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy

# 1. Create sample data
print("Creating sample data...")
dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
np.random.seed(42)

# Generate price data
price = 100
prices = []
for _ in range(len(dates)):
    price *= (1 + np.random.normal(0.001, 0.02))  # 0.1% drift, 2% volatility
    prices.append(price)

# Create OHLCV DataFrame
df = pd.DataFrame(index=dates)
df['Close'] = prices
df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.005, len(dates)))
df['High'] = np.maximum(df['Open'], df['Close']) * (1 + abs(np.random.normal(0.01, 0.005, len(dates))))
df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - abs(np.random.normal(0.01, 0.005, len(dates))))
df['Volume'] = 1000000

print(f"Created {len(df)} days of data from ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

# 2. Run backtest
print("\nRunning backtest...")
bt = Backtest(
    df, 
    CamarillaStrategy,
    cash=10000,
    commission=0.002
)

# Use try/except to handle any errors
try:
    stats = bt.run()
    
    # 3. Show results
    print("\n📊 BACKTEST RESULTS:")
    print("="*40)
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
    print(f"# Trades: {stats['# Trades']}")
    
    # 4. Parameter combinations to test
    print("\n🔧 Testing different parameters...")
    
    # Test conservative settings
    stats2 = bt.run(
        range_threshold=0.0005,  # Tighter range
        stop_loss_buffer=0.002,  # Wider stops
        take_profit_ratio=1.5    # Lower targets
    )
    print(f"\nConservative: Return = {stats2['Return [%]']:.2f}%, Trades = {stats2['# Trades']}")
    
    # Test aggressive settings  
    stats3 = bt.run(
        range_threshold=0.002,   # Wider range
        stop_loss_buffer=0.0005, # Tighter stops
        take_profit_ratio=3.0    # Higher targets
    )
    print(f"Aggressive: Return = {stats3['Return [%]']:.2f}%, Trades = {stats3['# Trades']}")
    
    # Test range-only
    stats4 = bt.run(
        use_range_trading=True,
        use_breakout_trading=False
    )
    print(f"Range Only: Return = {stats4['Return [%]']:.2f}%, Trades = {stats4['# Trades']}")
    
    # Test breakout-only
    stats5 = bt.run(
        use_range_trading=False,
        use_breakout_trading=True
    )
    print(f"Breakout Only: Return = {stats5['Return [%]']:.2f}%, Trades = {stats5['# Trades']}")
    
    print("\n✅ Backtest complete!")
    
except Exception as e:
    print(f"\n❌ Error during backtest: {e}")
    print("\nTroubleshooting:")
    print("1. Check if camarilla_strategy.py is in the same directory")
    print("2. Ensure all required libraries are installed: pip install backtesting pandas numpy")
    print("3. The error details above will help identify the issue")

print("\n💡 To use with your own data:")
print("1. Load your CSV: df = pd.read_csv('your_data.csv')")
print("2. Ensure columns: Date (index), Open, High, Low, Close, Volume")
print("3. Run: bt = Backtest(df, CamarillaStrategy, cash=10000)")
print("4. Get results: stats = bt.run()")