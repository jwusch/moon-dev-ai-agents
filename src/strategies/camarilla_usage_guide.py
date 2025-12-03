"""
🌙 Camarilla Strategy Usage Guide
Step-by-step guide for using the Camarilla strategy with backtesting.py

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from datetime import datetime
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy

print("🌙 CAMARILLA STRATEGY - USAGE GUIDE")
print("="*60)

print("\n📚 METHOD 1: Load Your Own CSV Data")
print("-"*40)
print("""
# Load your data
df = pd.read_csv('your_data.csv')

# Ensure it has the required columns:
# - Date (as index)
# - Open, High, Low, Close, Volume

# Set date as index if needed:
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Run backtest
bt = Backtest(df, CamarillaStrategy, cash=10000, commission=0.001)
stats = bt.run()
print(f"Return: {stats['Return [%]']:.2f}%")
""")

print("\n📚 METHOD 2: Use with Crypto/Forex (High Value Assets)")
print("-"*40)
print("""
# For Bitcoin or other high-value assets, increase initial cash
# This prevents "insufficient margin" errors

bt = Backtest(df, CamarillaStrategy, 
              cash=1000000,  # Use 1M for BTC
              commission=0.001)
stats = bt.run()
""")

print("\n📚 METHOD 3: Optimize Strategy Parameters")
print("-"*40)
print("""
# Find best parameters for your data
stats = bt.optimize(
    range_threshold=[0.0005, 0.001, 0.002],
    breakout_confirmation=[0, 1, 2],
    stop_loss_buffer=[0.0005, 0.001, 0.002],
    take_profit_ratio=[1.5, 2.0, 2.5, 3.0],
    maximize='Sharpe Ratio'  # or 'Return [%]'
)
""")

print("\n📚 METHOD 4: Custom Parameters")
print("-"*40)
print("""
# Run with specific parameters
stats = bt.run(
    # S3/R3 proximity threshold (0.1% = 0.001)
    range_threshold=0.001,
    
    # Extra stop loss buffer beyond S4/R4
    stop_loss_buffer=0.001,
    
    # Risk/reward ratio for targets
    take_profit_ratio=2.0,
    
    # Enable/disable trading modes
    use_range_trading=True,
    use_breakout_trading=True,
    
    # Breakout confirmation bars
    breakout_confirmation=1
)
""")

print("\n📚 METHOD 5: Analyze Results")
print("-"*40)
print("""
# Print key metrics
print(f"Total Return: {stats['Return [%]']:.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Total Trades: {stats['# Trades']}")
print(f"Profit Factor: {stats['Profit Factor']:.2f}")

# Generate HTML plot
bt.plot(filename='results.html')
""")

print("\n📚 METHOD 6: Compare Different Markets")
print("-"*40)
print("""
markets = {
    'BTC': load_btc_data(),
    'ETH': load_eth_data(),
    'EURUSD': load_forex_data()
}

for name, data in markets.items():
    bt = Backtest(data, CamarillaStrategy, cash=100000)
    stats = bt.run()
    print(f"{name}: Return = {stats['Return [%]']:.2f}%")
""")

print("\n⚠️  COMMON ISSUES AND SOLUTIONS:")
print("-"*40)
print("""
1. "Insufficient margin" errors:
   → Increase initial cash or reduce position sizing
   → Modify _calculate_position_size() in strategy

2. Poor performance on trending markets:
   → Camarilla works best in ranging markets
   → Consider disabling range trading in strong trends
   
3. Too many/few trades:
   → Adjust range_threshold parameter
   → Tighter = fewer trades, Wider = more trades

4. Missing data columns:
   → Ensure your DataFrame has: Open, High, Low, Close, Volume
   → Date should be the index
""")

print("\n🎯 BEST PRACTICES:")
print("-"*40)
print("""
1. Backtest on at least 1-2 years of data
2. Use appropriate commission rates (0.1-0.2% for crypto)
3. Test on both trending and ranging market periods
4. Optimize parameters for your specific market
5. Always validate with out-of-sample data
""")

print("\n📋 QUICK START TEMPLATE:")
print("-"*40)
print("""
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy
import pandas as pd

# Load data
df = pd.read_csv('BTCUSD_daily.csv', index_col='Date', parse_dates=True)

# Run backtest
bt = Backtest(df, CamarillaStrategy, cash=100000, commission=0.001)
stats = bt.run()

# Show results
print(f"Return: {stats['Return [%]']:.2f}%")
print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
print(f"Trades: {stats['# Trades']}")

# Plot
bt.plot()
""")

print("\n✅ Ready to use the Camarilla strategy with your own data!")
print("📁 Files needed: camarilla_strategy.py + your OHLCV data")
print("🚀 Happy trading!")