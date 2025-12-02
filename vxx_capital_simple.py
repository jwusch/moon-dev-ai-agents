"""
🎯 VXX Returns - Simple Calculation
Clear answer on capital and returns

Author: Claude (Anthropic)
"""

import pandas as pd

# Load trades
trades_df = pd.read_csv('vxx_simple_trades.csv')

print("="*70)
print("📊 VXX STRATEGY RETURNS - SIMPLE BREAKDOWN")
print("="*70)

# Basic info
initial_capital = 10000
position_size = 0.95  # 95% of capital per trade
num_trades = len(trades_df)
total_pnl = trades_df['PnL%'].sum()

# Calculate compound returns (how backtesting.py does it)
capital = initial_capital
for pnl in trades_df['PnL%']:
    trade_size = capital * position_size
    profit = trade_size * (pnl / 100)
    capital += profit

print(f"\n💰 CAPITAL AND RETURNS:")
print(f"  Starting Capital: ${initial_capital:,}")
print(f"  Position Size: {position_size*100:.0f}% of capital per trade")
print(f"  Total Trades: {num_trades}")
print(f"  Period: 50 trading days")

print(f"\n📈 PERFORMANCE:")
print(f"  Simple Return (sum): {total_pnl:.1f}%")
print(f"  Compound Return: {(capital/initial_capital - 1)*100:.1f}%")
print(f"  Final Capital: ${capital:,.2f}")
print(f"  Profit: ${capital - initial_capital:,.2f}")

# Annualized returns
trading_days = 50
annual_trading_days = 252

simple_daily = total_pnl / trading_days
simple_annual = simple_daily * annual_trading_days

compound_daily = (capital/initial_capital)**(1/trading_days) - 1
compound_annual = (1 + compound_daily)**annual_trading_days - 1

print(f"\n📊 ANNUALIZED RETURNS:")
print(f"  Simple:")
print(f"    Daily: {simple_daily:.3f}%")
print(f"    Annual: {simple_annual:.1f}%")
print(f"  Compound (CAGR):")
print(f"    Daily: {compound_daily*100:.3f}%") 
print(f"    Annual: {compound_annual*100:.1f}%")

print(f"\n⏱️ TIME IN MARKET:")
avg_duration = trades_df['Bars'].mean() * 15  # minutes
total_duration = trades_df['Bars'].sum() * 15  # minutes
market_hours_per_day = 6.5
total_market_minutes = trading_days * market_hours_per_day * 60
time_in_market_pct = (total_duration / total_market_minutes) * 100

print(f"  Average trade duration: {avg_duration:.0f} minutes")
print(f"  Time in positions: {time_in_market_pct:.1f}% of market hours")
print(f"  Time flat: {100-time_in_market_pct:.1f}% of market hours")

print(f"\n🎯 BOTTOM LINE:")
print(f"""
Starting with $10,000:
- After 50 days: ${capital:,.2f} (up ${capital-initial_capital:,.2f})
- That's a {(capital/initial_capital - 1)*100:.1f}% return
- Annualized: {compound_annual*100:.1f}% CAGR

The strategy only uses capital {time_in_market_pct:.1f}% of the time,
so most of your money sits idle waiting for signals.
""")