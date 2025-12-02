"""
🎯 VXX Capital Utilization Analysis
Understanding actual returns vs capital employed

Author: Claude (Anthropic)
"""

import pandas as pd

# Load the trades
trades_df = pd.read_csv('vxx_simple_trades.csv')
trades_df['Entry'] = pd.to_datetime(trades_df['Entry'], utc=True).dt.tz_localize(None)
trades_df['Exit'] = pd.to_datetime(trades_df['Exit'], utc=True).dt.tz_localize(None)

print("="*70)
print("📊 VXX CAPITAL UTILIZATION ANALYSIS")
print("="*70)

# Calculate time in market
trades_df['Duration_Min'] = (trades_df['Exit'] - trades_df['Entry']).dt.total_seconds() / 60
total_minutes = trades_df['Duration_Min'].sum()

# Calculate market hours
start_date = trades_df['Entry'].min().date()
end_date = trades_df['Exit'].max().date()
trading_days = pd.bdate_range(start_date, end_date).shape[0]
total_market_minutes = trading_days * 6.5 * 60  # 6.5 hours per day

time_in_market_pct = (total_minutes / total_market_minutes) * 100

print(f"\nTRADING STATISTICS:")
print(f"  Period: {start_date} to {end_date}")
print(f"  Trading days: {trading_days}")
print(f"  Total trades: {len(trades_df)}")
print(f"  Total return: {trades_df['PnL%'].sum():.1f}%")

print(f"\nTIME UTILIZATION:")
print(f"  Total time in positions: {total_minutes:.0f} minutes")
print(f"  Total market time available: {total_market_minutes:.0f} minutes")
print(f"  Time in market: {time_in_market_pct:.1f}%")
print(f"  Time flat (no position): {100 - time_in_market_pct:.1f}%")

# Calculate returns
initial_capital = 10000
position_size_pct = 95  # 95% of capital per trade

# Method 1: Compound returns (what backtesting.py does)
compound_capital = initial_capital
for _, trade in trades_df.iterrows():
    trade_capital = compound_capital * (position_size_pct / 100)
    trade_profit = trade_capital * (trade['PnL%'] / 100)
    compound_capital += trade_profit

compound_return_pct = ((compound_capital - initial_capital) / initial_capital) * 100

# Method 2: Simple sum of returns
simple_return_pct = trades_df['PnL%'].sum()

# Method 3: Average capital deployed
avg_capital_deployed = initial_capital * (position_size_pct / 100) * (time_in_market_pct / 100)

print(f"\nCAPITAL & RETURNS:")
print(f"  Initial capital: ${initial_capital:,.0f}")
print(f"  Position size per trade: {position_size_pct}% (${initial_capital * position_size_pct / 100:,.0f})")
print(f"  ")
print(f"  Simple return (sum of trades): {simple_return_pct:.1f}%")
print(f"  Compound return: {compound_return_pct:.1f}%")
print(f"  Final capital: ${compound_capital:,.2f}")

print(f"\nANNUALIZED RETURNS:")
days_in_period = (end_date - start_date).days
years = days_in_period / 365

print(f"  Period length: {days_in_period} days ({years:.2f} years)")
print(f"  ")
print(f"  Simple annualized: {simple_return_pct / years:.1f}%")
print(f"  Compound annualized (CAGR): {((compound_capital/initial_capital)**(1/years) - 1) * 100:.1f}%")

print(f"\nCAPITAL EFFICIENCY:")
print(f"  Average capital deployed: ${avg_capital_deployed:,.0f} ({avg_capital_deployed/initial_capital*100:.1f}% of total)")
print(f"  Return on deployed capital: {simple_return_pct / (time_in_market_pct/100):.1f}%")
print(f"  Return on total capital: {simple_return_pct:.1f}%")

# Risk metrics
print(f"\nRISK METRICS:")
print(f"  Max drawdown: {trades_df['PnL%'].cumsum().min():.1f}%")
print(f"  Win rate: {(trades_df['PnL%'] > 0).sum() / len(trades_df) * 100:.1f}%")
print(f"  Average win: {trades_df[trades_df['PnL%'] > 0]['PnL%'].mean():.2f}%")
print(f"  Average loss: {trades_df[trades_df['PnL%'] < 0]['PnL%'].mean():.2f}%")
print(f"  Profit factor: {abs(trades_df[trades_df['PnL%'] > 0]['PnL%'].sum() / trades_df[trades_df['PnL%'] < 0]['PnL%'].sum()):.2f}")

# Sharpe ratio approximation
daily_returns = trades_df.groupby(trades_df['Entry'].dt.date)['PnL%'].sum()
sharpe = daily_returns.mean() / daily_returns.std() * (252**0.5)
print(f"  Sharpe ratio (approx): {sharpe:.2f}")

print("\n" + "="*70)
print("💡 KEY INSIGHTS")
print("="*70)

print(f"""
The strategy is only in the market {time_in_market_pct:.1f}% of the time, meaning:

1. On $10,000 capital: 
   - You made ${compound_capital - initial_capital:.2f} ({compound_return_pct:.1f}%)
   - Annualized CAGR: {((compound_capital/initial_capital)**(1/years) - 1) * 100:.1f}%

2. Capital efficiency:
   - Only ${avg_capital_deployed:,.0f} was at risk on average
   - The rest sat idle {100 - time_in_market_pct:.1f}% of the time

3. To improve returns you could:
   - Run multiple uncorrelated strategies
   - Trade multiple volatility products (VIXY, UVXY, SVXY)
   - Use the idle capital for other strategies
""")

print("\n✅ Analysis complete!")