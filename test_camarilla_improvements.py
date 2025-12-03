"""
🚀 Test Camarilla Strategy Improvements on TSLA
Compare original vs improved strategies

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

import yfinance as yf
import pandas as pd
from backtesting import Backtest
from src.strategies.camarilla_strategy import CamarillaStrategy
from src.strategies.camarilla_trend_simple import CamarillaTrendSimple

# Get TSLA data
print("📊 Fetching TSLA data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")
print(f"✅ Got {len(df)} days of TSLA data")
print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
print(f"   Current price: ${df['Close'].iloc[-1]:.2f}")

# Test original strategy
print("\n" + "="*60)
print("1️⃣ ORIGINAL CAMARILLA STRATEGY")
print("="*60)

bt_original = Backtest(df, CamarillaStrategy, cash=10000, commission=0.002)
original_results = bt_original.run()

print(f"Return.............. {original_results['Return [%]']:+8.2f}%")
print(f"Sharpe Ratio........ {original_results['Sharpe Ratio']:8.2f}")
print(f"Max Drawdown........ {original_results['Max. Drawdown [%]']:8.2f}%")
print(f"Win Rate............ {original_results['Win Rate [%]']:8.2f}%")
print(f"Total Trades........ {original_results['# Trades']:8.0f}")

# Test improved trend-aware strategy
print("\n" + "="*60)
print("2️⃣ TREND-AWARE CAMARILLA STRATEGY")
print("="*60)

bt_trend = Backtest(df, CamarillaTrendSimple, cash=10000, commission=0.002)
trend_results = bt_trend.run()

print(f"Return.............. {trend_results['Return [%]']:+8.2f}%")
print(f"Sharpe Ratio........ {trend_results['Sharpe Ratio']:8.2f}")
print(f"Max Drawdown........ {trend_results['Max. Drawdown [%]']:8.2f}%")
print(f"Win Rate............ {trend_results['Win Rate [%]']:8.2f}%")
print(f"Total Trades........ {trend_results['# Trades']:8.0f}")

# Calculate improvements
print("\n" + "="*60)
print("3️⃣ IMPROVEMENTS")
print("="*60)

return_improvement = trend_results['Return [%]'] - original_results['Return [%]']
sharpe_improvement = trend_results['Sharpe Ratio'] - original_results['Sharpe Ratio']
winrate_improvement = trend_results['Win Rate [%]'] - original_results['Win Rate [%]']

print(f"Return Improvement... {return_improvement:+8.2f} percentage points")
print(f"Sharpe Improvement... {sharpe_improvement:+8.2f}")
print(f"Win Rate Improvement. {winrate_improvement:+8.2f} percentage points")

# Optimize the trend strategy
print("\n" + "="*60)
print("4️⃣ PARAMETER OPTIMIZATION")
print("="*60)

print("Testing different trend periods...")
optimal = bt_trend.optimize(
    trend_ma=[20, 50, 100, 200],
    position_size_pct=[0.1, 0.25, 0.5],
    maximize='Sharpe Ratio',
    constraint=lambda p: p.trend_ma >= 20
)

print(f"\nOptimal Parameters:")
print(f"  Trend MA........... {optimal._strategy.trend_ma}")
print(f"  Position Size...... {optimal._strategy.position_size_pct * 100:.0f}%")
print(f"  Optimal Return..... {optimal['Return [%]']:+8.2f}%")
print(f"  Optimal Sharpe..... {optimal['Sharpe Ratio']:8.2f}")
print(f"  Optimal Win Rate... {optimal['Win Rate [%]']:8.2f}%")

# Test on different market conditions
print("\n" + "="*60)
print("5️⃣ PERFORMANCE IN DIFFERENT MARKET CONDITIONS")
print("="*60)

# Split data into periods
mid_point = len(df) // 2

# First half
df_first = df.iloc[:mid_point]
bt_first = Backtest(df_first, CamarillaTrendSimple, cash=10000, commission=0.002)
first_results = bt_first.run()

# Second half
df_second = df.iloc[mid_point:]
bt_second = Backtest(df_second, CamarillaTrendSimple, cash=10000, commission=0.002)
second_results = bt_second.run()

print(f"First Year Performance:")
print(f"  Period: {df_first.index[0].date()} to {df_first.index[-1].date()}")
print(f"  Return: {first_results['Return [%]']:+.2f}%")
print(f"  Sharpe: {first_results['Sharpe Ratio']:.2f}")

print(f"\nSecond Year Performance:")
print(f"  Period: {df_second.index[0].date()} to {df_second.index[-1].date()}")
print(f"  Return: {second_results['Return [%]']:+.2f}%")
print(f"  Sharpe: {second_results['Sharpe Ratio']:.2f}")

# Analyze trades
if len(trend_results._trades) > 0:
    trades = trend_results._trades
    print("\n" + "="*60)
    print("6️⃣ TRADE ANALYSIS")
    print("="*60)
    
    # Calculate trade statistics
    winning_trades = trades[trades['PnL'] > 0]
    losing_trades = trades[trades['PnL'] < 0]
    
    print(f"Total Trades......... {len(trades)}")
    print(f"Winning Trades....... {len(winning_trades)}")
    print(f"Losing Trades........ {len(losing_trades)}")
    
    if len(winning_trades) > 0:
        print(f"Avg Win.............. {winning_trades['ReturnPct'].mean():.2%}")
        print(f"Largest Win.......... {winning_trades['ReturnPct'].max():.2%}")
    
    if len(losing_trades) > 0:
        print(f"Avg Loss............. {losing_trades['ReturnPct'].mean():.2%}")
        print(f"Largest Loss......... {losing_trades['ReturnPct'].min():.2%}")
    
    # Profit factor
    if len(losing_trades) > 0 and losing_trades['PnL'].sum() < 0:
        profit_factor = winning_trades['PnL'].sum() / abs(losing_trades['PnL'].sum())
        print(f"Profit Factor........ {profit_factor:.2f}")

# Key improvements summary
print("\n" + "="*60)
print("🎯 KEY IMPROVEMENTS FOR TRENDING MARKETS")
print("="*60)

improvements = [
    "✅ Trend detection using moving average",
    "✅ Trade with trend direction (buy pullbacks in uptrends)",
    "✅ Dynamic stops based on ATR (volatility-adjusted)",
    "✅ Trailing stops to capture trends",
    "✅ Reduced position size for counter-trend trades",
    "✅ RSI filter to avoid overbought/oversold extremes"
]

for improvement in improvements:
    print(f"  {improvement}")

print("\n💡 CONCLUSION:")
if return_improvement > 0:
    print(f"  The trend-aware strategy improved returns by {return_improvement:.1f} percentage points!")
    print(f"  Better suited for trending stocks like TSLA")
else:
    print(f"  Original strategy performed better in this period")
    print(f"  Consider different parameters or market conditions")

# Save results
try:
    optimal.plot(filename='tsla_camarilla_improved.html', open_browser=False)
    print("\n✅ Performance chart saved as: tsla_camarilla_improved.html")
except:
    pass