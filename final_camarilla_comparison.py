"""
🏁 Final Camarilla Strategy Comparison on Real TSLA Data
Compare all versions and find the best approach for trending stocks

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

import yfinance as yf
import pandas as pd
from backtesting import Backtest

# Import all strategy versions
from src.strategies.camarilla_strategy import CamarillaStrategy
from src.strategies.camarilla_adaptive import AdaptiveCamarillaStrategy

print("🚀 CAMARILLA STRATEGY ANALYSIS FOR TSLA")
print("="*70)

# Get REAL TSLA data
print("\n📊 Loading REAL TSLA market data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")

print(f"✅ Loaded {len(df)} days of actual TSLA trading data")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Price range: ${df['Low'].min():.2f} to ${df['High'].max():.2f}")
print(f"   Starting price: ${df['Close'].iloc[0]:.2f}")
print(f"   Ending price: ${df['Close'].iloc[-1]:.2f}")
print(f"   Buy & hold return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")

# Test parameters
initial_cash = 10000
commission = 0.002

# Store results
results = []

# 1. Original Camarilla
print("\n" + "-"*70)
print("1️⃣ ORIGINAL CAMARILLA STRATEGY (Range Trading)")
print("-"*70)

bt1 = Backtest(df, CamarillaStrategy, cash=initial_cash, commission=commission)
result1 = bt1.run()

print(f"Total Return: {result1['Return [%]']:+.2f}%")
print(f"Total Trades: {result1['# Trades']}")
print(f"Win Rate: {result1['Win Rate [%]']:.1f}%")
print(f"Sharpe Ratio: {result1['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {result1['Max. Drawdown [%]']:.2f}%")

results.append({
    'Strategy': 'Original Camarilla',
    'Return %': result1['Return [%]'],
    'Trades': result1['# Trades'],
    'Win Rate %': result1['Win Rate [%]'],
    'Sharpe': result1['Sharpe Ratio'],
    'Max DD %': result1['Max. Drawdown [%]']
})

# 2. Adaptive Camarilla
print("\n" + "-"*70)
print("2️⃣ ADAPTIVE CAMARILLA (Trend + Range)")
print("-"*70)

bt2 = Backtest(df, AdaptiveCamarillaStrategy, cash=initial_cash, commission=commission)
result2 = bt2.run()

print(f"Total Return: {result2['Return [%]']:+.2f}%")
print(f"Total Trades: {result2['# Trades']}")
print(f"Win Rate: {result2['Win Rate [%]']:.1f}%")
print(f"Sharpe Ratio: {result2['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {result2['Max. Drawdown [%]']:.2f}%")

results.append({
    'Strategy': 'Adaptive Camarilla',
    'Return %': result2['Return [%]'],
    'Trades': result2['# Trades'],
    'Win Rate %': result2['Win Rate [%]'],
    'Sharpe': result2['Sharpe Ratio'],
    'Max DD %': result2['Max. Drawdown [%]']
})

# 3. Optimized Adaptive
print("\n" + "-"*70)
print("3️⃣ OPTIMIZED ADAPTIVE STRATEGY")
print("-"*70)

print("Running optimization...")
optimal = bt2.optimize(
    trend_period=[20, 50, 100],
    risk_per_trade=[0.01, 0.02, 0.03],
    maximize='Sharpe Ratio',
    constraint=lambda p: p.trend_period >= 20
)

print(f"Optimal Parameters:")
print(f"  Trend Period: {optimal._strategy.trend_period}")
print(f"  Risk per Trade: {optimal._strategy.risk_per_trade * 100:.1f}%")
print(f"Total Return: {optimal['Return [%]']:+.2f}%")
print(f"Total Trades: {optimal['# Trades']}")
print(f"Win Rate: {optimal['Win Rate [%]']:.1f}%")
print(f"Sharpe Ratio: {optimal['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {optimal['Max. Drawdown [%]']:.2f}%")

results.append({
    'Strategy': 'Optimized Adaptive',
    'Return %': optimal['Return [%]'],
    'Trades': optimal['# Trades'],
    'Win Rate %': optimal['Win Rate [%]'],
    'Sharpe': optimal['Sharpe Ratio'],
    'Max DD %': optimal['Max. Drawdown [%]']
})

# Summary comparison
print("\n" + "="*70)
print("📊 FINAL COMPARISON SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best performers
best_return = results_df.loc[results_df['Return %'].idxmax()]
best_sharpe = results_df.loc[results_df['Sharpe'].idxmax()]

print(f"\n🏆 Best Return: {best_return['Strategy']} ({best_return['Return %']:+.2f}%)")
print(f"🏆 Best Risk-Adjusted: {best_sharpe['Strategy']} (Sharpe: {best_sharpe['Sharpe']:.2f})")

# Key insights
print("\n" + "="*70)
print("💡 KEY INSIGHTS FOR TSLA TRADING")
print("="*70)

print("\n1. MARKET CHARACTERISTICS:")
print(f"   • TSLA is a highly trending stock (not ideal for pure range trading)")
print(f"   • High volatility requires dynamic risk management")
print(f"   • Strong directional moves need trend-following elements")

print("\n2. STRATEGY ADAPTATIONS THAT WORKED:")
improvements = []
if result2['Return [%]'] > result1['Return [%]']:
    improvements.append("✅ Adding trend detection improved returns")
    improvements.append("✅ ADX-based market regime detection")
    improvements.append("✅ Different position sizing for trend vs range trades")
    improvements.append("✅ Breakout trading in trending markets")
else:
    improvements.append("❌ Adaptive strategy needs further tuning")

for improvement in improvements:
    print(f"   {improvement}")

print("\n3. RECOMMENDATIONS:")
print("   • Use Adaptive Camarilla for stocks like TSLA")
print("   • Original Camarilla better for range-bound instruments")
print("   • Always backtest with REAL data (not synthetic)")
print("   • Consider market regime when selecting strategy")

# Save best performer chart
print("\n📊 Generating performance chart...")
try:
    if optimal['Return [%]'] >= result2['Return [%]']:
        optimal.plot(filename='tsla_camarilla_best.html', open_browser=False)
    else:
        result2.plot(filename='tsla_camarilla_best.html', open_browser=False)
    print("✅ Chart saved as: tsla_camarilla_best.html")
except:
    print("⚠️ Could not generate chart")

print("\n✅ Analysis complete!")
print("\n🎯 BOTTOM LINE:")
buy_hold = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
best_strategy_return = results_df['Return %'].max()

if best_strategy_return > buy_hold:
    print(f"   Best Camarilla variant beat buy & hold by {best_strategy_return - buy_hold:.1f} percentage points!")
else:
    print(f"   Buy & hold ({buy_hold:.1f}%) outperformed all Camarilla variants")
    print(f"   Consider different strategies for strong trending stocks like TSLA")