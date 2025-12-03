"""
🚀 Simple Camarilla Backtest with Real TSLA Data
Direct and simple implementation

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

import yfinance as yf
import pandas as pd
from backtesting import Backtest
from src.strategies.camarilla_strategy import CamarillaStrategy

# Get real TSLA data directly
print("🚀 Fetching real TSLA data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")  # 2 years of data

print(f"✅ Got {len(df)} days of TSLA data")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")

# Run backtest
print("\n📊 Running Camarilla backtest on real TSLA data...")
bt = Backtest(
    df,
    CamarillaStrategy,
    cash=10000,
    commission=0.002
)

# Test default parameters
results = bt.run()

print("\n📈 Results with REAL TSLA data:")
print("="*50)
print(f"Total Return........ {results['Return [%]']:8.2f}%")
print(f"Annual Return....... {results['Return (Ann.) [%]']:8.2f}%")
print(f"Sharpe Ratio........ {results['Sharpe Ratio']:8.2f}")
print(f"Max Drawdown........ {results['Max. Drawdown [%]']:8.2f}%")
print(f"Total Trades........ {results['# Trades']:8.0f}")
print(f"Win Rate............ {results['Win Rate [%]']:8.2f}%")

# Save detailed results
print(f"\n💾 Saving detailed trade log...")
trades_df = results._trades
if len(trades_df) > 0:
    trades_df.to_csv('tsla_camarilla_trades.csv')
    print(f"   Saved {len(trades_df)} trades to tsla_camarilla_trades.csv")
    
    # Show sample trades
    print(f"\n📋 Sample trades:")
    print(trades_df[['EntryTime', 'ExitTime', 'PnL', 'ReturnPct']].head())

# Try optimization
print(f"\n🔧 Optimizing parameters...")
optimal = bt.optimize(
    range_threshold=[0.001, 0.002, 0.003, 0.004, 0.005],
    stop_loss_buffer=[0.001, 0.0015, 0.002, 0.0025],
    maximize='Return [%]'
)

print(f"\n✅ Optimal parameters:")
print(f"   Range threshold: {optimal._strategy.range_threshold}")
print(f"   Stop loss buffer: {optimal._strategy.stop_loss_buffer}")
print(f"   Optimized return: {optimal['Return [%]']:.2f}%")

# Generate plot
print(f"\n📊 Generating performance plot...")
try:
    optimal.plot(filename='tsla_camarilla_real_results.html', open_browser=False)
    print("   ✅ Saved as tsla_camarilla_real_results.html")
except:
    print("   ⚠️ Could not generate plot")

print(f"\n🎯 Summary:")
print(f"   • Used REAL TSLA historical data from YFinance")
print(f"   • NOT synthetic/random data") 
print(f"   • Results reflect actual market conditions")
print(f"   • {len(df)} days of real trading data analyzed")