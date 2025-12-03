"""
🚀 Test Camarilla + VWAP Strategy on TSLA
Compare performance with and without VWAP

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from src.strategies.camarilla_strategy import CamarillaStrategy
from src.strategies.camarilla_vwap_strategy import CamarillaVWAPStrategy
import talib

# Simple VWAP-only strategy for comparison
class VWAPOnlyStrategy(Strategy):
    """Trade purely based on VWAP crossovers"""
    
    def init(self):
        # Calculate simple VWAP
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        
        # Daily VWAP calculation
        def calc_vwap():
            cumulative_tpv = (typical_price * self.data.Volume).cumsum()
            cumulative_volume = self.data.Volume.cumsum()
            return cumulative_tpv / cumulative_volume
        
        self.vwap = self.I(calc_vwap)
        self.sma20 = self.I(talib.SMA, self.data.Close, 20)
        
    def next(self):
        if len(self.data) < 20 or self.position:
            return
            
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        sma = self.sma20[-1]
        
        if pd.isna(vwap) or pd.isna(sma):
            return
            
        # Buy when price crosses above VWAP in uptrend
        if price > vwap and price > sma and self.data.Close[-2] <= self.vwap[-2]:
            self.buy(size=0.5)
        # Sell when price crosses below VWAP in downtrend
        elif price < vwap and price < sma and self.data.Close[-2] >= self.vwap[-2]:
            self.sell(size=0.5)

# Load TSLA data
print("📊 Loading TSLA data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="1y")  # 1 year for cleaner VWAP

print(f"✅ Loaded {len(df)} days of TSLA data")
print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")

# Calculate buy & hold
buy_hold = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

# Run backtests
print("\n" + "="*70)
print("🔬 STRATEGY COMPARISON")
print("="*70)

strategies = [
    ("Original Camarilla", CamarillaStrategy),
    ("Camarilla + VWAP", CamarillaVWAPStrategy),
    ("VWAP Only", VWAPOnlyStrategy)
]

results = []

for name, strategy_class in strategies:
    print(f"\n{name}:")
    bt = Backtest(df, strategy_class, cash=10000, commission=0.002)
    result = bt.run()
    
    results.append({
        'Strategy': name,
        'ROI %': result['Return [%]'],
        'Sharpe': result['Sharpe Ratio'],
        'Trades': result['# Trades'],
        'Win Rate %': result['Win Rate [%]'],
        'Max DD %': result['Max. Drawdown [%]']
    })
    
    print(f"  ROI: {result['Return [%]']:+.2f}%")
    print(f"  Sharpe: {result['Sharpe Ratio']:.2f}")
    print(f"  Trades: {result['# Trades']}")
    print(f"  Win Rate: {result['Win Rate [%]']:.1f}%")

# Summary comparison
print("\n" + "="*70)
print("📊 FULL COMPARISON")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n📈 Buy & Hold TSLA: {buy_hold:+.1f}%")

# Find improvements
original_roi = results_df[results_df['Strategy'] == 'Original Camarilla']['ROI %'].iloc[0]
vwap_roi = results_df[results_df['Strategy'] == 'Camarilla + VWAP']['ROI %'].iloc[0]
improvement = vwap_roi - original_roi

print("\n" + "="*70)
print("💡 VWAP ENHANCEMENT ANALYSIS")
print("="*70)

print(f"\n✅ ROI Improvement with VWAP: {improvement:+.1f} percentage points")

if improvement > 0:
    print("\n🎯 How VWAP helps:")
    print("  1. Better entry timing - wait for price to pull back to VWAP")
    print("  2. Trend confirmation - price above VWAP = bullish bias")
    print("  3. Dynamic support/resistance - VWAP acts as moving pivot")
    print("  4. Volume validation - high volume at VWAP = stronger signal")
    print("  5. Institutional levels - large traders often use VWAP")
else:
    print("\n⚠️ VWAP didn't improve results in this test")
    print("  Possible reasons:")
    print("  - Daily VWAP resets may not suit swing trading")
    print("  - Need intraday data for better VWAP signals")

# Test VWAP parameters optimization
print("\n" + "="*70)
print("🔧 OPTIMIZING VWAP STRATEGY")
print("="*70)

bt_vwap = Backtest(df, CamarillaVWAPStrategy, cash=10000, commission=0.002)
optimal = bt_vwap.optimize(
    trend_period=[20, 50, 100],
    risk_per_trade=[0.01, 0.02, 0.03],
    maximize='Sharpe Ratio'
)

print(f"Optimal parameters:")
print(f"  Trend Period: {optimal._strategy.trend_period}")
print(f"  Risk per Trade: {optimal._strategy.risk_per_trade * 100:.1f}%")
print(f"  Optimized ROI: {optimal['Return [%]']:+.2f}%")
print(f"  Optimized Sharpe: {optimal['Sharpe Ratio']:.2f}")

# Show example of VWAP in action
print("\n" + "="*70)
print("📈 VWAP IN ACTION")
print("="*70)

# Calculate VWAP for visualization
df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
df['TPV'] = df['TypicalPrice'] * df['Volume']
df['CumTPV'] = df['TPV'].cumsum()
df['CumVolume'] = df['Volume'].cumsum()
df['VWAP'] = df['CumTPV'] / df['CumVolume']

# Find recent examples
recent = df.tail(20)
above_vwap = recent[recent['Close'] > recent['VWAP']]
below_vwap = recent[recent['Close'] < recent['VWAP']]

print(f"\nLast 20 days VWAP analysis:")
print(f"  Days above VWAP: {len(above_vwap)} ({len(above_vwap)/20*100:.0f}%)")
print(f"  Days below VWAP: {len(below_vwap)} ({len(below_vwap)/20*100:.0f}%)")

# Current status
current_price = df['Close'].iloc[-1]
current_vwap = df['VWAP'].iloc[-1]
distance = (current_price - current_vwap) / current_vwap * 100

print(f"\nCurrent TSLA status:")
print(f"  Price: ${current_price:.2f}")
print(f"  VWAP: ${current_vwap:.2f}")
print(f"  Distance from VWAP: {distance:+.1f}%")

if distance > 0:
    print("  📈 Price above VWAP - Bullish bias")
else:
    print("  📉 Price below VWAP - Bearish bias")

print("\n✅ Analysis complete!")

# Save optimal strategy chart
try:
    optimal.plot(filename='camarilla_vwap_results.html', open_browser=False)
    print("📊 Chart saved as: camarilla_vwap_results.html")
except:
    pass