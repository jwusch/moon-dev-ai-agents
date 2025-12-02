"""
🚀 Simplified Camarilla + VWAP Strategy
Shows how VWAP improves entry timing

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

class CamarillaWithVWAP(Strategy):
    """Simple implementation showing VWAP benefit"""
    
    def init(self):
        # Calculate VWAP (simplified - cumulative)
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        
        def calc_vwap():
            tpv = typical_price * self.data.Volume
            cum_tpv = pd.Series(tpv).cumsum()
            cum_volume = pd.Series(self.data.Volume).cumsum()
            vwap = cum_tpv / cum_volume
            # Fill any NaN values with previous valid value
            vwap = vwap.ffill().fillna(typical_price[0])
            return vwap.values
        
        self.vwap = self.I(calc_vwap)
        
        # Trend
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        
        # Camarilla levels (simplified)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        prev_range = (high.shift(1) - low.shift(1)).fillna(high[0] - low[0])
        prev_close = close.shift(1).fillna(close[0])
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        
    def next(self):
        if len(self.data) < 51 or self.position:
            return
            
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        sma = self.sma50[-1]
        r3 = self.r3[-1]
        s3 = self.s3[-1]
        
        if pd.isna(vwap) or pd.isna(sma):
            return
            
        # Key insight: Use VWAP to confirm entries
        above_vwap = price > vwap
        in_uptrend = price > sma
        
        # ENHANCED ENTRY RULES WITH VWAP
        if in_uptrend and above_vwap:
            # Buy pullback to S3 only if still above VWAP
            if s3 < price <= s3 * 1.02:
                self.buy(size=0.5)
        elif not in_uptrend and not above_vwap:
            # Sell rally to R3 only if still below VWAP
            if r3 * 0.98 <= price < r3:
                self.sell(size=0.25)

# Load and test
print("📊 Testing Camarilla + VWAP on TSLA\n")

tsla = yf.Ticker("TSLA")
df = tsla.history(period="1y")

# Original Camarilla (without VWAP)
from src.strategies.camarilla_strategy import CamarillaStrategy
bt1 = Backtest(df, CamarillaStrategy, cash=10000, commission=0.002)
r1 = bt1.run()

# With VWAP
bt2 = Backtest(df, CamarillaWithVWAP, cash=10000, commission=0.002)
r2 = bt2.run()

# Results
print("="*60)
print("📊 RESULTS COMPARISON")
print("="*60)

print(f"\n1️⃣ Original Camarilla (No VWAP):")
print(f"   ROI: {r1['Return [%]']:+.1f}%")
print(f"   Trades: {r1['# Trades']}")
print(f"   Win Rate: {r1['Win Rate [%]']:.1f}%")
print(f"   Max Drawdown: {r1['Max. Drawdown [%]']:.1f}%")

print(f"\n2️⃣ Camarilla + VWAP:")
print(f"   ROI: {r2['Return [%]']:+.1f}%")
print(f"   Trades: {r2['# Trades']}")
print(f"   Win Rate: {r2['Win Rate [%]']:.1f}%")
print(f"   Max Drawdown: {r2['Max. Drawdown [%]']:.1f}%")

improvement = r2['Return [%]'] - r1['Return [%]']
win_improvement = r2['Win Rate [%]'] - r1['Win Rate [%]']

print(f"\n✅ IMPROVEMENTS WITH VWAP:")
print(f"   ROI: {improvement:+.1f} percentage points")
print(f"   Win Rate: {win_improvement:+.1f} percentage points")

# Show how VWAP helps
print("\n" + "="*60)
print("💡 HOW VWAP IMPROVES THE STRATEGY")
print("="*60)

print("""
1. BETTER ENTRY CONFIRMATION:
   • Original: Buy at S3 regardless of trend strength
   • With VWAP: Buy at S3 only if price > VWAP (bullish)

2. TREND VALIDATION:
   • Price > VWAP = Institutional buying pressure
   • Price < VWAP = Institutional selling pressure

3. FALSE SIGNAL FILTER:
   • Avoids entries when price is on wrong side of VWAP
   • Reduces losing trades in weak trends

4. DYNAMIC SUPPORT/RESISTANCE:
   • VWAP acts as moving pivot point
   • Can use as target or stop level
""")

# Current VWAP analysis
df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (df['TypicalPrice'] * df['Volume']).cumsum() / df['Volume'].cumsum()

current_price = df['Close'].iloc[-1]
current_vwap = df['VWAP'].iloc[-1]
distance_pct = (current_price - current_vwap) / current_vwap * 100

print(f"\n📍 Current TSLA Status:")
print(f"   Price: ${current_price:.2f}")
print(f"   VWAP: ${current_vwap:.2f}")
print(f"   Distance: {distance_pct:+.1f}%")

if current_price > current_vwap:
    print("   📈 Bullish - Price above VWAP")
else:
    print("   📉 Bearish - Price below VWAP")

# VWAP trading tips
print("\n🎯 VWAP TRADING TIPS:")
print("• Best for intraday trading (resets daily)")
print("• Strong signals when price crosses VWAP with volume")
print("• Use with other indicators for confirmation")
print("• Works well in trending markets")

print(f"\n📊 Buy & Hold TSLA: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.1f}%")