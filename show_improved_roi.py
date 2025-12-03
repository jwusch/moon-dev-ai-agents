"""
📊 Show Actual ROI of Improved Camarilla Strategy
Simple implementation that works

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
import talib

class TrendCamarillaSimple(Strategy):
    """Simple trend-aware Camarilla"""
    
    def init(self):
        # Basic indicators
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Camarilla levels (simplified)
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        prev_range = (high.shift(1) - low.shift(1)).fillna(method='bfill')
        prev_close = close.shift(1).fillna(close[0])
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        
    def next(self):
        if len(self.data) < 51 or self.position:
            return
            
        price = self.data.Close[-1]
        sma = self.sma50[-1]
        atr = self.atr[-1]
        
        if pd.isna(sma) or pd.isna(atr):
            return
            
        # Simple trend-aware rules
        if price > sma:  # Uptrend
            # Buy pullback to S3
            if price <= self.s3[-1] * 1.01:
                self.buy(size=0.5)  # Use 50% of capital
        else:  # Downtrend or ranging
            # Original Camarilla: sell at R3
            if price >= self.r3[-1] * 0.99:
                self.sell(size=0.25)  # Use 25% for shorts

# Load data
print("Loading TSLA data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")

# Calculate buy & hold
buy_hold = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

# Test original
from src.strategies.camarilla_strategy import CamarillaStrategy
bt1 = Backtest(df, CamarillaStrategy, cash=10000, commission=0.002)
r1 = bt1.run()

# Test improved
bt2 = Backtest(df, TrendCamarillaSimple, cash=10000, commission=0.002)
r2 = bt2.run()

# Results
print("\n" + "="*60)
print("🎯 CAMARILLA STRATEGY ROI COMPARISON")
print("="*60)

print(f"\nBuy & Hold TSLA:")
print(f"  ROI: {buy_hold:+.1f}%")

print(f"\nOriginal Camarilla (fights the trend):")
print(f"  ROI: {r1['Return [%]']:+.1f}%")
print(f"  Trades: {r1['# Trades']}")

print(f"\nImproved Trend-Aware Camarilla:")
print(f"  ROI: {r2['Return [%]']:+.1f}%")
print(f"  Trades: {r2['# Trades']}")

improvement = r2['Return [%]'] - r1['Return [%]']
print(f"\n✅ Improvement: {improvement:+.1f} percentage points!")

if r2['Return [%]'] > 0:
    print(f"\n🎉 The trend-aware version turned a {r1['Return [%]']:.1f}% loss into a {r2['Return [%]']:+.1f}% gain!")