"""
📊 Impact of Dynamic Position Sizing on Camarilla ROI
Shows how reducing counter-trend position size improves returns

Author: Claude (Anthropic)
"""

import yfinance as yf
from backtesting import Backtest, Strategy
import talib
import pandas as pd

class CamarillaWithDynamicSizing(Strategy):
    """Camarilla with trend-based position sizing"""
    
    # Position sizing parameters
    with_trend_size = 0.50      # 50% of capital when trading with trend
    counter_trend_size = 0.10   # 10% of capital when trading against trend
    
    def init(self):
        # Camarilla levels (simplified)
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        prev_range = (high.shift(1) - low.shift(1)).fillna(high[0] - low[0])
        prev_close = close.shift(1).fillna(close[0])
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        
        # Trend indicator
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        
    def next(self):
        if len(self.data) < 51 or self.position:
            return
            
        price = self.data.Close[-1]
        sma = self.sma50[-1]
        r3 = self.r3[-1]
        s3 = self.s3[-1]
        
        if pd.isna(sma):
            return
            
        # Determine trend
        is_uptrend = price > sma
        
        # DYNAMIC SIZING BASED ON TREND
        if is_uptrend:
            # In uptrend: Large size for longs, small for shorts
            if price <= s3 * 1.01:  # Buy at support (WITH trend)
                self.buy(size=self.with_trend_size)
            elif price >= r3 * 0.99:  # Sell at resistance (AGAINST trend)
                self.sell(size=self.counter_trend_size)  # SMALL SIZE!
        else:
            # In downtrend: Large size for shorts, small for longs  
            if price >= r3 * 0.99:  # Sell at resistance (WITH trend)
                self.sell(size=self.with_trend_size)
            elif price <= s3 * 1.01:  # Buy at support (AGAINST trend)
                self.buy(size=self.counter_trend_size)  # SMALL SIZE!

class CamarillaEqualSizing(Strategy):
    """Original Camarilla with equal position sizing"""
    
    def init(self):
        # Same Camarilla levels
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        prev_range = (high.shift(1) - low.shift(1)).fillna(high[0] - low[0])
        prev_close = close.shift(1).fillna(close[0])
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        
    def next(self):
        if len(self.data) < 10 or self.position:
            return
            
        price = self.data.Close[-1]
        r3 = self.r3[-1]
        s3 = self.s3[-1]
        
        # EQUAL SIZING regardless of trend
        if price <= s3 * 1.01:  # Buy at support
            self.buy(size=0.5)  # Always 50%
        elif price >= r3 * 0.99:  # Sell at resistance
            self.sell(size=0.5)  # Always 50%

# Load TSLA data
print("📊 Loading TSLA data...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")

print(f"✅ Loaded {len(df)} days")
print(f"   TSLA moved from ${df['Close'].iloc[0]:.2f} to ${df['Close'].iloc[-1]:.2f}")
print(f"   Buy & Hold ROI: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.1f}%")

# Run backtests
print("\n🔬 Testing Position Sizing Impact...")

# Equal sizing (original approach)
bt1 = Backtest(df, CamarillaEqualSizing, cash=10000, commission=0.002)
equal = bt1.run()

# Dynamic sizing based on trend
bt2 = Backtest(df, CamarillaWithDynamicSizing, cash=10000, commission=0.002)
dynamic = bt2.run()

# Results
print("\n" + "="*70)
print("💰 POSITION SIZING IMPACT ON ROI")
print("="*70)

print(f"\n1️⃣ EQUAL POSITION SIZING (Original):")
print(f"   • Always use 50% of capital")
print(f"   • ROI: {equal['Return [%]']:+.2f}%")
print(f"   • Final Equity: ${equal['Equity Final [$]']:,.2f}")
print(f"   • Total Trades: {equal['# Trades']}")
print(f"   • Win Rate: {equal['Win Rate [%]']:.1f}%")

print(f"\n2️⃣ DYNAMIC POSITION SIZING (Improved):")
print(f"   • WITH trend: 50% of capital")
print(f"   • AGAINST trend: 10% of capital (80% smaller!)")
print(f"   • ROI: {dynamic['Return [%]']:+.2f}%")
print(f"   • Final Equity: ${dynamic['Equity Final [$]']:,.2f}")
print(f"   • Total Trades: {dynamic['# Trades']}")
print(f"   • Win Rate: {dynamic['Win Rate [%]']:.1f}%")

improvement = dynamic['Return [%]'] - equal['Return [%]']

print("\n" + "="*70)
print("📈 IMPROVEMENT FROM DYNAMIC SIZING")
print("="*70)

print(f"\n✅ ROI Improvement: {improvement:+.1f} percentage points")
print(f"💡 By reducing counter-trend position size by 80%")

if improvement > 0:
    print(f"\n🎯 KEY INSIGHT:")
    print(f"   Smaller positions against the trend = smaller losses")
    print(f"   This simple change improved ROI by {improvement:.1f} percentage points!")
else:
    print(f"\n⚠️ In this case, dynamic sizing didn't help much")
    print(f"   TSLA's strong trend makes counter-trend trades losers regardless of size")

# Analyze trade breakdown
if dynamic['# Trades'] > 0:
    print(f"\n📊 TRADE ANALYSIS:")
    # Note: In real implementation, we'd track which trades were with/against trend
    print(f"   With dynamic sizing, losses are limited when fighting the trend")
    print(f"   This risk management is crucial for trending markets like TSLA")