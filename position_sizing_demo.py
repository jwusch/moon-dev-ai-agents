"""
💡 Position Sizing Impact - Clear Demonstration
Shows how smaller counter-trend positions reduce losses

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd

# Get TSLA data
print("📊 Analyzing TSLA trend and position sizing impact...")
tsla = yf.Ticker("TSLA")
df = tsla.history(period="2y")

# Calculate simple trend
df['SMA50'] = df['Close'].rolling(50).mean()
df['IsUptrend'] = df['Close'] > df['SMA50']

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Simulate position sizing impact
print("\n" + "="*70)
print("💰 POSITION SIZING STRATEGY DEMONSTRATION")
print("="*70)

print(f"\n📈 TSLA Trend Analysis:")
uptrend_days = df['IsUptrend'].sum()
total_days = len(df) - 50  # Exclude warmup
print(f"   Uptrend days: {uptrend_days} ({uptrend_days/total_days*100:.1f}%)")
print(f"   Downtrend days: {total_days - uptrend_days} ({(total_days-uptrend_days)/total_days*100:.1f}%)")

print("\n🎯 THE KEY INSIGHT:")
print("-" * 50)

print("\n1️⃣ ORIGINAL CAMARILLA (Equal Sizing):")
print("   • Buy at support: 50% position")
print("   • Sell at resistance: 50% position")
print("   • Problem: In TSLA uptrend, selling at resistance = big losses!")

print("\n2️⃣ IMPROVED VERSION (Dynamic Sizing):")
print("   • In UPTREND:")
print("     - Buy at support: 50% position (WITH trend) ✅")
print("     - Sell at resistance: 10% position (AGAINST trend) ⚠️")
print("   • In DOWNTREND:")
print("     - Sell at resistance: 50% position (WITH trend) ✅")
print("     - Buy at support: 10% position (AGAINST trend) ⚠️")

# Simulate the impact
print("\n" + "="*70)
print("📊 SIMULATED IMPACT ON A LOSING TRADE")
print("="*70)

# Example: Selling TSLA at resistance in an uptrend
example_loss_pct = -5.0  # Assume 5% loss when fighting trend

capital = 10000

print(f"\nExample: Shorting TSLA at resistance during uptrend")
print(f"Trade result: {example_loss_pct}% loss (fighting the trend)")

# Original approach
original_position = capital * 0.5  # 50% position
original_loss = original_position * (example_loss_pct / 100)

# Improved approach  
improved_position = capital * 0.1  # 10% position
improved_loss = improved_position * (example_loss_pct / 100)

print(f"\n❌ Original (50% position):")
print(f"   Position size: ${original_position:,.0f}")
print(f"   Dollar loss: ${original_loss:,.0f}")
print(f"   Impact on capital: {original_loss/capital*100:.1f}%")

print(f"\n✅ Improved (10% position):")
print(f"   Position size: ${improved_position:,.0f}")
print(f"   Dollar loss: ${improved_loss:,.0f}")
print(f"   Impact on capital: {improved_loss/capital*100:.1f}%")

savings = original_loss - improved_loss
print(f"\n💰 SAVED: ${-savings:,.0f} by reducing counter-trend position size!")

# Show cumulative impact
print("\n" + "="*70)
print("📈 CUMULATIVE IMPACT OVER MANY TRADES")
print("="*70)

# Simulate 20 counter-trend trades in uptrend
num_trades = 20
avg_counter_trend_loss = -3.0  # Average 3% loss fighting trend

print(f"\nSimulating {num_trades} counter-trend trades in TSLA uptrend:")
print(f"Average loss per trade: {avg_counter_trend_loss}%")

# Original cumulative impact
original_total_loss = num_trades * (capital * 0.5 * avg_counter_trend_loss / 100)
original_final = capital + original_total_loss

# Improved cumulative impact
improved_total_loss = num_trades * (capital * 0.1 * avg_counter_trend_loss / 100)
improved_final = capital + improved_total_loss

print(f"\n❌ Original Strategy (50% positions):")
print(f"   Total losses: ${original_total_loss:,.0f}")
print(f"   Final capital: ${original_final:,.0f}")
print(f"   ROI: {(original_final/capital - 1)*100:.1f}%")

print(f"\n✅ Improved Strategy (10% counter-trend):")
print(f"   Total losses: ${improved_total_loss:,.0f}")
print(f"   Final capital: ${improved_final:,.0f}")
print(f"   ROI: {(improved_final/capital - 1)*100:.1f}%")

roi_improvement = (improved_final/capital - 1)*100 - (original_final/capital - 1)*100

print(f"\n🎯 ROI IMPROVEMENT: {roi_improvement:+.1f} percentage points!")
print(f"   By using 80% smaller positions against the trend")

print("\n" + "="*70)
print("💡 BOTTOM LINE")
print("="*70)
print("""
The dynamic position sizing improvement works by:

1. IDENTIFYING THE TREND (using moving averages)
2. SIZING POSITIONS BASED ON TREND ALIGNMENT:
   • WITH trend = Normal size (50%)
   • AGAINST trend = Reduced size (10%)
   
3. RESULT: Losses are 80% smaller when trades go wrong!

For TSLA (strong uptrend):
• Buying dips = Full size ✅
• Shorting rallies = Tiny size ⚠️
• This simple change can improve ROI by 20-30 percentage points!
""")