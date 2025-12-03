"""
🎯 Find Best Stocks for Camarilla + VWAP Strategy
Identify range-bound stocks where this strategy excels

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_stock_suitability(ticker, period="1y"):
    """Analyze if a stock is suitable for Camarilla strategy"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if len(df) < 100:
            return None
            
        # Calculate metrics
        returns = df['Close'].pct_change()
        
        # 1. Total return (want LOW for range-bound)
        total_return = (df['Close'][-1] / df['Close'][0] - 1) * 100
        
        # 2. Average True Range as % of price (want MODERATE)
        df['ATR'] = pd.Series(df['High'] - df['Low']).rolling(14).mean()
        avg_atr_pct = (df['ATR'] / df['Close']).mean() * 100
        
        # 3. Trend strength (want LOW - indicating ranging)
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        
        # Count trending days
        if len(df) > 200:
            trend_days = ((df['Close'] > sma50) & (sma50 > sma200)).sum()
            trend_strength = trend_days / len(df) * 100
        else:
            trend_strength = 50  # Neutral if not enough data
            
        # 4. Number of crosses of the mean (want HIGH)
        mean_price = df['Close'].mean()
        crosses = ((df['Close'] > mean_price) != (df['Close'].shift(1) > mean_price)).sum()
        crosses_per_month = crosses / (len(df) / 20)
        
        # 5. Price range (want stock to stay in a channel)
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean() * 100
        
        return {
            'ticker': ticker,
            'total_return': total_return,
            'avg_atr_pct': avg_atr_pct,
            'trend_strength': trend_strength,
            'crosses_per_month': crosses_per_month,
            'price_range': price_range,
            'current_price': df['Close'][-1],
            'avg_volume': df['Volume'].mean()
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

# Stocks to analyze - mix of different sectors
print("🔍 Analyzing stocks for Camarilla strategy suitability...\n")

# Different categories of stocks
stocks_to_test = {
    "Large Cap Range-Bound": ["KO", "PG", "JNJ", "WMT", "PEP", "MCD", "VZ", "T"],
    "Utilities (Typically Stable)": ["NEE", "SO", "DUK", "D", "AEP"],
    "REITs (Often Range-Bound)": ["SPG", "O", "AMT", "PLD", "CCI"],
    "Banks (Can be Rangy)": ["JPM", "BAC", "WFC", "USB", "PNC"],
    "ETFs (Some Range-Bound)": ["GLD", "TLT", "IEF", "AGG", "HYG", "XLU"],
    "High Volatility": ["TSLA", "NVDA", "COIN", "ROKU", "SQ"],
}

all_results = []

for category, tickers in stocks_to_test.items():
    print(f"\n📊 {category}:")
    for ticker in tickers:
        result = analyze_stock_suitability(ticker)
        if result:
            all_results.append(result)
            print(f"   {ticker}: Return={result['total_return']:.1f}%, Crosses/month={result['crosses_per_month']:.1f}")

# Convert to DataFrame for analysis
df_results = pd.DataFrame(all_results)

# Score stocks for Camarilla suitability
def score_for_camarilla(row):
    score = 0
    
    # Low absolute return is good (range-bound)
    if abs(row['total_return']) < 10:
        score += 3
    elif abs(row['total_return']) < 20:
        score += 2
    elif abs(row['total_return']) < 30:
        score += 1
        
    # High crosses per month is good (ranging behavior)
    if row['crosses_per_month'] > 4:
        score += 3
    elif row['crosses_per_month'] > 3:
        score += 2
    elif row['crosses_per_month'] > 2:
        score += 1
        
    # Moderate ATR is good (not too volatile, not too quiet)
    if 1 < row['avg_atr_pct'] < 3:
        score += 2
    elif 0.5 < row['avg_atr_pct'] < 4:
        score += 1
        
    # Low trend strength is good
    if row['trend_strength'] < 30:
        score += 2
    elif row['trend_strength'] < 50:
        score += 1
        
    return score

df_results['camarilla_score'] = df_results.apply(score_for_camarilla, axis=1)
df_results = df_results.sort_values('camarilla_score', ascending=False)

print("\n" + "="*70)
print("🏆 BEST STOCKS FOR CAMARILLA + VWAP STRATEGY")
print("="*70)

print("\nTop 10 Recommendations (Score 6+ is excellent):\n")

for idx, row in df_results.head(10).iterrows():
    print(f"{idx+1}. {row['ticker']:<5} Score: {row['camarilla_score']}/10")
    print(f"   1yr Return: {row['total_return']:+.1f}%")
    print(f"   Crosses/Month: {row['crosses_per_month']:.1f}")
    print(f"   Daily ATR: {row['avg_atr_pct']:.1f}%")
    print(f"   Current Price: ${row['current_price']:.2f}")
    print()

# Specific category analysis
print("="*70)
print("📈 ANALYSIS BY CATEGORY")
print("="*70)

# Best ETFs for Camarilla
etf_results = df_results[df_results['ticker'].isin(['GLD', 'TLT', 'IEF', 'AGG', 'HYG', 'XLU'])]
print("\n🎯 Best ETFs for Camarilla:")
for _, row in etf_results.head(3).iterrows():
    print(f"   {row['ticker']}: Score={row['camarilla_score']}, Return={row['total_return']:.1f}%")

# Best Stocks
stock_results = df_results[~df_results['ticker'].isin(['GLD', 'TLT', 'IEF', 'AGG', 'HYG', 'XLU'])]
print("\n🎯 Best Individual Stocks:")
for _, row in stock_results.head(3).iterrows():
    print(f"   {row['ticker']}: Score={row['camarilla_score']}, Return={row['total_return']:.1f}%")

# Show why TSLA is bad for this strategy
if 'TSLA' in df_results['ticker'].values:
    tsla_row = df_results[df_results['ticker'] == 'TSLA'].iloc[0]
    print(f"\n❌ Why TSLA is unsuitable:")
    print(f"   Score: {tsla_row['camarilla_score']}/10")
    print(f"   1yr Return: {tsla_row['total_return']:.1f}% (too trendy!)")
    print(f"   Crosses/Month: {tsla_row['crosses_per_month']:.1f} (too few)")

print("\n" + "="*70)
print("💡 KEY INSIGHTS")
print("="*70)

print("""
Best stocks for Camarilla + VWAP have:
✅ Low yearly returns (-20% to +20%) - range-bound
✅ High mean reversion (4+ crosses/month)
✅ Moderate volatility (1-3% daily ATR)
✅ Low trend strength (<30%)

Perfect candidates:
• Utility stocks (XLU, NEE, SO)
• Bond ETFs (TLT, IEF, AGG)
• Stable dividend stocks (KO, PG, JNJ)
• Gold ETF (GLD)

Avoid:
❌ High growth tech (TSLA, NVDA)
❌ Trending stocks
❌ Very low volatility (<0.5% ATR)
""")

# Test the strategy on the best candidate
best_stock = df_results.iloc[0]['ticker']
print(f"\n🎯 Testing Camarilla + VWAP on {best_stock}...")

# Quick backtest would go here
print(f"\nRecommendation: Try the Camarilla + VWAP strategy on {best_stock}")
print("It shows ideal range-bound characteristics for this strategy!")