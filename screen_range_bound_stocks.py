"""
📊 Screen for Range-Bound Stocks
Find stocks that actually trade sideways, not trending

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def analyze_stock_for_range(ticker, period="1y"):
    """Analyze if a stock is range-bound"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if len(df) < 200:
            return None
        
        # Calculate metrics for range-bound detection
        
        # 1. Total return (want LOW)
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        
        # 2. How much time spent in a range
        sma50 = df['Close'].rolling(50).mean()
        upper_band = sma50 * 1.10  # 10% bands
        lower_band = sma50 * 0.90
        
        in_range = ((df['Close'] < upper_band) & (df['Close'] > lower_band)).sum()
        pct_in_range = in_range / len(df) * 100
        
        # 3. Number of times crossed the mean
        mean_price = df['Close'].mean()
        crosses = ((df['Close'] > mean_price) != (df['Close'].shift(1) > mean_price)).sum()
        
        # 4. Standard deviation of returns
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 5. Trend strength (want LOW)
        # Linear regression slope
        x = np.arange(len(df))
        y = df['Close'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend_strength = abs(slope) / df['Close'].mean() * 252 * 100  # Annualized
        
        # 6. High-Low range consistency
        daily_range = (df['High'] - df['Low']) / df['Close'] * 100
        range_consistency = daily_range.std()
        
        # 7. Support/Resistance levels
        # Find peaks and troughs
        highs = df['High'].rolling(20).max()
        lows = df['Low'].rolling(20).min()
        
        # Count how many times we hit similar levels
        resistance_hits = 0
        support_hits = 0
        
        for level in highs.unique()[-10:]:  # Last 10 resistance levels
            hits = ((df['High'] > level * 0.98) & (df['High'] < level * 1.02)).sum()
            if hits > 2:
                resistance_hits += hits
                
        for level in lows.unique()[-10:]:  # Last 10 support levels  
            hits = ((df['Low'] > level * 0.98) & (df['Low'] < level * 1.02)).sum()
            if hits > 2:
                support_hits += hits
        
        # Range-bound score (0-100)
        score = 0
        
        # Low total return is good (max 30 points)
        if abs(total_return) < 5:
            score += 30
        elif abs(total_return) < 10:
            score += 20
        elif abs(total_return) < 20:
            score += 10
        
        # High time in range (max 20 points)
        if pct_in_range > 80:
            score += 20
        elif pct_in_range > 60:
            score += 15
        elif pct_in_range > 40:
            score += 10
        
        # Many mean crosses (max 20 points)
        crosses_per_year = crosses * 252 / len(df)
        if crosses_per_year > 20:
            score += 20
        elif crosses_per_year > 15:
            score += 15
        elif crosses_per_year > 10:
            score += 10
            
        # Low trend strength (max 15 points)
        if trend_strength < 5:
            score += 15
        elif trend_strength < 10:
            score += 10
        elif trend_strength < 20:
            score += 5
            
        # Multiple S/R hits (max 15 points)
        total_sr_hits = resistance_hits + support_hits
        if total_sr_hits > 20:
            score += 15
        elif total_sr_hits > 15:
            score += 10
        elif total_sr_hits > 10:
            score += 5
        
        return {
            'ticker': ticker,
            'score': score,
            'total_return': total_return,
            'pct_in_range': pct_in_range,
            'crosses_per_year': crosses_per_year,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'sr_hits': total_sr_hits,
            'current_price': df['Close'].iloc[-1],
            'avg_volume': df['Volume'].mean()
        }
        
    except Exception as e:
        return None

# Extended stock universe
print("="*70)
print("📊 SCREENING FOR RANGE-BOUND STOCKS")
print("="*70)

stock_universe = [
    # Previous test stocks
    "PEP", "KO", "PG", "JNJ", "CL", "GIS", "K", "MCD", "WMT",
    
    # Utilities - often range-bound
    "NEE", "SO", "DUK", "D", "AEP", "XEL", "ED", "EXC", "PEG", "ES", "WEC", "DTE", "ETR", "FE",
    
    # Consumer defensive
    "KMB", "CLX", "CPB", "HSY", "MDLZ", "KHC", "CAG", "SJM", "MKC", "KR", "SYY", "CHD",
    
    # REITs - often range-bound
    "O", "WPC", "NNN", "STOR", "ADC", "STAG", "PSA", "EXR", "LSI", "CUBE", "NSA",
    
    # Telecoms
    "VZ", "T", "TMUS", "LUMN",
    
    # Mature tech
    "IBM", "CSCO", "INTC", "HPQ", "ORCL", "HPE",
    
    # Healthcare - mature
    "BMY", "GILD", "AMGN", "ABBV", "GSK", "SNY", "NVS",
    
    # Financials - stable
    "USB", "PNC", "TFC", "RF", "CFG", "FITB", "KEY", "CMA",
    
    # Materials - cyclical
    "DD", "DOW", "LYB", "EMN", "PPG", "SHW", "ECL", "APD",
    
    # ETFs - potentially range-bound
    "XLU", "XLP", "XLRE", "IYR", "VNQ", "REM", "MORT",
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "IAU", "DBC", "USO", "UNG",
    
    # International - ADRs
    "NGG", "BTI", "GSK", "BP", "RDS.A", "TOT", "E", "SAN", "TEF", "TU"
]

# Remove duplicates
stock_universe = list(set(stock_universe))
print(f"Screening {len(stock_universe)} unique stocks...\n")

# Create directory for results
results_dir = "range_bound_screening"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Screen stocks in parallel
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_stock = {executor.submit(analyze_stock_for_range, ticker): ticker for ticker in stock_universe}
    
    completed = 0
    for future in as_completed(future_to_stock):
        result = future.result()
        if result and result['score'] > 0:
            results.append(result)
        completed += 1
        
        if completed % 10 == 0:
            print(f"Processed {completed}/{len(stock_universe)} stocks...")

# Convert to DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values('score', ascending=False)

# Save full results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"{results_dir}/range_bound_scores_{timestamp}.csv", index=False)

print("\n" + "="*70)
print("🏆 TOP 20 RANGE-BOUND STOCKS")
print("="*70)
print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':<7} {'Return':<10} {'In Range':<10} {'Crosses/Yr':<12} {'Trend':<8}")
print("-"*70)

for idx, row in df.head(20).iterrows():
    print(f"{idx+1:<5} {row['ticker']:<8} {row['score']:<7.0f} "
          f"{row['total_return']:>8.1f}% {row['pct_in_range']:>8.1f}% "
          f"{row['crosses_per_year']:>10.1f} {row['trend_strength']:>7.1f}%")

# Category analysis
print("\n" + "="*70)
print("📈 ANALYSIS BY CATEGORY")
print("="*70)

categories = {
    'Utilities': ['NEE', 'SO', 'DUK', 'D', 'AEP', 'XEL', 'ED', 'XLU'],
    'Consumer Defensive': ['PEP', 'KO', 'PG', 'CL', 'GIS', 'KMB', 'CLX', 'HSY'],
    'REITs': ['O', 'WPC', 'NNN', 'PSA', 'IYR', 'VNQ', 'XLRE'],
    'Bonds/Fixed Income': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
    'Healthcare': ['JNJ', 'BMY', 'GILD', 'AMGN', 'ABBV'],
    'Mature Tech': ['IBM', 'CSCO', 'INTC', 'ORCL']
}

for cat_name, tickers in categories.items():
    cat_stocks = df[df['ticker'].isin(tickers)]
    if len(cat_stocks) > 0:
        avg_score = cat_stocks['score'].mean()
        best = cat_stocks.iloc[0] if len(cat_stocks) > 0 else None
        
        print(f"\n{cat_name}:")
        print(f"   Average score: {avg_score:.1f}")
        if best is not None:
            print(f"   Best: {best['ticker']} (Score: {best['score']}, Return: {best['total_return']:.1f}%)")

# Find truly range-bound stocks
excellent = df[df['score'] >= 70]
good = df[(df['score'] >= 50) & (df['score'] < 70)]

print("\n" + "="*70)
print("💎 BEST CANDIDATES FOR CAMARILLA STRATEGY")
print("="*70)

if len(excellent) > 0:
    print(f"\nEXCELLENT (Score 70+): {len(excellent)} stocks")
    for _, stock in excellent.iterrows():
        print(f"   {stock['ticker']}: Score {stock['score']}, "
              f"Return {stock['total_return']:+.1f}%, "
              f"{stock['pct_in_range']:.0f}% time in range")

if len(good) > 0:
    print(f"\nGOOD (Score 50-70): {len(good)} stocks")
    for _, stock in good.head(5).iterrows():
        print(f"   {stock['ticker']}: Score {stock['score']}, "
              f"Return {stock['total_return']:+.1f}%")

# Key insights
print("\n" + "="*70)
print("📊 KEY INSIGHTS")
print("="*70)

avg_return_top10 = df.head(10)['total_return'].mean()
avg_trend_top10 = df.head(10)['trend_strength'].mean()

print(f"""
Top 10 Range-Bound Stocks:
- Average return: {avg_return_top10:+.1f}% (close to 0 is good)
- Average trend strength: {avg_trend_top10:.1f}% per year
- Average time in range: {df.head(10)['pct_in_range'].mean():.1f}%

These stocks show genuine sideways movement, not strong trends.
They are better candidates for mean reversion strategies.
""")

print(f"\n✅ Full results saved to: {results_dir}/range_bound_scores_{timestamp}.csv")
print("✅ Use these stocks for further Camarilla strategy testing")