"""
🎯 Find Naturally Range-Bound Instruments & Pairs
Looking for ETFs and pairs that are designed to mean-revert

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 FINDING NATURALLY RANGE-BOUND INSTRUMENTS")
print("="*70)

# Categories of potentially range-bound instruments
range_bound_candidates = {
    
    "Currency-Hedged ETFs": {
        "description": "Remove currency risk, focus on underlying assets",
        "tickers": ["HEDJ", "DXJ", "DBJP", "HEFA", "HEZU", "DBEZ"]
    },
    
    "Bond Spreads": {
        "description": "Yield curve plays that mean-revert",
        "pairs": [
            ("TLT", "IEF"),  # 20yr vs 10yr treasuries
            ("IEF", "SHY"),  # 10yr vs 1yr treasuries
            ("LQD", "HYG"),  # Investment grade vs high yield
            ("AGG", "TLT"),  # Total bond vs long bond
        ]
    },
    
    "Sector Pairs": {
        "description": "Correlated sectors that diverge and converge",
        "pairs": [
            ("XLU", "XLP"),  # Utilities vs Staples (defensive pair)
            ("XLF", "KRE"),  # Large banks vs Regional banks
            ("VNQ", "IYR"),  # Two REIT ETFs
            ("GLD", "GDX"),  # Gold vs Gold miners
            ("XLE", "OIH"),  # Energy vs Oil services
        ]
    },
    
    "Country Pairs": {
        "description": "Similar economies that move together",
        "pairs": [
            ("EWC", "EWA"),  # Canada vs Australia (commodity)
            ("EWG", "EWQ"),  # Germany vs France
            ("EWJ", "EWY"),  # Japan vs South Korea
        ]
    },
    
    "Volatility Products": {
        "description": "Mean-reverting by nature",
        "tickers": ["VXX", "VIXY", "SVXY", "ZIV", "VXZ"]
    },
    
    "Market Neutral ETFs": {
        "description": "Designed to be market neutral",
        "tickers": ["BTAL", "MOM", "CHEP", "QAI", "MCRO"]
    },
    
    "Commodity Spreads": {
        "description": "Commodity pairs that mean revert",
        "pairs": [
            ("USO", "BNO"),   # WTI vs Brent oil
            ("GLD", "SLV"),   # Gold vs Silver
            ("CORN", "WEAT"), # Corn vs Wheat
        ]
    },
    
    "Style Box Pairs": {
        "description": "Value vs Growth tends to cycle",
        "pairs": [
            ("VTV", "VUG"),   # Value vs Growth
            ("IWD", "IWF"),   # Russell Value vs Growth
            ("IWM", "MDY"),   # Small cap vs Mid cap
        ]
    }
}

def analyze_pair(ticker1, ticker2, period="2y"):
    """Analyze a pair for mean reversion characteristics"""
    try:
        # Download data
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        df1 = stock1.history(period=period)
        df2 = stock2.history(period=period)
        
        # Align the data
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 200:
            return None
            
        df1 = df1.loc[common_dates]
        df2 = df2.loc[common_dates]
        
        # Calculate spread (ratio)
        spread = df1['Close'] / df2['Close']
        
        # Mean reversion metrics
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Z-score
        zscore = (spread - spread_mean) / spread_std
        
        # Count reversions
        crosses = ((zscore > 0) != (zscore.shift(1) > 0)).sum()
        
        # Time spent in range
        in_1std = ((zscore > -1) & (zscore < 1)).sum() / len(zscore) * 100
        in_2std = ((zscore > -2) & (zscore < 2)).sum() / len(zscore) * 100
        
        # Current z-score
        current_zscore = zscore.iloc[-1]
        
        # Cointegration test (simplified)
        correlation = df1['Close'].pct_change().corr(df2['Close'].pct_change())
        
        return {
            'pair': f"{ticker1}/{ticker2}",
            'correlation': correlation,
            'mean_reversion_score': crosses / len(zscore) * 100,
            'pct_in_1std': in_1std,
            'pct_in_2std': in_2std,
            'current_zscore': current_zscore,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'days_analyzed': len(common_dates)
        }
        
    except Exception as e:
        return None

def analyze_single_ticker(ticker, period="2y"):
    """Analyze single ticker for range-bound behavior"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if len(df) < 100:
            return None
            
        # Check if it oscillates around a mean
        returns = df['Close'].pct_change()
        
        # Hurst exponent (simplified) - measures trending vs mean reverting
        # < 0.5 = mean reverting, > 0.5 = trending
        lags = range(2, 20)
        tau = []
        for lag in lags:
            price_diff = df['Close'].diff(lag).dropna()
            tau.append(price_diff.std())
        
        # Linear fit to log-log plot
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        hurst = np.polyfit(log_lags, log_tau, 1)[0]
        
        # Price range
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].mean()
        
        # Total return (want low)
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        
        return {
            'ticker': ticker,
            'hurst': hurst,
            'is_mean_reverting': hurst < 0.5,
            'total_return': total_return,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'price_range': price_range * 100,
            'current_price': df['Close'].iloc[-1]
        }
        
    except Exception as e:
        return None

# Analyze single instruments
print("\n📊 ANALYZING SINGLE INSTRUMENTS\n")

single_results = []

for category, data in range_bound_candidates.items():
    if 'tickers' in data:
        print(f"\n{category}:")
        for ticker in data['tickers']:
            result = analyze_single_ticker(ticker)
            if result:
                single_results.append({**result, 'category': category})
                mr_status = "✓ Mean Reverting" if result['is_mean_reverting'] else "✗ Trending"
                print(f"  {ticker}: Hurst={result['hurst']:.2f} {mr_status}, "
                      f"Return={result['total_return']:.1f}%")

# Analyze pairs
print("\n\n📊 ANALYZING PAIRS\n")

pair_results = []

for category, data in range_bound_candidates.items():
    if 'pairs' in data:
        print(f"\n{category}:")
        for ticker1, ticker2 in data['pairs']:
            result = analyze_pair(ticker1, ticker2)
            if result:
                pair_results.append({**result, 'category': category})
                print(f"  {result['pair']}: Correlation={result['correlation']:.2f}, "
                      f"Mean Reversion={result['mean_reversion_score']:.1f}, "
                      f"Current Z={result['current_zscore']:.2f}")

# Best single instruments
print("\n" + "="*70)
print("🏆 BEST SINGLE INSTRUMENTS FOR CAMARILLA")
print("="*70)

if single_results:
    single_df = pd.DataFrame(single_results)
    
    # Filter for mean reverting
    mr_instruments = single_df[single_df['is_mean_reverting']]
    
    if len(mr_instruments) > 0:
        print("\nMean Reverting Instruments (Hurst < 0.5):")
        for _, inst in mr_instruments.iterrows():
            print(f"  {inst['ticker']} ({inst['category']})")
            print(f"    Hurst: {inst['hurst']:.3f}")
            print(f"    Volatility: {inst['volatility']:.1f}%")
            print(f"    Total Return: {inst['total_return']:.1f}%")

# Best pairs
print("\n" + "="*70)
print("🏆 BEST PAIRS FOR MEAN REVERSION")
print("="*70)

if pair_results:
    pairs_df = pd.DataFrame(pair_results)
    pairs_df = pairs_df.sort_values('mean_reversion_score', ascending=False)
    
    print("\nTop 10 Mean Reverting Pairs:")
    print(f"{'Pair':<15} {'Category':<20} {'Correlation':<12} {'MR Score':<10} {'In 1σ':<8} {'Current Z':<10}")
    print("-"*85)
    
    for _, pair in pairs_df.head(10).iterrows():
        print(f"{pair['pair']:<15} {pair['category']:<20} {pair['correlation']:>10.2f} "
              f"{pair['mean_reversion_score']:>8.1f} {pair['pct_in_1std']:>6.1f}% "
              f"{pair['current_zscore']:>8.2f}")

# Trading opportunities
print("\n" + "="*70)
print("💡 CURRENT TRADING OPPORTUNITIES")
print("="*70)

if pair_results:
    # Find pairs with extreme z-scores
    extreme_pairs = [p for p in pair_results if abs(p['current_zscore']) > 1.5]
    
    if extreme_pairs:
        print("\nPairs with Extreme Z-scores (>1.5σ):")
        for pair in extreme_pairs:
            if pair['current_zscore'] > 1.5:
                print(f"  {pair['pair']}: Z={pair['current_zscore']:.2f} - Consider SELLING the spread")
            else:
                print(f"  {pair['pair']}: Z={pair['current_zscore']:.2f} - Consider BUYING the spread")

# Recommendations
print("\n" + "="*70)
print("📋 RECOMMENDATIONS FOR CAMARILLA STRATEGY")
print("="*70)

print("""
1. BEST SINGLE INSTRUMENTS:
   - Volatility products (VXX, VIXY) - naturally mean reverting
   - Market neutral ETFs - designed to oscillate
   - Currency-hedged ETFs - remove trending FX component

2. BEST PAIRS:
   - Treasury spreads (TLT/IEF) - yield curve trades
   - Sector pairs (XLU/XLP) - defensive sectors
   - Country pairs (EWC/EWA) - correlated economies

3. HOW TO TRADE:
   - Use z-score instead of price for signals
   - Buy spread at -2σ, sell at +2σ
   - Use Camarilla levels on the spread ratio
   - Tighter stops (1σ move against position)

4. ADVANTAGES:
   - True mean reversion behavior
   - Lower directional risk
   - More predictable ranges
   - Can use options for income
""")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if pair_results:
    pd.DataFrame(pair_results).to_csv(f"range_bound_pairs_{timestamp}.csv", index=False)
    print(f"\n✅ Pair analysis saved to: range_bound_pairs_{timestamp}.csv")

if single_results:
    pd.DataFrame(single_results).to_csv(f"range_bound_singles_{timestamp}.csv", index=False)
    print(f"✅ Single instrument analysis saved to: range_bound_singles_{timestamp}.csv")