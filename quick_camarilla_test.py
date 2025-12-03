"""
🎯 Quick Camarilla Strategy Test on Multiple Stocks
Simple implementation to find best candidates

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np

def test_camarilla_simple(ticker, days=500):
    """Test simple Camarilla mean reversion on a stock"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if len(df) < days:
            return None
            
        df = df.tail(days)
        
        # Calculate Camarilla levels
        df['PrevHigh'] = df['High'].shift(1)
        df['PrevLow'] = df['Low'].shift(1)
        df['PrevClose'] = df['Close'].shift(1)
        df['Range'] = df['PrevHigh'] - df['PrevLow']
        
        # Camarilla levels
        df['R3'] = df['PrevClose'] + df['Range'] * 1.1 / 4
        df['S3'] = df['PrevClose'] - df['Range'] * 1.1 / 4
        
        # Simple 20-day SMA as trend filter
        df['SMA20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Trading signals
        df['Buy'] = (df['Close'] <= df['S3'] * 1.01) & (df['RSI'] < 35)
        df['Sell'] = (df['Close'] >= df['R3'] * 0.99) & (df['RSI'] > 65)
        
        # Calculate returns
        position = 0
        trades = 0
        wins = 0
        total_return = 0
        
        for i in range(20, len(df)):
            if df['Buy'].iloc[i] and position == 0:
                entry_price = df['Close'].iloc[i]
                position = 1
                trades += 1
                
            elif df['Sell'].iloc[i] and position == 0:
                entry_price = df['Close'].iloc[i]
                position = -1
                trades += 1
                
            elif position == 1:
                # Exit long
                if df['Close'].iloc[i] >= df['R3'].iloc[i] or df['RSI'].iloc[i] > 70:
                    exit_price = df['Close'].iloc[i]
                    ret = (exit_price - entry_price) / entry_price
                    total_return += ret
                    if ret > 0:
                        wins += 1
                    position = 0
                    
            elif position == -1:
                # Exit short
                if df['Close'].iloc[i] <= df['S3'].iloc[i] or df['RSI'].iloc[i] < 30:
                    exit_price = df['Close'].iloc[i]
                    ret = (entry_price - exit_price) / entry_price
                    total_return += ret
                    if ret > 0:
                        wins += 1
                    position = 0
        
        # Buy & hold
        buy_hold = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)
        
        # Results
        strategy_return = total_return
        alpha = strategy_return - buy_hold
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return {
            'Ticker': ticker,
            'Strategy': strategy_return * 100,
            'Buy_Hold': buy_hold * 100,
            'Alpha': alpha * 100,
            'Trades': trades,
            'Win_Rate': win_rate,
            'Current_Price': df['Close'].iloc[-1],
            'Volatility': df['Close'].pct_change().std() * np.sqrt(252) * 100
        }
        
    except Exception as e:
        return None

# Test stocks
print("🎯 CAMARILLA STRATEGY - MULTI-STOCK ANALYSIS")
print("="*70)

stocks_to_test = [
    # Consumer Defensive
    "PEP", "KO", "PG", "CL", "GIS", "K", "MCD", "WMT", "COST",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABT", "CVS",
    # Utilities
    "SO", "DUK", "NEE", "ED", "XEL",
    # REITs
    "O", "PSA", "AMT", "SPG", "PLD",
    # Telecom
    "VZ", "T", "TMUS",
    # ETFs
    "XLU", "XLP", "XLRE", "AGG", "TLT"
]

print(f"Testing {len(stocks_to_test)} stocks...\n")

results = []
for ticker in stocks_to_test:
    print(f"Testing {ticker}...", end=" ")
    result = test_camarilla_simple(ticker)
    if result:
        results.append(result)
        print(f"Alpha: {result['Alpha']:+.1f}%")
    else:
        print("Failed")

# Sort by alpha
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Alpha', ascending=False)

print("\n" + "="*70)
print("📊 TOP PERFORMERS BY ALPHA")
print("="*70)
print()

print(f"{'Ticker':<8} {'Strategy':>10} {'Buy&Hold':>10} {'Alpha':>10} {'Trades':>8} {'Win%':>8} {'Vol':>8}")
print("-"*70)

for idx, row in results_df.head(15).iterrows():
    print(f"{row['Ticker']:<8} {row['Strategy']:>9.1f}% {row['Buy_Hold']:>9.1f}% "
          f"{row['Alpha']:>9.1f}% {row['Trades']:>8} {row['Win_Rate']:>7.1f}% {row['Volatility']:>7.1f}%")

# Statistics
positive_alpha = results_df[results_df['Alpha'] > 0]

print(f"\n📊 SUMMARY:")
print(f"   Total stocks tested: {len(results_df)}")
print(f"   Stocks with positive alpha: {len(positive_alpha)} ({len(positive_alpha)/len(results_df)*100:.0f}%)")
print(f"   Average alpha (all): {results_df['Alpha'].mean():.1f}%")
print(f"   Average alpha (positive): {positive_alpha['Alpha'].mean():.1f}%")
print(f"   Best performer: {results_df.iloc[0]['Ticker']} (+{results_df.iloc[0]['Alpha']:.1f}% alpha)")

# By sector
print(f"\n📈 BEST BY CATEGORY:")
categories = {
    'Consumer': ['PEP', 'KO', 'PG', 'CL', 'GIS', 'K', 'MCD', 'WMT', 'COST'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'CVS'],
    'Utilities': ['SO', 'DUK', 'NEE', 'ED', 'XEL', 'XLU'],
    'REITs': ['O', 'PSA', 'AMT', 'SPG', 'PLD', 'XLRE'],
    'Fixed Income': ['AGG', 'TLT']
}

for cat, tickers in categories.items():
    cat_df = results_df[results_df['Ticker'].isin(tickers)]
    if len(cat_df) > 0:
        best = cat_df.iloc[0]
        avg = cat_df['Alpha'].mean()
        print(f"   {cat}: {best['Ticker']} ({best['Alpha']:+.1f}%), Avg: {avg:+.1f}%")

# Portfolio
print(f"\n💰 RECOMMENDED PORTFOLIO (Equal Weight):")
portfolio = positive_alpha.head(8)
if len(portfolio) > 0:
    for idx, stock in portfolio.iterrows():
        print(f"   {stock['Ticker']} - Alpha: {stock['Alpha']:+.1f}%, Vol: {stock['Volatility']:.1f}%")
    
    print(f"\n   Average portfolio alpha: {portfolio['Alpha'].mean():.1f}%")
    print(f"   Expected annual return: {portfolio['Strategy'].mean():.1f}%")

print("\n✅ Analysis complete!")