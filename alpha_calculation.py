"""
🎯 Alpha Calculation for Mean Reversion 15 Strategy
Comparing strategy returns to buy-and-hold benchmark

Author: Claude (Anthropic)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from yfinance_cache_demo import YFinanceCache

def calculate_alpha():
    """Calculate alpha (excess return over buy-and-hold)"""
    cache = YFinanceCache()
    
    print("="*70)
    print("📊 ALPHA CALCULATION - MEAN REVERSION 15")
    print("="*70)
    print("\nAlpha = Strategy Return - Buy & Hold Return")
    print("(Testing period: 59 days)\n")
    
    # Load optimization results
    results_df = pd.read_csv('profit_target_optimization.csv')
    
    # Get unique symbols
    symbols = results_df['Symbol'].unique()
    
    alpha_results = []
    
    for symbol in symbols:
        # Get best result for this symbol
        symbol_results = results_df[results_df['Symbol'] == symbol]
        best_result = symbol_results.loc[symbol_results['Total_Return_%'].idxmax()]
        
        # Calculate buy and hold return
        try:
            df = cache.get_data(symbol, period="59d", interval="1d")
            if len(df) > 0:
                buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            else:
                buy_hold_return = 0
        except:
            buy_hold_return = 0
            
        # Calculate alpha
        strategy_return = best_result['Total_Return_%']
        alpha = strategy_return - buy_hold_return
        
        # Annualized values
        annual_factor = 252 / 59
        annual_strategy = strategy_return * annual_factor
        annual_buy_hold = buy_hold_return * annual_factor
        annual_alpha = alpha * annual_factor
        
        alpha_results.append({
            'Symbol': symbol,
            'Strategy_Return_%': strategy_return,
            'Buy_Hold_Return_%': buy_hold_return,
            'Alpha_%': alpha,
            'Best_Target_%': best_result['Profit_Target'],
            'Annual_Strategy_%': annual_strategy,
            'Annual_Buy_Hold_%': annual_buy_hold,
            'Annual_Alpha_%': annual_alpha
        })
    
    # Convert to DataFrame and sort by alpha
    alpha_df = pd.DataFrame(alpha_results)
    alpha_df = alpha_df.sort_values('Alpha_%', ascending=False)
    
    # Display results
    print(f"{'Symbol':<8} {'Strategy':>10} {'Buy&Hold':>10} {'Alpha':>8} {'Best PT':>8} {'Annual α':>10}")
    print("-"*70)
    
    for _, row in alpha_df.iterrows():
        print(f"{row['Symbol']:<8} {row['Strategy_Return_%']:>9.1f}% {row['Buy_Hold_Return_%']:>9.1f}% "
              f"{row['Alpha_%']:>7.1f}% {row['Best_Target_%']:>7.0f}% {row['Annual_Alpha_%']:>9.0f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("📊 ALPHA ANALYSIS")
    print("="*70)
    
    # Positive alpha symbols
    positive_alpha = alpha_df[alpha_df['Alpha_%'] > 0]
    print(f"\n✅ SYMBOLS WITH POSITIVE ALPHA: {len(positive_alpha)}/{len(alpha_df)}")
    if len(positive_alpha) > 0:
        print("\nTop 10 by Alpha:")
        for i, (_, row) in enumerate(positive_alpha.head(10).iterrows()):
            print(f"{i+1}. {row['Symbol']}: {row['Alpha_%']:+.1f}% alpha "
                  f"({row['Strategy_Return_%']:.1f}% strategy vs {row['Buy_Hold_Return_%']:.1f}% B&H) "
                  f"→ {row['Annual_Alpha_%']:+.0f}% annual")
    
    # Negative alpha symbols
    negative_alpha = alpha_df[alpha_df['Alpha_%'] < 0]
    if len(negative_alpha) > 0:
        print(f"\n❌ SYMBOLS WITH NEGATIVE ALPHA: {len(negative_alpha)}")
        print("(Strategy underperformed buy & hold)")
    
    # Best profit targets for alpha
    print("\n💡 PROFIT TARGETS FOR MAXIMUM ALPHA:")
    target_analysis = {}
    for _, row in alpha_df.iterrows():
        target = row['Best_Target_%']
        if target not in target_analysis:
            target_analysis[target] = []
        target_analysis[target].append(row['Alpha_%'])
    
    for target, alphas in sorted(target_analysis.items()):
        avg_alpha = sum(alphas) / len(alphas)
        print(f"  {target}% target: {len(alphas)} symbols, avg alpha: {avg_alpha:.1f}%")
    
    # Category analysis
    print("\n📈 ALPHA BY CATEGORY:")
    
    categories = {
        'Volatility': ['VXX', 'VIXY', 'UVXY'],
        'Leveraged': ['TQQQ', 'SQQQ', 'SPXL', 'SPXS'],
        'Individual': ['AMD', 'TSLA', 'NVDA'],
        'Commodities': ['GLD', 'SLV'],
        'Sectors': ['XLF', 'XLE', 'XLI']
    }
    
    for cat_name, symbols in categories.items():
        cat_df = alpha_df[alpha_df['Symbol'].isin(symbols)]
        if len(cat_df) > 0:
            avg_alpha = cat_df['Alpha_%'].mean()
            print(f"  {cat_name}: {avg_alpha:+.1f}% average alpha")
    
    # Save results
    alpha_df.to_csv('alpha_results.csv', index=False)
    print(f"\n✅ Alpha results saved to: alpha_results.csv")
    
    return alpha_df

if __name__ == "__main__":
    alpha_results = calculate_alpha()