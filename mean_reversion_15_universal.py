"""
🎯 Mean Reversion 15 - Universal Testing
Testing the proven VXX strategy on multiple symbols

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from yfinance_cache_demo import YFinanceCache

class MeanReversion15Strategy:
    """The proven VXX strategy, now tested on any symbol"""
    
    def __init__(self):
        self.cache = YFinanceCache()
        
    def run_strategy(self, symbol, period="59d"):
        """Run the exact VXX strategy on any symbol"""
        try:
            # Get 15-minute data
            df = self.cache.get_data(symbol, period=period, interval="15m")
            
            if len(df) < 50:
                return None
            
            # Calculate indicators (exact same as VXX)
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
            df['RSI'] = talib.RSI(df['Close'].values, 14)
            
            trades = []
            position = None
            entry_price = 0
            entry_time = None
            entry_bar = 0
            
            for i in range(50, len(df)):
                current_time = df.index[i]
                current_price = df['Close'].iloc[i]
                distance = df['Distance%'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                if pd.isna(distance) or pd.isna(rsi):
                    continue
                
                # Market hours only
                if current_time.hour < 9 or current_time.hour >= 16:
                    continue
                
                if position is None:
                    # Entry signals - exact VXX parameters
                    if distance < -1.0 and rsi < 40:
                        position = 'Long'
                        entry_price = current_price
                        entry_time = current_time
                        entry_bar = i
                        
                    elif distance > 1.0 and rsi > 60:
                        position = 'Short'
                        entry_price = current_price
                        entry_time = current_time
                        entry_bar = i
                
                else:
                    # Exit conditions
                    bars_held = i - entry_bar
                    hours_held = bars_held * 15 / 60
                    
                    if position == 'Long':
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        
                        if (distance > -0.2 or      # Near mean
                            pnl_pct > 1.0 or        # 1% profit
                            pnl_pct < -1.5 or       # 1.5% loss
                            hours_held > 3):        # 3 hours max
                            
                            trades.append({
                                'Entry': entry_time,
                                'Exit': current_time,
                                'Type': position,
                                'PnL%': pnl_pct
                            })
                            position = None
                            
                    elif position == 'Short':
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        if (distance < 0.2 or       # Near mean
                            pnl_pct > 1.0 or        # 1% profit
                            pnl_pct < -1.5 or       # 1.5% loss
                            hours_held > 3):        # 3 hours max
                            
                            trades.append({
                                'Entry': entry_time,
                                'Exit': current_time,
                                'Type': position,
                                'PnL%': pnl_pct
                            })
                            position = None
            
            if len(trades) == 0:
                return None
                
            # Calculate results
            trades_df = pd.DataFrame(trades)
            total_return = trades_df['PnL%'].sum()
            num_trades = len(trades_df)
            win_rate = (trades_df['PnL%'] > 0).sum() / num_trades * 100
            avg_trade = trades_df['PnL%'].mean()
            
            # Calculate volatility
            daily_volatility = df['Close'].pct_change().std() * np.sqrt(26) * 100  # 26 15-min bars per day
            
            return {
                'Symbol': symbol,
                'Total_Return_%': total_return,
                'Num_Trades': num_trades,
                'Win_Rate_%': win_rate,
                'Avg_Trade_%': avg_trade,
                'Daily_Vol_%': daily_volatility,
                'Trades_Per_Day': num_trades / (len(df) / 26 / 6.5)  # 6.5 hours per day
            }
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            return None

def test_symbols():
    """Test strategy on multiple symbols"""
    print("="*70)
    print("🎯 MEAN REVERSION 15 - UNIVERSAL TESTING")
    print("="*70)
    
    strategy = MeanReversion15Strategy()
    
    # Test symbols by category
    test_list = {
        'Volatility': ['VXX', 'VIXY', 'UVXY', 'SVXY'],
        'Leveraged_ETFs': ['TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'TNA', 'TZA'],
        'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
        'Bonds': ['TLT', 'TBT', 'IEF', 'SHY', 'AGG'],
        'Sectors': ['XLF', 'XLE', 'XLK', 'XLV', 'XLI'],
        'Crypto': ['BITO', 'BITQ'],
        'Individual_Stocks': ['TSLA', 'AAPL', 'NVDA', 'AMD', 'SPY']
    }
    
    all_results = []
    
    for category, symbols in test_list.items():
        print(f"\n📊 Testing {category}...")
        
        for symbol in symbols:
            print(f"  {symbol}...", end=" ")
            result = strategy.run_strategy(symbol)
            
            if result:
                all_results.append(result)
                print(f"✓ {result['Total_Return_%']:.1f}%")
            else:
                print("✗ No trades or insufficient data")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nNo results to analyze!")
        return
    
    # Sort by return
    results_df = results_df.sort_values('Total_Return_%', ascending=False)
    
    print("\n" + "="*70)
    print("📊 STRATEGY RESULTS BY SYMBOL")
    print("="*70)
    print(f"{'Symbol':<8} {'Return%':>10} {'Trades':>8} {'Win%':>8} {'Avg%':>8} {'Daily Vol':>10}")
    print("-"*70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Symbol']:<8} {row['Total_Return_%']:>9.1f}% {row['Num_Trades']:>8} "
              f"{row['Win_Rate_%']:>7.1f}% {row['Avg_Trade_%']:>7.2f}% {row['Daily_Vol_%']:>9.1f}%")
    
    # Analysis by characteristics
    print("\n" + "="*70)
    print("📈 ANALYSIS")
    print("="*70)
    
    # Winners vs Losers
    winners = results_df[results_df['Total_Return_%'] > 0]
    losers = results_df[results_df['Total_Return_%'] <= 0]
    
    print(f"\n✅ PROFITABLE SYMBOLS: {len(winners)}/{len(results_df)}")
    if len(winners) > 0:
        print(f"  Average return: {winners['Total_Return_%'].mean():.1f}%")
        print(f"  Best: {winners.iloc[0]['Symbol']} ({winners.iloc[0]['Total_Return_%']:.1f}%)")
        print(f"  Symbols: {', '.join(winners['Symbol'].tolist())}")
    
    print(f"\n❌ UNPROFITABLE SYMBOLS: {len(losers)}/{len(results_df)}")
    if len(losers) > 0:
        print(f"  Average return: {losers['Total_Return_%'].mean():.1f}%")
        print(f"  Worst: {losers.iloc[-1]['Symbol']} ({losers.iloc[-1]['Total_Return_%']:.1f}%)")
    
    # Correlation with volatility
    correlation = results_df['Total_Return_%'].corr(results_df['Daily_Vol_%'])
    print(f"\n📊 CORRELATION WITH VOLATILITY: {correlation:.2f}")
    
    # Best categories
    print("\n🏆 KEY FINDINGS:")
    
    # Find patterns
    high_vol = results_df[results_df['Daily_Vol_%'] > 2.0]
    if len(high_vol) > 0:
        print(f"  High volatility (>2% daily) symbols: {high_vol['Total_Return_%'].mean():.1f}% avg return")
        print(f"    {', '.join(high_vol['Symbol'].tolist())}")
    
    low_vol = results_df[results_df['Daily_Vol_%'] <= 2.0]
    if len(low_vol) > 0:
        print(f"  Low volatility (≤2% daily) symbols: {low_vol['Total_Return_%'].mean():.1f}% avg return")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    top_5 = results_df.head(5)
    print(f"  Best symbols for Mean Reversion 15:")
    for i, row in top_5.iterrows():
        annualized = row['Total_Return_%'] * (252/59)  # Annualize
        print(f"    {row['Symbol']}: {row['Total_Return_%']:.1f}% (≈{annualized:.0f}% annual)")
    
    # Save results
    results_df.to_csv('mean_reversion_15_results.csv', index=False)
    print(f"\n✅ Results saved to: mean_reversion_15_results.csv")

if __name__ == "__main__":
    test_symbols()