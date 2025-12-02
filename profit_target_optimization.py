"""
🎯 Profit Target Optimization for Mean Reversion 15
Testing 1%, 2%, 5%, 10% profit targets across all symbols

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from yfinance_cache_demo import YFinanceCache

class ProfitTargetOptimizer:
    def __init__(self):
        self.cache = YFinanceCache()
        self.profit_targets = [1.0, 2.0, 5.0, 10.0]
        self.stop_loss_ratios = {
            1.0: 1.5,   # Original: 1% profit, 1.5% stop
            2.0: 3.0,   # 2% profit, 3% stop  
            5.0: 7.5,   # 5% profit, 7.5% stop
            10.0: 15.0  # 10% profit, 15% stop
        }
        
    def run_strategy_with_target(self, symbol, profit_target, period="59d"):
        """Run strategy with specific profit target"""
        try:
            # Get data
            df = self.cache.get_data(symbol, period=period, interval="15m")
            
            if len(df) < 50:
                return None
            
            # Calculate indicators
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
            df['RSI'] = talib.RSI(df['Close'].values, 14)
            
            # Get corresponding stop loss
            stop_loss = self.stop_loss_ratios[profit_target]
            
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
                    # Entry signals - same as original
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
                    # Exit conditions with new profit target
                    bars_held = i - entry_bar
                    hours_held = bars_held * 15 / 60
                    
                    if position == 'Long':
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        
                        if (pnl_pct >= profit_target or      # New profit target
                            pnl_pct <= -stop_loss or         # Proportional stop loss
                            distance > -0.2 or                # Near mean
                            hours_held > 3):                  # 3 hours max
                            
                            trades.append({
                                'PnL%': pnl_pct,
                                'Exit_Reason': 'Profit' if pnl_pct >= profit_target else 
                                             'Stop' if pnl_pct <= -stop_loss else 
                                             'Mean' if distance > -0.2 else 'Time'
                            })
                            position = None
                            
                    elif position == 'Short':
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        if (pnl_pct >= profit_target or      # New profit target
                            pnl_pct <= -stop_loss or         # Proportional stop loss
                            distance < 0.2 or                 # Near mean
                            hours_held > 3):                  # 3 hours max
                            
                            trades.append({
                                'PnL%': pnl_pct,
                                'Exit_Reason': 'Profit' if pnl_pct >= profit_target else 
                                             'Stop' if pnl_pct <= -stop_loss else 
                                             'Mean' if distance < 0.2 else 'Time'
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
            
            # Exit reason breakdown
            exit_counts = trades_df['Exit_Reason'].value_counts()
            profit_exits = exit_counts.get('Profit', 0)
            
            return {
                'Symbol': symbol,
                'Profit_Target': profit_target,
                'Total_Return_%': total_return,
                'Num_Trades': num_trades,
                'Win_Rate_%': win_rate,
                'Avg_Trade_%': avg_trade,
                'Profit_Exits': profit_exits,
                'Profit_Exit_%': profit_exits / num_trades * 100 if num_trades > 0 else 0
            }
            
        except Exception as e:
            print(f"Error with {symbol} at {profit_target}%: {e}")
            return None
    
    def optimize_all_symbols(self):
        """Test all profit targets on all symbols"""
        # Top performing symbols from previous test
        test_symbols = [
            'AMD', 'SQQQ', 'TSLA', 'VIXY', 'NVDA', 'VXX',
            'XLI', 'SPY', 'XLE', 'XLF', 'TLT', 'UVXY',
            'TQQQ', 'GLD', 'SLV', 'SPXL', 'SPXS'
        ]
        
        all_results = []
        
        print("="*70)
        print("🎯 PROFIT TARGET OPTIMIZATION")
        print("="*70)
        
        for symbol in test_symbols:
            print(f"\n📊 Testing {symbol}...")
            symbol_results = []
            
            for target in self.profit_targets:
                print(f"  {target}% target...", end=" ")
                result = self.run_strategy_with_target(symbol, target)
                
                if result:
                    all_results.append(result)
                    symbol_results.append(result)
                    print(f"✓ {result['Total_Return_%']:.1f}%")
                else:
                    print("✗")
            
            # Show best target for this symbol
            if symbol_results:
                best = max(symbol_results, key=lambda x: x['Total_Return_%'])
                print(f"  Best: {best['Profit_Target']}% target → {best['Total_Return_%']:.1f}% return")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Analysis
        self.analyze_results(results_df)
        
        # Save results
        results_df.to_csv('profit_target_optimization.csv', index=False)
        print(f"\n✅ Full results saved to: profit_target_optimization.csv")
        
        return results_df
    
    def analyze_results(self, results_df):
        """Analyze optimization results"""
        print("\n" + "="*70)
        print("📊 OPTIMIZATION ANALYSIS")
        print("="*70)
        
        # Average by profit target
        print("\n📈 AVERAGE RETURNS BY PROFIT TARGET:")
        avg_by_target = results_df.groupby('Profit_Target').agg({
            'Total_Return_%': 'mean',
            'Num_Trades': 'mean',
            'Win_Rate_%': 'mean',
            'Profit_Exit_%': 'mean'
        }).round(1)
        
        print(avg_by_target)
        
        # Best performers at each target
        print("\n🏆 TOP 5 PERFORMERS BY PROFIT TARGET:")
        for target in self.profit_targets:
            target_df = results_df[results_df['Profit_Target'] == target]
            top_5 = target_df.nlargest(5, 'Total_Return_%')
            
            print(f"\n{target}% Profit Target:")
            for _, row in top_5.iterrows():
                print(f"  {row['Symbol']}: {row['Total_Return_%']:.1f}% "
                      f"({row['Num_Trades']} trades, {row['Win_Rate_%']:.0f}% win)")
        
        # Optimal targets by symbol
        print("\n💡 OPTIMAL PROFIT TARGET BY SYMBOL:")
        print("-"*50)
        print(f"{'Symbol':<8} {'Best Target':>12} {'Return%':>10} {'Improvement':>12}")
        print("-"*50)
        
        for symbol in results_df['Symbol'].unique():
            symbol_df = results_df[results_df['Symbol'] == symbol]
            if len(symbol_df) == 0:
                continue
                
            best = symbol_df.loc[symbol_df['Total_Return_%'].idxmax()]
            baseline = symbol_df[symbol_df['Profit_Target'] == 1.0]['Total_Return_%'].values[0] if 1.0 in symbol_df['Profit_Target'].values else 0
            improvement = best['Total_Return_%'] - baseline
            
            print(f"{symbol:<8} {best['Profit_Target']:>11.0f}% {best['Total_Return_%']:>9.1f}% "
                  f"{improvement:>11.1f}%")
        
        # Key findings
        print("\n📊 KEY FINDINGS:")
        
        # Which target works best overall
        best_target_overall = avg_by_target['Total_Return_%'].idxmax()
        print(f"1. Best profit target overall: {best_target_overall}% "
              f"(avg return: {avg_by_target.loc[best_target_overall, 'Total_Return_%']:.1f}%)")
        
        # Trade frequency impact
        print(f"2. Trade frequency decreases with higher targets:")
        for target in self.profit_targets:
            avg_trades = avg_by_target.loc[target, 'Num_Trades']
            print(f"   {target}% target: {avg_trades:.0f} trades average")
        
        # Win rate analysis
        print(f"3. Win rates by target:")
        for target in self.profit_targets:
            win_rate = avg_by_target.loc[target, 'Win_Rate_%']
            profit_exits = avg_by_target.loc[target, 'Profit_Exit_%']
            print(f"   {target}% target: {win_rate:.0f}% wins, {profit_exits:.0f}% hit profit target")


if __name__ == "__main__":
    optimizer = ProfitTargetOptimizer()
    results = optimizer.optimize_all_symbols()