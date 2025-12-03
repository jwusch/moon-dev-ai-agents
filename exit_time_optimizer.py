"""
⏰ Exit Time Limit Optimizer
Comprehensive optimization of time-based exits for VXX Mean Reversion strategy

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class OptimizationResult:
    time_limit: float
    total_trades: int
    win_rate: float
    total_return_pct: float
    avg_return_per_trade: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    time_limit_exits: int
    time_limit_exit_pct: float

class ExitTimeOptimizer:
    """
    Optimize time-based exit strategy for mean reversion trading
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 commission: float = 7.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
        
    def prepare_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Prepare data with all indicators"""
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="1d", progress=False)
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Ensure we have the right column structure
        if df.columns.nlevels > 1:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Core indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance_Pct'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Simplified quality score calculation"""
        score = 0
        
        # Distance from SMA
        distance = df['Distance_Pct'].iloc[idx]
        if distance < -3.0:
            score += 25
        elif distance < -2.0:
            score += 20
        elif distance < -1.0:
            score += 15
        
        # RSI level
        rsi = df['RSI'].iloc[idx]
        if rsi < 25:
            score += 25
        elif rsi < 30:
            score += 20
        elif rsi < 35:
            score += 15
        elif rsi < 40:
            score += 10
        
        # Volume confirmation
        volume_ratio = df['Volume_Ratio'].iloc[idx]
        if not pd.isna(volume_ratio):
            if volume_ratio > 2.0:
                score += 20
            elif volume_ratio > 1.5:
                score += 15
            elif volume_ratio > 1.0:
                score += 10
        
        return score
    
    def run_backtest_with_time_limit(self, df: pd.DataFrame, max_hold_days: float, 
                                   min_quality_score: int = 50) -> Dict:
        """Run backtest with specific time limit"""
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        for i in range(50, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check for new entry
            if position is None:
                distance = df['Distance_Pct'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                if distance < -1.5 and rsi < 35:
                    score = self.calculate_quality_score(df, i)
                    
                    if score >= min_quality_score:
                        position = {
                            'entry_date': current_date,
                            'entry_price': current_price,
                            'entry_idx': i
                        }
            
            # Check for exit
            elif position is not None:
                days_held = (current_date - position['entry_date']).days
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Exit conditions
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance_Pct'].iloc[i] > 0:
                    exit_reason = 'Above Mean'
                elif days_held >= max_hold_days:
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    # Calculate trade results
                    shares = (current_capital * 0.95) / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price']) - (2 * self.commission)
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'win': pnl_pct > 0
                    })
                    position = None
            
            equity_curve.append({
                'date': current_date,
                'equity': current_capital
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Analyze results
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_return_pct = (current_capital / self.initial_capital - 1) * 100
        
        # Calculate other metrics
        returns = trades_df['pnl_pct'].values
        wins = trades_df[trades_df['win']]['pnl_pct']
        losses = trades_df[~trades_df['win']]['pnl_pct']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
        
        # Max drawdown
        equity = equity_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        daily_returns = equity.pct_change().dropna()
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Time limit exit analysis
        time_limit_exits = len(trades_df[trades_df['exit_reason'] == 'Time Limit'])
        time_limit_exit_pct = (time_limit_exits / total_trades) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'avg_return_per_trade': total_return_pct / total_trades if total_trades > 0 else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'time_limit_exits': time_limit_exits,
            'time_limit_exit_pct': time_limit_exit_pct,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
    
    def optimize_time_limits(self, symbol: str, 
                           time_limits: List[float] = None) -> List[OptimizationResult]:
        """Optimize across different time limits"""
        if time_limits is None:
            # Test range from 1 day to 15 days
            time_limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
        
        print(f"🎯 Optimizing time limits for {symbol}...")
        print(f"📊 Testing {len(time_limits)} different time limits...")
        
        # Prepare data
        df = self.prepare_data(symbol)
        if df is None:
            return []
        
        results = []
        
        for time_limit in time_limits:
            print(f"   Testing {time_limit} days...", end="")
            
            try:
                result = self.run_backtest_with_time_limit(df, time_limit)
                
                if 'error' not in result:
                    opt_result = OptimizationResult(
                        time_limit=time_limit,
                        total_trades=result['total_trades'],
                        win_rate=result['win_rate'],
                        total_return_pct=result['total_return_pct'],
                        avg_return_per_trade=result['avg_return_per_trade'],
                        profit_factor=result['profit_factor'],
                        max_drawdown=result['max_drawdown'],
                        sharpe_ratio=result['sharpe_ratio'],
                        time_limit_exits=result['time_limit_exits'],
                        time_limit_exit_pct=result['time_limit_exit_pct']
                    )
                    results.append(opt_result)
                    print(f" ✓ {result['total_trades']} trades, {result['win_rate']:.1f}% win rate, {result['total_return_pct']:+.1f}% return")
                else:
                    print(f" ❌ {result['error']}")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        return results
    
    def create_optimization_visualizations(self, results: List[OptimizationResult], symbol: str):
        """Create comprehensive optimization visualizations"""
        if not results:
            print("No results to visualize")
            return None
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'Time_Limit': r.time_limit,
                'Total_Trades': r.total_trades,
                'Win_Rate': r.win_rate,
                'Total_Return': r.total_return_pct,
                'Avg_Return_Per_Trade': r.avg_return_per_trade,
                'Profit_Factor': r.profit_factor if r.profit_factor != float('inf') else 10,
                'Max_Drawdown': r.max_drawdown,
                'Sharpe_Ratio': r.sharpe_ratio,
                'Time_Limit_Exits': r.time_limit_exits,
                'Time_Limit_Exit_Pct': r.time_limit_exit_pct
            }
            for r in results
        ])
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{symbol} - Time Limit Optimization Analysis', fontsize=20, fontweight='bold')
        
        # 1. Total Return vs Time Limit
        ax = axes[0, 0]
        ax.plot(df['Time_Limit'], df['Total_Return'], 'o-', linewidth=3, markersize=8, color='#00ff88')
        ax.set_title('Total Return vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # Highlight best return
        best_return_idx = df['Total_Return'].idxmax()
        ax.scatter(df.loc[best_return_idx, 'Time_Limit'], df.loc[best_return_idx, 'Total_Return'], 
                  s=200, color='gold', marker='*', zorder=5)
        ax.text(df.loc[best_return_idx, 'Time_Limit'], df.loc[best_return_idx, 'Total_Return'] + 1,
                f"Best: {df.loc[best_return_idx, 'Time_Limit']:.0f}d", ha='center', fontweight='bold')
        
        # 2. Win Rate vs Time Limit
        ax = axes[0, 1]
        ax.plot(df['Time_Limit'], df['Win_Rate'], 'o-', linewidth=3, markersize=8, color='#4ecdc4')
        ax.set_title('Win Rate vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Win Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='white', linestyle='--', alpha=0.5, label='50% Break-even')
        
        # Highlight best win rate
        best_winrate_idx = df['Win_Rate'].idxmax()
        ax.scatter(df.loc[best_winrate_idx, 'Time_Limit'], df.loc[best_winrate_idx, 'Win_Rate'], 
                  s=200, color='gold', marker='*', zorder=5)
        
        # 3. Sharpe Ratio vs Time Limit
        ax = axes[0, 2]
        ax.plot(df['Time_Limit'], df['Sharpe_Ratio'], 'o-', linewidth=3, markersize=8, color='#ffd93d')
        ax.set_title('Sharpe Ratio vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # 4. Number of Trades vs Time Limit
        ax = axes[1, 0]
        ax.plot(df['Time_Limit'], df['Total_Trades'], 'o-', linewidth=3, markersize=8, color='#ff6b6b')
        ax.set_title('Trade Frequency vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Total Trades')
        ax.grid(True, alpha=0.3)
        
        # 5. Average Return per Trade
        ax = axes[1, 1]
        ax.plot(df['Time_Limit'], df['Avg_Return_Per_Trade'], 'o-', linewidth=3, markersize=8, color='#a8e6cf')
        ax.set_title('Avg Return per Trade vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Avg Return per Trade (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # 6. Time Limit Exit Percentage
        ax = axes[1, 2]
        ax.plot(df['Time_Limit'], df['Time_Limit_Exit_Pct'], 'o-', linewidth=3, markersize=8, color='#ff9999')
        ax.set_title('% of Trades Exiting on Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Time Limit Exits (%)')
        ax.grid(True, alpha=0.3)
        
        # 7. Max Drawdown vs Time Limit
        ax = axes[2, 0]
        ax.plot(df['Time_Limit'], df['Max_Drawdown'], 'o-', linewidth=3, markersize=8, color='#ffb3ba')
        ax.set_title('Max Drawdown vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Days')
        ax.set_ylabel('Max Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # 8. Risk-Return Scatter
        ax = axes[2, 1]
        scatter = ax.scatter(df['Max_Drawdown'], df['Total_Return'], 
                           c=df['Time_Limit'], s=100, cmap='viridis', alpha=0.8)
        ax.set_title('Risk vs Return by Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Max Hold Days')
        
        # 9. Optimization Summary Table
        ax = axes[2, 2]
        ax.axis('off')
        
        # Find best configurations
        best_return = df.loc[df['Total_Return'].idxmax()]
        best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
        best_winrate = df.loc[df['Win_Rate'].idxmax()]
        
        summary_text = f"""OPTIMIZATION SUMMARY

🏆 BEST CONFIGURATIONS:

Best Total Return:
  Time Limit: {best_return['Time_Limit']:.0f} days
  Return: {best_return['Total_Return']:+.1f}%
  Win Rate: {best_return['Win_Rate']:.1f}%
  Trades: {best_return['Total_Trades']:.0f}

Best Sharpe Ratio:
  Time Limit: {best_sharpe['Time_Limit']:.0f} days
  Sharpe: {best_sharpe['Sharpe_Ratio']:.2f}
  Return: {best_sharpe['Total_Return']:+.1f}%
  Win Rate: {best_sharpe['Win_Rate']:.1f}%

Best Win Rate:
  Time Limit: {best_winrate['Time_Limit']:.0f} days
  Win Rate: {best_winrate['Win_Rate']:.1f}%
  Return: {best_winrate['Total_Return']:+.1f}%
  Trades: {best_winrate['Total_Trades']:.0f}

💡 INSIGHTS:
• Test Range: {df['Time_Limit'].min():.0f} to {df['Time_Limit'].max():.0f} days
• Time Exits: {df['Time_Limit_Exit_Pct'].mean():.1f}% avg
• Optimal Range: {best_return['Time_Limit']:.0f}-{best_sharpe['Time_Limit']:.0f} days
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig, df

def main():
    """Run comprehensive time limit optimization"""
    print("⏰ VXX MEAN REVERSION TIME LIMIT OPTIMIZATION")
    print("=" * 60)
    
    # Test symbols
    symbols = ['VXX', 'AMD', 'SQQQ']
    
    # Time limits to test (in days)
    time_limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    
    optimizer = ExitTimeOptimizer(
        initial_capital=10000,
        profit_target_pct=5.0,
        stop_loss_pct=7.5,
        commission=7.0
    )
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        
        try:
            results = optimizer.optimize_time_limits(symbol, time_limits)
            
            if results:
                all_results[symbol] = results
                
                # Create visualizations
                print("   📊 Creating optimization charts...")
                fig, df = optimizer.create_optimization_visualizations(results, symbol)
                
                # Save plot
                filename = f'{symbol}_time_limit_optimization.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='#1a1a1a', edgecolor='none')
                print(f"   ✅ Saved: {filename}")
                
                # Print best results
                best_return = max(results, key=lambda x: x.total_return_pct)
                best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
                
                print(f"\n   🏆 BEST CONFIGURATIONS:")
                print(f"   Best Return: {best_return.time_limit:.0f} days → {best_return.total_return_pct:+.1f}% return")
                print(f"   Best Sharpe: {best_sharpe.time_limit:.0f} days → {best_sharpe.sharpe_ratio:.2f} Sharpe")
                
                plt.close()
                
        except Exception as e:
            print(f"   ❌ Error optimizing {symbol}: {e}")
    
    # Overall summary
    if all_results:
        print(f"\n{'='*60}")
        print("📊 OPTIMIZATION SUMMARY ACROSS ALL SYMBOLS")
        print(f"{'='*60}")
        
        print(f"{'Symbol':<8} {'Best Days':<10} {'Return%':<10} {'Win%':<8} {'Sharpe':<8}")
        print("-" * 50)
        
        for symbol, results in all_results.items():
            best = max(results, key=lambda x: x.total_return_pct)
            print(f"{symbol:<8} {best.time_limit:<10.0f} {best.total_return_pct:<10.1f} "
                  f"{best.win_rate:<8.1f} {best.sharpe_ratio:<8.2f}")
    
    print(f"\n✅ Time limit optimization complete!")
    print(f"📁 Generated {len(all_results)} detailed optimization reports")

if __name__ == "__main__":
    main()