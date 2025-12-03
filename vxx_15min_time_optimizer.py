"""
⚡ VXX 15-Minute Time Limit Optimizer
Optimizes time-based exits for the original VXX Mean Reversion 15-minute strategy

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
class TimeOptimizationResult:
    time_limit_periods: int
    time_limit_minutes: int
    total_trades: int
    win_rate: float
    total_return_pct: float
    avg_return_per_trade: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    time_limit_exits: int
    time_limit_exit_pct: float
    avg_hold_minutes: float

class VXX15MinTimeOptimizer:
    """
    Optimize time-based exit strategy for VXX Mean Reversion 15-minute strategy
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 commission: float = 1.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
        
    def prepare_data(self, symbol: str = "VXX", period: str = "60d") -> pd.DataFrame:
        """Prepare 15-minute data with all indicators"""
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="15m", progress=False)
            print(f"📊 Downloaded {len(df)} bars of 15-minute {symbol} data")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Ensure we have the right column structure
        if df.columns.nlevels > 1:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Core indicators (same as original strategy)
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
        
        # Momentum indicators for quality scoring
        df['ROC_1'] = talib.ROC(df['Close'].values, 1)
        df['ROC_3'] = talib.ROC(df['Close'].values, 3)
        
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Calculate entry quality score (same as improved strategy)"""
        score = 0
        
        # Distance from SMA (0-25 points)
        distance = df['Distance_Pct'].iloc[idx]
        if distance < -2.0:
            score += 25
        elif distance < -1.5:
            score += 20
        elif distance < -1.0:
            score += 15
        
        # RSI level (0-25 points)
        rsi = df['RSI'].iloc[idx]
        if rsi < 25:
            score += 25
        elif rsi < 30:
            score += 20
        elif rsi < 35:
            score += 15
        elif rsi < 40:
            score += 10
        
        # Volume confirmation (0-20 points)
        volume_ratio = df['Volume_Ratio'].iloc[idx]
        if not pd.isna(volume_ratio):
            if volume_ratio > 1.5:
                score += 20
            elif volume_ratio > 1.2:
                score += 15
            elif volume_ratio > 1.0:
                score += 10
        
        # Momentum (0-15 points)
        if 'ROC_1' in df.columns and 'ROC_3' in df.columns:
            roc_1 = df['ROC_1'].iloc[idx]
            roc_3 = df['ROC_3'].iloc[idx]
            if not pd.isna(roc_1) and not pd.isna(roc_3):
                if roc_1 > roc_3 and roc_1 > -0.5:
                    score += 15
                elif roc_1 > -1.0:
                    score += 8
        
        return score
    
    def run_backtest_with_time_limit(self, df: pd.DataFrame, max_hold_periods: int,
                                   min_quality_score: int = 40) -> Dict:
        """Run backtest with specific time limit in 15-minute periods"""
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        # Start after sufficient data for indicators
        start_idx = max(50, len(df) // 4)
        
        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Market hours filter (9:30 AM - 4:00 PM ET, Mon-Fri)
            if (current_time.weekday() >= 5 or  # Weekend
                current_time.hour < 9 or 
                (current_time.hour == 9 and current_time.minute < 30) or
                current_time.hour >= 16):
                continue
            
            # Check for new entry
            if position is None:
                distance = df['Distance_Pct'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                # Original entry conditions
                if distance < -1.0 and rsi < 40:
                    score = self.calculate_quality_score(df, i)
                    
                    if score >= min_quality_score:
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'entry_idx': i,
                            'score': score
                        }
            
            # Check for exit
            elif position is not None:
                periods_held = i - position['entry_idx']
                minutes_held = periods_held * 15
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Exit conditions (same as original strategy)
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance_Pct'].iloc[i] > -0.3:
                    exit_reason = 'Mean Reversion'
                elif periods_held >= max_hold_periods:
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    # Calculate trade results
                    position_size = current_capital * 0.95  # 95% of capital
                    shares = position_size / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price']) - (2 * self.commission)
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'minutes_held': minutes_held,
                        'periods_held': periods_held,
                        'exit_reason': exit_reason,
                        'entry_score': position['score'],
                        'win': pnl_pct > 0
                    })
                    position = None
            
            # Record equity
            equity_curve.append({
                'time': current_time,
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
        
        # Sharpe ratio (annualized)
        if len(returns) > 1:
            # Calculate based on trade frequency
            start_time = trades_df['entry_time'].min()
            end_time = trades_df['exit_time'].max()
            total_days = (end_time - start_time).total_seconds() / (24 * 3600)
            trades_per_day = total_trades / total_days if total_days > 0 else 0
            trade_frequency_per_year = trades_per_day * 252
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trade_frequency_per_year) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
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
            'avg_hold_minutes': trades_df['minutes_held'].mean(),
            'trades_df': trades_df,
            'equity_df': equity_df
        }
    
    def optimize_time_limits(self, symbol: str = "VXX", 
                           time_limits_periods: List[int] = None) -> List[TimeOptimizationResult]:
        """Optimize across different time limits (in 15-minute periods)"""
        if time_limits_periods is None:
            # Test range from 1 period (15min) to 20 periods (5 hours)
            time_limits_periods = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40]
        
        print(f"🎯 Optimizing time limits for {symbol} on 15-minute timeframe...")
        print(f"📊 Testing {len(time_limits_periods)} different time limits...")
        print(f"⏱️ Range: {time_limits_periods[0]*15}min to {time_limits_periods[-1]*15}min")
        
        # Prepare data
        df = self.prepare_data(symbol)
        if df is None:
            return []
        
        results = []
        
        for time_limit in time_limits_periods:
            minutes = time_limit * 15
            hours = minutes / 60
            print(f"   Testing {time_limit} periods ({minutes:.0f}min / {hours:.1f}h)...", end="")
            
            try:
                result = self.run_backtest_with_time_limit(df, time_limit)
                
                if 'error' not in result:
                    opt_result = TimeOptimizationResult(
                        time_limit_periods=time_limit,
                        time_limit_minutes=minutes,
                        total_trades=result['total_trades'],
                        win_rate=result['win_rate'],
                        total_return_pct=result['total_return_pct'],
                        avg_return_per_trade=result['avg_return_per_trade'],
                        profit_factor=result['profit_factor'],
                        max_drawdown=result['max_drawdown'],
                        sharpe_ratio=result['sharpe_ratio'],
                        time_limit_exits=result['time_limit_exits'],
                        time_limit_exit_pct=result['time_limit_exit_pct'],
                        avg_hold_minutes=result['avg_hold_minutes']
                    )
                    results.append(opt_result)
                    print(f" ✅ {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return_pct']:+.1f}% return")
                else:
                    print(f" ❌ {result['error']}")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        return results
    
    def create_optimization_visualizations(self, results: List[TimeOptimizationResult], symbol: str):
        """Create comprehensive optimization visualizations"""
        if not results:
            print("No results to visualize")
            return None
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'Time_Limit_Periods': r.time_limit_periods,
                'Time_Limit_Minutes': r.time_limit_minutes,
                'Time_Limit_Hours': r.time_limit_minutes / 60,
                'Total_Trades': r.total_trades,
                'Win_Rate': r.win_rate,
                'Total_Return': r.total_return_pct,
                'Avg_Return_Per_Trade': r.avg_return_per_trade,
                'Profit_Factor': r.profit_factor if r.profit_factor != float('inf') else 10,
                'Max_Drawdown': r.max_drawdown,
                'Sharpe_Ratio': r.sharpe_ratio,
                'Time_Limit_Exits': r.time_limit_exits,
                'Time_Limit_Exit_Pct': r.time_limit_exit_pct,
                'Avg_Hold_Minutes': r.avg_hold_minutes
            }
            for r in results
        ])
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{symbol} - 15-Minute Time Limit Optimization', fontsize=20, fontweight='bold')
        
        # 1. Total Return vs Time Limit (in hours for readability)
        ax = axes[0, 0]
        ax.plot(df['Time_Limit_Hours'], df['Total_Return'], 'o-', linewidth=3, markersize=8, color='#00ff88')
        ax.set_title('Total Return vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # Highlight best return
        best_return_idx = df['Total_Return'].idxmax()
        ax.scatter(df.loc[best_return_idx, 'Time_Limit_Hours'], df.loc[best_return_idx, 'Total_Return'], 
                  s=200, color='gold', marker='*', zorder=5)
        ax.text(df.loc[best_return_idx, 'Time_Limit_Hours'], df.loc[best_return_idx, 'Total_Return'] + 1,
                f"Best: {df.loc[best_return_idx, 'Time_Limit_Hours']:.1f}h", ha='center', fontweight='bold')
        
        # 2. Win Rate vs Time Limit
        ax = axes[0, 1]
        ax.plot(df['Time_Limit_Hours'], df['Win_Rate'], 'o-', linewidth=3, markersize=8, color='#4ecdc4')
        ax.set_title('Win Rate vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Win Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='white', linestyle='--', alpha=0.5, label='50% Break-even')
        
        # 3. Sharpe Ratio vs Time Limit
        ax = axes[0, 2]
        ax.plot(df['Time_Limit_Hours'], df['Sharpe_Ratio'], 'o-', linewidth=3, markersize=8, color='#ffd93d')
        ax.set_title('Sharpe Ratio vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # 4. Number of Trades vs Time Limit
        ax = axes[1, 0]
        ax.plot(df['Time_Limit_Hours'], df['Total_Trades'], 'o-', linewidth=3, markersize=8, color='#ff6b6b')
        ax.set_title('Trade Frequency vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Total Trades')
        ax.grid(True, alpha=0.3)
        
        # 5. Average Return per Trade
        ax = axes[1, 1]
        ax.plot(df['Time_Limit_Hours'], df['Avg_Return_Per_Trade'], 'o-', linewidth=3, markersize=8, color='#a8e6cf')
        ax.set_title('Avg Return per Trade vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Avg Return per Trade (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # 6. Time Limit Exit Percentage
        ax = axes[1, 2]
        ax.plot(df['Time_Limit_Hours'], df['Time_Limit_Exit_Pct'], 'o-', linewidth=3, markersize=8, color='#ff9999')
        ax.set_title('% of Trades Exiting on Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Time Limit Exits (%)')
        ax.grid(True, alpha=0.3)
        
        # 7. Max Drawdown vs Time Limit
        ax = axes[2, 0]
        ax.plot(df['Time_Limit_Hours'], df['Max_Drawdown'], 'o-', linewidth=3, markersize=8, color='#ffb3ba')
        ax.set_title('Max Drawdown vs Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Max Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # 8. Actual vs Limit Hold Time Comparison
        ax = axes[2, 1]
        ax.plot(df['Time_Limit_Minutes'], df['Avg_Hold_Minutes'], 'o-', linewidth=3, markersize=8, color='#b19cd9')
        ax.plot([0, df['Time_Limit_Minutes'].max()], [0, df['Time_Limit_Minutes'].max()], 'r--', alpha=0.5, label='Time Limit Line')
        ax.set_title('Actual vs Max Hold Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (minutes)')
        ax.set_ylabel('Actual Avg Hold Time (minutes)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 9. Optimization Summary
        ax = axes[2, 2]
        ax.axis('off')
        
        # Find best configurations
        best_return = df.loc[df['Total_Return'].idxmax()]
        best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
        best_winrate = df.loc[df['Win_Rate'].idxmax()]
        
        # Original strategy (12 periods = 3 hours)
        original_periods = 12
        original_row = df[df['Time_Limit_Periods'] == original_periods]
        if len(original_row) > 0:
            orig = original_row.iloc[0]
            orig_text = f"""
ORIGINAL (3h limit):
  Return: {orig['Total_Return']:+.1f}%
  Win Rate: {orig['Win_Rate']:.1f}%
  Trades: {orig['Total_Trades']:.0f}
  Time Exits: {orig['Time_Limit_Exit_Pct']:.1f}%"""
        else:
            orig_text = "\nORIGINAL: Not tested"
        
        summary_text = f"""15-MIN TIMEFRAME OPTIMIZATION
        
🏆 BEST CONFIGURATIONS:

Best Total Return:
  Time Limit: {best_return['Time_Limit_Hours']:.1f}h ({best_return['Time_Limit_Periods']:.0f} periods)
  Return: {best_return['Total_Return']:+.1f}%
  Win Rate: {best_return['Win_Rate']:.1f}%
  Trades: {best_return['Total_Trades']:.0f}

Best Sharpe Ratio:
  Time Limit: {best_sharpe['Time_Limit_Hours']:.1f}h ({best_sharpe['Time_Limit_Periods']:.0f} periods)
  Sharpe: {best_sharpe['Sharpe_Ratio']:.2f}
  Return: {best_sharpe['Total_Return']:+.1f}%
  Win Rate: {best_sharpe['Win_Rate']:.1f}%{orig_text}

💡 INSIGHTS:
• Test Range: {df['Time_Limit_Hours'].min():.1f}h to {df['Time_Limit_Hours'].max():.1f}h
• Avg Time Exits: {df['Time_Limit_Exit_Pct'].mean():.1f}%
• Optimal: {best_return['Time_Limit_Hours']:.1f}h limit
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig, df

def main():
    """Run 15-minute timeframe time limit optimization for VXX"""
    print("⚡ VXX 15-MINUTE TIME LIMIT OPTIMIZATION")
    print("=" * 60)
    print("Optimizing time-based exit limits for the original VXX Mean Reversion strategy")
    print("Original strategy used 12 periods (3 hours) - testing 1-40 periods")
    
    # Time limits to test (in 15-minute periods)
    time_limits_periods = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40]
    
    optimizer = VXX15MinTimeOptimizer(
        initial_capital=10000,
        profit_target_pct=5.0,  # Original strategy setting
        stop_loss_pct=7.5,      # Slightly wider than original 1.5%
        commission=1.0          # Lower commission for 15-min trading
    )
    
    print(f"\n{'='*20} VXX 15-MINUTE OPTIMIZATION {'='*20}")
    
    try:
        results = optimizer.optimize_time_limits("VXX", time_limits_periods)
        
        if results:
            # Create visualizations
            print("\n📊 Creating optimization charts...")
            fig, df = optimizer.create_optimization_visualizations(results, "VXX")
            
            # Save plot
            filename = 'VXX_15min_time_limit_optimization.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"✅ Saved: {filename}")
            
            # Print detailed results
            best_return = max(results, key=lambda x: x.total_return_pct)
            best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
            original_result = next((r for r in results if r.time_limit_periods == 12), None)
            
            print(f"\n🏆 OPTIMIZATION RESULTS:")
            print("=" * 80)
            print(f"{'Periods':<8} {'Hours':<6} {'Return%':<8} {'Win%':<6} {'Trades':<7} {'Sharpe':<7} {'TimeExit%':<9}")
            print("-" * 80)
            
            for result in results:
                print(f"{result.time_limit_periods:<8} {result.time_limit_minutes/60:<6.1f} "
                      f"{result.total_return_pct:<8.1f} {result.win_rate:<6.1f} "
                      f"{result.total_trades:<7} {result.sharpe_ratio:<7.2f} "
                      f"{result.time_limit_exit_pct:<9.1f}")
            
            print("\n🎯 KEY FINDINGS:")
            print(f"• Best Return: {best_return.time_limit_periods} periods ({best_return.time_limit_minutes/60:.1f}h) → {best_return.total_return_pct:+.1f}%")
            print(f"• Best Sharpe: {best_sharpe.time_limit_periods} periods ({best_sharpe.time_limit_minutes/60:.1f}h) → {best_sharpe.sharpe_ratio:.2f}")
            
            if original_result:
                print(f"• Original (12 periods / 3h): {original_result.total_return_pct:+.1f}% return, {original_result.win_rate:.1f}% win rate")
                improvement = best_return.total_return_pct - original_result.total_return_pct
                print(f"• Improvement: {improvement:+.1f}% better with {best_return.time_limit_periods} periods")
            
            plt.close()
            
    except Exception as e:
        print(f"❌ Error optimizing VXX: {e}")
    
    print(f"\n✅ VXX 15-minute time limit optimization complete!")

if __name__ == "__main__":
    main()