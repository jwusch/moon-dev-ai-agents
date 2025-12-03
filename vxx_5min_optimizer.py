"""
⚡ VXX 5-Minute Strategy Optimizer
Testing VXX Mean Reversion strategy on 5-minute timeframe with time limit optimization

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
class TimeOptimizationResult5Min:
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
    trades_per_day: float

class VXX5MinOptimizer:
    """
    Test VXX Mean Reversion strategy on 5-minute timeframe
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
        """Prepare 5-minute data with all indicators"""
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="5m", progress=False)
            print(f"📊 Downloaded {len(df)} bars of 5-minute {symbol} data")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Ensure we have the right column structure
        if df.columns.nlevels > 1:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Core indicators (adapted for 5-minute timeframe)
        # Use 60 periods for 5-hour SMA (equivalent to 20 periods on 15-min)
        df['SMA'] = df['Close'].rolling(60).mean()
        df['Distance_Pct'] = (df['Close'] - df['SMA']) / df['SMA'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 42)  # 42 periods = ~3.5 hours (14 on 15-min)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 42)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(60).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        # Momentum indicators for quality scoring
        df['ROC_1'] = talib.ROC(df['Close'].values, 1)
        df['ROC_3'] = talib.ROC(df['Close'].values, 9)  # 9 periods = 45min (3 on 15-min)
        
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Calculate entry quality score for 5-minute timeframe"""
        score = 0
        
        # Distance from SMA (0-25 points) - same thresholds
        distance = df['Distance_Pct'].iloc[idx]
        if distance < -2.0:
            score += 25
        elif distance < -1.5:
            score += 20
        elif distance < -1.0:
            score += 15
        
        # RSI level (0-25 points) - same thresholds
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
        """Run backtest with specific time limit in 5-minute periods"""
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        # Start after sufficient data for indicators
        start_idx = max(100, len(df) // 4)
        
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
                
                # Entry conditions (same as 15-min strategy)
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
                minutes_held = periods_held * 5
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Exit conditions
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance_Pct'].iloc[i] > -0.1:  # Slightly tighter for 5-min
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
        
        # Time-based metrics
        start_time = trades_df['entry_time'].min()
        end_time = trades_df['exit_time'].max()
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        
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
            'trades_per_day': trades_per_day,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
    
    def optimize_time_limits(self, symbol: str = "VXX", 
                           time_limits_periods: List[int] = None) -> List[TimeOptimizationResult5Min]:
        """Optimize across different time limits (in 5-minute periods)"""
        if time_limits_periods is None:
            # Test range from 3 periods (15min) to 120 periods (10 hours)
            # Equivalent to 1-40 periods on 15-min timeframe
            time_limits_periods = [3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
        
        print(f"🎯 Optimizing time limits for {symbol} on 5-minute timeframe...")
        print(f"📊 Testing {len(time_limits_periods)} different time limits...")
        print(f"⏱️ Range: {time_limits_periods[0]*5}min to {time_limits_periods[-1]*5}min")
        
        # Prepare data
        df = self.prepare_data(symbol)
        if df is None:
            return []
        
        results = []
        
        for time_limit in time_limits_periods:
            minutes = time_limit * 5
            hours = minutes / 60
            print(f"   Testing {time_limit} periods ({minutes:.0f}min / {hours:.1f}h)...", end="")
            
            try:
                result = self.run_backtest_with_time_limit(df, time_limit)
                
                if 'error' not in result:
                    opt_result = TimeOptimizationResult5Min(
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
                        avg_hold_minutes=result['avg_hold_minutes'],
                        trades_per_day=result['trades_per_day']
                    )
                    results.append(opt_result)
                    print(f" ✅ {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return_pct']:+.1f}% return, {result['trades_per_day']:.1f} tr/day")
                else:
                    print(f" ❌ {result['error']}")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        return results
    
    def create_comparison_with_15min(self, results_5min: List[TimeOptimizationResult5Min]):
        """Create comparison chart between 5-min and 15-min results"""
        if not results_5min:
            print("No 5-minute results to compare")
            return None
        
        # Convert to DataFrame
        df_5min = pd.DataFrame([
            {
                'Timeframe': '5min',
                'Time_Limit_Hours': r.time_limit_minutes / 60,
                'Total_Return': r.total_return_pct,
                'Win_Rate': r.win_rate,
                'Total_Trades': r.total_trades,
                'Trades_Per_Day': r.trades_per_day,
                'Avg_Hold_Minutes': r.avg_hold_minutes,
                'Sharpe_Ratio': r.sharpe_ratio,
                'Time_Limit_Exit_Pct': r.time_limit_exit_pct
            }
            for r in results_5min
        ])
        
        # Create comparison data (approximated 15-min results for key points)
        comparison_data = [
            {'Time_Limit_Hours': 0.25, 'Return_15min': 5.4, 'Return_5min': None},
            {'Time_Limit_Hours': 0.5, 'Return_15min': 2.2, 'Return_5min': None},
            {'Time_Limit_Hours': 0.75, 'Return_15min': 7.8, 'Return_5min': None},
            {'Time_Limit_Hours': 1.0, 'Return_15min': 1.2, 'Return_5min': None},
            {'Time_Limit_Hours': 1.5, 'Return_15min': 7.9, 'Return_5min': None},
            {'Time_Limit_Hours': 2.0, 'Return_15min': 7.5, 'Return_5min': None},
            {'Time_Limit_Hours': 3.0, 'Return_15min': 5.2, 'Return_5min': None},
        ]
        
        # Match 5-min results to comparison times
        for comp in comparison_data:
            closest_5min = min(df_5min.iterrows(), 
                             key=lambda x: abs(x[1]['Time_Limit_Hours'] - comp['Time_Limit_Hours']),
                             default=(None, None))
            if closest_5min[1] is not None:
                comp['Return_5min'] = closest_5min[1]['Total_Return']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('5-Minute vs 15-Minute Strategy Comparison', fontsize=18, fontweight='bold')
        
        # 1. Return Comparison
        ax = axes[0, 0]
        hours = [c['Time_Limit_Hours'] for c in comparison_data if c['Return_5min'] is not None]
        returns_15min = [c['Return_15min'] for c in comparison_data if c['Return_5min'] is not None]
        returns_5min = [c['Return_5min'] for c in comparison_data if c['Return_5min'] is not None]
        
        ax.plot(hours, returns_15min, 'o-', linewidth=3, markersize=8, color='#ff6b6b', label='15-minute')
        ax.plot(hours, returns_5min, 's-', linewidth=3, markersize=8, color='#4ecdc4', label='5-minute')
        ax.set_title('Return Comparison by Time Limit', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Total Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Trading Frequency
        ax = axes[0, 1]
        ax.plot(df_5min['Time_Limit_Hours'], df_5min['Trades_Per_Day'], 
                'o-', linewidth=3, markersize=8, color='#ffd93d')
        ax.set_title('Trading Frequency (5-minute)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Max Hold Time (hours)')
        ax.set_ylabel('Trades per Day')
        ax.grid(True, alpha=0.3)
        
        # 3. Win Rate vs Total Trades Scatter
        ax = axes[1, 0]
        scatter = ax.scatter(df_5min['Total_Trades'], df_5min['Win_Rate'], 
                           c=df_5min['Time_Limit_Hours'], s=100, cmap='viridis', alpha=0.8)
        ax.set_title('Win Rate vs Trade Volume (5-minute)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Trades')
        ax.set_ylabel('Win Rate (%)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time Limit (hours)')
        
        # 4. Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Find best 5-min results
        best_return_5min = max(results_5min, key=lambda x: x.total_return_pct)
        best_sharpe_5min = max(results_5min, key=lambda x: x.sharpe_ratio)
        
        # Approximate best 15-min results for comparison
        best_return_15min_val = 7.9  # From 1.5h limit
        best_return_15min_time = 1.5
        
        summary_text = f"""5-MINUTE vs 15-MINUTE COMPARISON

🏆 BEST 5-MINUTE RESULTS:
Best Return: {best_return_5min.total_return_pct:+.1f}%
  Time Limit: {best_return_5min.time_limit_minutes/60:.1f}h
  Win Rate: {best_return_5min.win_rate:.1f}%
  Trades: {best_return_5min.total_trades}
  Trades/Day: {best_return_5min.trades_per_day:.1f}

Best Sharpe: {best_sharpe_5min.sharpe_ratio:.2f}
  Time Limit: {best_sharpe_5min.time_limit_minutes/60:.1f}h
  Return: {best_sharpe_5min.total_return_pct:+.1f}%

📊 COMPARISON:
15-min Best: +{best_return_15min_val:.1f}% ({best_return_15min_time:.1f}h)
5-min Best:  {best_return_5min.total_return_pct:+.1f}% ({best_return_5min.time_limit_minutes/60:.1f}h)

Improvement: {best_return_5min.total_return_pct - best_return_15min_val:+.1f}%

💡 INSIGHTS:
• 5-min = {best_return_5min.trades_per_day:.1f}x more trades/day
• Faster entries and exits
• More trading opportunities
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig, df_5min

def main():
    """Run 5-minute timeframe optimization for VXX"""
    print("⚡ VXX 5-MINUTE STRATEGY OPTIMIZATION")
    print("=" * 60)
    print("Testing VXX Mean Reversion strategy on 5-minute timeframe")
    print("Hypothesis: 5-min should outperform 15-min (more opportunities)")
    
    # Time limits to test (in 5-minute periods)
    # Equivalent to key 15-min periods: 1,2,3,4,6,8,10,12,16,20,24,28,32,36,40
    time_limits_periods = [3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
    
    optimizer = VXX5MinOptimizer(
        initial_capital=10000,
        profit_target_pct=5.0,
        stop_loss_pct=7.5,
        commission=1.0
    )
    
    print(f"\n{'='*20} VXX 5-MINUTE OPTIMIZATION {'='*20}")
    
    try:
        results = optimizer.optimize_time_limits("VXX", time_limits_periods)
        
        if results:
            # Create comparison visualization
            print("\n📊 Creating 5-min vs 15-min comparison...")
            fig, df = optimizer.create_comparison_with_15min(results)
            
            # Save plot
            filename = 'VXX_5min_vs_15min_comparison.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"✅ Saved: {filename}")
            
            # Print detailed results
            best_return = max(results, key=lambda x: x.total_return_pct)
            best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
            
            print(f"\n🏆 5-MINUTE OPTIMIZATION RESULTS:")
            print("=" * 100)
            print(f"{'Periods':<8} {'Hours':<6} {'Return%':<8} {'Win%':<6} {'Trades':<7} {'Tr/Day':<7} {'Sharpe':<7} {'TimeExit%':<9}")
            print("-" * 100)
            
            for result in results:
                print(f"{result.time_limit_periods:<8} {result.time_limit_minutes/60:<6.1f} "
                      f"{result.total_return_pct:<8.1f} {result.win_rate:<6.1f} "
                      f"{result.total_trades:<7} {result.trades_per_day:<7.1f} "
                      f"{result.sharpe_ratio:<7.2f} {result.time_limit_exit_pct:<9.1f}")
            
            print("\n🎯 KEY FINDINGS:")
            print(f"• Best Return: {best_return.time_limit_periods} periods ({best_return.time_limit_minutes/60:.1f}h) → {best_return.total_return_pct:+.1f}%")
            print(f"• Best Sharpe: {best_sharpe.time_limit_periods} periods ({best_sharpe.time_limit_minutes/60:.1f}h) → {best_sharpe.sharpe_ratio:.2f}")
            print(f"• Trading Frequency: {best_return.trades_per_day:.1f} trades/day vs ~0.2 trades/day on 15-min")
            
            # Compare to 15-min best
            print(f"\n📊 TIMEFRAME COMPARISON:")
            print(f"• 15-minute best: +7.9% (1.5h limit)")
            print(f"• 5-minute best:  {best_return.total_return_pct:+.1f}% ({best_return.time_limit_minutes/60:.1f}h limit)")
            improvement = best_return.total_return_pct - 7.9
            print(f"• Improvement: {improvement:+.1f}% with 5-minute timeframe")
            print(f"• Trading volume: {best_return.total_trades} trades vs ~15 on 15-min")
            
            plt.close()
            
    except Exception as e:
        print(f"❌ Error optimizing VXX on 5-minute: {e}")
    
    print(f"\n✅ VXX 5-minute strategy optimization complete!")

if __name__ == "__main__":
    main()