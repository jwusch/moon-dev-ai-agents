"""
⚡ Intraday Timeframe Optimizer
Testing VXX Mean Reversion strategy on 1min, 5min, and 15min timeframes

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
class TimeframeResult:
    timeframe: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    trades_per_day: float
    avg_hold_minutes: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    best_exit_reason: str

class IntradayTimeframeOptimizer:
    """
    Test VXX Mean Reversion across different intraday timeframes
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 commission: float = 1.0):  # Lower commission for frequent trading
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
        
    def prepare_data(self, symbol: str, interval: str, period: str = "60d") -> pd.DataFrame:
        """Prepare data for different timeframes"""
        try:
            import yfinance as yf
            
            # Adjust period based on interval to get sufficient data
            if interval == "1m":
                period = "7d"  # 1-min data limited to 7 days
            elif interval == "5m":
                period = "60d"  # 5-min data limited to 60 days
            else:  # 15m
                period = "60d"
            
            print(f"      📊 Downloading {symbol} {interval} data for {period}...")
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
        except Exception as e:
            print(f"      ❌ Error downloading {symbol} {interval}: {e}")
            return None
        
        if len(df) < 100:
            print(f"      ⚠️ Insufficient data: {len(df)} bars")
            return None
        
        # Ensure proper column structure
        if df.columns.nlevels > 1:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Calculate indicators (adjust periods for timeframe)
        if interval == "1m":
            sma_period = 20  # 20 minutes
            rsi_period = 14  # 14 minutes
            atr_period = 14
        elif interval == "5m":
            sma_period = 20  # 100 minutes (~1.7 hours)
            rsi_period = 14  # 70 minutes
            atr_period = 14
        else:  # 15m
            sma_period = 20  # 5 hours
            rsi_period = 14  # 3.5 hours
            atr_period = 14
        
        df['SMA'] = df['Close'].rolling(sma_period).mean()
        df['Distance_Pct'] = (df['Close'] - df['SMA']) / df['SMA'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, rsi_period)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, atr_period)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(sma_period).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        # Momentum indicators
        df['ROC_1'] = talib.ROC(df['Close'].values, 1)
        df['ROC_3'] = talib.ROC(df['Close'].values, 3)
        
        print(f"      ✅ Prepared {len(df)} bars of {interval} data")
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Calculate entry quality score"""
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
    
    def run_backtest(self, df: pd.DataFrame, interval: str, min_quality_score: int = 40) -> Dict:
        """Run backtest for specific timeframe"""
        
        # Adjust time limits based on timeframe
        if interval == "1m":
            max_hold_periods = 60  # 1 hour
            mean_reversion_threshold = -0.2  # More sensitive
        elif interval == "5m":
            max_hold_periods = 36  # 3 hours (36 * 5min)
            mean_reversion_threshold = -0.3
        else:  # 15m
            max_hold_periods = 12  # 3 hours (12 * 15min)
            mean_reversion_threshold = -0.3
        
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        # Skip weekends and outside market hours
        start_idx = max(50, len(df) // 4)  # Start after enough data for indicators
        
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
                
                # Entry conditions
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
                minutes_held = periods_held * (1 if interval == "1m" else 5 if interval == "5m" else 15)
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Exit conditions
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance_Pct'].iloc[i] > mean_reversion_threshold:
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
                'equity': current_capital,
                'price': current_price
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Analyze results
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_return_pct = (current_capital / self.initial_capital - 1) * 100
        
        # Time-adjusted metrics
        start_time = trades_df['entry_time'].min()
        end_time = trades_df['exit_time'].max()
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        wins = trades_df[trades_df['win']]['pnl_pct']
        losses = trades_df[~trades_df['win']]['pnl_pct']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
        
        # Drawdown
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
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        best_exit_reason = exit_reasons.index[0] if len(exit_reasons) > 0 else 'None'
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'trades_per_day': trades_per_day,
            'avg_hold_minutes': trades_df['minutes_held'].mean(),
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'best_exit_reason': best_exit_reason,
            'trades_df': trades_df,
            'equity_df': equity_df,
            'total_days': total_days
        }
    
    def optimize_timeframes(self, symbol: str, timeframes: List[str] = None) -> List[TimeframeResult]:
        """Test strategy across multiple timeframes"""
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]
        
        print(f"\n🎯 TIMEFRAME OPTIMIZATION: {symbol}")
        print("=" * 50)
        
        results = []
        
        for timeframe in timeframes:
            print(f"\n⏱️ Testing {timeframe} timeframe...")
            
            try:
                # Prepare data
                df = self.prepare_data(symbol, timeframe)
                
                if df is None:
                    continue
                
                # Run backtest
                print("      🔄 Running backtest...")
                result = self.run_backtest(df, timeframe)
                
                if 'error' in result:
                    print(f"      ❌ {result['error']}")
                    continue
                
                # Create result object
                tf_result = TimeframeResult(
                    timeframe=timeframe,
                    total_trades=result['total_trades'],
                    win_rate=result['win_rate'],
                    total_return_pct=result['total_return_pct'],
                    trades_per_day=result['trades_per_day'],
                    avg_hold_minutes=result['avg_hold_minutes'],
                    profit_factor=result['profit_factor'],
                    max_drawdown=result['max_drawdown'],
                    sharpe_ratio=result['sharpe_ratio'],
                    best_exit_reason=result['best_exit_reason']
                )
                results.append(tf_result)
                
                # Print summary
                print(f"      ✅ Results: {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"{result['total_return_pct']:+.1f}% return")
                print(f"         Trading: {result['trades_per_day']:.1f} trades/day, "
                      f"{result['avg_hold_minutes']:.0f}min avg hold")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        return results
    
    def create_comparison_chart(self, results: List[TimeframeResult], symbol: str):
        """Create comprehensive timeframe comparison"""
        if not results:
            print("No results to visualize")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{symbol} - Timeframe Optimization Comparison', fontsize=18, fontweight='bold')
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Timeframe': r.timeframe,
                'Total_Return': r.total_return_pct,
                'Win_Rate': r.win_rate,
                'Trades_Per_Day': r.trades_per_day,
                'Avg_Hold_Minutes': r.avg_hold_minutes,
                'Profit_Factor': r.profit_factor if r.profit_factor != float('inf') else 10,
                'Max_Drawdown': r.max_drawdown,
                'Sharpe_Ratio': r.sharpe_ratio,
                'Total_Trades': r.total_trades
            }
            for r in results
        ])
        
        # 1. Total Return Comparison
        ax = axes[0, 0]
        bars = ax.bar(df['Timeframe'], df['Total_Return'], 
                     color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
        ax.set_title('Total Return by Timeframe', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = df['Total_Return'].idxmax()
        bars[best_idx].set_color('gold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{df.iloc[i]["Total_Return"]:.1f}%', ha='center', fontweight='bold')
        
        # 2. Win Rate Comparison
        ax = axes[0, 1]
        bars = ax.bar(df['Timeframe'], df['Win_Rate'], 
                     color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        ax.set_title('Win Rate by Timeframe', fontweight='bold')
        ax.set_ylabel('Win Rate (%)')
        ax.axhline(y=50, color='white', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{df.iloc[i]["Win_Rate"]:.1f}%', ha='center', fontweight='bold')
        
        # 3. Trading Frequency
        ax = axes[0, 2]
        bars = ax.bar(df['Timeframe'], df['Trades_Per_Day'], 
                     color=['#ffcc99', '#ff99cc', '#ccff99'], alpha=0.8)
        ax.set_title('Trading Frequency', fontweight='bold')
        ax.set_ylabel('Trades per Day')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{df.iloc[i]["Trades_Per_Day"]:.1f}', ha='center', fontweight='bold')
        
        # 4. Average Hold Time
        ax = axes[1, 0]
        bars = ax.bar(df['Timeframe'], df['Avg_Hold_Minutes'], 
                     color=['#ffd93d', '#6bcf7f', '#4d96ff'], alpha=0.8)
        ax.set_title('Average Hold Time', fontweight='bold')
        ax.set_ylabel('Minutes')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            minutes = df.iloc[i]['Avg_Hold_Minutes']
            if minutes >= 60:
                label = f'{minutes/60:.1f}h'
            else:
                label = f'{minutes:.0f}m'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   label, ha='center', fontweight='bold')
        
        # 5. Risk-Return Scatter
        ax = axes[1, 1]
        scatter = ax.scatter(df['Max_Drawdown'], df['Total_Return'], 
                           c=['red', 'blue', 'green'], s=200, alpha=0.8)
        
        # Add labels for each point
        for i, row in df.iterrows():
            ax.annotate(row['Timeframe'], 
                       (row['Max_Drawdown'], row['Total_Return']),
                       xytext=(5, 5), textcoords='offset points', 
                       fontweight='bold', fontsize=12)
        
        ax.set_title('Risk vs Return', fontweight='bold')
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Find best performers
        best_return = df.loc[df['Total_Return'].idxmax()]
        best_winrate = df.loc[df['Win_Rate'].idxmax()]
        best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
        
        summary_text = f"""TIMEFRAME COMPARISON

🏆 BEST PERFORMERS:

Highest Return:
  {best_return['Timeframe']}: {best_return['Total_Return']:+.1f}%
  Win Rate: {best_return['Win_Rate']:.1f}%
  Trades/Day: {best_return['Trades_Per_Day']:.1f}

Best Win Rate:
  {best_winrate['Timeframe']}: {best_winrate['Win_Rate']:.1f}%
  Return: {best_winrate['Total_Return']:+.1f}%
  Hold Time: {best_winrate['Avg_Hold_Minutes']:.0f}min

Best Sharpe:
  {best_sharpe['Timeframe']}: {best_sharpe['Sharpe_Ratio']:.2f}
  Return: {best_sharpe['Total_Return']:+.1f}%
  Drawdown: {best_sharpe['Max_Drawdown']:.1f}%

💡 INSIGHTS:
• Faster = More Trades
• Shorter Holds = Less Risk
• Higher Frequency Trading
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig, df

def main():
    """Run comprehensive timeframe optimization"""
    print("⚡ INTRADAY TIMEFRAME OPTIMIZATION")
    print("=" * 60)
    print("Testing VXX Mean Reversion on 1min, 5min, and 15min timeframes")
    print("Expected: Smaller timeframes = More opportunities + Better returns")
    
    # Test symbols
    symbols = ['VXX', 'SQQQ', 'AMD']  # Focus on volatility products
    timeframes = ["1m", "5m", "15m"]
    
    optimizer = IntradayTimeframeOptimizer(
        initial_capital=10000,
        profit_target_pct=5.0,
        stop_loss_pct=7.5,
        commission=1.0  # Lower for frequent trading
    )
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} TIMEFRAME TEST {'='*20}")
        
        try:
            results = optimizer.optimize_timeframes(symbol, timeframes)
            
            if results:
                all_results[symbol] = results
                
                # Create comparison chart
                print("\n   📊 Creating timeframe comparison chart...")
                fig, df = optimizer.create_comparison_chart(results, symbol)
                
                # Save chart
                filename = f'{symbol}_timeframe_optimization.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='#1a1a1a', edgecolor='none')
                print(f"   ✅ Saved: {filename}")
                
                # Print results summary
                print(f"\n   📊 {symbol} TIMEFRAME RESULTS:")
                print("   " + "-" * 70)
                print(f"   {'TF':<4} {'Return%':<8} {'Win%':<6} {'Trades/D':<8} {'Hold':<8} {'Sharpe':<7}")
                print("   " + "-" * 70)
                
                for result in results:
                    hold_str = f"{result.avg_hold_minutes:.0f}m" if result.avg_hold_minutes < 60 else f"{result.avg_hold_minutes/60:.1f}h"
                    print(f"   {result.timeframe:<4} {result.total_return_pct:<8.1f} "
                          f"{result.win_rate:<6.1f} {result.trades_per_day:<8.1f} "
                          f"{hold_str:<8} {result.sharpe_ratio:<7.2f}")
                
                plt.close()
                
        except Exception as e:
            print(f"   ❌ Error testing {symbol}: {e}")
    
    # Overall comparison
    if all_results:
        print(f"\n{'='*60}")
        print("🏆 BEST TIMEFRAME BY SYMBOL")
        print(f"{'='*60}")
        
        for symbol, results in all_results.items():
            best_result = max(results, key=lambda x: x.total_return_pct)
            print(f"\n{symbol}:")
            print(f"  Best Timeframe: {best_result.timeframe}")
            print(f"  Return: {best_result.total_return_pct:+.1f}%")
            print(f"  Win Rate: {best_result.win_rate:.1f}%")
            print(f"  Trading: {best_result.trades_per_day:.1f} trades/day")
            print(f"  Hold Time: {best_result.avg_hold_minutes:.0f} minutes")
        
        # Overall insights
        print(f"\n💡 KEY INSIGHTS:")
        
        all_timeframe_results = [result for results in all_results.values() for result in results]
        
        # Group by timeframe
        tf_1m = [r for r in all_timeframe_results if r.timeframe == "1m"]
        tf_5m = [r for r in all_timeframe_results if r.timeframe == "5m"]
        tf_15m = [r for r in all_timeframe_results if r.timeframe == "15m"]
        
        if tf_1m:
            avg_return_1m = sum(r.total_return_pct for r in tf_1m) / len(tf_1m)
            avg_trades_1m = sum(r.trades_per_day for r in tf_1m) / len(tf_1m)
            print(f"• 1-minute: {avg_return_1m:+.1f}% avg return, {avg_trades_1m:.1f} trades/day")
        
        if tf_5m:
            avg_return_5m = sum(r.total_return_pct for r in tf_5m) / len(tf_5m)
            avg_trades_5m = sum(r.trades_per_day for r in tf_5m) / len(tf_5m)
            print(f"• 5-minute: {avg_return_5m:+.1f}% avg return, {avg_trades_5m:.1f} trades/day")
        
        if tf_15m:
            avg_return_15m = sum(r.total_return_pct for r in tf_15m) / len(tf_15m)
            avg_trades_15m = sum(r.trades_per_day for r in tf_15m) / len(tf_15m)
            print(f"• 15-minute: {avg_return_15m:+.1f}% avg return, {avg_trades_15m:.1f} trades/day")
    
    print(f"\n✅ Timeframe optimization complete!")
    print(f"📁 Generated {len(all_results)} detailed timeframe comparisons")

if __name__ == "__main__":
    main()