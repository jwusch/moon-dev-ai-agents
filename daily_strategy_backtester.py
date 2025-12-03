"""
📈 VXX Mean Reversion Daily Strategy Backtester
Comprehensive backtesting using daily data with detailed visualizations

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
class Trade:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_type: str
    pnl_pct: float
    pnl_dollars: float
    hold_days: int
    entry_score: int
    exit_reason: str

class VXXMeanReversionDailyBacktester:
    """
    Advanced backtester for VXX Mean Reversion strategy using daily data
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 max_hold_days: int = 5,
                 commission: float = 7.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.commission = commission
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        
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
            # Multi-level columns, flatten them
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Core indicators (adapted for daily timeframe)
        df['SMA20'] = df['Close'].rolling(20).mean()  # 20-day moving average
        df['Distance_Pct'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        # Additional quality indicators
        df['ROC_1'] = talib.ROC(df['Close'].values, 1)
        df['ROC_3'] = talib.ROC(df['Close'].values, 3)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle * 100
        
        # Market structure
        df['Close_5d_ago'] = df['Close'].shift(5)
        df['Price_Change_5d'] = (df['Close'] - df['Close_5d_ago']) / df['Close_5d_ago'] * 100
        
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> Tuple[int, Dict]:
        """Calculate entry quality score"""
        score = 0
        details = {}
        
        # 1. Distance from SMA (0-25 points) - more weight for daily
        distance = df['Distance_Pct'].iloc[idx]
        if distance < -3.0:
            score += 25
            details['distance_score'] = 25
        elif distance < -2.0:
            score += 20
            details['distance_score'] = 20
        elif distance < -1.0:
            score += 15
            details['distance_score'] = 15
        else:
            details['distance_score'] = 0
            
        # 2. RSI level (0-25 points)
        rsi = df['RSI'].iloc[idx]
        if rsi < 25:
            score += 25
            details['rsi_score'] = 25
        elif rsi < 30:
            score += 20
            details['rsi_score'] = 20
        elif rsi < 35:
            score += 15
            details['rsi_score'] = 15
        elif rsi < 40:
            score += 10
            details['rsi_score'] = 10
        else:
            details['rsi_score'] = 0
            
        # 3. Volume confirmation (0-20 points)
        if 'Volume_Ratio' in df.columns:
            volume_ratio = df['Volume_Ratio'].iloc[idx]
            if not pd.isna(volume_ratio):
                if volume_ratio > 2.0:
                    score += 20
                    details['volume_score'] = 20
                elif volume_ratio > 1.5:
                    score += 15
                    details['volume_score'] = 15
                elif volume_ratio > 1.0:
                    score += 10
                    details['volume_score'] = 10
        
        # 4. Recent momentum (0-20 points)
        if 'Price_Change_5d' in df.columns:
            change_5d = df['Price_Change_5d'].iloc[idx]
            if not pd.isna(change_5d):
                if change_5d < -10:  # Strong decline
                    score += 20
                    details['momentum_score'] = 20
                elif change_5d < -5:
                    score += 15
                    details['momentum_score'] = 15
                elif change_5d < 0:
                    score += 10
                    details['momentum_score'] = 10
        
        # 5. Volatility environment (0-10 points)
        if 'BB_Width' in df.columns:
            bb_width = df['BB_Width'].iloc[idx]
            if not pd.isna(bb_width):
                if bb_width > 8:  # High volatility
                    score += 10
                    details['volatility_score'] = 10
                elif bb_width > 5:
                    score += 5
                    details['volatility_score'] = 5
        
        details['total_score'] = score
        return score, details
    
    def run_backtest(self, df: pd.DataFrame, min_quality_score: int = 40) -> Dict:
        """Run the backtest"""
        self.trades = []
        current_capital = self.initial_capital
        position = None
        
        # Track equity curve
        equity_curve = []
        
        for i in range(50, len(df)):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check for new entry
            if position is None:
                distance = df['Distance_Pct'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                # Entry conditions (adapted for daily)
                if distance < -1.5 and rsi < 35:  # Slightly more conservative for daily
                    # Calculate quality score
                    score, details = self.calculate_quality_score(df, i)
                    
                    if score >= min_quality_score:
                        position = {
                            'type': 'LONG',
                            'entry_date': current_date,
                            'entry_price': current_price,
                            'entry_idx': i,
                            'score': score,
                            'details': details
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
                elif df['Distance_Pct'].iloc[i] > 0:  # Above SMA
                    exit_reason = 'Above Mean'
                elif days_held >= self.max_hold_days:
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    # Calculate trade results
                    shares = (current_capital * 0.95) / position['entry_price']  # 95% position sizing
                    pnl_dollars = shares * (current_price - position['entry_price']) - (2 * self.commission)
                    current_capital += pnl_dollars
                    
                    # Create trade record
                    trade = Trade(
                        entry_date=position['entry_date'],
                        exit_date=current_date,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        position_type=position['type'],
                        pnl_pct=pnl_pct,
                        pnl_dollars=pnl_dollars,
                        hold_days=days_held,
                        entry_score=position['score'],
                        exit_reason=exit_reason
                    )
                    self.trades.append(trade)
                    position = None
            
            # Record equity
            equity_curve.append({
                'date': current_date,
                'equity': current_capital,
                'price': current_price
            })
        
        self.equity_curve = pd.DataFrame(equity_curve)
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze backtest results"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame([
            {
                'Entry_Date': t.entry_date,
                'Exit_Date': t.exit_date,
                'PnL_%': t.pnl_pct,
                'PnL_$': t.pnl_dollars,
                'Days_Held': t.hold_days,
                'Entry_Score': t.entry_score,
                'Exit_Reason': t.exit_reason,
                'Win': t.pnl_pct > 0
            }
            for t in self.trades
        ])
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df['Win'].sum()
        win_rate = winning_trades / total_trades * 100
        
        total_return_pct = (self.equity_curve['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        avg_win = trades_df[trades_df['Win']]['PnL_%'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[~trades_df['Win']]['PnL_%'].mean() if winning_trades < total_trades else 0
        
        # Risk metrics
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        
        # Trading frequency
        start_date = trades_df['Entry_Date'].min()
        end_date = trades_df['Exit_Date'].max()
        days_traded = (end_date - start_date).days
        trades_per_month = total_trades / (days_traded / 30) if days_traded > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'final_capital': self.equity_curve['equity'].iloc[-1],
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'trades_per_month': trades_per_month,
            'avg_hold_days': trades_df['Days_Held'].mean(),
            'trades_df': trades_df,
            'start_date': start_date,
            'end_date': end_date,
            'days_traded': days_traded
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        return drawdown.min()
    
    def create_visualizations(self, results: Dict, symbol: str):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{symbol} - VXX Mean Reversion Daily Strategy Backtest', fontsize=20, fontweight='bold')
        
        # 1. Equity Curve
        ax = axes[0, 0]
        equity = self.equity_curve['equity']
        returns = (equity / self.initial_capital - 1) * 100
        
        ax.plot(self.equity_curve['date'], returns, linewidth=3, color='#00ff88', label='Strategy', alpha=0.9)
        
        # Buy and hold comparison
        start_price = self.equity_curve['price'].iloc[0]
        end_price = self.equity_curve['price'].iloc[-1]
        buy_hold_return = (end_price / start_price - 1) * 100
        ax.axhline(y=buy_hold_return, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Buy & Hold ({buy_hold_return:.1f}%)')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        ax.fill_between(self.equity_curve['date'], drawdown, 0, 
                       color='#ff6b6b', alpha=0.7, label='Drawdown')
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Trade PnL Distribution
        ax = axes[0, 2]
        trades_df = results['trades_df']
        wins = trades_df[trades_df['Win']]['PnL_%']
        losses = trades_df[~trades_df['Win']]['PnL_%']
        
        ax.hist(wins, bins=15, alpha=0.8, color='#00ff88', label=f'Wins ({len(wins)})')
        ax.hist(losses, bins=15, alpha=0.8, color='#ff6b6b', label=f'Losses ({len(losses)})')
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.8)
        ax.set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('PnL (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax = axes[1, 0]
        equity_monthly = self.equity_curve.set_index('date')['equity'].resample('M').last()
        monthly_returns = equity_monthly.pct_change().dropna() * 100
        
        if len(monthly_returns) > 12:
            # Create year-month matrix
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            pivot_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).mean()
            pivot_table = pivot_data.unstack(level=1)
            
            im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(12))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index)
            plt.colorbar(im, ax=ax, label='Return (%)')
        else:
            ax.bar(range(len(monthly_returns)), monthly_returns.values, 
                  color=['#00ff88' if x > 0 else '#ff6b6b' for x in monthly_returns])
        
        ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
        
        # 5. Trade Timeline
        ax = axes[1, 1]
        win_trades = trades_df[trades_df['Win']]
        loss_trades = trades_df[~trades_df['Win']]
        
        if len(win_trades) > 0:
            ax.scatter(win_trades['Entry_Date'], win_trades['PnL_%'], 
                      color='#00ff88', alpha=0.7, s=50, label='Wins')
        if len(loss_trades) > 0:
            ax.scatter(loss_trades['Entry_Date'], loss_trades['PnL_%'], 
                      color='#ff6b6b', alpha=0.7, s=50, label='Losses')
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.set_title('Trade Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Entry Date')
        ax.set_ylabel('PnL (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Hold Duration Analysis
        ax = axes[1, 2]
        duration_bins = [0, 1, 2, 3, 4, 5]
        duration_labels = ['1 day', '2 days', '3 days', '4 days', '5+ days']
        
        duration_counts = []
        duration_win_rates = []
        
        for i in range(len(duration_bins)-1):
            mask = (trades_df['Days_Held'] >= duration_bins[i]) & (trades_df['Days_Held'] < duration_bins[i+1])
            subset = trades_df[mask]
            duration_counts.append(len(subset))
            duration_win_rates.append(subset['Win'].mean() * 100 if len(subset) > 0 else 0)
        
        x = np.arange(len(duration_labels))
        bars = ax.bar(x, duration_win_rates, color='#4ecdc4', alpha=0.8)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, duration_counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'n={count}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(duration_labels)
        ax.set_title('Win Rate by Hold Duration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Win Rate (%)')
        ax.grid(True, alpha=0.3)
        
        # 7. Score vs Performance
        ax = axes[2, 0]
        colors = ['#00ff88' if win else '#ff6b6b' for win in trades_df['Win']]
        ax.scatter(trades_df['Entry_Score'], trades_df['PnL_%'], 
                  c=colors, alpha=0.7, s=60)
        ax.set_title('Entry Score vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Entry Quality Score')
        ax.set_ylabel('PnL (%)')
        ax.grid(True, alpha=0.3)
        
        # 8. Cumulative Trades
        ax = axes[2, 1]
        cumulative_trades = np.arange(1, len(trades_df) + 1)
        cumulative_wins = trades_df['Win'].cumsum()
        win_rate_evolution = cumulative_wins / cumulative_trades * 100
        
        ax.plot(cumulative_trades, win_rate_evolution, linewidth=2, color='#ffd93d')
        ax.axhline(y=50, color='white', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax.set_title('Win Rate Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative Win Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Performance Summary
        ax = axes[2, 2]
        ax.axis('off')
        
        # Performance summary
        summary_text = f"""PERFORMANCE SUMMARY

📊 TRADING STATISTICS
Total Trades: {results['total_trades']:,}
Win Rate: {results['win_rate']:.1f}%
Profit Factor: {results['profit_factor']:.2f}

💰 RETURNS
Total Return: {results['total_return_pct']:+.1f}%
Final Capital: ${results['final_capital']:,.0f}
Average Win: {results['avg_win_pct']:+.1f}%
Average Loss: {results['avg_loss_pct']:+.1f}%

📈 RISK METRICS
Sharpe Ratio: {results['sharpe_ratio']:.2f}
Max Drawdown: {results['max_drawdown_pct']:.1f}%
Avg Hold Time: {results['avg_hold_days']:.1f} days

⏱️ FREQUENCY
Trades/Month: {results['trades_per_month']:.1f}
Period: {results['days_traded']} days
From: {results['start_date'].strftime('%Y-%m-%d')}
To: {results['end_date'].strftime('%Y-%m-%d')}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig

def main():
    """Run comprehensive backtest"""
    print("📈 VXX MEAN REVERSION DAILY STRATEGY BACKTEST")
    print("=" * 60)
    
    # Test symbols
    symbols = ['VXX', 'SQQQ', 'AMD', 'VIXY', 'UVXY']
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n🎯 Testing {symbol}...")
        
        try:
            # Initialize backtester
            backtester = VXXMeanReversionDailyBacktester(
                initial_capital=10000,
                profit_target_pct=5.0,
                stop_loss_pct=7.5,
                max_hold_days=5,
                commission=7.0
            )
            
            # Prepare data
            print("   📊 Preparing data...")
            df = backtester.prepare_data(symbol, period="2y")
            
            if df is None:
                continue
                
            # Run backtest
            print("   🔄 Running backtest...")
            results = backtester.run_backtest(df, min_quality_score=50)
            
            if 'error' in results:
                print(f"   ❌ {results['error']}")
                continue
            
            all_results[symbol] = results
            
            # Create visualizations
            print("   📊 Creating visualizations...")
            fig = backtester.create_visualizations(results, symbol)
            
            # Save plot
            filename = f'{symbol}_daily_backtest.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"   ✅ Saved: {filename}")
            
            # Print summary
            print(f"   📋 Results: {results['total_trades']} trades, "
                  f"{results['win_rate']:.1f}% win rate, "
                  f"{results['total_return_pct']:+.1f}% return")
            
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Create comparison summary
    if all_results:
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Symbol':<8} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'Sharpe':<8} {'Max DD%':<8}")
        print("-" * 60)
        
        for symbol, results in all_results.items():
            print(f"{symbol:<8} {results['total_trades']:<8} "
                  f"{results['win_rate']:<8.1f} {results['total_return_pct']:<10.1f} "
                  f"{results['sharpe_ratio']:<8.2f} {results['max_drawdown_pct']:<8.1f}")
    
    print("\n✅ Backtesting complete!")
    print(f"📁 Generated {len(all_results)} detailed backtest reports")

if __name__ == "__main__":
    main()