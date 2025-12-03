"""
📈 VXX Mean Reversion 15 Strategy Backtester
Comprehensive backtesting with detailed visualizations and analytics

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
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str  # 'LONG' or 'SHORT'
    pnl_pct: float
    pnl_dollars: float
    hold_duration: float  # hours
    entry_score: int
    exit_reason: str

class VXXMeanReversionBacktester:
    """
    Advanced backtester for VXX Mean Reversion 15 strategy
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 max_hold_hours: int = 3,
                 commission: float = 1.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_hours = max_hold_hours
        self.commission = commission
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = []
        
    def prepare_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Prepare data with all indicators"""
        try:
            from yfinance_cache_demo import YFinanceCache
            cache = YFinanceCache()
            df = cache.get_data(symbol, period=period, interval="15m")
        except:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="15m")
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Core indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Additional quality indicators
        df['ROC_1'] = talib.ROC(df['Close'].values, 1)
        df['ROC_3'] = talib.ROC(df['Close'].values, 3)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle * 100
        
        return df.dropna()
    
    def calculate_quality_score(self, df: pd.DataFrame, idx: int) -> Tuple[int, Dict]:
        """Calculate entry quality score"""
        score = 0
        details = {}
        
        # 1. Distance from SMA (0-20 points)
        distance = df['Distance%'].iloc[idx]
        if distance < -2.0:
            score += 20
            details['distance_score'] = 20
        elif distance < -1.5:
            score += 15
            details['distance_score'] = 15
        elif distance < -1.0:
            score += 10
            details['distance_score'] = 10
        else:
            details['distance_score'] = 0
            
        # 2. RSI level (0-20 points)
        rsi = df['RSI'].iloc[idx]
        if rsi < 25:
            score += 20
            details['rsi_score'] = 20
        elif rsi < 30:
            score += 15
            details['rsi_score'] = 15
        elif rsi < 40:
            score += 10
            details['rsi_score'] = 10
        else:
            details['rsi_score'] = 0
            
        # 3. Volume confirmation (0-15 points)
        if 'Volume_Ratio' in df.columns:
            volume_ratio = df['Volume_Ratio'].iloc[idx]
            if not pd.isna(volume_ratio):
                if volume_ratio > 1.5:
                    score += 15
                    details['volume_score'] = 15
                elif volume_ratio > 1.2:
                    score += 10
                    details['volume_score'] = 10
                elif volume_ratio > 1.0:
                    score += 5
                    details['volume_score'] = 5
        
        # 4. Momentum (0-15 points)
        if 'ROC_1' in df.columns and 'ROC_3' in df.columns:
            roc_1 = df['ROC_1'].iloc[idx]
            roc_3 = df['ROC_3'].iloc[idx]
            if not pd.isna(roc_1) and not pd.isna(roc_3):
                if roc_1 > roc_3 and roc_1 > -0.5:
                    score += 15
                    details['momentum_score'] = 15
                elif roc_1 > -1.0:
                    score += 8
                    details['momentum_score'] = 8
        
        # 5. Volatility environment (0-10 points)
        if 'BB_Width' in df.columns:
            bb_width = df['BB_Width'].iloc[idx]
            if not pd.isna(bb_width):
                if bb_width > 5:  # High volatility
                    score += 10
                    details['volatility_score'] = 10
                elif bb_width > 3:
                    score += 5
                    details['volatility_score'] = 5
        
        details['total_score'] = score
        return score, details
    
    def run_backtest(self, df: pd.DataFrame, min_quality_score: int = 40) -> Dict:
        """Run the backtest"""
        self.trades = []
        current_capital = self.initial_capital
        position = None
        entry_idx = 0
        
        # Track equity curve
        equity_curve = []
        
        for i in range(100, len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Skip non-market hours (simplified)
            if current_time.hour < 9 or current_time.hour >= 16:
                continue
            
            # Check for new entry
            if position is None:
                distance = df['Distance%'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                # Only LONG entries for simplicity
                if distance < -1.0 and rsi < 40:
                    # Calculate quality score
                    score, details = self.calculate_quality_score(df, i)
                    
                    if score >= min_quality_score:
                        position = {
                            'type': 'LONG',
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'entry_idx': i,
                            'score': score,
                            'details': details
                        }
                        entry_idx = i
            
            # Check for exit
            elif position is not None:
                hours_held = (current_time - position['entry_time']).total_seconds() / 3600
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Exit conditions
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance%'].iloc[i] > -0.2:
                    exit_reason = 'Return to Mean'
                elif hours_held >= self.max_hold_hours:
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    # Calculate trade results
                    pnl_dollars = current_capital * (pnl_pct / 100) - (2 * self.commission)
                    current_capital += pnl_dollars
                    
                    # Create trade record
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        position_type=position['type'],
                        pnl_pct=pnl_pct,
                        pnl_dollars=pnl_dollars,
                        hold_duration=hours_held,
                        entry_score=position['score'],
                        exit_reason=exit_reason
                    )
                    self.trades.append(trade)
                    position = None
            
            # Record equity
            equity_curve.append({
                'timestamp': current_time,
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
                'Entry_Time': t.entry_time,
                'Exit_Time': t.exit_time,
                'PnL_%': t.pnl_pct,
                'PnL_$': t.pnl_dollars,
                'Duration_Hours': t.hold_duration,
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
        returns = trades_df['PnL_%'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        
        # Trading frequency
        start_date = trades_df['Entry_Time'].min()
        end_date = trades_df['Exit_Time'].max()
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
            'avg_hold_hours': trades_df['Duration_Hours'].mean(),
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
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Equity Curve
        ax1 = plt.subplot(4, 2, 1)
        equity = self.equity_curve['equity']
        returns = (equity / self.initial_capital - 1) * 100
        
        plt.plot(self.equity_curve['timestamp'], returns, linewidth=2, color='#00ff88', label='Strategy')
        
        # Buy and hold comparison
        start_price = self.equity_curve['price'].iloc[0]
        end_price = self.equity_curve['price'].iloc[-1]
        buy_hold_return = (end_price / start_price - 1) * 100
        plt.axhline(y=buy_hold_return, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Buy & Hold ({buy_hold_return:.1f}%)')
        
        plt.title(f'{symbol} - Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(4, 2, 2)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        plt.fill_between(self.equity_curve['timestamp'], drawdown, 0, 
                        color='#ff6b6b', alpha=0.7, label='Drawdown')
        plt.title('Drawdown', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Trade PnL Distribution
        ax3 = plt.subplot(4, 2, 3)
        trades_df = results['trades_df']
        wins = trades_df[trades_df['Win']]['PnL_%']
        losses = trades_df[~trades_df['Win']]['PnL_%']
        
        plt.hist(wins, bins=20, alpha=0.7, color='#00ff88', label=f'Wins ({len(wins)})')
        plt.hist(losses, bins=20, alpha=0.7, color='#ff6b6b', label=f'Losses ({len(losses)})')
        plt.axvline(x=0, color='white', linestyle='--', alpha=0.8)
        plt.title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('PnL (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Win Rate by Entry Score
        ax4 = plt.subplot(4, 2, 4)
        score_bins = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        win_rates = []
        trade_counts = []
        labels = []
        
        for min_score, max_score in score_bins:
            mask = (trades_df['Entry_Score'] >= min_score) & (trades_df['Entry_Score'] < max_score)
            subset = trades_df[mask]
            if len(subset) > 0:
                win_rate = subset['Win'].mean() * 100
                win_rates.append(win_rate)
                trade_counts.append(len(subset))
                labels.append(f'{min_score}-{max_score}')
        
        if win_rates:
            bars = plt.bar(labels, win_rates, color='#4ecdc4', alpha=0.8)
            
            # Add trade count labels
            for i, (bar, count) in enumerate(zip(bars, trade_counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'n={count}', ha='center', va='bottom', fontsize=10)
            
            plt.title('Win Rate by Entry Quality Score', fontsize=14, fontweight='bold')
            plt.xlabel('Entry Score Range')
            plt.ylabel('Win Rate (%)')
            plt.grid(True, alpha=0.3)
        
        # 5. Cumulative PnL Over Time
        ax5 = plt.subplot(4, 2, 5)
        trades_df_sorted = trades_df.sort_values('Exit_Time')
        trades_df_sorted['Cumulative_PnL'] = trades_df_sorted['PnL_$'].cumsum()
        
        plt.plot(trades_df_sorted['Exit_Time'], trades_df_sorted['Cumulative_PnL'], 
                linewidth=2, color='#ffd93d', marker='o', markersize=4)
        plt.title('Cumulative PnL Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL ($)')
        plt.grid(True, alpha=0.3)
        
        # 6. Hold Duration vs PnL
        ax6 = plt.subplot(4, 2, 6)
        colors = ['#00ff88' if win else '#ff6b6b' for win in trades_df['Win']]
        plt.scatter(trades_df['Duration_Hours'], trades_df['PnL_%'], 
                   c=colors, alpha=0.7, s=50)
        plt.title('Hold Duration vs PnL', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (Hours)')
        plt.ylabel('PnL (%)')
        plt.grid(True, alpha=0.3)
        
        # 7. Exit Reason Analysis
        ax7 = plt.subplot(4, 2, 7)
        exit_reasons = trades_df['Exit_Reason'].value_counts()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(exit_reasons)))
        
        wedges, texts, autotexts = plt.pie(exit_reasons.values, labels=exit_reasons.index, 
                                          autopct='%1.1f%%', colors=colors_pie)
        plt.title('Exit Reason Distribution', fontsize=14, fontweight='bold')
        
        # 8. Performance Summary
        ax8 = plt.subplot(4, 2, 8)
        ax8.axis('off')
        
        # Create performance summary text
        summary_text = f"""
PERFORMANCE SUMMARY

Total Trades: {results['total_trades']:,}
Win Rate: {results['win_rate']:.1f}%
Total Return: {results['total_return_pct']:+.1f}%
Final Capital: ${results['final_capital']:,.0f}

Average Win: {results['avg_win_pct']:+.1f}%
Average Loss: {results['avg_loss_pct']:+.1f}%
Profit Factor: {results['profit_factor']:.2f}
Sharpe Ratio: {results['sharpe_ratio']:.2f}

Max Drawdown: {results['max_drawdown_pct']:.1f}%
Avg Hold Time: {results['avg_hold_hours']:.1f} hours
Trades/Month: {results['trades_per_month']:.1f}

Period: {results['start_date'].strftime('%Y-%m-%d')}
to {results['end_date'].strftime('%Y-%m-%d')}
({results['days_traded']} days)
        """
        
        plt.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle(f'{symbol} - VXX Mean Reversion 15 Backtest Results', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        return fig

def main():
    """Run comprehensive backtest"""
    print("📈 VXX MEAN REVERSION 15 STRATEGY BACKTEST")
    print("=" * 60)
    
    # Test symbols
    symbols = ['VXX', 'SQQQ', 'AMD', 'VIXY']
    
    for symbol in symbols:
        print(f"\n🎯 Testing {symbol}...")
        
        try:
            # Initialize backtester
            backtester = VXXMeanReversionBacktester(
                initial_capital=10000,
                profit_target_pct=5.0,  # Using optimized 5% target
                stop_loss_pct=7.5,
                max_hold_hours=3,
                commission=1.0
            )
            
            # Prepare data
            print("   📊 Preparing data...")
            df = backtester.prepare_data(symbol, period="6mo")
            
            # Run backtest
            print("   🔄 Running backtest...")
            results = backtester.run_backtest(df, min_quality_score=50)
            
            if 'error' in results:
                print(f"   ❌ {results['error']}")
                continue
            
            # Create visualizations
            print("   📊 Creating visualizations...")
            fig = backtester.create_visualizations(results, symbol)
            
            # Save plot
            filename = f'{symbol}_backtest_results.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"   ✅ Saved: {filename}")
            
            # Print summary
            print(f"   📋 Results: {results['total_trades']} trades, "
                  f"{results['win_rate']:.1f}% win rate, "
                  f"{results['total_return_pct']:+.1f}% return")
            
            plt.show()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Backtesting complete!")

if __name__ == "__main__":
    main()