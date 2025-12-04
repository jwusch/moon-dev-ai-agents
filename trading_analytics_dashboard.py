#!/usr/bin/env python
"""
📈 TRADING PERFORMANCE ANALYTICS DASHBOARD
Comprehensive performance analysis with risk-adjusted metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import argparse
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

from src.data.position_tracker import PositionTracker


class TradingAnalytics:
    """Comprehensive trading performance analytics"""
    
    def __init__(self):
        self.tracker = PositionTracker()
        self.risk_free_rate = 0.05  # 5% annual risk-free rate
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        # Get all positions
        all_positions = self.tracker.get_all_positions()
        open_positions = self.tracker.get_open_positions()
        closed_positions = all_positions[all_positions['status'] == 'CLOSED']
        
        if all_positions.empty:
            return {
                'error': 'No trading data available',
                'total_positions': 0
            }
            
        # Basic metrics
        total_positions = len(all_positions)
        open_count = len(open_positions)
        closed_count = len(closed_positions)
        
        # P&L Analysis
        total_realized_pnl = closed_positions['profit_loss'].sum() if not closed_positions.empty else 0
        total_unrealized_pnl = self._calculate_unrealized_pnl(open_positions)
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        # Win/Loss Statistics
        if not closed_positions.empty:
            winning_trades = closed_positions[closed_positions['profit_loss'] > 0]
            losing_trades = closed_positions[closed_positions['profit_loss'] <= 0]
            
            win_rate = len(winning_trades) / len(closed_positions) if len(closed_positions) > 0 else 0
            avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf')
            
            # Percentage returns
            avg_win_pct = winning_trades['profit_loss_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss_pct = losing_trades['profit_loss_pct'].mean() if len(losing_trades) > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
            avg_win_pct = avg_loss_pct = 0
            
        # Risk Metrics
        if not closed_positions.empty:
            returns = closed_positions['profit_loss_pct'] / 100  # Convert to decimal
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(closed_positions)
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        else:
            sharpe_ratio = max_drawdown = volatility = 0
            
        # Position Sizing Analysis
        position_sizes = all_positions['position_size']
        avg_position_size = position_sizes.mean()
        max_position_size = position_sizes.max()
        min_position_size = position_sizes.min()
        
        # Holding Period Analysis
        if not closed_positions.empty:
            holding_periods = self._calculate_holding_periods(closed_positions)
            avg_holding_period = holding_periods.mean()
            max_holding_period = holding_periods.max()
            min_holding_period = holding_periods.min()
        else:
            avg_holding_period = max_holding_period = min_holding_period = 0
            
        # Symbol Performance
        symbol_performance = self._analyze_symbol_performance(all_positions)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_positions': total_positions,
                'open_positions': open_count,
                'closed_positions': closed_count,
                'total_realized_pnl': total_realized_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_pnl': total_pnl
            },
            'performance_metrics': {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_win_pct': avg_win_pct,
                'avg_loss_pct': avg_loss_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility
            },
            'position_analysis': {
                'avg_position_size': avg_position_size,
                'max_position_size': max_position_size,
                'min_position_size': min_position_size,
                'avg_holding_period_days': avg_holding_period,
                'max_holding_period_days': max_holding_period,
                'min_holding_period_days': min_holding_period
            },
            'symbol_performance': symbol_performance
        }
        
    def _calculate_unrealized_pnl(self, open_positions: pd.DataFrame) -> float:
        """Calculate unrealized P&L for open positions"""
        if open_positions.empty:
            return 0
            
        try:
            import yfinance as yf
            total_unrealized = 0
            
            for _, position in open_positions.iterrows():
                symbol = position['symbol']
                shares = position['shares']
                entry_price = position['entry_price']
                
                # Get current price
                ticker = yf.download(symbol, period='1d', progress=False)
                if not ticker.empty:
                    # Fix multi-level column issue if present
                    if isinstance(ticker.columns, pd.MultiIndex):
                        ticker.columns = ticker.columns.get_level_values(0)
                    current_price = ticker['Close'].iloc[-1]
                    unrealized_pnl = (current_price - entry_price) * shares
                    total_unrealized += unrealized_pnl
                    
            return total_unrealized
        except Exception:
            return 0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
    def _calculate_max_drawdown(self, closed_positions: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if closed_positions.empty:
            return 0
            
        # Sort by exit date
        positions_sorted = closed_positions.sort_values('exit_date')
        
        # Calculate cumulative returns
        cumulative_returns = (1 + positions_sorted['profit_loss_pct'] / 100).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(drawdown.min()) if not drawdown.empty else 0
        
    def _calculate_holding_periods(self, closed_positions: pd.DataFrame) -> pd.Series:
        """Calculate holding periods in days"""
        holding_periods = []
        
        for _, position in closed_positions.iterrows():
            entry_date = pd.to_datetime(position['entry_date'])
            exit_date = pd.to_datetime(position['exit_date'])
            holding_period = (exit_date - entry_date).days
            holding_periods.append(holding_period)
            
        return pd.Series(holding_periods)
        
    def _analyze_symbol_performance(self, all_positions: pd.DataFrame) -> Dict:
        """Analyze performance by symbol"""
        symbol_stats = {}
        
        for symbol in all_positions['symbol'].unique():
            symbol_positions = all_positions[all_positions['symbol'] == symbol]
            closed_symbol_positions = symbol_positions[symbol_positions['status'] == 'CLOSED']
            
            if not closed_symbol_positions.empty:
                total_pnl = closed_symbol_positions['profit_loss'].sum()
                win_rate = len(closed_symbol_positions[closed_symbol_positions['profit_loss'] > 0]) / len(closed_symbol_positions)
                avg_return = closed_symbol_positions['profit_loss_pct'].mean()
                trade_count = len(closed_symbol_positions)
            else:
                total_pnl = win_rate = avg_return = trade_count = 0
                
            symbol_stats[symbol] = {
                'total_trades': len(symbol_positions),
                'closed_trades': len(closed_symbol_positions),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_return_pct': avg_return,
                'trade_count': trade_count
            }
            
        return symbol_stats
        
    def generate_performance_report(self) -> str:
        """Generate detailed performance report"""
        
        performance = self.get_performance_summary()
        
        if 'error' in performance:
            return f"❌ {performance['error']}"
            
        report = []
        report.append(colored("📈 TRADING PERFORMANCE ANALYTICS DASHBOARD", 'cyan', attrs=['bold']))
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary Section
        summary = performance['summary']
        report.append(colored("📊 PORTFOLIO SUMMARY", 'yellow', attrs=['bold']))
        report.append("-" * 40)
        report.append(f"Total Positions: {summary['total_positions']}")
        report.append(f"Open Positions: {summary['open_positions']}")
        report.append(f"Closed Positions: {summary['closed_positions']}")
        report.append("")
        
        # P&L Section
        total_pnl = summary['total_pnl']
        pnl_color = 'green' if total_pnl >= 0 else 'red'
        report.append(colored("💰 P&L ANALYSIS", 'yellow', attrs=['bold']))
        report.append("-" * 40)
        report.append(f"Realized P&L: ${summary['total_realized_pnl']:+,.2f}")
        report.append(f"Unrealized P&L: ${summary['total_unrealized_pnl']:+,.2f}")
        report.append(colored(f"Total P&L: ${total_pnl:+,.2f}", pnl_color, attrs=['bold']))
        report.append("")
        
        # Performance Metrics
        metrics = performance['performance_metrics']
        report.append(colored("🎯 PERFORMANCE METRICS", 'yellow', attrs=['bold']))
        report.append("-" * 40)
        
        if summary['closed_positions'] > 0:
            report.append(f"Win Rate: {metrics['win_rate']:.1%}")
            report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            report.append(f"Expectancy: ${metrics['expectancy']:+,.2f}")
            report.append(f"Average Win: ${metrics['avg_win']:+,.2f} ({metrics['avg_win_pct']:+.1f}%)")
            report.append(f"Average Loss: ${metrics['avg_loss']:+,.2f} ({metrics['avg_loss_pct']:+.1f}%)")
            report.append("")
            
            # Risk Metrics
            report.append(colored("⚠️ RISK METRICS", 'yellow', attrs=['bold']))
            report.append("-" * 40)
            report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report.append(f"Maximum Drawdown: {metrics['max_drawdown']:.1%}")
            report.append(f"Volatility (Annual): {metrics['volatility']:.1%}")
            report.append("")
        else:
            report.append("No closed positions available for performance metrics")
            report.append("")
            
        # Position Analysis
        pos_analysis = performance['position_analysis']
        report.append(colored("📋 POSITION ANALYSIS", 'yellow', attrs=['bold']))
        report.append("-" * 40)
        report.append(f"Average Position Size: ${pos_analysis['avg_position_size']:,.2f}")
        report.append(f"Largest Position: ${pos_analysis['max_position_size']:,.2f}")
        report.append(f"Smallest Position: ${pos_analysis['min_position_size']:,.2f}")
        
        if summary['closed_positions'] > 0:
            report.append(f"Average Holding Period: {pos_analysis['avg_holding_period_days']:.1f} days")
            report.append(f"Longest Hold: {pos_analysis['max_holding_period_days']:.0f} days")
            report.append(f"Shortest Hold: {pos_analysis['min_holding_period_days']:.0f} days")
        report.append("")
        
        # Symbol Performance
        symbol_perf = performance['symbol_performance']
        if symbol_perf:
            report.append(colored("🏆 TOP PERFORMING SYMBOLS", 'yellow', attrs=['bold']))
            report.append("-" * 40)
            
            # Sort symbols by total P&L
            sorted_symbols = sorted(symbol_perf.items(), 
                                  key=lambda x: x[1]['total_pnl'], reverse=True)
            
            for symbol, stats in sorted_symbols[:10]:  # Top 10
                pnl = stats['total_pnl']
                win_rate = stats['win_rate']
                trades = stats['closed_trades']
                
                if trades > 0:
                    pnl_color = 'green' if pnl >= 0 else 'red'
                    report.append(colored(
                        f"  {symbol}: ${pnl:+,.2f} | {win_rate:.1%} win rate | {trades} trades",
                        pnl_color
                    ))
                else:
                    report.append(f"  {symbol}: No closed trades")
            report.append("")
            
        # Performance Grading
        report.append(colored("📊 PERFORMANCE GRADE", 'yellow', attrs=['bold']))
        report.append("-" * 40)
        
        if summary['closed_positions'] > 0:
            grade, grade_color = self._calculate_performance_grade(metrics)
            report.append(colored(f"Overall Grade: {grade}", grade_color, attrs=['bold']))
            report.append(self._get_grade_explanation(grade))
        else:
            report.append("Insufficient data for grading")
            
        report.append("")
        report.append("=" * 80)
        
        return '\n'.join(report)
        
    def _calculate_performance_grade(self, metrics: Dict) -> Tuple[str, str]:
        """Calculate overall performance grade"""
        score = 0
        
        # Win rate scoring (0-30 points)
        win_rate = metrics['win_rate']
        if win_rate >= 0.6:
            score += 30
        elif win_rate >= 0.5:
            score += 20
        elif win_rate >= 0.4:
            score += 10
            
        # Profit factor scoring (0-25 points)
        profit_factor = metrics['profit_factor']
        if profit_factor >= 2.0:
            score += 25
        elif profit_factor >= 1.5:
            score += 20
        elif profit_factor >= 1.2:
            score += 15
        elif profit_factor >= 1.0:
            score += 10
            
        # Sharpe ratio scoring (0-25 points)
        sharpe = metrics['sharpe_ratio']
        if sharpe >= 2.0:
            score += 25
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 15
        elif sharpe >= 0.5:
            score += 10
            
        # Max drawdown scoring (0-20 points)
        max_dd = metrics['max_drawdown']
        if max_dd <= 0.05:  # 5%
            score += 20
        elif max_dd <= 0.10:  # 10%
            score += 15
        elif max_dd <= 0.20:  # 20%
            score += 10
        elif max_dd <= 0.30:  # 30%
            score += 5
            
        # Assign grade
        if score >= 85:
            return "A+", "green"
        elif score >= 80:
            return "A", "green"
        elif score >= 75:
            return "A-", "green"
        elif score >= 70:
            return "B+", "yellow"
        elif score >= 65:
            return "B", "yellow"
        elif score >= 60:
            return "B-", "yellow"
        elif score >= 55:
            return "C+", "yellow"
        elif score >= 50:
            return "C", "yellow"
        elif score >= 45:
            return "C-", "yellow"
        elif score >= 40:
            return "D", "red"
        else:
            return "F", "red"
            
    def _get_grade_explanation(self, grade: str) -> str:
        """Get explanation for performance grade"""
        explanations = {
            "A+": "🌟 Exceptional performance! Professional-level results.",
            "A": "🎯 Excellent performance! Very strong risk-adjusted returns.",
            "A-": "👍 Strong performance with good risk management.",
            "B+": "✅ Good performance, room for optimization.",
            "B": "📈 Above average performance.",
            "B-": "⚖️ Decent performance, focus on risk management.",
            "C+": "🔧 Average performance, needs improvement.",
            "C": "📊 Mediocre performance, review strategy.",
            "C-": "⚠️ Below average, significant improvements needed.",
            "D": "🚨 Poor performance, major strategy overhaul required.",
            "F": "💥 Failing performance, stop trading and reassess."
        }
        return explanations.get(grade, "Unknown grade")
        
    def save_analytics_report(self, filename: str = None):
        """Save analytics to file"""
        if filename is None:
            filename = f'trading_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
        performance = self.get_performance_summary()
        
        with open(filename, 'w') as f:
            json.dump(performance, f, indent=2, default=str)
            
        print(f"📊 Analytics saved to {filename}")
        
    def create_performance_charts(self):
        """Create performance visualization charts"""
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Get data
            all_positions = self.tracker.get_all_positions()
            closed_positions = all_positions[all_positions['status'] == 'CLOSED']
            
            if closed_positions.empty:
                print("❌ No closed positions available for charting")
                return
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Trading Performance Analytics', fontsize=16, fontweight='bold')
            
            # 1. P&L Distribution
            axes[0, 0].hist(closed_positions['profit_loss'], bins=20, alpha=0.7, 
                          color='skyblue', edgecolor='black')
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('P&L Distribution')
            axes[0, 0].set_xlabel('Profit/Loss ($)')
            axes[0, 0].set_ylabel('Frequency')
            
            # 2. Cumulative Returns
            closed_positions_sorted = closed_positions.sort_values('exit_date')
            cumulative_pnl = closed_positions_sorted['profit_loss'].cumsum()
            axes[0, 1].plot(cumulative_pnl, color='green', linewidth=2)
            axes[0, 1].set_title('Cumulative P&L')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Cumulative P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Win/Loss by Symbol
            symbol_pnl = closed_positions.groupby('symbol')['profit_loss'].sum().sort_values(ascending=False)
            if len(symbol_pnl) > 0:
                colors = ['green' if x >= 0 else 'red' for x in symbol_pnl.values]
                axes[1, 0].bar(symbol_pnl.index, symbol_pnl.values, color=colors, alpha=0.7)
                axes[1, 0].set_title('P&L by Symbol')
                axes[1, 0].set_xlabel('Symbol')
                axes[1, 0].set_ylabel('Total P&L ($)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Holding Period Distribution
            holding_periods = self._calculate_holding_periods(closed_positions)
            axes[1, 1].hist(holding_periods, bins=15, alpha=0.7, 
                          color='orange', edgecolor='black')
            axes[1, 1].set_title('Holding Period Distribution')
            axes[1, 1].set_xlabel('Days Held')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f'trading_performance_charts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Performance charts saved to {chart_filename}")
            
        except Exception as e:
            print(colored(f"❌ Error creating charts: {str(e)}", 'red'))


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Trading Analytics Dashboard')
    parser.add_argument('--report', '-r', action='store_true', 
                       help='Generate performance report')
    parser.add_argument('--charts', '-c', action='store_true',
                       help='Create performance charts') 
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save analytics to file')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all analytics (report + charts + save)')
    
    args = parser.parse_args()
    
    # Create analytics instance
    analytics = TradingAnalytics()
    
    if args.all or (not any([args.report, args.charts, args.save])):
        # Default: run everything
        print(analytics.generate_performance_report())
        analytics.save_analytics_report()
        analytics.create_performance_charts()
    else:
        if args.report:
            print(analytics.generate_performance_report())
        if args.save:
            analytics.save_analytics_report()
        if args.charts:
            analytics.create_performance_charts()


if __name__ == "__main__":
    main()