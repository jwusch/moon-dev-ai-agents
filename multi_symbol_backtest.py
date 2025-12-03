"""
📈 Multi-Symbol Ensemble Strategy Backtest
Test ensemble strategy on symbols better suited for mean reversion and volatility strategies

Target symbols:
- VXX (volatility ETF - mean reversion friendly)
- GLD (gold - sideways/cyclical) 
- XLE (energy sector - cyclical)
- TLT (bonds - mean reverting)
- IWM (small caps - more volatile)
- SPY (broad market - comparison)

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import concurrent.futures
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from comprehensive_qqq_backtest import ComprehensiveBacktester, ComprehensiveBacktestResults

plt.style.use('dark_background')
sns.set_palette("husl")

class MultiSymbolBacktester:
    """
    Test ensemble strategy across multiple symbols to find where it works best
    """
    
    def __init__(self):
        # Symbols chosen for different characteristics
        self.test_symbols = {
            'VXX': {
                'name': 'VXX (Volatility ETF)',
                'type': 'volatility',
                'expectation': 'mean_reversion',
                'reasoning': 'Volatility spikes revert, perfect for our strategies'
            },
            'GLD': {
                'name': 'GLD (Gold ETF)', 
                'type': 'commodity',
                'expectation': 'cyclical',
                'reasoning': 'Gold cycles between fear/greed, sideways trending'
            },
            'XLE': {
                'name': 'XLE (Energy Sector)',
                'type': 'sector',
                'expectation': 'cyclical',
                'reasoning': 'Energy is highly cyclical, oil price driven'
            },
            'TLT': {
                'name': 'TLT (20+ Year Treasury)',
                'type': 'bonds',
                'expectation': 'mean_reversion',
                'reasoning': 'Interest rate cycles create mean reversion opportunities'
            },
            'IWM': {
                'name': 'IWM (Russell 2000 Small Caps)',
                'type': 'equity',
                'expectation': 'moderate',
                'reasoning': 'More volatile than large caps, some mean reversion'
            },
            'SPY': {
                'name': 'SPY (S&P 500)',
                'type': 'equity',
                'expectation': 'poor',
                'reasoning': 'Strong uptrend, buy & hold should win (control test)'
            },
            'USO': {
                'name': 'USO (Oil ETF)',
                'type': 'commodity',
                'expectation': 'cyclical',
                'reasoning': 'Oil price cycles, supply/demand shocks'
            },
            'SLV': {
                'name': 'SLV (Silver ETF)',
                'type': 'commodity', 
                'expectation': 'volatile_cyclical',
                'reasoning': 'Silver more volatile than gold, industrial/monetary dual nature'
            }
        }
        
        self.results = {}
        
    def run_all_backtests(self) -> Dict[str, ComprehensiveBacktestResults]:
        """Run backtests on all symbols"""
        
        print("🎯 MULTI-SYMBOL ENSEMBLE STRATEGY BACKTEST")
        print("=" * 70)
        print("Testing ensemble strategy on symbols better suited for mean reversion")
        
        # Test each symbol
        for symbol, info in self.test_symbols.items():
            print(f"\n{'='*20} {symbol} - {info['name']} {'='*20}")
            print(f"Type: {info['type'].title()}")
            print(f"Expectation: {info['expectation'].replace('_', ' ').title()}")
            print(f"Reasoning: {info['reasoning']}")
            
            try:
                # Initialize backtester for this symbol
                backtester = ComprehensiveBacktester(symbol)
                
                # Download data
                df = backtester.download_maximum_data()
                
                if df.empty or len(df) < 1000:
                    print(f"❌ Insufficient data for {symbol}")
                    continue
                
                # Run backtest
                results = backtester.comprehensive_backtest(df)
                self.results[symbol] = {
                    'results': results,
                    'info': info,
                    'data_points': len(df)
                }
                
                # Print quick summary
                self._print_quick_summary(symbol, results, info)
                
                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save detailed results
                results_filename = f'{symbol}_backtest_results_{timestamp}.json'
                with open(results_filename, 'w') as f:
                    json.dump(results.__dict__, f, indent=2, default=str)
                print(f"✅ Results saved: {results_filename}")
                
            except Exception as e:
                print(f"❌ Error testing {symbol}: {e}")
                continue
        
        # Create comprehensive comparison
        self._create_comparison_analysis()
        
        return self.results
    
    def _print_quick_summary(self, symbol: str, results: ComprehensiveBacktestResults, info: Dict):
        """Print quick summary of results"""
        
        strategy_return = results.strategy_total_return_pct
        buy_hold_return = results.buy_hold_total_return_pct
        excess_return = results.excess_return_pct
        win_rate = results.win_rate
        sharpe = results.strategy_sharpe
        
        print(f"\n📊 QUICK RESULTS:")
        print(f"   Strategy Return: {strategy_return:+.1f}%")
        print(f"   Buy & Hold Return: {buy_hold_return:+.1f}%") 
        print(f"   Excess Return: {excess_return:+.1f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Years Tested: {results.total_years:.1f}")
        
        # Assessment
        if excess_return > 0 and sharpe > 0.5:
            assessment = "🟢 SUCCESS - Strategy beats buy & hold!"
        elif excess_return > 0:
            assessment = "🟡 MODERATE - Beats buy & hold but low Sharpe"
        elif strategy_return > 0 and sharpe > 0:
            assessment = "🟠 MIXED - Positive returns but underperforms"
        else:
            assessment = "🔴 POOR - Strategy underperforms"
        
        print(f"   Assessment: {assessment}")
        
        # Check against expectation
        expected = info['expectation']
        if excess_return > 0 and expected in ['mean_reversion', 'cyclical', 'volatile_cyclical']:
            print(f"   ✅ MATCHES EXPECTATION: {expected} strategy worked")
        elif excess_return <= 0 and expected == 'poor':
            print(f"   ✅ MATCHES EXPECTATION: {expected} performance as expected")
        else:
            print(f"   ❌ UNEXPECTED: Expected {expected}, got different results")
    
    def _create_comparison_analysis(self):
        """Create comprehensive comparison analysis"""
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        print(f"\n{'='*70}")
        print("🏆 COMPREHENSIVE COMPARISON ANALYSIS")
        print(f"{'='*70}")
        
        # Prepare comparison data
        comparison_data = []
        for symbol, data in self.results.items():
            results = data['results']
            info = data['info']
            
            comparison_data.append({
                'Symbol': symbol,
                'Name': info['name'],
                'Type': info['type'],
                'Expectation': info['expectation'],
                'Strategy Return %': results.strategy_total_return_pct,
                'Buy Hold Return %': results.buy_hold_total_return_pct,
                'Excess Return %': results.excess_return_pct,
                'Strategy Sharpe': results.strategy_sharpe,
                'Win Rate %': results.win_rate,
                'Total Trades': results.total_trades,
                'Years': results.total_years,
                'Max Drawdown %': results.strategy_max_drawdown,
                'Volatility %': results.strategy_volatility
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by excess return (best performance first)
        df_comparison = df_comparison.sort_values('Excess Return %', ascending=False)
        
        # Print formatted table
        print(f"\n📊 PERFORMANCE RANKING (by Excess Return):")
        print("-" * 120)
        
        print(f"{'Rank':<4} {'Symbol':<6} {'Type':<12} {'Strategy':<8} {'Buy&Hold':<8} {'Excess':<8} {'Sharpe':<7} {'Win%':<5} {'Trades':<7}")
        print("-" * 120)
        
        for i, row in df_comparison.iterrows():
            rank = df_comparison.index.get_loc(i) + 1
            symbol = row['Symbol']
            type_name = row['Type'][:11]
            strategy_ret = row['Strategy Return %']
            bh_ret = row['Buy Hold Return %']
            excess = row['Excess Return %']
            sharpe = row['Strategy Sharpe']
            win_rate = row['Win Rate %']
            trades = row['Total Trades']
            
            print(f"{rank:<4} {symbol:<6} {type_name:<12} {strategy_ret:>+7.1f}% {bh_ret:>+7.1f}% {excess:>+7.1f}% {sharpe:>6.2f} {win_rate:>4.1f}% {trades:>6}")
        
        # Winners and losers
        winners = df_comparison[df_comparison['Excess Return %'] > 0]
        losers = df_comparison[df_comparison['Excess Return %'] <= 0]
        
        print(f"\n🏆 WINNERS (Strategy beats Buy & Hold): {len(winners)}/{len(df_comparison)}")
        if not winners.empty:
            for _, row in winners.iterrows():
                excess = row['Excess Return %']
                sharpe = row['Strategy Sharpe']
                symbol = row['Symbol']
                expectation = row['Expectation']
                print(f"   {symbol}: +{excess:.1f}% excess, {sharpe:.2f} Sharpe (expected: {expectation})")
        
        print(f"\n❌ LOSERS (Strategy underperforms): {len(losers)}/{len(df_comparison)}")
        if not losers.empty:
            for _, row in losers.iterrows():
                excess = row['Excess Return %']
                symbol = row['Symbol']
                expectation = row['Expectation']
                print(f"   {symbol}: {excess:.1f}% excess (expected: {expectation})")
        
        # Category analysis
        print(f"\n📈 ANALYSIS BY ASSET TYPE:")
        category_analysis = df_comparison.groupby('Type').agg({
            'Excess Return %': ['mean', 'count'],
            'Strategy Sharpe': 'mean',
            'Win Rate %': 'mean'
        }).round(2)
        
        for asset_type in category_analysis.index:
            avg_excess = category_analysis.loc[asset_type, ('Excess Return %', 'mean')]
            count = int(category_analysis.loc[asset_type, ('Excess Return %', 'count')])
            avg_sharpe = category_analysis.loc[asset_type, ('Strategy Sharpe', 'mean')]
            avg_win_rate = category_analysis.loc[asset_type, ('Win Rate %', 'mean')]
            
            print(f"   {asset_type.title()}: {avg_excess:+.1f}% avg excess ({count} symbols), {avg_sharpe:.2f} Sharpe, {avg_win_rate:.1f}% win rate")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        best_performer = df_comparison.iloc[0]
        worst_performer = df_comparison.iloc[-1]
        
        print(f"   🥇 Best Performer: {best_performer['Symbol']} ({best_performer['Type']}) with {best_performer['Excess Return %']:+.1f}% excess return")
        print(f"   🥉 Worst Performer: {worst_performer['Symbol']} ({worst_performer['Type']}) with {worst_performer['Excess Return %']:+.1f}% excess return")
        
        # Strategy effectiveness by expectation
        expectation_performance = df_comparison.groupby('Expectation')['Excess Return %'].mean()
        print(f"\n📊 STRATEGY EFFECTIVENESS BY EXPECTATION:")
        for expectation, avg_performance in expectation_performance.sort_values(ascending=False).items():
            print(f"   {expectation.replace('_', ' ').title()}: {avg_performance:+.1f}% average excess return")
        
        # Overall assessment
        positive_excess = len(winners)
        total_tested = len(df_comparison)
        success_rate = positive_excess / total_tested * 100
        
        overall_avg_excess = df_comparison['Excess Return %'].mean()
        avg_sharpe = df_comparison['Strategy Sharpe'].mean()
        
        print(f"\n🎯 OVERALL ENSEMBLE STRATEGY ASSESSMENT:")
        print(f"   Success Rate: {success_rate:.1f}% ({positive_excess}/{total_tested} symbols beat buy & hold)")
        print(f"   Average Excess Return: {overall_avg_excess:+.1f}%")
        print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        if success_rate >= 50 and overall_avg_excess > 0:
            final_assessment = "🟢 STRATEGY WORKS - Suitable for certain asset classes"
        elif success_rate >= 30 or overall_avg_excess > -5:
            final_assessment = "🟡 MIXED RESULTS - Works for some symbols, needs refinement"
        else:
            final_assessment = "🔴 STRATEGY FAILS - Consistently underperforms across asset classes"
        
        print(f"   Final Assessment: {final_assessment}")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = f'multi_symbol_comparison_{timestamp}.json'
        df_comparison.to_json(comparison_filename, indent=2)
        print(f"\n✅ Comparison saved: {comparison_filename}")
        
        # Create visualization
        self._create_comparison_visualization(df_comparison, timestamp)
    
    def _create_comparison_visualization(self, df: pd.DataFrame, timestamp: str):
        """Create comparison visualization"""
        
        print(f"📊 Creating comparison visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Multi-Symbol Ensemble Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Excess Return by Symbol
        ax = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in df['Excess Return %']]
        bars = ax.bar(df['Symbol'], df['Excess Return %'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax.set_title('Excess Return by Symbol', fontweight='bold')
        ax.set_ylabel('Excess Return (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['Excess Return %']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 2. Sharpe Ratio Comparison
        ax = axes[0, 1]
        ax.scatter(df['Strategy Sharpe'], df['Excess Return %'], 
                  s=100, c=df.index, cmap='viridis', alpha=0.7)
        
        for i, row in df.iterrows():
            ax.annotate(row['Symbol'], (row['Strategy Sharpe'], row['Excess Return %']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        ax.set_title('Risk-Adjusted Performance', fontweight='bold')
        ax.set_xlabel('Strategy Sharpe Ratio')
        ax.set_ylabel('Excess Return (%)')
        ax.grid(True, alpha=0.3)
        
        # 3. Performance by Asset Type
        ax = axes[0, 2]
        type_performance = df.groupby('Type')['Excess Return %'].mean().sort_values()
        colors = ['green' if x > 0 else 'red' for x in type_performance.values]
        bars = ax.barh(type_performance.index, type_performance.values, color=colors, alpha=0.7)
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
        ax.set_title('Average Excess Return by Asset Type', fontweight='bold')
        ax.set_xlabel('Average Excess Return (%)')
        
        # 4. Win Rate vs Excess Return
        ax = axes[1, 0]
        scatter = ax.scatter(df['Win Rate %'], df['Excess Return %'], 
                           s=df['Total Trades']/2, c=df['Strategy Sharpe'], 
                           cmap='RdYlGn', alpha=0.7)
        
        for i, row in df.iterrows():
            ax.annotate(row['Symbol'], (row['Win Rate %'], row['Excess Return %']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.set_title('Win Rate vs Performance (size=trades, color=sharpe)', fontweight='bold')
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel('Excess Return (%)')
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # 5. Strategy vs Buy & Hold Returns
        ax = axes[1, 1]
        ax.scatter(df['Buy Hold Return %'], df['Strategy Return %'], s=100, alpha=0.7)
        
        for i, row in df.iterrows():
            ax.annotate(row['Symbol'], (row['Buy Hold Return %'], row['Strategy Return %']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add diagonal line (equal performance)
        min_val = min(df['Buy Hold Return %'].min(), df['Strategy Return %'].min())
        max_val = max(df['Buy Hold Return %'].max(), df['Strategy Return %'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'white', linestyle='--', alpha=0.5, label='Equal Performance')
        
        ax.set_title('Strategy vs Buy & Hold Total Returns', fontweight='bold')
        ax.set_xlabel('Buy & Hold Return (%)')
        ax.set_ylabel('Strategy Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        winners = df[df['Excess Return %'] > 0]
        losers = df[df['Excess Return %'] <= 0]
        
        summary_text = f"""MULTI-SYMBOL BACKTEST SUMMARY

📊 PERFORMANCE OVERVIEW:
• Symbols Tested: {len(df)}
• Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)
• Losers: {len(losers)} ({len(losers)/len(df)*100:.1f}%)

📈 AGGREGATE METRICS:
• Avg Excess Return: {df['Excess Return %'].mean():+.1f}%
• Avg Sharpe Ratio: {df['Strategy Sharpe'].mean():.2f}
• Avg Win Rate: {df['Win Rate %'].mean():.1f}%
• Total Trades (all): {df['Total Trades'].sum():,}

🏆 BEST PERFORMERS:
"""
        
        for i, (_, row) in enumerate(winners.head(3).iterrows(), 1):
            summary_text += f"{i}. {row['Symbol']}: {row['Excess Return %']:+.1f}%\n"
        
        summary_text += f"\n❌ WORST PERFORMERS:\n"
        for i, (_, row) in enumerate(losers.tail(3).iterrows(), 1):
            summary_text += f"{i}. {row['Symbol']}: {row['Excess Return %']:+.1f}%\n"
        
        summary_text += f"\n💡 ASSET TYPE WINNERS:\n"
        type_performance = df.groupby('Type')['Excess Return %'].mean().sort_values(ascending=False)
        for asset_type, avg_return in type_performance.head(3).items():
            summary_text += f"• {asset_type.title()}: {avg_return:+.1f}%\n"
        
        # Overall conclusion
        success_rate = len(winners) / len(df) * 100
        if success_rate >= 50:
            conclusion = "✅ STRATEGY VIABLE"
        elif success_rate >= 30:
            conclusion = "⚠️ MIXED RESULTS"
        else:
            conclusion = "❌ STRATEGY INEFFECTIVE"
        
        summary_text += f"\n🎯 CONCLUSION: {conclusion}"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f'multi_symbol_comparison_{timestamp}.png'
        fig.savefig(chart_filename, dpi=300, bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        print(f"✅ Chart saved: {chart_filename}")
        
        plt.close()

def main():
    """Run multi-symbol backtest"""
    
    # Initialize and run backtests
    multi_backtester = MultiSymbolBacktester()
    results = multi_backtester.run_all_backtests()
    
    print(f"\n✅ Multi-symbol backtest complete!")
    print(f"Tested {len(results)} symbols to find where the ensemble strategy works best")

if __name__ == "__main__":
    main()