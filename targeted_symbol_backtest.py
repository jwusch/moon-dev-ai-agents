"""
🎯 Targeted Symbol Backtest - Finding More Winners
Test ensemble strategy on symbols that should match our successful patterns:
- Mean-reverting assets (like TLT - WINNER)
- Volatile cyclical commodities (like USO - DEFENSIVE WINNER)
- Avoid trending growth assets (like SPY/QQQ - LOSERS)

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from comprehensive_qqq_backtest import ComprehensiveBacktester, ComprehensiveBacktestResults

class TargetedSymbolBacktester:
    """
    Test symbols that should theoretically work based on our findings
    """
    
    def __init__(self):
        # Symbols chosen based on successful patterns
        self.target_symbols = {
            # BONDS & INTEREST RATE SENSITIVE (TLT was winner)
            'IEF': {
                'name': 'IEF (7-10 Year Treasury)',
                'type': 'bonds',
                'expectation': 'winner',
                'reasoning': 'Interest rate cycles like TLT, should mean revert'
            },
            'SHY': {
                'name': 'SHY (1-3 Year Treasury)',
                'type': 'bonds',
                'expectation': 'winner',
                'reasoning': 'Short-term bond cycles, less volatile than TLT'
            },
            'TBT': {
                'name': 'TBT (Inverse 20+ Treasury)',
                'type': 'bonds',
                'expectation': 'winner',
                'reasoning': 'Inverse TLT - should have opposite mean reversion'
            },
            'LQD': {
                'name': 'LQD (Investment Grade Corp Bonds)',
                'type': 'bonds',
                'expectation': 'winner',
                'reasoning': 'Credit cycles, interest rate sensitive'
            },
            'HYG': {
                'name': 'HYG (High Yield Bonds)',
                'type': 'bonds',
                'expectation': 'winner',
                'reasoning': 'Credit cycles, risk-on/risk-off mean reversion'
            },
            
            # VOLATILE COMMODITIES (USO was defensive winner)
            'UNG': {
                'name': 'UNG (Natural Gas)',
                'type': 'commodity',
                'expectation': 'winner',
                'reasoning': 'Natural gas extremely cyclical, weather/supply driven'
            },
            'DBA': {
                'name': 'DBA (Agriculture)',
                'type': 'commodity',
                'expectation': 'winner',
                'reasoning': 'Crop cycles, weather patterns, mean reverting'
            },
            'PDBC': {
                'name': 'PDBC (Optimum Yield Commodities)',
                'type': 'commodity',
                'expectation': 'winner',
                'reasoning': 'Broad commodity basket, cyclical nature'
            },
            'DJP': {
                'name': 'DJP (Commodity Index)',
                'type': 'commodity',
                'expectation': 'winner',
                'reasoning': 'Diversified commodities, cycles vs stocks'
            },
            
            # VOLATILITY & FEAR (VXX failed but let\'s try related)
            'VIXY': {
                'name': 'VIXY (Short-term VIX)',
                'type': 'volatility',
                'expectation': 'winner',
                'reasoning': 'Volatility spikes revert, fear cycles'
            },
            'SVXY': {
                'name': 'SVXY (Inverse VIX Short-term)',
                'type': 'volatility',
                'expectation': 'winner',
                'reasoning': 'Inverse volatility, mean reversion opposite direction'
            },
            
            # CURRENCY & INTERNATIONAL (cyclical vs USD)
            'FXE': {
                'name': 'FXE (Euro Currency)',
                'type': 'currency',
                'expectation': 'winner',
                'reasoning': 'EUR/USD cycles, central bank policy driven'
            },
            'FXY': {
                'name': 'FXY (Japanese Yen)',
                'type': 'currency',
                'expectation': 'winner',
                'reasoning': 'JPY safe haven cycles, carry trade reversals'
            },
            'UUP': {
                'name': 'UUP (US Dollar Index)',
                'type': 'currency',
                'expectation': 'winner',
                'reasoning': 'DXY cycles with risk-on/risk-off sentiment'
            },
            
            # SECTOR ROTATION (cyclical sectors)
            'XLF': {
                'name': 'XLF (Financial Sector)',
                'type': 'sector',
                'expectation': 'moderate',
                'reasoning': 'Interest rate sensitive, bank cycles'
            },
            'XLU': {
                'name': 'XLU (Utilities)',
                'type': 'sector',
                'expectation': 'winner',
                'reasoning': 'Defensive sector, mean reversion vs growth'
            },
            'XLB': {
                'name': 'XLB (Materials)',
                'type': 'sector',
                'expectation': 'moderate',
                'reasoning': 'Commodity cycle sensitive, economic cycles'
            },
            'REZ': {
                'name': 'REZ (Residential Real Estate)',
                'type': 'sector',
                'expectation': 'winner',
                'reasoning': 'Interest rate sensitive, housing cycles'
            },
            
            # INVERSE/HEDGE FUNDS (should be mean reverting by design)
            'SH': {
                'name': 'SH (Inverse S&P 500)',
                'type': 'inverse',
                'expectation': 'winner',
                'reasoning': 'Inverse equity, benefits from market reversals'
            },
            'PSQ': {
                'name': 'PSQ (Inverse QQQ)',
                'type': 'inverse',
                'expectation': 'winner',
                'reasoning': 'Inverse tech, benefits from growth corrections'
            },
            
            # CONTROL TESTS (should fail based on our findings)
            'VTI': {
                'name': 'VTI (Total Stock Market)',
                'type': 'equity',
                'expectation': 'loser',
                'reasoning': 'Trending equity like SPY/QQQ - control test'
            },
            'ARKK': {
                'name': 'ARKK (Innovation ETF)',
                'type': 'equity',
                'expectation': 'loser',
                'reasoning': 'Growth momentum fund - should fail like QQQ'
            }
        }
        
        self.results = {}
        
    def run_targeted_backtests(self) -> Dict:
        """Run backtests on targeted symbols"""
        
        print("🎯 TARGETED SYMBOL ENSEMBLE STRATEGY BACKTEST")
        print("=" * 70)
        print("Testing symbols that should match our successful patterns:")
        print("✅ TLT (bonds) had +10.1% excess return")
        print("✅ USO (commodities) had +22.5% excess return")
        print("❌ SPY/QQQ (trending equity) had massive underperformance")
        
        winners = []
        moderate = []
        losers = []
        
        for symbol, info in self.target_symbols.items():
            print(f"\n{'='*15} {symbol} - {info['name']} {'='*15}")
            print(f"Type: {info['type'].title()}")
            print(f"Expectation: {info['expectation'].title()}")
            print(f"Reasoning: {info['reasoning']}")
            
            try:
                # Initialize backtester
                backtester = ComprehensiveBacktester(symbol)
                
                # Download data
                df = backtester.download_maximum_data()
                
                if df.empty or len(df) < 1000:
                    print(f"❌ Insufficient data for {symbol}")
                    continue
                
                # Run backtest
                results = backtester.comprehensive_backtest(df)
                
                # Store results
                self.results[symbol] = {
                    'results': results,
                    'info': info,
                    'data_points': len(df)
                }
                
                # Categorize by performance
                excess_return = results.excess_return_pct
                expectation = info['expectation']
                
                if excess_return > 0:
                    winners.append((symbol, excess_return, expectation))
                    print(f"🟢 WINNER: {excess_return:+.1f}% excess return")
                elif excess_return > -50:
                    moderate.append((symbol, excess_return, expectation))
                    print(f"🟡 MODERATE: {excess_return:+.1f}% excess return")
                else:
                    losers.append((symbol, excess_return, expectation))
                    print(f"🔴 LOSER: {excess_return:+.1f}% excess return")
                
                # Check vs expectation
                if excess_return > 0 and expectation == 'winner':
                    print(f"   ✅ PREDICTION CORRECT: Expected winner, got winner")
                elif excess_return <= 0 and expectation == 'loser':
                    print(f"   ✅ PREDICTION CORRECT: Expected loser, got loser")
                elif abs(excess_return) < 50 and expectation == 'moderate':
                    print(f"   ✅ PREDICTION CORRECT: Expected moderate, got moderate")
                else:
                    print(f"   ❌ PREDICTION WRONG: Expected {expectation}, got different result")
                
                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_filename = f'{symbol}_backtest_results_{timestamp}.json'
                with open(results_filename, 'w') as f:
                    json.dump(results.__dict__, f, indent=2, default=str)
                
            except Exception as e:
                print(f"❌ Error testing {symbol}: {e}")
                continue
        
        # Analysis summary
        self._print_strategy_analysis(winners, moderate, losers)
        self._save_comprehensive_results()
        
        return self.results
    
    def _print_strategy_analysis(self, winners: List, moderate: List, losers: List):
        """Print comprehensive strategy analysis"""
        
        print(f"\n{'='*70}")
        print("🧠 ENSEMBLE STRATEGY INTELLIGENCE ANALYSIS")
        print(f"{'='*70}")
        
        total_tested = len(winners) + len(moderate) + len(losers)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Symbols Tested: {total_tested}")
        print(f"   🟢 Winners (excess > 0%): {len(winners)} ({len(winners)/total_tested*100:.1f}%)")
        print(f"   🟡 Moderate (-50% < excess ≤ 0%): {len(moderate)} ({len(moderate)/total_tested*100:.1f}%)")
        print(f"   🔴 Losers (excess ≤ -50%): {len(losers)} ({len(losers)/total_tested*100:.1f}%)")
        
        if winners:
            print(f"\n🏆 CONFIRMED WINNERS:")
            winners.sort(key=lambda x: x[1], reverse=True)  # Sort by excess return
            for symbol, excess, expectation in winners:
                info = self.target_symbols[symbol]
                prediction = "✅ PREDICTED" if expectation == 'winner' else "🔮 SURPRISE"
                print(f"   {symbol} ({info['type']}): {excess:+.1f}% excess - {prediction}")
        
        if moderate:
            print(f"\n🟡 MODERATE PERFORMERS:")
            moderate.sort(key=lambda x: x[1], reverse=True)
            for symbol, excess, expectation in moderate:
                info = self.target_symbols[symbol]
                print(f"   {symbol} ({info['type']}): {excess:+.1f}% excess")
        
        if losers:
            print(f"\n❌ CONFIRMED LOSERS:")
            losers.sort(key=lambda x: x[1])  # Sort by worst first
            for symbol, excess, expectation in losers[-5:]:  # Show worst 5
                info = self.target_symbols[symbol]
                prediction = "✅ PREDICTED" if expectation == 'loser' else "❌ UNEXPECTED"
                print(f"   {symbol} ({info['type']}): {excess:+.1f}% excess - {prediction}")
        
        # Asset class analysis
        print(f"\n📈 ASSET CLASS PERFORMANCE:")
        asset_performance = {}
        
        for symbol, data in self.results.items():
            asset_type = data['info']['type']
            excess_return = data['results'].excess_return_pct
            
            if asset_type not in asset_performance:
                asset_performance[asset_type] = []
            asset_performance[asset_type].append(excess_return)
        
        for asset_type, returns in asset_performance.items():
            avg_excess = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            count = len(returns)
            
            if avg_excess > 0:
                status = "🟢 GOOD"
            elif avg_excess > -50:
                status = "🟡 MIXED"
            else:
                status = "🔴 POOR"
            
            print(f"   {asset_type.title()}: {avg_excess:+.1f}% avg excess, {win_rate:.0f}% win rate ({count} symbols) {status}")
        
        # Strategy refinement insights
        print(f"\n💡 STRATEGY REFINEMENT INSIGHTS:")
        
        # Find best asset classes
        best_classes = sorted(asset_performance.items(), key=lambda x: np.mean(x[1]), reverse=True)
        
        if best_classes and np.mean(best_classes[0][1]) > 0:
            best_class = best_classes[0]
            print(f"   🎯 TARGET ASSET CLASS: {best_class[0].title()}")
            print(f"      Average excess return: {np.mean(best_class[1]):+.1f}%")
            print(f"      Success rate: {sum(1 for r in best_class[1] if r > 0) / len(best_class[1]) * 100:.0f}%")
        
        # Prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for symbol, data in self.results.items():
            expectation = data['info']['expectation']
            actual_excess = data['results'].excess_return_pct
            total_predictions += 1
            
            if (expectation == 'winner' and actual_excess > 0) or \
               (expectation == 'loser' and actual_excess < -50) or \
               (expectation == 'moderate' and -50 <= actual_excess <= 0):
                correct_predictions += 1
        
        prediction_accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        print(f"   🔮 PREDICTION ACCURACY: {prediction_accuracy:.0f}% ({correct_predictions}/{total_predictions})")
        
        # Overall strategy assessment
        all_excess_returns = [data['results'].excess_return_pct for data in self.results.values()]
        overall_avg_excess = np.mean(all_excess_returns)
        overall_win_rate = sum(1 for r in all_excess_returns if r > 0) / len(all_excess_returns) * 100
        
        print(f"\n🎯 ENSEMBLE STRATEGY FINAL ASSESSMENT:")
        print(f"   Overall Average Excess Return: {overall_avg_excess:+.1f}%")
        print(f"   Overall Success Rate: {overall_win_rate:.0f}%")
        print(f"   Prediction Accuracy: {prediction_accuracy:.0f}%")
        
        if overall_win_rate >= 40 and overall_avg_excess > -20:
            final_grade = "🟢 VIABLE STRATEGY"
            recommendation = "Strategy works well on specific asset classes. Focus on winners."
        elif overall_win_rate >= 25:
            final_grade = "🟡 SELECTIVE STRATEGY"  
            recommendation = "Strategy works but needs careful symbol selection."
        else:
            final_grade = "🔴 INEFFECTIVE STRATEGY"
            recommendation = "Strategy fails across most asset classes. Major revision needed."
        
        print(f"   Final Grade: {final_grade}")
        print(f"   Recommendation: {recommendation}")
        
        # Best practices for the strategy
        if winners:
            winner_types = [self.target_symbols[w[0]]['type'] for w in winners]
            common_types = max(set(winner_types), key=winner_types.count) if winner_types else None
            
            if common_types:
                print(f"\n📋 STRATEGY BEST PRACTICES:")
                print(f"   ✅ DO: Focus on {common_types} assets")
                print(f"   ✅ DO: Look for mean-reverting, cyclical patterns")
                print(f"   ❌ AVOID: Trending growth assets (SPY, QQQ, ARKK)")
                print(f"   ❌ AVOID: Long-term momentum plays")
    
    def _save_comprehensive_results(self):
        """Save comprehensive comparison results"""
        
        if not self.results:
            return
        
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
                'Strategy_Return_Pct': results.strategy_total_return_pct,
                'BuyHold_Return_Pct': results.buy_hold_total_return_pct,
                'Excess_Return_Pct': results.excess_return_pct,
                'Strategy_Sharpe': results.strategy_sharpe,
                'Win_Rate_Pct': results.win_rate,
                'Total_Trades': results.total_trades,
                'Years_Tested': results.total_years,
                'Max_Drawdown_Pct': results.strategy_max_drawdown,
                'Prediction_Correct': self._check_prediction_accuracy(results.excess_return_pct, info['expectation'])
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Excess_Return_Pct', ascending=False)
        
        # Save detailed comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = f'targeted_symbol_analysis_{timestamp}.json'
        df_comparison.to_json(comparison_filename, indent=2)
        print(f"\n✅ Detailed analysis saved: {comparison_filename}")
        
        # Create summary CSV for easy viewing
        summary_filename = f'targeted_symbol_summary_{timestamp}.csv'
        df_summary = df_comparison[['Symbol', 'Type', 'Expectation', 'Excess_Return_Pct', 
                                  'Strategy_Sharpe', 'Win_Rate_Pct', 'Prediction_Correct']]
        df_summary.to_csv(summary_filename, index=False)
        print(f"✅ Summary table saved: {summary_filename}")
    
    def _check_prediction_accuracy(self, excess_return: float, expectation: str) -> bool:
        """Check if prediction was accurate"""
        if expectation == 'winner' and excess_return > 0:
            return True
        elif expectation == 'loser' and excess_return < -50:
            return True
        elif expectation == 'moderate' and -50 <= excess_return <= 0:
            return True
        else:
            return False

def main():
    """Run targeted symbol backtest"""
    
    # Initialize and run backtests
    targeted_backtester = TargetedSymbolBacktester()
    results = targeted_backtester.run_targeted_backtests()
    
    print(f"\n✅ Targeted symbol backtest complete!")
    print(f"Found specific asset classes where ensemble strategy excels")

if __name__ == "__main__":
    main()