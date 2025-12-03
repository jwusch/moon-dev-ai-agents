"""
🔥💎 AEGS BACKTEST ORCHESTRATOR AGENT 💎🔥
Efficiently processes discovered candidates through AEGS backtesting

Features:
- Parallel processing for speed
- Automatic result collection
- Error handling and retry logic
- Progress tracking
- Auto-registration of successful symbols
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict
import concurrent.futures
from termcolor import colored
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from comprehensive_qqq_backtest_cached import CachedComprehensiveBacktester as ComprehensiveBacktester
from aegs_auto_registry import register_backtest_result
from src.agents.backtest_history import BacktestHistory
from src.agents.invalid_symbol_tracker import InvalidSymbolTracker

class AEGSBacktestAgent:
    """
    Orchestrates parallel backtesting of AEGS candidates
    """
    
    def __init__(self, max_workers=5):
        self.name = "AEGS Backtest Orchestrator"
        self.max_workers = max_workers
        self.results = []
        self.failed_symbols = []
        self.history = BacktestHistory()
        self.invalid_tracker = InvalidSymbolTracker()
        
    def run(self, discovery_file: str = None):
        """Process discovered candidates through backtesting"""
        
        print(colored(f"🧪 {self.name} Starting...", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        # Load candidates
        candidates = self._load_candidates(discovery_file)
        
        if not candidates:
            print("❌ No candidates to process")
            return
        
        print(f"📊 Processing {len(candidates)} candidates with {self.max_workers} workers...")
        
        # Process in parallel
        start_time = time.time()
        self._process_candidates_parallel(candidates)
        elapsed = time.time() - start_time
        
        # Summary
        self._print_summary(elapsed)
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _load_candidates(self, discovery_file: str = None) -> List[Dict]:
        """Load candidates from discovery file"""
        
        if discovery_file:
            filepath = discovery_file
        else:
            # Find most recent discovery file
            import glob
            discovery_files = glob.glob("aegs_discoveries_*.json")
            if not discovery_files:
                print("❌ No discovery files found")
                return []
            filepath = sorted(discovery_files)[-1]
        
        print(f"📂 Loading candidates from: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data['candidates']
    
    def _process_candidates_parallel(self, candidates: List[Dict]):
        """Process multiple candidates in parallel"""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_candidate = {}
            
            for candidate in candidates:
                future = executor.submit(self._backtest_symbol, candidate)
                future_to_candidate[future] = candidate
            
            # Process completed tasks
            completed = 0
            total = len(candidates)
            
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                        
                        # Show progress
                        excess = result['excess_return']
                        if excess > 1000:
                            status = colored("💎 GOLDMINE!", 'red', attrs=['bold'])
                        elif excess > 100:
                            status = colored("🚀 HIGH POTENTIAL", 'yellow')
                        elif excess > 0:
                            status = colored("✅ Positive", 'green')
                        else:
                            status = "❌ Negative"
                        
                        print(f"[{completed}/{total}] {candidate['symbol']}: {status} ({excess:+.0f}% excess)")
                    else:
                        self.failed_symbols.append(candidate['symbol'])
                        # Track as invalid symbol
                        self.invalid_tracker.add_invalid_symbol(candidate['symbol'], "Backtest failed - insufficient data", "backtest_failed")
                        # Record in history to prevent retesting
                        failed_result = {
                            'symbol': candidate['symbol'],
                            'discovery_reason': candidate.get('reason', 'Unknown'),
                            'excess_return': 0,
                            'status': 'parallel_processing_failed',
                            'win_rate': 0,
                            'total_trades': 0,
                            'error': 'Backtest failed - insufficient data'
                        }
                        self.history.record_test(candidate['symbol'], failed_result)
                        print(f"[{completed}/{total}] {candidate['symbol']}: ❌ Failed")
                        
                except Exception as e:
                    self.failed_symbols.append(candidate['symbol'])
                    # Track as invalid symbol
                    error_msg = str(e)
                    if "possibly delisted" in error_msg or "No data found" in error_msg:
                        self.invalid_tracker.add_invalid_symbol(candidate['symbol'], "No data found - possibly delisted", "no_data")
                    else:
                        self.invalid_tracker.add_invalid_symbol(candidate['symbol'], f"Backtest error: {error_msg[:100]}", "backtest_error")
                    # Record in history to prevent retesting
                    exception_result = {
                        'symbol': candidate['symbol'],
                        'discovery_reason': candidate.get('reason', 'Unknown'),
                        'excess_return': 0,
                        'status': 'parallel_processing_exception',
                        'win_rate': 0,
                        'total_trades': 0,
                        'error': error_msg[:100]
                    }
                    self.history.record_test(candidate['symbol'], exception_result)
                    print(f"[{completed}/{total}] {candidate['symbol']}: ❌ Error - {str(e)[:50]}")
    
    def _backtest_symbol(self, candidate: Dict) -> Dict:
        """Backtest a single symbol"""
        
        symbol = candidate['symbol']
        reason = candidate.get('reason', 'Unknown')
        
        try:
            # Initialize backtester
            backtester = ComprehensiveBacktester(symbol)
            
            # Download data
            df = backtester.download_maximum_data()
            
            if df is None or len(df) == 0:
                # Data download failed - likely delisted
                self.invalid_tracker.add_invalid_symbol(symbol, "No data available - possibly delisted", "no_data")
                # Record failed test in history to prevent retesting
                failed_result = {
                    'symbol': symbol,
                    'discovery_reason': reason,
                    'excess_return': 0,
                    'status': 'failed_no_data',
                    'win_rate': 0,
                    'total_trades': 0
                }
                self.history.record_test(symbol, failed_result)
                return None
            elif len(df) < 250:  # Lowered from 500 to ~1 year of trading days
                # Only mark as invalid if it's truly insufficient
                # Check if it's a legitimate symbol with recent data
                if len(df) > 0:
                    days_of_data = (df.index[-1] - df.index[0]).days
                    if days_of_data < 365:  # Less than a year of data
                        # Don't mark as invalid - just skip for now
                        print(f"   ⚠️  {symbol}: Only {len(df)} data points ({days_of_data} days) - skipping")
                        # Record insufficient data test to prevent immediate retry
                        insufficient_result = {
                            'symbol': symbol,
                            'discovery_reason': reason,
                            'excess_return': 0,
                            'status': 'insufficient_data_recent',
                            'win_rate': 0,
                            'total_trades': 0,
                            'data_points': len(df),
                            'days_of_data': days_of_data
                        }
                        self.history.record_test(symbol, insufficient_result)
                        return None
                    else:
                        # Has been around for a while but sparse data
                        self.invalid_tracker.add_invalid_symbol(symbol, f"Insufficient data - only {len(df)} data points over {days_of_data} days", "insufficient_data")
                        # Record sparse data test
                        sparse_result = {
                            'symbol': symbol,
                            'discovery_reason': reason,
                            'excess_return': 0,
                            'status': 'insufficient_data_sparse',
                            'win_rate': 0,
                            'total_trades': 0,
                            'data_points': len(df),
                            'days_of_data': days_of_data
                        }
                        self.history.record_test(symbol, sparse_result)
                        return None
                else:
                    self.invalid_tracker.add_invalid_symbol(symbol, "No data available", "no_data")
                    # Record no data test
                    no_data_result = {
                        'symbol': symbol,
                        'discovery_reason': reason,
                        'excess_return': 0,
                        'status': 'no_data',
                        'win_rate': 0,
                        'total_trades': 0
                    }
                    self.history.record_test(symbol, no_data_result)
                    return None
            
            # Run backtest
            results = backtester.comprehensive_backtest(df)
            
            # Prepare result
            result = {
                'symbol': symbol,
                'discovery_reason': reason,
                'excess_return': results.excess_return_pct,
                'strategy_return': results.strategy_total_return_pct,
                'buy_hold_return': results.buy_hold_total_return_pct,
                'win_rate': results.win_rate,
                'total_trades': results.total_trades,
                'sharpe': results.strategy_sharpe,
                'years': results.total_years,
                'data_points': len(df),
                'status': 'success'
            }
            
            # Record in backtest history (SUCCESSFUL test)
            self.history.record_test(symbol, result)
            
            # Auto-register if successful
            if results.excess_return_pct > 10:
                # Guess category from reason
                category = self._guess_category(reason)
                
                # Register in goldmine registry
                register_backtest_result(symbol, results, category)
            
            return result
            
        except ValueError as e:
            if "No profitable alpha sources" in str(e):
                # This is not an invalid symbol - just not suitable for AEGS
                print(f"   ℹ️  {symbol}: Not suitable for AEGS strategy (no profitable alphas)")
                # Record as tested but not suitable for AEGS to prevent retesting
                unsuitable_result = {
                    'symbol': symbol,
                    'discovery_reason': reason,
                    'excess_return': 0,
                    'status': 'unsuitable_for_aegs',
                    'win_rate': 0,
                    'total_trades': 0,
                    'error': 'No profitable alpha sources'
                }
                self.history.record_test(symbol, unsuitable_result)
                return None
            else:
                # Other ValueError - might be data issue
                self.invalid_tracker.add_invalid_symbol(symbol, f"Backtest error: {str(e)}", "backtest_error")
                # Record ValueError in history
                error_result = {
                    'symbol': symbol,
                    'discovery_reason': reason,
                    'excess_return': 0,
                    'status': 'value_error',
                    'win_rate': 0,
                    'total_trades': 0,
                    'error': str(e)[:100]
                }
                self.history.record_test(symbol, error_result)
                return None
        except Exception as e:
            # Generic error - track it
            error_msg = str(e)
            if "possibly delisted" in error_msg or "No data found" in error_msg:
                self.invalid_tracker.add_invalid_symbol(symbol, "No data found - possibly delisted", "no_data")
            else:
                self.invalid_tracker.add_invalid_symbol(symbol, f"Backtest error: {error_msg[:100]}", "backtest_error")
            
            # Record generic error in history
            generic_error_result = {
                'symbol': symbol,
                'discovery_reason': reason,
                'excess_return': 0,
                'status': 'exception_error',
                'win_rate': 0,
                'total_trades': 0,
                'error': error_msg[:100]
            }
            self.history.record_test(symbol, generic_error_result)
            return None
    
    def _guess_category(self, reason: str) -> str:
        """Guess category from discovery reason"""
        
        reason_lower = reason.lower()
        
        if 'crypto' in reason_lower or 'bitcoin' in reason_lower:
            return 'Cryptocurrency'
        elif 'volatility' in reason_lower:
            return 'High Volatility'
        elif 'volume' in reason_lower:
            return 'Volume Anomaly'
        elif 'recovery' in reason_lower:
            return 'Recovery Play'
        elif 'sector' in reason_lower:
            return 'Sector Rotation'
        elif 'ai' in reason_lower:
            return 'AI Discovery'
        else:
            return 'Unknown'
    
    def _print_summary(self, elapsed_time: float):
        """Print processing summary"""
        
        print("\n" + "=" * 80)
        print(colored("📊 BACKTEST SUMMARY", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if self.results:
            # Sort by excess return
            sorted_results = sorted(self.results, key=lambda x: x['excess_return'], reverse=True)
            
            # Count by category
            goldmines = [r for r in sorted_results if r['excess_return'] > 1000]
            high_potential = [r for r in sorted_results if 100 < r['excess_return'] <= 1000]
            positive = [r for r in sorted_results if 0 < r['excess_return'] <= 100]
            negative = [r for r in sorted_results if r['excess_return'] <= 0]
            
            print(f"\nResults Distribution:")
            print(f"   💎 Goldmines (>1000%): {len(goldmines)}")
            print(f"   🚀 High Potential (100-1000%): {len(high_potential)}")
            print(f"   ✅ Positive (<100%): {len(positive)}")
            print(f"   ❌ Negative: {len(negative)}")
            print(f"   🔧 Failed: {len(self.failed_symbols)}")
            
            # Show top performers
            if goldmines:
                print(colored("\n💎 NEW GOLDMINES DISCOVERED:", 'red', attrs=['bold']))
                for r in goldmines:
                    print(f"   {r['symbol']}: {r['excess_return']:+,.0f}% excess | {r['win_rate']:.1f}% win rate")
                    print(f"      Discovery: {r['discovery_reason']}")
            
            if high_potential:
                print(colored("\n🚀 HIGH POTENTIAL FOUND:", 'yellow'))
                for r in high_potential[:5]:  # Top 5
                    print(f"   {r['symbol']}: {r['excess_return']:+.0f}% excess | {r['win_rate']:.1f}% win rate")
            
            # Performance stats
            avg_excess = sum(r['excess_return'] for r in sorted_results) / len(sorted_results)
            success_rate = len([r for r in sorted_results if r['excess_return'] > 0]) / len(sorted_results) * 100
            
            print(f"\nPerformance Metrics:")
            print(f"   Average Excess Return: {avg_excess:+.1f}%")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Processing Time: {elapsed_time:.1f} seconds")
            print(f"   Symbols/second: {len(self.results) / elapsed_time:.2f}")
        
        else:
            print("❌ No successful backtests completed")
    
    def _save_results(self):
        """Save backtest results"""
        
        if not self.results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_data = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent': self.name,
            'total_processed': len(self.results) + len(self.failed_symbols),
            'successful': len(self.results),
            'failed': len(self.failed_symbols),
            'results': sorted(self.results, key=lambda x: x['excess_return'], reverse=True),
            'failed_symbols': self.failed_symbols
        }
        
        filename = f'aegs_backtest_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Save summary for quick reference
        summary_data = []
        for r in sorted(self.results, key=lambda x: x['excess_return'], reverse=True):
            summary_data.append({
                'symbol': r['symbol'],
                'excess_return': round(r['excess_return'], 1),
                'win_rate': round(r['win_rate'], 1),
                'category': self._guess_category(r['discovery_reason'])
            })
        
        summary_file = f'aegs_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_file}")


def main():
    """Run the backtest agent"""
    
    print(colored("🧪 AEGS BACKTEST ORCHESTRATOR", 'cyan', attrs=['bold']))
    
    # Check for command line arguments
    import sys
    discovery_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Initialize agent with 5 parallel workers
    agent = AEGSBacktestAgent(max_workers=5)
    
    # Process candidates
    results = agent.run(discovery_file)
    
    if results:
        goldmines = [r for r in results if r['excess_return'] > 1000]
        if goldmines:
            print(colored(f"\n🎉 Discovered {len(goldmines)} new goldmines!", 'red', attrs=['bold']))
            print("🚀 These have been automatically added to the registry!")


if __name__ == "__main__":
    main()