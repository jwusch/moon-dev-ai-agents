#!/usr/bin/env python3
"""
Multi-timeframe AEGS testing
Tests symbols across different resolutions to find optimal trading timeframes
"""

import subprocess
import json
import time
from datetime import datetime
from termcolor import colored
import pandas as pd

class MultiTimeframeAEGSTester:
    def __init__(self):
        self.timeframe_configs = {
            '5m': {'days': 30, 'description': '5-minute bars (30 days)'},
            '15m': {'days': 60, 'description': '15-minute bars (60 days)'},
            '1h': {'days': 180, 'description': 'Hourly bars (180 days)'},
            '1d': {'days': 730, 'description': 'Daily bars (2 years)'}
        }
        
    def test_symbol_timeframe(self, symbol, timeframe):
        """Test a symbol with specific timeframe using AEGS"""
        print(colored(f"\n⏱️ Testing {symbol} on {timeframe} timeframe...", 'yellow'))
        
        try:
            # Modify the AEGS backtest to use specific timeframe
            # We'll use environment variable to pass timeframe
            import os
            env = os.environ.copy()
            env['AEGS_TIMEFRAME'] = timeframe
            env['AEGS_DAYS'] = str(self.timeframe_configs[timeframe]['days'])
            
            # Run AEGS backtest
            cmd = ['python', 'run_aegs_backtest.py', symbol]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=180,
                env=env
            )
            
            # Parse results
            output = result.stdout
            
            if "Excess Return:" in output:
                # Extract metrics
                metrics = self.parse_aegs_output(output)
                metrics['timeframe'] = timeframe
                
                # Display summary
                print(colored(f"   ✅ Success!", 'green'))
                print(f"   Excess Return: {metrics.get('excess_return', 'N/A')}%")
                print(f"   Win Rate: {metrics.get('win_rate', 'N/A')}%")
                print(f"   Trades: {metrics.get('num_trades', 'N/A')}")
                
                return metrics
            else:
                print(colored(f"   ❌ No valid results", 'red'))
                return None
                
        except subprocess.TimeoutExpired:
            print(colored(f"   ⏱️ Timeout", 'yellow'))
            return None
        except Exception as e:
            print(colored(f"   ❌ Error: {str(e)[:100]}", 'red'))
            return None
    
    def parse_aegs_output(self, output):
        """Parse AEGS output for key metrics"""
        metrics = {}
        
        # Extract excess return
        if "Excess Return:" in output:
            try:
                line = [l for l in output.split('\n') if 'Excess Return:' in l][0]
                metrics['excess_return'] = float(line.split('+')[1].split('%')[0])
            except:
                pass
        
        # Extract strategy return
        if "Strategy Return:" in output:
            try:
                line = [l for l in output.split('\n') if 'Strategy Return:' in l][0]
                value = line.split(':')[1].strip().replace('%', '')
                metrics['strategy_return'] = float(value)
            except:
                pass
        
        # Extract win rate
        if "Win Rate:" in output:
            try:
                line = [l for l in output.split('\n') if 'Win Rate:' in l][0]
                metrics['win_rate'] = float(line.split(':')[1].split('%')[0])
            except:
                pass
        
        # Extract number of trades
        if "Total Trades:" in output:
            try:
                line = [l for l in output.split('\n') if 'Total Trades:' in l][0]
                metrics['num_trades'] = int(line.split(':')[1].strip())
            except:
                pass
        
        return metrics
    
    def compare_symbol(self, symbol, timeframes=None):
        """Test a symbol across multiple timeframes"""
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '1d']
        
        print(colored(f"\n{'='*60}", 'cyan'))
        print(colored(f"📊 MULTI-TIMEFRAME ANALYSIS: {symbol}", 'cyan', attrs=['bold']))
        print(colored(f"{'='*60}", 'cyan'))
        
        results = {}
        
        for tf in timeframes:
            metrics = self.test_symbol_timeframe(symbol, tf)
            if metrics:
                results[tf] = metrics
            time.sleep(2)  # Rate limiting
        
        # Display comparison
        if results:
            self.display_comparison(symbol, results)
            
            # Find optimal timeframe
            optimal = self.find_optimal_timeframe(results)
            print(colored(f"\n⭐ OPTIMAL TIMEFRAME: {optimal['timeframe']}", 'green', attrs=['bold']))
            print(f"   Reasoning: Best combination of excess return ({optimal['metrics']['excess_return']:.1f}%) ")
            print(f"             and win rate ({optimal['metrics']['win_rate']:.1f}%)")
            
            return results
        else:
            print(colored("❌ No valid results for any timeframe", 'red'))
            return None
    
    def display_comparison(self, symbol, results):
        """Display timeframe comparison table"""
        print(colored(f"\n📊 {symbol} TIMEFRAME COMPARISON", 'yellow'))
        print("-"*80)
        print(f"{'Timeframe':<15} {'Excess Return':<15} {'Win Rate':<12} {'Trades':<10} {'Score':<10}")
        print("-"*80)
        
        for tf, metrics in sorted(results.items()):
            excess = metrics.get('excess_return', 0)
            win_rate = metrics.get('win_rate', 0)
            trades = metrics.get('num_trades', 0)
            
            # Calculate composite score
            score = (excess * 0.5 + win_rate * 0.3 + min(trades/20, 1) * 20)
            
            print(f"{tf:<15} {excess:+.1f}%".ljust(15) + 
                  f" {win_rate:.1f}%".ljust(12) + 
                  f" {trades}".ljust(10) + 
                  f" {score:.1f}")
    
    def find_optimal_timeframe(self, results):
        """Find the optimal timeframe based on composite score"""
        best_score = -float('inf')
        best_tf = None
        
        for tf, metrics in results.items():
            # Composite score emphasizing excess return and win rate
            score = (
                metrics.get('excess_return', 0) * 0.5 +
                metrics.get('win_rate', 0) * 0.3 +
                min(metrics.get('num_trades', 0) / 20, 1) * 20
            )
            
            if score > best_score:
                best_score = score
                best_tf = tf
        
        return {
            'timeframe': best_tf,
            'metrics': results[best_tf],
            'score': best_score
        }
    
    def batch_test(self, symbols, timeframes=None):
        """Test multiple symbols across timeframes"""
        print(colored("\n🚀 MULTI-TIMEFRAME BATCH TESTING", 'cyan', attrs=['bold']))
        print(f"Testing {len(symbols)} symbols")
        
        all_results = {}
        optimal_timeframes = {}
        
        for symbol in symbols:
            results = self.compare_symbol(symbol, timeframes)
            if results:
                all_results[symbol] = results
                optimal = self.find_optimal_timeframe(results)
                optimal_timeframes[symbol] = optimal['timeframe']
        
        # Summary report
        self.generate_summary_report(all_results, optimal_timeframes)
    
    def generate_summary_report(self, all_results, optimal_timeframes):
        """Generate final summary report"""
        print(colored("\n" + "="*80, 'yellow'))
        print(colored("📊 TIMEFRAME OPTIMIZATION SUMMARY", 'yellow', attrs=['bold']))
        print("="*80)
        
        # Group by optimal timeframe
        tf_groups = {}
        for symbol, tf in optimal_timeframes.items():
            if tf not in tf_groups:
                tf_groups[tf] = []
            tf_groups[tf].append(symbol)
        
        print("\n🎯 Optimal Timeframes by Symbol:")
        for tf in ['5m', '15m', '1h', '1d']:
            if tf in tf_groups:
                print(colored(f"\n{tf} timeframe:", 'cyan'))
                for symbol in tf_groups[tf]:
                    metrics = all_results[symbol][tf]
                    print(f"   {symbol}: +{metrics.get('excess_return', 0):.1f}% excess, "
                          f"{metrics.get('win_rate', 0):.1f}% win rate")
        
        print(colored("\n💡 KEY INSIGHTS:", 'green'))
        print("• 5m/15m: Best for volatile/meme stocks (AMC, GME)")
        print("• 1h: Balanced for most stocks")
        print("• 1d: Best for trending stocks with clear patterns")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'timeframe_optimization_{timestamp}.json'
        
        save_data = {
            'test_date': datetime.now().isoformat(),
            'results': all_results,
            'optimal_timeframes': optimal_timeframes
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(colored(f"\n💾 Results saved to: {filename}", 'green'))

def main():
    """Run multi-timeframe testing"""
    tester = MultiTimeframeAEGSTester()
    
    # Test your current positions first
    print(colored("Testing current positions...", 'cyan'))
    current_positions = ['SLAI', 'TLRY']
    
    for symbol in current_positions:
        # For faster testing, only test shorter timeframes
        tester.compare_symbol(symbol, timeframes=['15m', '1h', '1d'])
    
    # Ask if user wants to test more symbols
    print("\nWould you like to test additional symbols? (y/n): ", end='')
    response = input()
    
    if response.lower() == 'y':
        additional_symbols = ['RIOT', 'GME', 'AMC']
        tester.batch_test(additional_symbols, timeframes=['15m', '1h', '1d'])

if __name__ == "__main__":
    main()