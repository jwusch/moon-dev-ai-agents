#!/usr/bin/env python3
"""
Test Previously Excluded Symbols
Runs AEGS backtest on symbols that were permanently excluded to see if issues are resolved
"""

import subprocess
import time
import json
from datetime import datetime
from termcolor import colored
import concurrent.futures
import sqlite3

class ExcludedSymbolsTester:
    def __init__(self):
        self.results = {}
        self.db_path = 'aegs_data.db'
        
    def get_top_excluded_symbols(self, limit=30):
        """Get symbols with highest failure counts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get symbols with 10+ failures (previously permanent)
        cursor.execute("""
            SELECT symbol, fail_count, reason 
            FROM invalid_symbols 
            WHERE fail_count >= 10 
            ORDER BY fail_count DESC 
            LIMIT ?
        """, (limit,))
        
        symbols = []
        for row in cursor.fetchall():
            symbol, fail_count, reason = row
            symbols.append({
                'symbol': symbol,
                'fail_count': fail_count,
                'previous_reason': reason
            })
            
        conn.close()
        return symbols
    
    def test_symbol(self, symbol_info):
        """Test a single symbol"""
        symbol = symbol_info['symbol']
        if symbol == 'TEST':  # Skip test symbol
            return None
            
        print(colored(f"\n🔍 Testing {symbol} (previously failed {symbol_info['fail_count']} times)...", 'cyan'))
        
        start_time = time.time()
        try:
            # Run AEGS backtest
            cmd = ['python', 'run_aegs_backtest.py', symbol]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            elapsed = time.time() - start_time
            
            # Check for JSON parsing error
            if "Extra data: line" in result.stderr or "Extra data: line" in result.stdout:
                status = "JSON_ERROR"
                print(colored(f"❌ {symbol}: Still has JSON parsing error", 'red'))
            elif "insufficient data" in result.stdout.lower():
                status = "INSUFFICIENT_DATA"
                print(colored(f"⚠️ {symbol}: Insufficient data", 'yellow'))
            elif "backtest complete" in result.stdout.lower():
                status = "SUCCESS"
                print(colored(f"✅ {symbol}: Backtest completed successfully!", 'green'))
                
                # Extract key metrics if successful
                metrics = self.extract_metrics(result.stdout)
                return {
                    'symbol': symbol,
                    'status': status,
                    'elapsed_time': elapsed,
                    'previous_fails': symbol_info['fail_count'],
                    'metrics': metrics
                }
            else:
                status = "OTHER_ERROR"
                print(colored(f"❓ {symbol}: Unknown error", 'yellow'))
                
            return {
                'symbol': symbol,
                'status': status,
                'elapsed_time': elapsed,
                'previous_fails': symbol_info['fail_count'],
                'error': result.stderr[:200] if result.stderr else result.stdout[:200]
            }
            
        except subprocess.TimeoutExpired:
            print(colored(f"⏱️ {symbol}: Timeout after 5 minutes", 'yellow'))
            return {
                'symbol': symbol,
                'status': 'TIMEOUT',
                'elapsed_time': 300,
                'previous_fails': symbol_info['fail_count']
            }
        except Exception as e:
            print(colored(f"💥 {symbol}: Exception - {str(e)}", 'red'))
            return {
                'symbol': symbol,
                'status': 'EXCEPTION',
                'error': str(e),
                'previous_fails': symbol_info['fail_count']
            }
    
    def extract_metrics(self, output):
        """Extract key metrics from successful backtest"""
        metrics = {}
        
        # Extract excess return
        if "Excess Return:" in output:
            try:
                excess_line = [line for line in output.split('\n') if 'Excess Return:' in line][0]
                metrics['excess_return'] = float(excess_line.split('+')[1].split('%')[0])
            except:
                pass
                
        # Extract win rate
        if "Win Rate:" in output:
            try:
                win_line = [line for line in output.split('\n') if 'Win Rate:' in line][0]
                metrics['win_rate'] = float(win_line.split(':')[1].split('%')[0])
            except:
                pass
                
        return metrics
    
    def run_batch_test(self, max_workers=3):
        """Run tests on multiple symbols concurrently"""
        symbols = self.get_top_excluded_symbols(30)
        
        print(colored(f"\n🚀 Testing {len(symbols)} Previously Excluded Symbols", 'yellow', attrs=['bold']))
        print("="*60)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.test_symbol, sym): sym for sym in symbols}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def generate_report(self, results):
        """Generate summary report"""
        print(colored("\n📊 TEST RESULTS SUMMARY", 'yellow', attrs=['bold']))
        print("="*60)
        
        # Categorize results
        success = [r for r in results if r['status'] == 'SUCCESS']
        json_errors = [r for r in results if r['status'] == 'JSON_ERROR']
        insufficient = [r for r in results if r['status'] == 'INSUFFICIENT_DATA']
        other = [r for r in results if r['status'] not in ['SUCCESS', 'JSON_ERROR', 'INSUFFICIENT_DATA']]
        
        print(colored(f"\n✅ SUCCESSFUL: {len(success)} symbols", 'green'))
        for r in sorted(success, key=lambda x: x.get('metrics', {}).get('excess_return', 0), reverse=True):
            metrics = r.get('metrics', {})
            excess = metrics.get('excess_return', 'N/A')
            print(f"   {r['symbol']}: Excess Return: {excess}% (was {r['previous_fails']} failures)")
        
        print(colored(f"\n❌ STILL JSON ERRORS: {len(json_errors)} symbols", 'red'))
        for r in sorted(json_errors, key=lambda x: x['previous_fails'], reverse=True):
            print(f"   {r['symbol']}: {r['previous_fails']} previous failures")
        
        print(colored(f"\n⚠️ INSUFFICIENT DATA: {len(insufficient)} symbols", 'yellow'))
        for r in insufficient[:5]:  # Show first 5
            print(f"   {r['symbol']}")
        
        if other:
            print(colored(f"\n❓ OTHER ERRORS: {len(other)} symbols", 'yellow'))
            for r in other[:5]:
                print(f"   {r['symbol']}: {r['status']}")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'excluded_symbols_test_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'total_tested': len(results),
                'results': results,
                'summary': {
                    'successful': len(success),
                    'json_errors': len(json_errors),
                    'insufficient_data': len(insufficient),
                    'other_errors': len(other)
                }
            }, f, indent=2)
        
        print(colored(f"\n📄 Detailed results saved to: {filename}", 'cyan'))
        
        return {
            'success': success,
            'json_errors': json_errors,
            'insufficient': insufficient,
            'other': other
        }

def main():
    """Run the excluded symbols test"""
    tester = ExcludedSymbolsTester()
    
    # Run batch test
    results = tester.run_batch_test(max_workers=3)
    
    # Generate report
    summary = tester.generate_report(results)
    
    # Recommendation
    print(colored("\n💡 RECOMMENDATIONS:", 'cyan'))
    if summary['success']:
        print(f"1. ✅ {len(summary['success'])} symbols can be removed from exclusion list")
    if summary['json_errors']:
        print(f"2. ❌ {len(summary['json_errors'])} symbols should remain excluded (JSON errors persist)")
    print("3. 🔄 Consider re-testing in 24 hours as API issues may be temporary")

if __name__ == "__main__":
    main()