#!/usr/bin/env python3
"""
AEGS Multi-Timeframe Optimizer
Tests symbols across different time resolutions to find optimal trading timeframes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import concurrent.futures
from aegs_batch_backtest import AEGSBacktester
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class TimeframeOptimizer:
    def __init__(self):
        self.timeframes = {
            '1m': {'interval': '1m', 'days': 7, 'min_trades': 20},
            '5m': {'interval': '5m', 'days': 30, 'min_trades': 15},
            '15m': {'interval': '15m', 'days': 60, 'min_trades': 10},
            '1h': {'interval': '1h', 'days': 365, 'min_trades': 5},
            '1d': {'interval': '1d', 'days': 1825, 'min_trades': 3}  # 5 years for daily
        }
        
        self.results = {}
        
    def download_data(self, symbol, timeframe_config):
        """Download data for specific timeframe"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=timeframe_config['days'])
            
            # Download data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=timeframe_config['interval']
            )
            
            if df.empty or len(df) < 100:  # Need minimum data points
                return None
                
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            return df
            
        except Exception as e:
            print(colored(f"   ❌ Error downloading {timeframe_config['interval']} data: {e}", 'red'))
            return None
    
    def optimize_symbol(self, symbol, timeframes_to_test=None):
        """Test a symbol across multiple timeframes"""
        print(colored(f"\n🔍 Optimizing {symbol} across timeframes...", 'cyan', attrs=['bold']))
        print("="*60)
        
        if timeframes_to_test is None:
            timeframes_to_test = list(self.timeframes.keys())
        
        symbol_results = {}
        best_timeframe = None
        best_metric = -float('inf')
        
        for tf in timeframes_to_test:
            tf_config = self.timeframes[tf]
            print(colored(f"\n📊 Testing {tf} timeframe...", 'yellow'))
            
            # Download data
            df = self.download_data(symbol, tf_config)
            if df is None:
                print(f"   ⚠️ Insufficient data for {tf}")
                continue
            
            print(f"   ✅ Downloaded {len(df)} bars")
            
            try:
                # Run AEGS backtest
                backtester = AEGSBacktester()
                results = backtester.comprehensive_backtest(df)
                
                if results and results.get('num_trades', 0) >= tf_config['min_trades']:
                    # Extract key metrics
                    metrics = {
                        'timeframe': tf,
                        'excess_return': results.get('excess_return_pct', 0),
                        'strategy_return': results.get('strategy_return_pct', 0),
                        'buy_hold_return': results.get('buy_hold_return_pct', 0),
                        'win_rate': results.get('win_rate', 0),
                        'num_trades': results.get('num_trades', 0),
                        'sharpe_ratio': results.get('sharpe_ratio', 0),
                        'max_drawdown': results.get('max_drawdown', 0),
                        'avg_trade_return': results.get('avg_return', 0),
                        'avg_holding_days': results.get('avg_holding_days', 0),
                        'profit_factor': results.get('profit_factor', 0),
                        'data_points': len(df),
                        'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                    }
                    
                    # Calculate composite score for ranking
                    # Higher weight to excess return and win rate
                    composite_score = (
                        metrics['excess_return'] * 0.4 +
                        metrics['win_rate'] * 0.3 +
                        metrics['sharpe_ratio'] * 10 * 0.2 +
                        min(metrics['num_trades'] / 20, 1) * 0.1
                    )
                    metrics['composite_score'] = composite_score
                    
                    symbol_results[tf] = metrics
                    
                    # Track best timeframe
                    if composite_score > best_metric:
                        best_metric = composite_score
                        best_timeframe = tf
                    
                    # Display results
                    self.display_timeframe_results(tf, metrics)
                    
                else:
                    trades = results.get('num_trades', 0) if results else 0
                    print(f"   ❌ Insufficient trades ({trades}) for meaningful analysis")
                    
            except Exception as e:
                print(f"   ❌ Backtest error: {str(e)[:100]}")
                continue
        
        # Summary
        if symbol_results:
            print(colored(f"\n📊 {symbol} TIMEFRAME OPTIMIZATION SUMMARY", 'yellow', attrs=['bold']))
            print("="*60)
            
            # Sort by composite score
            sorted_tfs = sorted(symbol_results.items(), 
                              key=lambda x: x[1]['composite_score'], 
                              reverse=True)
            
            print("\nRanked Results:")
            for i, (tf, metrics) in enumerate(sorted_tfs, 1):
                print(f"\n{i}. {colored(tf, 'cyan', attrs=['bold'])} (Score: {metrics['composite_score']:.1f})")
                print(f"   Excess Return: {metrics['excess_return']:+.1f}%")
                print(f"   Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   Trades: {metrics['num_trades']}")
                print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
                
                if i == 1:
                    print(colored("   ⭐ OPTIMAL TIMEFRAME", 'green', attrs=['bold']))
            
            # Store results
            self.results[symbol] = {
                'timeframes': symbol_results,
                'optimal_timeframe': best_timeframe,
                'optimization_date': datetime.now().isoformat()
            }
            
            return symbol_results
        else:
            print(colored(f"\n❌ No valid timeframes found for {symbol}", 'red'))
            return None
    
    def display_timeframe_results(self, timeframe, metrics):
        """Display results for a single timeframe"""
        print(f"\n   📈 Results for {timeframe}:")
        print(f"      Excess Return: {metrics['excess_return']:+.1f}%")
        print(f"      Strategy Return: {metrics['strategy_return']:+.1f}%")
        print(f"      Win Rate: {metrics['win_rate']:.1f}%")
        print(f"      Number of Trades: {metrics['num_trades']}")
        print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"      Avg Trade Return: {metrics['avg_trade_return']:+.1f}%")
        
        # Holding period in appropriate units
        if timeframe in ['1m', '5m', '15m']:
            hold_hours = metrics['avg_holding_days'] * 24
            print(f"      Avg Holding: {hold_hours:.1f} hours")
        else:
            print(f"      Avg Holding: {metrics['avg_holding_days']:.1f} days")
    
    def get_top_performing_symbols(self, limit=10):
        """Get symbols that performed well in recent tests"""
        conn = sqlite3.connect('aegs_data.db')
        cursor = conn.cursor()
        
        # Get successful discoveries
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM discoveries 
            WHERE excess_return > 50 
            AND win_rate > 50
            ORDER BY discovered_at DESC
            LIMIT ?
        """, (limit,))
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Add some known good performers
        additional_symbols = ['GME', 'AMC', 'RIOT', 'TSLA', 'NVDA', 'BB', 'PLTR']
        
        # Combine and dedupe
        all_symbols = list(set(symbols + additional_symbols))[:limit]
        
        return all_symbols
    
    def batch_optimize(self, symbols=None, max_workers=2):
        """Optimize multiple symbols in parallel"""
        if symbols is None:
            symbols = self.get_top_performing_symbols()
        
        print(colored(f"\n🚀 BATCH TIMEFRAME OPTIMIZATION", 'cyan', attrs=['bold']))
        print(f"Testing {len(symbols)} symbols across timeframes")
        print("="*70)
        
        # Process symbols with limited parallelism (yfinance rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.optimize_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        print(colored(f"\n✅ Completed {symbol} ({completed}/{len(symbols)})", 'green'))
                except Exception as e:
                    print(colored(f"\n❌ Failed {symbol}: {str(e)[:100]}", 'red'))
        
        # Generate final report
        self.generate_optimization_report()
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.results:
            print("No results to report")
            return
        
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("📊 TIMEFRAME OPTIMIZATION FINAL REPORT", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        # Organize by optimal timeframe
        timeframe_groups = {}
        for symbol, data in self.results.items():
            optimal_tf = data.get('optimal_timeframe', 'Unknown')
            if optimal_tf not in timeframe_groups:
                timeframe_groups[optimal_tf] = []
            timeframe_groups[optimal_tf].append({
                'symbol': symbol,
                'metrics': data['timeframes'].get(optimal_tf, {})
            })
        
        # Display grouped results
        for tf in ['1m', '5m', '15m', '1h', '1d']:
            if tf in timeframe_groups:
                symbols = timeframe_groups[tf]
                print(colored(f"\n⏱️ {tf} Optimal Symbols ({len(symbols)}):", 'cyan', attrs=['bold']))
                
                # Sort by excess return
                symbols.sort(key=lambda x: x['metrics'].get('excess_return', 0), reverse=True)
                
                for sym_data in symbols[:5]:  # Top 5 for each timeframe
                    symbol = sym_data['symbol']
                    metrics = sym_data['metrics']
                    print(f"   {symbol}: +{metrics.get('excess_return', 0):.1f}% excess, "
                          f"{metrics.get('win_rate', 0):.1f}% win rate, "
                          f"{metrics.get('num_trades', 0)} trades")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'timeframe_optimization_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(colored(f"\n💾 Detailed results saved to: {filename}", 'green'))
        
        # Key insights
        print(colored("\n🔍 KEY INSIGHTS:", 'yellow', attrs=['bold']))
        print("1. Shorter timeframes (1m, 5m) best for: High volatility stocks, meme stocks")
        print("2. Medium timeframes (15m, 1h) best for: Balanced risk/reward, most stocks")
        print("3. Daily timeframes best for: Lower volatility, position trading")
        print("\n💡 Use these optimal timeframes when running AEGS for each symbol!")

def main():
    """Run timeframe optimization"""
    optimizer = TimeframeOptimizer()
    
    # You can test specific symbols
    # symbols = ['SLAI', 'TLRY', 'GME', 'AMC', 'RIOT']
    
    # Or let it find top performers automatically
    print("Finding top performing symbols for optimization...")
    symbols = optimizer.get_top_performing_symbols(10)
    
    print(f"Selected symbols: {', '.join(symbols)}")
    
    # Run optimization
    optimizer.batch_optimize(symbols, max_workers=2)

if __name__ == "__main__":
    main()