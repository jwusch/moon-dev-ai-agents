#!/usr/bin/env python3
"""
Compare AEGS performance across intraday timeframes
Focuses on 5m, 15m, 1h for active trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import colored
from aegs_batch_backtest import AEGSBacktester
import matplotlib.pyplot as plt
import seaborn as sns

class IntradayAEGSComparison:
    def __init__(self):
        self.timeframes = {
            '5m': {'interval': '5m', 'days': 30, 'label': '5 Minute'},
            '15m': {'interval': '15m', 'days': 60, 'label': '15 Minute'},
            '1h': {'interval': '1h', 'days': 180, 'label': '1 Hour'},
        }
        
    def compare_symbol(self, symbol):
        """Compare a symbol across intraday timeframes"""
        print(colored(f"\n📊 Analyzing {symbol} Across Intraday Timeframes", 'cyan', attrs=['bold']))
        print("="*60)
        
        results = {}
        
        for tf_name, tf_config in self.timeframes.items():
            print(colored(f"\n⏱️ Testing {tf_config['label']}...", 'yellow'))
            
            try:
                # Download data
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=tf_config['days'])
                
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=tf_config['interval']
                )
                
                if df.empty or len(df) < 100:
                    print(f"   ⚠️ Insufficient data")
                    continue
                
                # Clean columns
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                print(f"   ✅ Downloaded {len(df)} bars")
                print(f"   📅 Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
                # Run AEGS backtest
                backtester = AEGSBacktester()
                backtest_results = backtester.comprehensive_backtest(df)
                
                if backtest_results and backtest_results.get('num_trades', 0) > 3:
                    # Store results
                    results[tf_name] = {
                        'timeframe': tf_config['label'],
                        'excess_return': backtest_results.get('excess_return_pct', 0),
                        'strategy_return': backtest_results.get('strategy_return_pct', 0),
                        'win_rate': backtest_results.get('win_rate', 0),
                        'num_trades': backtest_results.get('num_trades', 0),
                        'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                        'avg_return': backtest_results.get('avg_return', 0),
                        'avg_holding_periods': backtest_results.get('avg_holding_days', 0) * 24,  # Convert to hours
                        'max_drawdown': backtest_results.get('max_drawdown', 0)
                    }
                    
                    # Print summary
                    print(colored(f"\n   📈 {tf_config['label']} Results:", 'green'))
                    print(f"      Excess Return: {results[tf_name]['excess_return']:+.1f}%")
                    print(f"      Win Rate: {results[tf_name]['win_rate']:.1f}%")
                    print(f"      Trades: {results[tf_name]['num_trades']}")
                    print(f"      Avg Holding: {results[tf_name]['avg_holding_periods']:.1f} hours")
                else:
                    print(f"   ❌ Insufficient trades or poor results")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")
                continue
        
        # Display comparison
        if results:
            self.display_comparison(symbol, results)
            return results
        else:
            print(colored(f"\n❌ No valid results for {symbol}", 'red'))
            return None
    
    def display_comparison(self, symbol, results):
        """Display side-by-side comparison of timeframes"""
        print(colored(f"\n📊 {symbol} TIMEFRAME COMPARISON", 'yellow', attrs=['bold']))
        print("="*80)
        
        # Create comparison table
        print(f"\n{'Metric':<20} {'5 Minute':<20} {'15 Minute':<20} {'1 Hour':<20}")
        print("-"*80)
        
        metrics = [
            ('Excess Return %', 'excess_return'),
            ('Win Rate %', 'win_rate'),
            ('Number of Trades', 'num_trades'),
            ('Avg Return %', 'avg_return'),
            ('Avg Hold (hours)', 'avg_holding_periods'),
            ('Sharpe Ratio', 'sharpe_ratio')
        ]
        
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<20}"
            for tf in ['5m', '15m', '1h']:
                if tf in results:
                    value = results[tf].get(metric_key, 0)
                    if metric_key in ['excess_return', 'avg_return']:
                        row += f"{value:+.1f}%".ljust(20)
                    elif metric_key == 'win_rate':
                        row += f"{value:.1f}%".ljust(20)
                    elif metric_key == 'num_trades':
                        row += f"{int(value)}".ljust(20)
                    elif metric_key == 'sharpe_ratio':
                        row += f"{value:.2f}".ljust(20)
                    elif metric_key == 'avg_holding_periods':
                        row += f"{value:.1f}h".ljust(20)
                else:
                    row += "N/A".ljust(20)
            print(row)
        
        # Find optimal timeframe
        best_tf = None
        best_score = -float('inf')
        
        for tf, data in results.items():
            # Composite score: weight excess return and win rate heavily
            score = (data['excess_return'] * 0.5 + 
                    data['win_rate'] * 0.3 + 
                    min(data['num_trades'] / 10, 1) * 0.2)
            
            if score > best_score:
                best_score = score
                best_tf = tf
        
        if best_tf:
            print(colored(f"\n⭐ OPTIMAL TIMEFRAME: {self.timeframes[best_tf]['label']}", 'green', attrs=['bold']))
            print(f"   Reasoning: Best combination of returns ({results[best_tf]['excess_return']:+.1f}%) ")
            print(f"             and win rate ({results[best_tf]['win_rate']:.1f}%)")
            
            # Trading recommendations
            print(colored(f"\n💡 TRADING RECOMMENDATIONS:", 'cyan'))
            
            if best_tf == '5m':
                print("   • Best for: Day trading with tight stops")
                print("   • Monitor: Every 5-15 minutes during market hours")
                print("   • Risk: Higher due to noise, use smaller position sizes")
                
            elif best_tf == '15m':
                print("   • Best for: Intraday swing trading")
                print("   • Monitor: Every 30-60 minutes")
                print("   • Risk: Balanced risk/reward, good for most traders")
                
            elif best_tf == '1h':
                print("   • Best for: Multi-day swing trading")
                print("   • Monitor: 2-3 times per day")
                print("   • Risk: Lower frequency, can use larger positions")

def main():
    """Run intraday comparison on key symbols"""
    comparator = IntradayAEGSComparison()
    
    # Test your positions and some high performers
    symbols = ['SLAI', 'TLRY', 'RIOT', 'GME', 'AMC']
    
    all_results = {}
    
    for symbol in symbols:
        results = comparator.compare_symbol(symbol)
        if results:
            all_results[symbol] = results
    
    # Summary recommendations
    if all_results:
        print(colored("\n" + "="*80, 'yellow'))
        print(colored("🎯 OVERALL TIMEFRAME RECOMMENDATIONS", 'yellow', attrs=['bold']))
        print("="*80)
        
        timeframe_counts = {'5m': 0, '15m': 0, '1h': 0}
        
        for symbol, results in all_results.items():
            # Find best timeframe for each symbol
            best_tf = max(results.items(), 
                         key=lambda x: x[1]['excess_return'] * 0.6 + x[1]['win_rate'] * 0.4)[0]
            timeframe_counts[best_tf] += 1
            
            print(f"{symbol}: {colored(results[best_tf]['timeframe'], 'green')} "
                  f"(+{results[best_tf]['excess_return']:.1f}% excess return)")
        
        print(colored("\n📊 Distribution:", 'cyan'))
        for tf, count in timeframe_counts.items():
            if count > 0:
                print(f"   {self.timeframes[tf]['label']}: {count} symbols")

if __name__ == "__main__":
    main()