#!/usr/bin/env python3
"""
AEGS Backtest with configurable timeframes
Allows testing different time resolutions (1m, 5m, 15m, 1h, 1d)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from termcolor import colored

# Import the AEGS backtester - we'll use the same logic
from aegs_batch_backtest import run_aegs_ensemble_backtest

def download_timeframe_data(symbol, interval='1d', days_back=730):
    """Download data for specific timeframe"""
    
    print(f"📊 Downloading {interval} data for {symbol} ({days_back} days)...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Download data
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        if df.empty:
            print(f"❌ No data available for {symbol} at {interval} timeframe")
            return None
            
        # Clean column names to match AEGS expectations
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        print(f"✅ Downloaded {len(df)} {interval} bars")
        print(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def run_timeframe_backtest(symbol, interval='1d', days_back=730):
    """Run AEGS backtest with specific timeframe"""
    
    print(colored(f"\n🔥💎 AEGS BACKTEST FOR {symbol} ({interval} timeframe) 💎🔥", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Download data for timeframe
    df = download_timeframe_data(symbol, interval, days_back)
    if df is None:
        return None
    
    # Run AEGS ensemble backtest
    print(f"\n🚀 Running AEGS backtest on {interval} data...")
    
    try:
        results = run_aegs_ensemble_backtest(symbol, df)
        
        if results:
            # Add timeframe info
            results['timeframe'] = interval
            results['data_points'] = len(df)
            results['date_range'] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            
            # Display results
            display_timeframe_results(symbol, interval, results)
            
            return results
        else:
            print(f"❌ Backtest failed for {interval} timeframe")
            return None
            
    except Exception as e:
        print(f"❌ Error during backtest: {str(e)[:200]}")
        return None

def display_timeframe_results(symbol, interval, results):
    """Display results for a specific timeframe"""
    
    print(colored(f"\n📊 {symbol} RESULTS ({interval} timeframe)", 'yellow', attrs=['bold']))
    print("="*60)
    
    # Key metrics
    print(f"\n⚡ Performance Metrics:")
    print(f"   Excess Return: {results.get('excess_return_pct', 0):+.1f}%")
    print(f"   Strategy Return: {results.get('strategy_return_pct', 0):+.1f}%")
    print(f"   Buy & Hold Return: {results.get('buy_hold_return_pct', 0):+.1f}%")
    print(f"   Win Rate: {results.get('win_rate', 0):.1f}%")
    print(f"   Total Trades: {results.get('num_trades', 0)}")
    
    if results.get('sharpe_ratio'):
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    # Holding period in appropriate units
    avg_holding = results.get('avg_holding_days', 0)
    if interval in ['1m', '5m', '15m']:
        print(f"   Avg Holding: {avg_holding * 24:.1f} hours")
    else:
        print(f"   Avg Holding: {avg_holding:.1f} days")
    
    # Trading recommendation
    print(colored(f"\n💡 TRADING RECOMMENDATION:", 'green'))
    
    excess_return = results.get('excess_return_pct', 0)
    win_rate = results.get('win_rate', 0)
    
    if excess_return > 100 and win_rate > 55:
        print(f"   🚀 STRONG BUY signal on {interval} timeframe")
        print(f"   Monitor every {get_monitoring_frequency(interval)}")
    elif excess_return > 50 and win_rate > 50:
        print(f"   ✅ BUY signal on {interval} timeframe")
        print(f"   Monitor every {get_monitoring_frequency(interval)}")
    elif excess_return > 0:
        print(f"   ⚠️ WATCH on {interval} timeframe")
        print(f"   Wait for better entry signals")
    else:
        print(f"   ❌ AVOID on {interval} timeframe")
        print(f"   Strategy underperforms buy & hold")

def get_monitoring_frequency(interval):
    """Get recommended monitoring frequency for timeframe"""
    frequencies = {
        '1m': '5-10 minutes',
        '5m': '15-30 minutes',
        '15m': '1 hour',
        '1h': '4 hours',
        '1d': 'daily'
    }
    return frequencies.get(interval, 'daily')

def compare_timeframes(symbol, timeframes=None):
    """Compare a symbol across multiple timeframes"""
    
    if timeframes is None:
        timeframes = {
            '5m': 30,    # 30 days
            '15m': 60,   # 60 days  
            '1h': 180,   # 180 days
            '1d': 730    # 2 years
        }
    
    print(colored(f"\n{'='*80}", 'cyan'))
    print(colored(f"🔄 MULTI-TIMEFRAME COMPARISON: {symbol}", 'cyan', attrs=['bold']))
    print(colored(f"{'='*80}", 'cyan'))
    
    results = {}
    
    for interval, days in timeframes.items():
        result = run_timeframe_backtest(symbol, interval, days)
        if result:
            results[interval] = result
    
    # Display comparison
    if results:
        display_comparison(symbol, results)
        find_optimal_timeframe(symbol, results)
    
    return results

def display_comparison(symbol, results):
    """Display side-by-side comparison"""
    
    print(colored(f"\n📊 {symbol} TIMEFRAME COMPARISON", 'yellow', attrs=['bold']))
    print("-"*80)
    print(f"{'Timeframe':<12} {'Excess %':<12} {'Strategy %':<12} {'Win Rate':<10} {'Trades':<8} {'Score':<8}")
    print("-"*80)
    
    for tf in ['5m', '15m', '1h', '1d']:
        if tf in results:
            r = results[tf]
            excess = r.get('excess_return_pct', 0)
            strategy = r.get('strategy_return_pct', 0)
            win_rate = r.get('win_rate', 0)
            trades = r.get('num_trades', 0)
            
            # Composite score
            score = excess * 0.5 + win_rate * 0.3 + min(trades/20, 1) * 20
            
            print(f"{tf:<12} {excess:+.1f}".ljust(12) + 
                  f" {strategy:+.1f}".ljust(12) + 
                  f" {win_rate:.1f}%".ljust(10) + 
                  f" {trades}".ljust(8) + 
                  f" {score:.1f}")

def find_optimal_timeframe(symbol, results):
    """Find the optimal timeframe for trading"""
    
    best_score = -float('inf')
    best_tf = None
    
    for tf, r in results.items():
        # Composite score emphasizing excess return and win rate
        score = (r.get('excess_return_pct', 0) * 0.5 + 
                r.get('win_rate', 0) * 0.3 + 
                min(r.get('num_trades', 0) / 20, 1) * 20)
        
        if score > best_score:
            best_score = score
            best_tf = tf
    
    if best_tf:
        print(colored(f"\n⭐ OPTIMAL TIMEFRAME: {best_tf}", 'green', attrs=['bold']))
        best_result = results[best_tf]
        print(f"   Excess Return: {best_result.get('excess_return_pct', 0):+.1f}%")
        print(f"   Win Rate: {best_result.get('win_rate', 0):.1f}%")
        print(f"   Recommendation: Use {best_tf} timeframe for {symbol}")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python aegs_timeframe_backtest.py SYMBOL [TIMEFRAME]")
        print("  python aegs_timeframe_backtest.py SYMBOL compare")
        print("\nExamples:")
        print("  python aegs_timeframe_backtest.py AAPL 1h")
        print("  python aegs_timeframe_backtest.py TSLA compare")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # Check for environment variable override
    interval = os.environ.get('AEGS_TIMEFRAME', '1d')
    days_back = int(os.environ.get('AEGS_DAYS', 730))
    
    if len(sys.argv) > 2:
        if sys.argv[2].lower() == 'compare':
            # Compare across multiple timeframes
            compare_timeframes(symbol)
        else:
            # Use specific timeframe
            interval = sys.argv[2]
            # Set appropriate days based on timeframe
            if interval == '1m':
                days_back = 7
            elif interval == '5m':
                days_back = 30
            elif interval == '15m':
                days_back = 60
            elif interval == '1h':
                days_back = 180
            
            run_timeframe_backtest(symbol, interval, days_back)
    else:
        # Single timeframe test
        run_timeframe_backtest(symbol, interval, days_back)

if __name__ == "__main__":
    main()