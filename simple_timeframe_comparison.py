#!/usr/bin/env python3
"""
Simple timeframe comparison showing how different resolutions affect trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import colored
import matplotlib.pyplot as plt

def analyze_timeframe_characteristics(symbol, timeframes=None):
    """Analyze how different timeframes affect trading characteristics"""
    
    if timeframes is None:
        timeframes = {
            '5m': {'days': 30, 'label': '5-Minute'},
            '15m': {'days': 60, 'label': '15-Minute'}, 
            '1h': {'days': 180, 'label': 'Hourly'},
            '1d': {'days': 730, 'label': 'Daily'}
        }
    
    print(colored(f"\n📊 TIMEFRAME ANALYSIS FOR {symbol}", 'cyan', attrs=['bold']))
    print("="*80)
    
    ticker = yf.Ticker(symbol)
    results = {}
    
    for tf_key, tf_config in timeframes.items():
        print(colored(f"\n⏱️ Analyzing {tf_config['label']} timeframe...", 'yellow'))
        
        try:
            # Download data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=tf_config['days'])
            
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=tf_key
            )
            
            if df.empty or len(df) < 20:
                print(f"   ⚠️ Insufficient data")
                continue
            
            # Calculate key characteristics
            analysis = analyze_data_characteristics(df, tf_key)
            results[tf_key] = analysis
            
            # Display findings
            print(f"   📊 Data points: {len(df)}")
            print(f"   📈 Avg daily volatility: {analysis['daily_volatility']:.1f}%")
            print(f"   🎯 Noise ratio: {analysis['noise_ratio']:.2f}")
            print(f"   💰 Avg move per bar: {analysis['avg_move']:.2f}%")
            print(f"   📉 Max drawdown: {analysis['max_drawdown']:.1f}%")
            print(f"   🔄 Mean reversion potential: {analysis['mean_reversion_score']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
            continue
    
    # Recommendations
    if results:
        display_timeframe_recommendations(symbol, results)
    
    return results

def analyze_data_characteristics(df, interval):
    """Analyze characteristics of data at specific timeframe"""
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (annualized based on timeframe)
    periods_per_day = {
        '1m': 390,   # 6.5 hours * 60
        '5m': 78,    # 6.5 hours * 12
        '15m': 26,   # 6.5 hours * 4
        '1h': 6.5,
        '1d': 1
    }
    
    periods = periods_per_day.get(interval, 1)
    daily_volatility = df['returns'].std() * np.sqrt(periods) * 100
    
    # Noise ratio (high frequency noise vs trend)
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['noise'] = (df['Close'] - df['sma_20']).abs() / df['sma_20']
    noise_ratio = df['noise'].mean()
    
    # Average move per bar
    df['move'] = (df['High'] - df['Low']) / df['Open'] * 100
    avg_move = df['move'].mean()
    
    # Maximum drawdown
    cumulative = (1 + df['returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Mean reversion score (how often price crosses SMA)
    df['above_sma'] = df['Close'] > df['sma_20']
    crosses = df['above_sma'].diff().abs().sum()
    mean_reversion_score = (crosses / len(df)) * 100
    
    # Calculate optimal holding period
    # Look for average time to reach +/- 2% move
    target_return = 0.02
    holding_periods = []
    
    for i in range(len(df) - 20):
        entry_price = df['Close'].iloc[i]
        for j in range(i + 1, min(i + 100, len(df))):
            exit_price = df['Close'].iloc[j]
            ret = (exit_price - entry_price) / entry_price
            if abs(ret) >= target_return:
                holding_periods.append(j - i)
                break
    
    avg_holding = np.mean(holding_periods) if holding_periods else 20
    
    return {
        'daily_volatility': daily_volatility,
        'noise_ratio': noise_ratio,
        'avg_move': avg_move,
        'max_drawdown': abs(max_drawdown),
        'mean_reversion_score': mean_reversion_score,
        'avg_holding_periods': avg_holding,
        'total_bars': len(df)
    }

def display_timeframe_recommendations(symbol, results):
    """Display recommendations based on timeframe analysis"""
    
    print(colored(f"\n🎯 TIMEFRAME RECOMMENDATIONS FOR {symbol}", 'green', attrs=['bold']))
    print("="*80)
    
    # Find best timeframe for different strategies
    print("\n📊 Strategy Recommendations:")
    
    # Mean reversion strategy
    best_mr_tf = max(results.items(), key=lambda x: x[1]['mean_reversion_score'])[0]
    print(f"\n1. Mean Reversion Strategy:")
    print(f"   Best timeframe: {colored(best_mr_tf, 'green')}")
    print(f"   Reason: Highest reversion score ({results[best_mr_tf]['mean_reversion_score']:.1f}%)")
    
    # Trend following
    best_trend_tf = min(results.items(), key=lambda x: x[1]['noise_ratio'])[0]
    print(f"\n2. Trend Following Strategy:")
    print(f"   Best timeframe: {colored(best_trend_tf, 'green')}")
    print(f"   Reason: Lowest noise ratio ({results[best_trend_tf]['noise_ratio']:.2f})")
    
    # Scalping
    if '5m' in results and results['5m']['avg_move'] > 0.2:
        print(f"\n3. Scalping Strategy:")
        print(f"   Best timeframe: {colored('5m', 'green')}")
        print(f"   Reason: High avg move per bar ({results['5m']['avg_move']:.2f}%)")
    
    # Risk assessment
    print(colored("\n⚠️ Risk Considerations:", 'yellow'))
    for tf, data in sorted(results.items()):
        print(f"\n{tf} timeframe:")
        print(f"   - Volatility: {data['daily_volatility']:.1f}% daily")
        print(f"   - Max drawdown: {data['max_drawdown']:.1f}%")
        print(f"   - Optimal hold: {data['avg_holding_periods']:.0f} bars")
        
        # Calculate position sizing recommendation
        # Kelly criterion approximation: f = edge/odds
        # Use volatility as proxy for risk
        position_pct = min(25, 100 / data['daily_volatility'])
        print(f"   - Suggested position size: {position_pct:.0f}% of capital")

def compare_symbols(symbols):
    """Compare multiple symbols to find optimal timeframes"""
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        results = analyze_timeframe_characteristics(symbol)
        if results:
            all_results[symbol] = results
    
    # Summary
    if all_results:
        print(colored("\n📊 SUMMARY ACROSS ALL SYMBOLS", 'yellow', attrs=['bold']))
        print("="*80)
        
        # Best timeframe by symbol
        for symbol, results in all_results.items():
            # Find timeframe with best risk/reward
            best_tf = None
            best_score = -float('inf')
            
            for tf, data in results.items():
                # Score based on low noise, moderate volatility, good mean reversion
                score = (data['mean_reversion_score'] / 10 + 
                        5 / (data['noise_ratio'] + 0.1) +
                        min(data['daily_volatility'] / 20, 2))
                
                if score > best_score:
                    best_score = score
                    best_tf = tf
            
            print(f"\n{symbol}:")
            print(f"   Optimal timeframe: {colored(best_tf, 'green')}")
            print(f"   Daily volatility: {results[best_tf]['daily_volatility']:.1f}%")
            print(f"   Strategy: {'Mean Reversion' if results[best_tf]['mean_reversion_score'] > 15 else 'Trend Following'}")

def main():
    """Run timeframe analysis"""
    
    # Your current positions
    print(colored("Analyzing your current positions...", 'cyan'))
    symbols = ['SLAI', 'TLRY']
    
    for symbol in symbols:
        analyze_timeframe_characteristics(symbol)
    
    # Additional high-interest symbols
    print(colored("\n\nWould you like to analyze additional symbols? (y/n): ", 'yellow'), end='')
    response = input()
    
    if response.lower() == 'y':
        additional = ['RIOT', 'GME', 'AMC']
        compare_symbols(symbols + additional)

if __name__ == "__main__":
    main()