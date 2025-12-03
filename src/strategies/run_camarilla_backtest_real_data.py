"""
🚀 Run Camarilla Backtest with Real Market Data
Uses YFinance to get actual historical data for backtesting

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest
from strategies.camarilla_strategy import CamarillaStrategy
from agents.yfinance_adapter import YFinanceAdapter

def get_real_market_data(symbol='TSLA', days=500):
    """Get real market data using YFinance"""
    print(f"📊 Fetching real {symbol} data from YFinance...")
    
    # Initialize YFinance
    yf = YFinanceAdapter()
    
    # Get daily data
    data = yf.get_ohlcv_data(symbol, '1d', days)
    
    if data.empty:
        raise ValueError(f"Could not fetch data for {symbol}")
    
    # Format for backtesting.py
    df = pd.DataFrame({
        'Open': data['open'],
        'High': data['high'],
        'Low': data['low'],
        'Close': data['close'],
        'Volume': data['volume']
    }, index=pd.to_datetime(data['timestamp']))
    
    # Remove any NaN values
    df = df.dropna()
    
    # Ensure all values are positive
    if (df <= 0).any().any():
        print("⚠️ Found non-positive values, cleaning data...")
        df = df[df > 0].dropna()
    
    return df

def run_real_data_backtest(symbol='TSLA'):
    """Run backtest with real market data"""
    
    print(f"🚀 Camarilla Strategy Backtest - Real {symbol} Data")
    print("="*60)
    
    # 1. Load real data
    print(f"\n1️⃣ Loading real {symbol} data...")
    df = get_real_market_data(symbol, days=500)
    
    print(f"✅ Loaded {len(df)} days of real {symbol} data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    print(f"   Current price: ${df['Close'].iloc[-1]:.2f}")
    
    # 2. Run backtest
    print("\n2️⃣ Running backtest...")
    bt = Backtest(
        df, 
        CamarillaStrategy,
        cash=10000,
        commission=0.002,  # 0.2% commission
        exclusive_orders=True
    )
    
    # Run with default parameters
    results = bt.run()
    
    # 3. Display results
    print("\n3️⃣ Results:")
    print("="*60)
    print(f"Total Return........  {results['Return [%]']:9.2f}%")
    print(f"Annual Return.......  {results['Return (Ann.) [%]']:9.2f}%")
    print(f"Sharpe Ratio........  {results['Sharpe Ratio']:9.2f}")
    print(f"Max Drawdown........  {results['Max. Drawdown [%]']:9.2f}%")
    print(f"Win Rate............  {results['Win Rate [%]']:9.2f}%")
    print(f"Total Trades........  {results['# Trades']:9.0f}")
    
    if results['# Trades'] > 0:
        print(f"Profit Factor.......  {results.get('Profit Factor', 0):9.2f}")
        print(f"Avg Trade...........  {results.get('Avg. Trade [%]', 0):9.2f}%")
    
    # 4. Compare different configurations
    print("\n4️⃣ Testing different parameter sets...")
    
    configs = {
        'Conservative': {
            'range_threshold': 0.003,
            'stop_loss_buffer': 0.002,
            'take_profit_multiplier': 1.5
        },
        'Moderate': {
            'range_threshold': 0.002,
            'stop_loss_buffer': 0.0015,
            'take_profit_multiplier': 2.0
        },
        'Aggressive': {
            'range_threshold': 0.001,
            'stop_loss_buffer': 0.001,
            'take_profit_multiplier': 3.0
        }
    }
    
    comparison = []
    
    for name, params in configs.items():
        result = bt.run(**params)
        comparison.append({
            'Strategy': name,
            'Return %': result['Return [%]'],
            'Sharpe': result['Sharpe Ratio'],
            'Trades': result['# Trades'],
            'Win Rate %': result['Win Rate [%]'],
            'Max DD %': result['Max. Drawdown [%]']
        })
    
    # Show comparison
    comparison_df = pd.DataFrame(comparison)
    print("\n📊 Parameter Comparison:")
    print(comparison_df.to_string(index=False))
    
    # 5. Find optimal parameters
    print("\n5️⃣ Running optimization...")
    
    # Optimize parameters
    optimal = bt.optimize(
        range_threshold=[0.001, 0.002, 0.003, 0.004],
        stop_loss_buffer=[0.001, 0.0015, 0.002, 0.0025],
        maximize='Return [%]',
        return_heatmap=True
    )
    
    print(f"\n✅ Optimal parameters found:")
    print(f"   Range Threshold: {optimal._strategy.range_threshold:.4f}")
    print(f"   Stop Loss Buffer: {optimal._strategy.stop_loss_buffer:.4f}")
    print(f"   Optimized Return: {optimal['Return [%]']:.2f}%")
    
    # 6. Generate plot
    print("\n6️⃣ Generating plot...")
    try:
        optimal.plot(filename=f'{symbol.lower()}_camarilla_results.html', open_browser=False)
        print(f"✅ Plot saved as: {symbol.lower()}_camarilla_results.html")
    except:
        print("⚠️ Could not generate plot")
    
    return optimal

def compare_multiple_symbols():
    """Compare strategy performance across different symbols"""
    print("\n\n🏁 Multi-Symbol Comparison")
    print("="*60)
    
    symbols = ['TSLA', 'AAPL', 'SPY', 'QQQ', 'NVDA']
    results = []
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            df = get_real_market_data(symbol, days=365)
            
            bt = Backtest(
                df, 
                CamarillaStrategy,
                cash=10000,
                commission=0.002
            )
            
            result = bt.run()
            
            results.append({
                'Symbol': symbol,
                'Return %': result['Return [%]'],
                'Sharpe': result['Sharpe Ratio'],
                'Win Rate %': result['Win Rate [%]'],
                'Trades': result['# Trades']
            })
            
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
    
    # Display comparison
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Return %', ascending=False)
        
        print("\n📊 Symbol Performance Comparison:")
        print(results_df.to_string(index=False))
        
        print(f"\n🏆 Best performer: {results_df.iloc[0]['Symbol']} with {results_df.iloc[0]['Return %']:.2f}% return")

def main():
    # Run backtest with real TSLA data
    optimal_result = run_real_data_backtest('TSLA')
    
    # Optional: Compare across multiple symbols
    compare_multiple_symbols()
    
    print("\n\n✅ Backtest complete with REAL market data!")
    print("💡 This uses actual historical prices from YFinance, not random data")
    print("📊 Try different symbols by changing the 'symbol' parameter")

if __name__ == "__main__":
    main()