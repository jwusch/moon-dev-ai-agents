"""
🌙 Run Camarilla Backtest - Simple Example
Shows how to use the Camarilla strategy with backtesting.py

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy

def create_sample_data(days=365, initial_price=100):
    """Create realistic sample OHLCV data"""
    np.random.seed(42)
    
    # Generate daily data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0002, 0.02, len(dates))  # 0.02% drift, 2% volatility
    price = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    
    # Add realistic intraday movements
    df['Close'] = price
    df['Open'] = price * (1 + np.random.normal(0, 0.005, len(dates)))
    
    # High/Low with realistic spread
    spread = np.abs(np.random.normal(0.01, 0.005, len(dates)))
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + spread)
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - spread * 0.8)
    
    # Volume
    df['Volume'] = np.random.lognormal(15, 0.5, len(dates))
    
    return df

def run_simple_backtest():
    """Run a simple backtest with the Camarilla strategy"""
    
    print("🌙 Camarilla Strategy Backtest")
    print("="*60)
    
    # 1. Load or create data
    print("\n1️⃣ Loading data...")
    df = create_sample_data(days=500, initial_price=100)
    print(f"✅ Loaded {len(df)} days of data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    
    # 2. Initialize backtest
    print("\n2️⃣ Initializing backtest...")
    bt = Backtest(
        df, 
        CamarillaStrategy,
        cash=10000,
        commission=0.002,  # 0.2% commission
        exclusive_orders=True,  # One position at a time
        hedging=False,
        trade_on_close=True
    )
    print("✅ Backtest initialized with $10,000 starting capital")
    
    # 3. Run with default parameters
    print("\n3️⃣ Running backtest with default parameters...")
    stats = bt.run()
    
    # 4. Display results
    print("\n4️⃣ Results:")
    print("="*60)
    
    # Key metrics
    metrics = [
        ('Total Return', 'Return [%]', '%'),
        ('Annual Return', 'Return (Ann.) [%]', '%'),
        ('Sharpe Ratio', 'Sharpe Ratio', ''),
        ('Max Drawdown', 'Max. Drawdown [%]', '%'),
        ('Win Rate', 'Win Rate [%]', '%'),
        ('Total Trades', '# Trades', ''),
        ('Profit Factor', 'Profit Factor', ''),
        ('Avg Trade', 'Avg. Trade [%]', '%'),
    ]
    
    for label, key, suffix in metrics:
        if key in stats:
            value = stats[key]
            if suffix == '%':
                print(f"{label:.<20} {value:>10.2f}{suffix}")
            elif key == '# Trades':
                print(f"{label:.<20} {value:>10.0f}")
            else:
                print(f"{label:.<20} {value:>10.2f}")
    
    # 5. Run optimization (optional)
    print("\n5️⃣ Running parameter optimization...")
    print("   Testing different parameter combinations...")
    
    optimized = bt.optimize(
        range_threshold=[0.0005, 0.001, 0.002],
        stop_loss_buffer=[0.0005, 0.001],
        maximize='Return [%]'
    )
    
    print("\n   Optimal parameters found:")
    print(f"   Range Threshold: {optimized._strategy.range_threshold:.4f}")
    print(f"   Stop Loss Buffer: {optimized._strategy.stop_loss_buffer:.4f}")
    print(f"   Optimized Return: {optimized['Return [%]']:.2f}%")
    
    # 6. Save plot
    print("\n6️⃣ Generating plot...")
    try:
        bt.plot(filename='camarilla_backtest_results.html', open_browser=False)
        print("✅ Plot saved as: camarilla_backtest_results.html")
    except:
        print("⚠️  Could not generate plot (requires browser)")
    
    return stats

def compare_strategies():
    """Compare different strategy configurations"""
    
    print("\n\n🔬 Strategy Comparison")
    print("="*60)
    
    # Load data once
    df = create_sample_data(days=500, initial_price=100)
    
    configurations = [
        {
            'name': 'Range Trading Only',
            'params': {
                'use_range_trading': True,
                'use_breakout_trading': False,
                'range_threshold': 0.001
            }
        },
        {
            'name': 'Breakout Trading Only', 
            'params': {
                'use_range_trading': False,
                'use_breakout_trading': True,
                'breakout_confirmation': 1
            }
        },
        {
            'name': 'Combined (Default)',
            'params': {}  # Use all defaults
        },
        {
            'name': 'Aggressive',
            'params': {
                'range_threshold': 0.002,  # Wider range
                'stop_loss_buffer': 0.0005,  # Tighter stops
                'take_profit_ratio': 3.0  # Higher targets
            }
        },
        {
            'name': 'Conservative',
            'params': {
                'range_threshold': 0.0005,  # Tighter range
                'stop_loss_buffer': 0.002,  # Wider stops
                'take_profit_ratio': 1.5  # Lower targets
            }
        }
    ]
    
    results = []
    
    for config in configurations:
        bt = Backtest(df, CamarillaStrategy, cash=10000, commission=0.002)
        stats = bt.run(**config['params'])
        
        results.append({
            'Strategy': config['name'],
            'Return %': stats['Return [%]'],
            'Sharpe': stats['Sharpe Ratio'],
            'Trades': stats['# Trades'],
            'Win Rate %': stats['Win Rate [%]'],
            'Max DD %': stats['Max. Drawdown [%]']
        })
        
        print(f"\n{config['name']}:")
        print(f"  Return: {stats['Return [%]']:>8.2f}%")
        print(f"  Sharpe: {stats['Sharpe Ratio']:>8.2f}")
        print(f"  Trades: {stats['# Trades']:>8.0f}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    print("\n\n📊 Full Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best by different metrics
    print(f"\n🏆 Best by Return: {comparison_df.loc[comparison_df['Return %'].idxmax(), 'Strategy']}")
    print(f"🏆 Best by Sharpe: {comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Strategy']}")
    print(f"🏆 Most Active: {comparison_df.loc[comparison_df['Trades'].idxmax(), 'Strategy']}")

if __name__ == "__main__":
    # Run simple backtest
    stats = run_simple_backtest()
    
    # Run strategy comparison
    compare_strategies()
    
    print("\n\n✅ Backtest complete!")
    print("\n💡 Next steps:")
    print("1. Load your own data (CSV with Date, Open, High, Low, Close, Volume)")
    print("2. Adjust strategy parameters in camarilla_strategy.py")
    print("3. Run optimization to find best parameters for your data")
    print("4. Use camarilla_live_trading.py for real-time signals")