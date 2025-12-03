"""
🌙 Clean Camarilla Backtest Runner
A working example without errors

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy

def create_realistic_data(days=252, trend=0.0005, volatility=0.02):
    """Create realistic market data with customizable trend and volatility"""
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with trend and volatility
    np.random.seed(42)
    returns = np.random.normal(trend, volatility, len(dates))
    
    # Start at a reasonable price
    initial_price = 100
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    
    # Generate realistic OHLC relationships
    daily_range = np.random.uniform(0.005, 0.02, len(dates))  # 0.5% to 2% daily range
    
    # Open can gap from previous close
    gaps = np.random.normal(0, 0.003, len(dates))  # Small overnight gaps
    df['Open'] = df['Close'].shift(1) * (1 + gaps)
    df.loc[df.index[0], 'Open'] = df.loc[df.index[0], 'Close']
    
    # High and Low based on range
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + daily_range * 0.6)
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - daily_range * 0.6)
    
    # Volume (log-normal distribution)
    df['Volume'] = np.random.lognormal(15, 0.5, len(dates))
    
    return df

def run_backtest_safely(df, strategy_class=CamarillaStrategy, **params):
    """Run backtest with error handling"""
    
    try:
        # Determine appropriate initial cash
        max_price = df['Close'].max()
        initial_cash = max(10000, max_price * 100)  # At least 100x max price
        
        # Initialize backtest
        bt = Backtest(
            df, 
            strategy_class,
            cash=initial_cash,
            commission=0.002,  # 0.2% commission
            exclusive_orders=True
        )
        
        # Run with parameters
        stats = bt.run(**params)
        
        # Display results safely
        print("\n📊 Results:")
        print("-" * 40)
        
        # Always available metrics
        safe_metrics = [
            ('Total Return', 'Return [%]'),
            ('Buy & Hold', 'Buy & Hold Return [%]'),
            ('Max Drawdown', 'Max. Drawdown [%]'),
            ('# Trades', '# Trades'),
            ('Win Rate', 'Win Rate [%]'),
            ('Exposure Time', 'Exposure Time [%]')
        ]
        
        for label, key in safe_metrics:
            if key in stats:
                if '%' in key and key != '# Trades':
                    print(f"{label:.<20} {stats[key]:>10.2f}%")
                else:
                    print(f"{label:.<20} {stats[key]:>10.0f}")
        
        # Optional metrics (might not be in all versions)
        optional_metrics = [
            ('Sharpe Ratio', 'Sharpe Ratio'),
            ('Profit Factor', 'Profit Factor'),
            ('Avg Trade', 'Avg. Trade [%]')
        ]
        
        for label, key in optional_metrics:
            if key in stats:
                if '%' in key:
                    print(f"{label:.<20} {stats[key]:>10.2f}%")
                else:
                    print(f"{label:.<20} {stats[key]:>10.2f}")
        
        return bt, stats
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        return None, None

def main():
    print("🌙 CAMARILLA STRATEGY - CLEAN BACKTEST")
    print("=" * 60)
    
    # Test different market conditions
    market_conditions = [
        {
            'name': 'Ranging Market',
            'trend': 0.0000,  # No trend
            'volatility': 0.015  # 1.5% daily vol
        },
        {
            'name': 'Bull Market',
            'trend': 0.0010,  # 0.1% daily drift
            'volatility': 0.020  # 2% daily vol
        },
        {
            'name': 'Bear Market',
            'trend': -0.0005,  # -0.05% daily drift
            'volatility': 0.025  # 2.5% daily vol
        },
        {
            'name': 'Low Volatility',
            'trend': 0.0002,
            'volatility': 0.008  # 0.8% daily vol
        }
    ]
    
    results_summary = []
    
    for condition in market_conditions:
        print(f"\n🔍 Testing: {condition['name']}")
        print("-" * 40)
        
        # Create data for this condition
        df = create_realistic_data(
            days=252,  # 1 year
            trend=condition['trend'],
            volatility=condition['volatility']
        )
        
        print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        # Run default strategy
        bt, stats = run_backtest_safely(df)
        
        if stats is not None:
            results_summary.append({
                'Market': condition['name'],
                'Return': stats.get('Return [%]', 0),
                'Sharpe': stats.get('Sharpe Ratio', 0),
                'Trades': stats.get('# Trades', 0),
                'Win Rate': stats.get('Win Rate [%]', 0)
            })
    
    # Summary comparison
    if results_summary:
        print("\n\n📊 MARKET CONDITIONS COMPARISON")
        print("=" * 60)
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        # Best performing conditions
        print(f"\n🏆 Best Return: {summary_df.loc[summary_df['Return'].idxmax(), 'Market']}")
        print(f"🏆 Best Sharpe: {summary_df.loc[summary_df['Sharpe'].idxmax(), 'Market']}")
        print(f"🏆 Highest Win Rate: {summary_df.loc[summary_df['Win Rate'].idxmax(), 'Market']}")
    
    # Example of parameter optimization
    print("\n\n🔧 PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Use ranging market for optimization
    df = create_realistic_data(days=252, trend=0, volatility=0.015)
    
    print("Testing different parameter combinations...")
    
    param_tests = [
        {'name': 'Default', 'params': {}},
        {'name': 'Conservative', 'params': {
            'range_threshold': 0.0005,
            'stop_loss_buffer': 0.002,
            'take_profit_ratio': 1.5
        }},
        {'name': 'Aggressive', 'params': {
            'range_threshold': 0.002,
            'stop_loss_buffer': 0.0005,
            'take_profit_ratio': 3.0
        }},
        {'name': 'Range Only', 'params': {
            'use_range_trading': True,
            'use_breakout_trading': False
        }},
        {'name': 'Breakout Only', 'params': {
            'use_range_trading': False,
            'use_breakout_trading': True
        }}
    ]
    
    for test in param_tests:
        print(f"\n{test['name']}:")
        bt, stats = run_backtest_safely(df, **test['params'])
        if stats is not None:
            print(f"  Return: {stats.get('Return [%]', 0):.2f}%")
            print(f"  Trades: {stats.get('# Trades', 0):.0f}")
    
    print("\n\n✅ Backtest complete!")
    print("\n📝 Next steps:")
    print("1. Replace create_realistic_data() with your own data loader")
    print("2. Adjust parameters based on your market's characteristics")
    print("3. Run on longer time periods for more reliable results")
    print("4. Always validate with out-of-sample data")

if __name__ == "__main__":
    main()