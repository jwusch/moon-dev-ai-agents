"""
🌙 Camarilla Strategy Backtesting Script
Test the Camarilla pivot strategy on historical data

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from backtesting import Backtest
from camarilla_strategy import CamarillaStrategy, CamarillaLevels
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def load_sample_data(symbol='BTC-USD', timeframe='15m'):
    """Load sample OHLCV data"""
    # Try multiple possible locations
    possible_paths = [
        f'../data/rbi/{symbol}-{timeframe}.csv',
        f'../../src/data/rbi/{symbol}-{timeframe}.csv',
        f'/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/{symbol}-{timeframe}.csv',
        f'/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/{symbol}-{timeframe}.csv'
    ]
    
    for file_path in possible_paths:
        try:
            df = pd.read_csv(file_path)
            # Ensure proper column names
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            print(f"Loaded data from: {file_path}")
            return df
        except:
            continue
    
    # Generate sample data if file not found
    print(f"Warning: Could not load data files, generating sample data...")
    return generate_sample_data()


def generate_sample_data(days=365):
    """Generate realistic sample OHLCV data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days*96, freq='15min')  # 96 bars per day
    
    # Generate realistic price movements
    returns = np.random.normal(0.0002, 0.015, len(dates))  # Small positive drift, 1.5% volatility
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    
    # Add intraday volatility
    intraday_vol = np.abs(np.random.normal(0, 0.003, len(dates)))
    
    df['Close'] = price
    df['Open'] = price * (1 + np.random.normal(0, 0.002, len(dates)))
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + intraday_vol)
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - intraday_vol)
    df['Volume'] = np.random.lognormal(10, 1, len(dates))
    
    return df


def analyze_camarilla_levels(df):
    """Analyze and visualize Camarilla levels"""
    # Calculate levels
    levels_df = CamarillaLevels.calculate_series(df.copy())
    
    # Plot last 30 days
    recent = levels_df.tail(30*96)  # 30 days of 15-min bars
    
    plt.figure(figsize=(15, 10))
    
    # Plot price
    plt.plot(recent.index, recent['Close'], 'k-', label='Close', linewidth=2)
    
    # Plot Camarilla levels
    plt.plot(recent.index, recent['R4'], 'r--', alpha=0.8, label='R4 (Breakout)')
    plt.plot(recent.index, recent['R3'], 'r-', alpha=0.7, label='R3 (Resistance)')
    plt.plot(recent.index, recent['R2'], 'r:', alpha=0.5, label='R2')
    plt.plot(recent.index, recent['R1'], 'r:', alpha=0.3, label='R1')
    
    plt.plot(recent.index, recent['Pivot'], 'b-', alpha=0.5, label='Pivot')
    
    plt.plot(recent.index, recent['S1'], 'g:', alpha=0.3, label='S1')
    plt.plot(recent.index, recent['S2'], 'g:', alpha=0.5, label='S2')
    plt.plot(recent.index, recent['S3'], 'g-', alpha=0.7, label='S3 (Support)')
    plt.plot(recent.index, recent['S4'], 'g--', alpha=0.8, label='S4 (Breakout)')
    
    plt.fill_between(recent.index, recent['S3'], recent['R3'], alpha=0.1, color='gray', label='Range Zone')
    
    plt.title('Camarilla Levels Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('camarilla_levels_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved as: camarilla_levels_plot.png")
    
    return levels_df


def run_backtest(df, strategy_params=None):
    """Run backtest with given parameters"""
    if strategy_params is None:
        strategy_params = {
            'range_threshold': 0.001,  # 0.1% proximity
            'breakout_confirmation': 1,
            'stop_loss_buffer': 0.001,
            'take_profit_ratio': 2.0,
            'use_range_trading': True,
            'use_breakout_trading': True
        }
    
    # Determine appropriate cash based on asset price
    max_price = df['Close'].max()
    initial_cash = max(100000, max_price * 10)  # At least 10x highest price
    
    # Run backtest
    bt = Backtest(df, CamarillaStrategy, 
                  cash=initial_cash,
                  commission=0.001,  # 0.1% commission
                  exclusive_orders=True)
    
    results = bt.run(**strategy_params)
    
    return bt, results


def optimize_strategy(df):
    """Optimize strategy parameters"""
    # Determine appropriate cash based on asset price
    max_price = df['Close'].max()
    initial_cash = max(100000, max_price * 10)
    
    bt = Backtest(df, CamarillaStrategy,
                  cash=initial_cash,
                  commission=0.001,
                  exclusive_orders=True)
    
    # Define parameter ranges
    params = dict(
        range_threshold=[0.0005, 0.001, 0.002],  # 0.05%, 0.1%, 0.2%
        breakout_confirmation=[0, 1, 2],
        stop_loss_buffer=[0.0005, 0.001, 0.002],
        take_profit_ratio=[1.5, 2.0, 2.5, 3.0],
        use_range_trading=[True],
        use_breakout_trading=[True]
    )
    
    # Run optimization
    results = bt.optimize(**params, maximize='Sharpe Ratio')
    
    return results


def analyze_results(results):
    """Print detailed analysis of backtest results"""
    print("\n" + "="*60)
    print("🌙 CAMARILLA STRATEGY BACKTEST RESULTS")
    print("="*60)
    
    # Key metrics
    print(f"\n📊 Performance Metrics:")
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {results['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {results['Profit Factor']:.2f}")
    
    print(f"\n📈 Trade Statistics:")
    print(f"Total Trades: {results['# Trades']}")
    print(f"Avg Trade Duration: {results['Avg. Trade Duration']}")
    print(f"Best Trade: {results['Best Trade [%]']:.2f}%")
    print(f"Worst Trade: {results['Worst Trade [%]']:.2f}%")
    
    print(f"\n💰 Risk Metrics:")
    print(f"Exposure Time: {results['Exposure Time [%]']:.2f}%")
    # Some versions of backtesting.py don't have all metrics
    if 'Avg. Exposure [%]' in results:
        print(f"Avg Exposure: {results['Avg. Exposure [%]']:.2f}%")
    if 'Max. Exposure [%]' in results:
        print(f"Max Exposure: {results['Max. Exposure [%]']:.2f}%")
    
    # Strategy parameters used
    print(f"\n⚙️ Strategy Parameters:")
    for param in ['range_threshold', 'breakout_confirmation', 'stop_loss_buffer', 
                  'take_profit_ratio', 'use_range_trading', 'use_breakout_trading']:
        if param in results._strategy_params:
            print(f"  {param}: {results._strategy_params[param]}")


def main():
    """Main execution function"""
    print("🌙 Camarilla Strategy Backtesting System")
    print("="*60)
    
    # Load data
    print("\n📊 Loading market data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} bars of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Analyze levels
    print("\n📈 Analyzing Camarilla levels...")
    levels_df = analyze_camarilla_levels(df)
    
    # Run default backtest
    print("\n🚀 Running backtest with default parameters...")
    bt, results = run_backtest(df)
    analyze_results(results)
    
    # Plot backtest results
    print("\n📊 Plotting backtest results...")
    # bt.plot()  # Skip interactive plot for now
    print("Skipping interactive plot in non-GUI environment")
    
    # Optimize parameters
    print("\n🔍 Optimizing strategy parameters...")
    print("This may take a few minutes...")
    optimal = optimize_strategy(df)
    
    print("\n🏆 Optimal Parameters Found:")
    analyze_results(optimal)
    
    # Test different market conditions
    print("\n🧪 Testing strategy variants...")
    
    # Range-only strategy
    print("\n1. Range-bound trading only (S3/R3 fade):")
    _, range_results = run_backtest(df, {
        'use_range_trading': True,
        'use_breakout_trading': False,
        'range_threshold': 0.001
    })
    print(f"   Return: {range_results['Return [%]']:.2f}%")
    print(f"   Sharpe: {range_results['Sharpe Ratio']:.2f}")
    print(f"   Trades: {range_results['# Trades']}")
    
    # Breakout-only strategy
    print("\n2. Breakout trading only (S4/R4 breakout):")
    _, breakout_results = run_backtest(df, {
        'use_range_trading': False,
        'use_breakout_trading': True,
        'breakout_confirmation': 1
    })
    print(f"   Return: {breakout_results['Return [%]']:.2f}%")
    print(f"   Sharpe: {breakout_results['Sharpe Ratio']:.2f}")
    print(f"   Trades: {breakout_results['# Trades']}")
    
    # Combined strategy
    print("\n3. Combined strategy (both approaches):")
    _, combined_results = run_backtest(df, {
        'use_range_trading': True,
        'use_breakout_trading': True
    })
    print(f"   Return: {combined_results['Return [%]']:.2f}%")
    print(f"   Sharpe: {combined_results['Sharpe Ratio']:.2f}")
    print(f"   Trades: {combined_results['# Trades']}")
    
    print("\n✅ Backtesting complete!")
    print("\n💡 Tips for live trading:")
    print("1. S3/R3 work best in ranging markets")
    print("2. S4/R4 breakouts signal potential trend days")
    print("3. Always use proper position sizing")
    print("4. Consider market conditions before trading")
    print("5. Backtest on your specific timeframe and instrument")


if __name__ == "__main__":
    main()