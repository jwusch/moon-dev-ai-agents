"""
🎯 Interactive Ensemble Strategy Builder CLI
Command-line interface for building ensemble trading strategies from top alpha sources

Usage:
python ensemble_builder_cli.py --symbol NVDA --top-n 5
python ensemble_builder_cli.py --interactive

Author: Claude (Anthropic)
"""

import argparse
import json
import os
from datetime import datetime
from ensemble_alpha_strategy import EnsembleAlphaStrategy

def interactive_mode():
    """Interactive mode for building ensemble strategies"""
    
    print("🎯 INTERACTIVE ENSEMBLE STRATEGY BUILDER")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Build ensemble strategy for a symbol")
        print("2. Load and display existing strategy")
        print("3. Compare multiple strategies")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            build_ensemble_interactive()
        elif choice == "2":
            load_strategy_interactive()
        elif choice == "3":
            compare_strategies_interactive()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

def build_ensemble_interactive():
    """Interactive ensemble building"""
    
    print("\n📊 BUILD ENSEMBLE STRATEGY")
    print("-" * 40)
    
    # Get symbol
    while True:
        symbol = input("Enter symbol (e.g., NVDA, AAPL, SPY): ").strip().upper()
        if symbol:
            break
        print("❌ Please enter a valid symbol")
    
    # Get number of signals
    while True:
        try:
            top_n = int(input("Number of top alpha sources to combine (1-10, default 5): ") or "5")
            if 1 <= top_n <= 10:
                break
            print("❌ Please enter a number between 1 and 10")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get timeframes
    print("\nAvailable timeframes: 1m, 5m, 15m, 1h, 1d")
    timeframes_input = input("Timeframes to analyze (comma-separated, default: 5m,15m,1h,1d): ").strip()
    if timeframes_input:
        timeframes = [tf.strip() for tf in timeframes_input.split(",")]
    else:
        timeframes = ["5m", "15m", "1h", "1d"]
    
    # Get data period
    period = input("Data period (7d, 30d, 60d, 1y, default: 60d): ").strip() or "60d"
    
    # Build strategy
    print(f"\n🚀 Building ensemble strategy for {symbol}...")
    build_ensemble_strategy(symbol, top_n, timeframes, period)

def build_ensemble_strategy(symbol: str, top_n: int, timeframes: list, period: str):
    """Build ensemble strategy with given parameters"""
    
    try:
        # Create ensemble strategy
        ensemble = EnsembleAlphaStrategy(symbol, top_n_signals=top_n)
        
        # Discover alpha sources
        print(f"\n🔍 Discovering alpha sources...")
        alpha_sources = ensemble.discover_alpha_sources(timeframes)
        
        if not alpha_sources:
            print(f"❌ No alpha sources found for {symbol} with current settings")
            print("💡 Try:")
            print("   - Different timeframes")
            print("   - Lower alpha threshold in scanner")
            print("   - Different symbol")
            return None
        
        # Prepare data and generate signals
        print(f"\n📊 Preparing ensemble data...")
        df = ensemble.prepare_ensemble_data(period=period)
        
        if len(df) < 100:
            print(f"❌ Insufficient data for {symbol} (got {len(df)} bars, need 100+)")
            return None
        
        # Generate ensemble signals
        print(f"\n🎯 Generating ensemble signals...")
        df = ensemble.generate_ensemble_signals(df)
        
        # Backtest strategy
        print(f"\n📈 Backtesting ensemble strategy...")
        result = ensemble.backtest_ensemble_strategy(df)
        
        if result is None:
            print(f"❌ No trades generated - strategy may need adjustment")
            return None
        
        # Create visualization
        print(f"\n📊 Creating visualization...")
        fig = ensemble.create_strategy_visualization(df, result)
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save chart
        chart_filename = f'{symbol}_ensemble_{timestamp}.png'
        fig.savefig(chart_filename, dpi=300, bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        print(f"✅ Chart saved: {chart_filename}")
        
        # Save strategy data
        strategy_data = {
            'symbol': symbol,
            'parameters': {
                'top_n_signals': top_n,
                'timeframes': timeframes,
                'period': period
            },
            'alpha_sources': [
                {
                    'name': source.strategy_name,
                    'timeframe': source.timeframe,
                    'alpha_score': source.alpha_score,
                    'win_rate': source.win_rate,
                    'confidence': source.confidence_score
                }
                for source in alpha_sources
            ],
            'weights': ensemble.weights,
            'performance': {
                'total_return_pct': result.total_return_pct,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'alpha_score': result.alpha_score,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'profit_factor': result.profit_factor
            },
            'created_at': datetime.now().isoformat()
        }
        
        strategy_filename = f'{symbol}_ensemble_strategy_{timestamp}.json'
        with open(strategy_filename, 'w') as f:
            json.dump(strategy_data, f, indent=2, default=str)
        print(f"✅ Strategy saved: {strategy_filename}")
        
        # Display summary
        print_strategy_summary(strategy_data)
        
        return strategy_data
        
    except Exception as e:
        print(f"❌ Error building ensemble strategy: {e}")
        return None

def load_strategy_interactive():
    """Load and display existing strategy"""
    
    print("\n📂 LOAD EXISTING STRATEGY")
    print("-" * 40)
    
    # Find strategy files
    strategy_files = [f for f in os.listdir('.') if f.endswith('_ensemble_strategy.json')]
    
    if not strategy_files:
        print("❌ No strategy files found")
        return
    
    print("Available strategies:")
    for i, filename in enumerate(strategy_files, 1):
        print(f"  {i}. {filename}")
    
    try:
        choice = int(input(f"\nSelect strategy (1-{len(strategy_files)}): ")) - 1
        if 0 <= choice < len(strategy_files):
            filename = strategy_files[choice]
            with open(filename, 'r') as f:
                strategy_data = json.load(f)
            
            print(f"\n📊 STRATEGY: {filename}")
            print_strategy_summary(strategy_data)
        else:
            print("❌ Invalid selection")
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error loading strategy: {e}")

def compare_strategies_interactive():
    """Compare multiple strategies"""
    
    print("\n⚖️ COMPARE STRATEGIES")
    print("-" * 40)
    
    # Find strategy files
    strategy_files = [f for f in os.listdir('.') if f.endswith('_ensemble_strategy.json')]
    
    if len(strategy_files) < 2:
        print("❌ Need at least 2 strategy files for comparison")
        return
    
    print("Available strategies:")
    for i, filename in enumerate(strategy_files, 1):
        print(f"  {i}. {filename}")
    
    print("\nSelect strategies to compare (comma-separated numbers):")
    try:
        choices = input("Selection: ").strip().split(',')
        selected_strategies = []
        
        for choice in choices:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(strategy_files):
                filename = strategy_files[idx]
                with open(filename, 'r') as f:
                    strategy_data = json.load(f)
                selected_strategies.append((filename, strategy_data))
        
        if len(selected_strategies) < 2:
            print("❌ Please select at least 2 strategies")
            return
        
        # Display comparison
        print(f"\n📊 STRATEGY COMPARISON")
        print("=" * 80)
        
        headers = ["Metric"] + [f"{s[1]['symbol']} ({s[0].split('_')[2][:8]})" for s in selected_strategies]
        print(f"{headers[0]:<20} {' '.join(f'{h:>15}' for h in headers[1:])}")
        print("-" * 80)
        
        metrics = [
            ('Total Return %', 'total_return_pct'),
            ('Win Rate %', 'win_rate'),
            ('Alpha Score', 'alpha_score'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Max Drawdown %', 'max_drawdown'),
            ('Total Trades', 'total_trades'),
            ('Profit Factor', 'profit_factor')
        ]
        
        for metric_name, metric_key in metrics:
            values = [f"{s[1]['performance'][metric_key]:>15.2f}" for s in selected_strategies]
            print(f"{metric_name:<20} {' '.join(values)}")
        
        # Find best performer
        best_return = max(selected_strategies, key=lambda x: x[1]['performance']['total_return_pct'])
        best_alpha = max(selected_strategies, key=lambda x: x[1]['performance']['alpha_score'])
        best_sharpe = max(selected_strategies, key=lambda x: x[1]['performance']['sharpe_ratio'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Highest Return: {best_return[1]['symbol']} ({best_return[1]['performance']['total_return_pct']:.1f}%)")
        print(f"   Highest Alpha:  {best_alpha[1]['symbol']} ({best_alpha[1]['performance']['alpha_score']:.2f})")
        print(f"   Best Sharpe:    {best_sharpe[1]['symbol']} ({best_sharpe[1]['performance']['sharpe_ratio']:.2f})")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error comparing strategies: {e}")

def print_strategy_summary(strategy_data):
    """Print formatted strategy summary"""
    
    symbol = strategy_data['symbol']
    performance = strategy_data['performance']
    alpha_sources = strategy_data['alpha_sources']
    
    print(f"\n📊 ENSEMBLE STRATEGY SUMMARY - {symbol}")
    print("=" * 60)
    
    print(f"📈 PERFORMANCE METRICS:")
    print(f"   Total Return:     {performance['total_return_pct']:+.1f}%")
    print(f"   Win Rate:         {performance['win_rate']:.1f}%")
    print(f"   Alpha Score:      {performance['alpha_score']:.2f}")
    print(f"   Sharpe Ratio:     {performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:     {performance['max_drawdown']:.1f}%")
    print(f"   Total Trades:     {performance['total_trades']}")
    print(f"   Profit Factor:    {performance['profit_factor']:.2f}")
    
    print(f"\n🎯 ALPHA SOURCES ({len(alpha_sources)}):")
    for i, source in enumerate(alpha_sources, 1):
        print(f"   {i}. {source['name']}: α={source['alpha_score']:.2f} ({source['win_rate']:.1f}% win)")
    
    print(f"\n⚙️ CONFIGURATION:")
    params = strategy_data['parameters']
    print(f"   Top N Signals:    {params['top_n_signals']}")
    print(f"   Timeframes:       {', '.join(params['timeframes'])}")
    print(f"   Data Period:      {params['period']}")
    
    # Strategy recommendation
    if performance['total_return_pct'] > 5 and performance['win_rate'] > 70:
        print(f"\n✅ STRONG STRATEGY - Consider for live trading")
    elif performance['total_return_pct'] > 2 and performance['win_rate'] > 60:
        print(f"\n⚠️  MODERATE STRATEGY - Needs optimization")
    else:
        print(f"\n❌ WEAK STRATEGY - Requires significant improvement")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Interactive Ensemble Strategy Builder")
    parser.add_argument('--symbol', type=str, help='Symbol to analyze (e.g., NVDA, AAPL)')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top alpha sources to combine (default: 5)')
    parser.add_argument('--timeframes', type=str, default='5m,15m,1h,1d', help='Comma-separated timeframes')
    parser.add_argument('--period', type=str, default='60d', help='Data period (default: 60d)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or not args.symbol:
        # Run interactive mode
        interactive_mode()
    else:
        # Run with command line arguments
        timeframes = [tf.strip() for tf in args.timeframes.split(',')]
        build_ensemble_strategy(args.symbol.upper(), args.top_n, timeframes, args.period)

if __name__ == "__main__":
    main()