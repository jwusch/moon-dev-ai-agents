"""
Demo script to populate the marketplace with sample strategies
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.strategy_registry_agent import StrategyRegistryAgent
from src.marketplace.analytics import StrategyAnalytics
from src.marketplace.exporter import StrategyExporter


def create_sample_performance_data():
    """Create sample performance data for demo"""
    # Create sample equity curve
    dates = pd.date_range(start='2024-01-01', end='2024-11-30', freq='D')
    initial_capital = 10000
    
    # Simulate a profitable strategy
    daily_returns = np.random.normal(0.002, 0.015, len(dates))  # 0.2% daily with 1.5% volatility
    equity_curve = pd.Series(
        initial_capital * (1 + daily_returns).cumprod(),
        index=dates
    )
    
    # Create sample trades
    num_trades = 50
    trade_indices = sorted(np.random.choice(len(dates)-1, num_trades, replace=False))
    
    trades = []
    for i in range(0, len(trade_indices)-1, 2):
        entry_idx = trade_indices[i]
        exit_idx = trade_indices[i+1]
        
        entry_price = equity_curve.iloc[entry_idx]
        exit_price = equity_curve.iloc[exit_idx]
        pnl = (exit_price - entry_price) / entry_price * 1000  # Assume $1000 per trade
        
        trades.append({
            'entry_time': dates[entry_idx],
            'exit_time': dates[exit_idx],
            'pnl': pnl,
            'side': 'long'
        })
    
    trades_df = pd.DataFrame(trades)
    
    return equity_curve, trades_df


def main():
    """Demo the marketplace functionality"""
    print("🌙 Moon Dev Strategy Marketplace Demo")
    print("=" * 50)
    
    # Initialize components
    registry = StrategyRegistryAgent()
    analytics = StrategyAnalytics()
    exporter = StrategyExporter(registry)
    
    # Register the sample RSI strategy
    print("\n📝 Registering sample RSI strategy...")
    
    try:
        strategy_path = os.path.join(project_root, "src", "strategies", "sample_rsi_strategy.py")
        
        metadata = registry.register_strategy(
            name="RSI Mean Reversion",
            description="A classic mean reversion strategy using RSI to identify oversold/overbought conditions",
            author="moon_dev",
            code_path=strategy_path,
            category=["mean_reversion", "technical"],
            timeframes=["15m", "1H", "4H"],
            instruments=["BTC", "ETH", "SOL"],
            min_capital=100.0,
            risk_level="low",
            dependencies=["pandas_ta"]
        )
        
        strategy_id = metadata['strategy_id']
        print(f"✅ Strategy registered with ID: {strategy_id}")
        
        # Generate and save performance data
        print("\n📊 Generating performance metrics...")
        equity_curve, trades = create_sample_performance_data()
        
        # Calculate metrics
        metrics = analytics.calculate_metrics(equity_curve, trades)
        
        # Save metrics
        analytics.save_metrics(strategy_id, metrics)
        
        # Update registry with performance summary
        registry.update_performance(strategy_id, metrics)
        
        # Generate performance report
        report = analytics.generate_performance_report(strategy_id, metrics)
        print(report)
        
        # Approve the strategy
        print("\n✅ Approving strategy for marketplace...")
        registry.approve_strategy(strategy_id)
        
        # Export strategy package
        print("\n📦 Creating strategy package...")
        export_path = os.path.join(project_root, "src", "data", "marketplace", "exports")
        os.makedirs(export_path, exist_ok=True)
        
        package_path = exporter.export_strategy_package(
            strategy_id,
            export_path,
            include_performance=True
        )
        print(f"✅ Package created: {package_path}")
        
        # Simulate some user activity
        print("\n👥 Simulating user activity...")
        
        # Add some ratings
        for _ in range(5):
            rating = np.random.uniform(4, 5)
            registry.update_rating(strategy_id, rating)
        
        # Increment downloads
        for _ in range(10):
            registry.increment_downloads(strategy_id)
        
        print("✅ Added 5 ratings and 10 downloads")
        
        # Run the registry agent to show stats
        print("\n📈 Current marketplace statistics:")
        registry.run()
        
        print("\n🎉 Demo complete!")
        print(f"\n💻 To view the marketplace dashboard, run:")
        print(f"   python {os.path.join('src', 'scripts', 'marketplace_dashboard.py')}")
        print(f"   Then open http://localhost:8002")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()