# 🌙 Moon Dev Strategy Marketplace - User Guide

## Overview

The Strategy Marketplace is a community-driven platform within the Moon Dev AI Trading ecosystem that enables users to share, discover, and analyze proven trading strategies with comprehensive performance metrics.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Browsing Strategies](#browsing-strategies)
3. [Submitting a Strategy](#submitting-a-strategy)
4. [Performance Analytics](#performance-analytics)
5. [Exporting & Importing](#exporting--importing)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)

## Getting Started

### Running the Marketplace

1. **Start the Dashboard**:
```bash
python src/scripts/marketplace_dashboard.py
```

2. **Access the Web Interface**:
Open your browser to `http://localhost:8002`

3. **Run Demo Data** (optional):
```bash
python src/scripts/demo_marketplace.py
```

### System Requirements

- Python 3.10+
- Active `tflow` conda environment
- Dependencies: `pip install flask pandas numpy`

## Browsing Strategies

### Dashboard Features

The marketplace dashboard provides:

- **Strategy Cards**: Visual overview of each strategy with key metrics
- **Search & Filters**: Find strategies by name, category, risk level
- **Performance Metrics**: View returns, Sharpe ratio, win rate, drawdown
- **Ratings**: Community ratings and download counts

### Search Options

- **Text Search**: Search by strategy name or description
- **Category Filter**: momentum, mean_reversion, technical, ml_based
- **Risk Level**: low, medium, high
- **Sort Options**: rating, return, downloads, recent

### Strategy Details

Click "View Details" on any strategy to see:
- Complete description and requirements
- Detailed performance metrics
- Author information and version history
- Download options

## Submitting a Strategy

### 1. Prepare Your Strategy

Your strategy must:
- Inherit from `BaseStrategy` class
- Implement `generate_signals()` method
- Be under 800 lines of code
- Include proper documentation

Example structure:
```python
from backtesting import Strategy

class YourStrategy(Strategy):
    # Parameters
    param1 = 20
    param2 = 50
    
    def init(self):
        # Initialize indicators
        pass
    
    def next(self):
        # Strategy logic
        pass

# Required metadata
STRATEGY_METADATA = {
    "name": "Your Strategy Name",
    "description": "What your strategy does",
    "author": "your_name",
    "category": ["momentum", "technical"],
    "timeframes": ["15m", "1H"],
    "instruments": ["BTC", "ETH"],
    "min_capital": 100.0,
    "risk_level": "medium",
    "dependencies": ["pandas_ta"]
}
```

### 2. Register Your Strategy

Using Python:
```python
from src.agents.strategy_registry_agent import StrategyRegistryAgent

registry = StrategyRegistryAgent()

metadata = registry.register_strategy(
    name="Your Strategy Name",
    description="Detailed description",
    author="your_username",
    code_path="path/to/your_strategy.py",
    category=["momentum", "technical"],
    timeframes=["15m", "1H"],
    instruments=["BTC", "ETH"],
    min_capital=100.0,
    risk_level="medium"
)
```

### 3. Add Performance Data

After backtesting:
```python
from src.marketplace.analytics import StrategyAnalytics

analytics = StrategyAnalytics()

# From your backtest results
metrics = analytics.calculate_metrics(
    equity_curve=your_equity_series,
    trades=your_trades_df
)

# Update registry
registry.update_performance(strategy_id, metrics)
```

## Performance Analytics

### Key Metrics Tracked

**Return Metrics:**
- Total Return %
- Annual Return %
- Monthly Return %

**Risk Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility

**Trade Statistics:**
- Win Rate
- Profit Factor
- Average Trade
- Total Trades

**Efficiency Metrics:**
- Time in Market %
- Risk/Reward Ratio
- Calmar Ratio
- Expectancy

### Performance Categories

Strategies are categorized based on performance:

**Return Tiers:**
- Excellent: >100% return
- Good: 50-100% return
- Moderate: 20-50% return
- Low: 0-20% return
- Negative: <0% return

**Risk-Adjusted Performance:**
- Excellent: Sharpe > 2.0
- Good: Sharpe 1.0-2.0
- Moderate: Sharpe 0.5-1.0
- Poor: Sharpe < 0.5

## Exporting & Importing

### Export a Strategy Package

```python
from src.marketplace.exporter import StrategyExporter

exporter = StrategyExporter(registry)

package_path = exporter.export_strategy_package(
    strategy_id="your_strategy_id",
    output_path="/path/to/save/",
    include_performance=True,
    include_backtest_data=False
)
```

Package includes:
- Strategy code
- Metadata JSON
- Performance metrics
- README with instructions
- Requirements file
- Installation script

### Import a Strategy Package

```python
result = exporter.import_strategy_package(
    package_path="strategy_package.zip",
    validate=True
)

print(f"Imported strategy ID: {result['strategy_id']}")
```

## API Reference

### Registry Agent API

```python
# Search strategies
results = registry.search_strategies(
    query="RSI",
    category="mean_reversion",
    min_rating=4.0,
    risk_level="low"
)

# Get specific strategy
strategy = registry.get_strategy(strategy_id)

# Update rating
registry.update_rating(strategy_id, rating=4.5)

# Mark as deprecated
registry.deprecate_strategy(strategy_id, reason="Outdated")
```

### Analytics API

```python
# Calculate metrics
metrics = analytics.calculate_metrics(
    equity_curve=equity_series,
    trades=trades_df,
    initial_capital=10000,
    risk_free_rate=0.02
)

# Compare strategies
comparison = analytics.compare_strategies([
    "strategy_id_1",
    "strategy_id_2",
    "strategy_id_3"
])

# Generate report
report = analytics.generate_performance_report(
    strategy_id,
    metrics
)
```

### Web API Endpoints

- `GET /api/stats` - Marketplace statistics
- `GET /api/strategies` - List all strategies
- `GET /api/strategies/search` - Search with filters
- `GET /api/strategies/{id}` - Get strategy details
- `GET /api/strategies/{id}/download` - Download package
- `POST /api/strategies/{id}/rate` - Rate strategy

## Best Practices

### For Strategy Authors

1. **Clear Documentation**
   - Explain the strategy logic clearly
   - List all parameters with descriptions
   - Provide example use cases

2. **Comprehensive Testing**
   - Test on multiple timeframes
   - Use different market conditions
   - Include transaction costs

3. **Risk Management**
   - Clearly state risk level
   - Include stop loss/take profit logic
   - Document maximum drawdown

4. **Version Control**
   - Use semantic versioning
   - Document changes between versions
   - Maintain backwards compatibility

### For Strategy Users

1. **Due Diligence**
   - Review performance metrics carefully
   - Check community ratings
   - Test with small capital first

2. **Understand the Strategy**
   - Read the full documentation
   - Understand entry/exit logic
   - Know the risk parameters

3. **Proper Testing**
   - Backtest on your own data
   - Paper trade before going live
   - Start with minimum capital

4. **Risk Management**
   - Never risk more than you can afford
   - Use proper position sizing
   - Monitor drawdowns closely

## Troubleshooting

### Common Issues

**Strategy Registration Fails:**
- Ensure strategy inherits from BaseStrategy
- Check for forbidden imports (os, subprocess, eval)
- Verify all required metadata fields

**Performance Metrics Missing:**
- Ensure equity curve is a pandas Series with datetime index
- Trades DataFrame must have required columns
- Check for NaN values in data

**Import Errors:**
- Verify package structure is correct
- Check manifest.json exists
- Ensure all dependencies are listed

### Getting Help

- Discord: https://discord.gg/8UPuVZ53bh
- GitHub Issues: Report bugs or request features
- YouTube: Watch tutorials on Moon Dev channel

## Future Enhancements

Planned features include:
- GitHub integration for version control
- Automated strategy discovery
- Machine learning optimization
- Social trading features
- Mobile app support

---

Built with ❤️ by Moon Dev - Democratizing AI-powered trading