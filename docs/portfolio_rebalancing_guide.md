# Portfolio Rebalancing Agent - User Guide

## Overview

The Portfolio Rebalancing Agent enables sophisticated portfolio management across multiple Moon Dev trading strategies. It automatically monitors allocations, detects drift, and executes rebalancing trades to maintain optimal portfolio composition.

## Quick Start

### 1. Create a Portfolio

```bash
python src/utils/portfolio_builder.py
```

Follow the interactive prompts to:
- Name your portfolio
- Select strategies from the marketplace
- Set target allocations
- Choose rebalancing method
- Configure risk limits

### 2. Run Portfolio Agent

```python
from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.models.portfolio_models import PortfolioConfig, RebalancingMethod

# Create configuration
config = PortfolioConfig(
    name="My Balanced Portfolio",
    target_allocations={
        "rsi_strategy": 0.40,
        "macd_strategy": 0.30,
        "ml_strategy": 0.30
    },
    rebalancing_method=RebalancingMethod.THRESHOLD,
    rebalancing_params={"min_drift": 0.05},
    risk_limits=DEFAULT_RISK_LIMITS
)

# Initialize agent
agent = PortfolioRebalancingAgent(config)

# Check and rebalance
result = agent.check_and_rebalance()
```

### 3. View Portfolio Status

```python
# Display comprehensive dashboard
agent.display_portfolio_dashboard()
```

## Rebalancing Methods

### 1. Threshold Rebalancing
Triggers when any position drifts beyond configured threshold.

```python
rebalancing_params = {
    "min_drift": 0.05,        # 5% drift triggers rebalancing
    "min_trade_size": 100,    # Minimum $100 trade
    "check_frequency": "daily"
}
```

### 2. Calendar Rebalancing
Rebalances on fixed schedule regardless of drift.

```python
rebalancing_params = {
    "frequency": "monthly",   # Options: daily, weekly, monthly, quarterly
    "rebalance_day": 1,      # Day of month/week
    "force_rebalance": True  # Rebalance even without drift
}
```

### 3. Adaptive Rebalancing
Dynamically adjusts weights based on strategy performance.

```python
rebalancing_params = {
    "volatility_threshold": 0.3,   # High volatility threshold
    "correlation_threshold": 0.8,  # High correlation threshold
    "performance_window": 30,      # Days for performance calc
    "min_improvement": 0.02        # 2% improvement needed
}
```

### 4. Risk Parity
Equalizes risk contribution from each strategy.

```python
rebalancing_params = {
    "target_risk_contribution": None,  # Equal if None
    "use_leverage": False,
    "rebalance_frequency": "weekly"
}
```

## Risk Management

### Default Risk Limits

```python
DEFAULT_RISK_LIMITS = {
    "max_single_strategy": 0.4,      # 40% max allocation
    "min_single_strategy": 0.05,     # 5% minimum
    "max_correlation_pair": 0.7,     # Max correlation between strategies
    "max_avg_correlation": 0.5,      # Average correlation limit
    "min_strategy_sharpe": 0.0,      # Remove negative Sharpe strategies
    "max_strategy_drawdown": 0.3,    # Remove if drawdown > 30%
    "max_portfolio_leverage": 1.0,   # No leverage
    "min_portfolio_sharpe": 0.5,     # Target portfolio Sharpe
    "max_portfolio_drawdown": 0.25   # 25% max portfolio drawdown
}
```

### Risk Validation

The agent validates all proposed allocations against risk limits:
- Single strategy concentration
- Portfolio-level metrics
- Correlation constraints
- Performance thresholds

## Integration with Moon Dev Ecosystem

### Strategy Marketplace Integration

```python
# Automatically pulls performance data from registered strategies
registry = StrategyRegistryAgent()
strategy_data = registry.get_strategy("strategy_id")
performance = strategy_data["performance_data"]
```

### Trading Agent Integration

```python
from src.agents.portfolio_integration import PortfolioIntegration

# Execute rebalancing orders
integrator = PortfolioIntegration()
orders = agent.rebalancing_history[-1].orders
results = integrator.execute_portfolio_orders(orders)
```

### Main Loop Integration

Add to `main.py`:

```python
ACTIVE_AGENTS = {
    'portfolio': True,  # Enable portfolio rebalancing
    # ... other agents
}

# In run loop
if ACTIVE_AGENTS['portfolio']:
    portfolio_agent.check_and_rebalance()
```

## Data Models

### PortfolioConfig
Core configuration for portfolio management.

### StrategyPosition
Current position and drift tracking for each strategy.

### RebalancingEvent
Complete record of each rebalancing action.

### Order
Individual buy/sell orders generated during rebalancing.

## Monitoring & Analytics

### Portfolio Metrics
- Total value and returns
- Sharpe ratio
- Maximum drawdown
- Volatility
- Strategy correlations
- Days since rebalance

### Performance Tracking
All rebalancing events are saved to:
```
src/data/portfolio/rebalance_YYYYMMDD_HHMMSS.json
```

### Dashboard Display
- Current allocations vs targets
- Drift visualization
- Performance metrics
- Risk indicators

## Example Portfolios

### Conservative Balanced
```python
{
    "btc_hodl": 0.40,
    "eth_stake": 0.30,
    "stable_yield": 0.30
}
```

### Aggressive Growth
```python
{
    "momentum_long": 0.35,
    "breakout_trades": 0.35,
    "alt_rotation": 0.30
}
```

### Market Neutral
```python
{
    "long_short_pairs": 0.40,
    "arbitrage": 0.30,
    "market_making": 0.30
}
```

## Best Practices

1. **Start Conservative**: Begin with higher thresholds and adjust based on experience
2. **Monitor Correlations**: High correlations reduce diversification benefits
3. **Regular Reviews**: Check strategy performance trends monthly
4. **Cost Awareness**: Consider transaction costs in rebalancing frequency
5. **Risk First**: Always prioritize risk limits over return optimization

## Troubleshooting

### No Rebalancing Triggered
- Check drift thresholds
- Verify calendar settings
- Ensure strategies have current data

### Risk Limits Violated
- Review allocation targets
- Check correlation matrix
- Adjust risk parameters

### Integration Issues
- Verify strategy registry connection
- Check trading agent status
- Review error logs in data/portfolio/

## Advanced Usage

### Custom Rebalancing Logic
Extend `RebalancingEngine` class:

```python
class CustomRebalancer(RebalancingEngine):
    def calculate_target_weights(self, ...):
        # Your custom logic
        pass
```

### Performance Attribution
Analyze contribution of each strategy:

```python
attribution = agent.calculate_performance_attribution()
```

### Backtesting Portfolios
Test historical performance:

```python
backtest_results = agent.backtest_portfolio(
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## Support

For issues or questions:
1. Check logs in `src/data/portfolio/`
2. Review strategy performance in marketplace
3. Join Moon Dev Discord
4. Watch YouTube tutorials

## Next Steps

1. Create your first portfolio
2. Run demo: `python src/scripts/demo_portfolio_rebalancing.py`
3. Monitor live performance
4. Optimize based on results
5. Share successful portfolios with community