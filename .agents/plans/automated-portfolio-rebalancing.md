# Feature: automated-portfolio-rebalancing

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

The Automated Portfolio Rebalancing System is an intelligent agent that manages multiple trading strategies as a cohesive portfolio. It automatically adjusts allocations based on performance metrics, risk thresholds, and user-defined targets. The system replaces the current simulation-based prototype with production-ready code that integrates with real wallet data, executes actual trades, and provides comprehensive portfolio analytics.

## User Story

As a cryptocurrency trader using multiple AI trading strategies
I want to have my portfolio automatically rebalanced based on performance and risk metrics
So that I can maintain optimal diversification and maximize risk-adjusted returns without manual intervention

## Problem Statement

Currently, users running multiple trading strategies from the Moon Dev ecosystem lack an automated way to:
- Manage overall portfolio allocation across strategies
- Rebalance based on performance drift or risk events
- Maintain diversification and correlation limits
- Execute rebalancing trades efficiently
- Track portfolio-level performance metrics

The existing portfolio_rebalancing_agent.py uses simulated data and lacks real wallet integration, making it unsuitable for production use.

## Solution Statement

Enhance the existing portfolio rebalancing agent to:
1. Integrate with real wallet data using proven patterns from risk_agent.py
2. Track actual strategy performance through position history
3. Calculate real correlation matrices from returns data
4. Execute rebalancing trades through the trading infrastructure
5. Provide portfolio-level risk management and alerts
6. Persist state and configuration for reliability

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: High
**Primary Systems Affected**: portfolio_rebalancing_agent.py, nice_funcs.py, strategy_registry_agent.py, risk_agent.py
**Dependencies**: pandas, numpy, existing Moon Dev trading infrastructure

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/agents/portfolio_rebalancing_agent.py` (lines 1-623) - Why: Existing portfolio agent to enhance
- `src/agents/risk_agent.py` (lines 114-165, 430-466) - Why: Pattern for real wallet integration and position fetching
- `src/models/portfolio_models.py` (lines 1-190) - Why: Data models for portfolio management
- `src/nice_funcs.py` (lines 430-472, 1184-1230) - Why: Wallet holdings and balance functions
- `src/agents/strategy_registry_agent.py` (lines 254-275) - Why: Strategy metadata and performance tracking
- `src/config.py` (lines 14-25, 59-83) - Why: Risk limits and trading configuration
- `src/agents/base_agent.py` - Why: Base agent pattern to follow

### New Files to Create

- `src/agents/portfolio_performance_tracker.py` - Track real strategy performance from position history
- `src/data/portfolio/state.json` - Persist portfolio state and configuration
- `src/data/portfolio/rebalancing_history.csv` - Log all rebalancing events

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [Pandas Documentation - DataFrame Operations](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
  - Specific section: DataFrame.corr() for correlation matrices
  - Why: Required for calculating strategy correlations
- [NumPy Financial Functions](https://numpy.org/doc/stable/reference/routines.financial.html)
  - Specific section: Portfolio optimization
  - Why: For risk parity and optimization calculations
- [Moon Dev Trading Docs](internal)
  - Specific section: Trading execution patterns
  - Why: Shows proper order execution flow

### Patterns to Follow

**Naming Conventions:**
```python
# From risk_agent.py - Portfolio value pattern
def get_portfolio_value(self):
    total_value = 0.0
    # Get USDC balance first
    usdc_value = n.get_token_balance_usd(config.USDC_ADDRESS)
    total_value += usdc_value
    # Get balance of each monitored token
    for token in config.MONITORED_TOKENS:
        if token != config.USDC_ADDRESS:
            token_value = n.get_token_balance_usd(token)
            total_value += token_value
    return total_value
```

**Error Handling:**
```python
# From risk_agent.py - Graceful error handling
try:
    positions = n.fetch_wallet_holdings_og(address)
except Exception as e:
    cprint(f"❌ Error getting positions: {str(e)}", "white", "on_red")
    return False
```

**Logging Pattern:**
```python
# From portfolio_rebalancing_agent.py - Color-coded logging
cprint("🌙 Portfolio Rebalancing Check", "cyan", attrs=["bold"])
cprint("=" * 50, "blue")
cprint(f"✅ Successfully rebalanced", "green")
cprint(f"⚠️ Warning: {message}", "yellow")
cprint(f"❌ Error: {message}", "red")
```

**Other Relevant Patterns:**
- Use BaseAgent inheritance for consistency
- Store data in src/data/{agent_name}/ directory
- Use ModelFactory for AI decisions when needed
- Follow existing validation patterns for safety

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation - Real Wallet Integration

Replace simulated data with real wallet positions and values. This involves updating the PortfolioMonitor to fetch actual positions and calculate real allocations.

**Tasks:**
- Update position tracking to use real wallet data
- Implement strategy-to-token mapping for position identification
- Add real-time value calculations
- Create position history tracking

### Phase 2: Core Implementation - Performance Tracking

Implement real strategy performance tracking by analyzing position history and calculating returns, Sharpe ratios, and other metrics.

**Tasks:**
- Create performance tracker component
- Calculate returns from position changes
- Implement Sharpe ratio and drawdown calculations
- Add correlation matrix computation

### Phase 3: Integration - Order Execution

Connect the rebalancing engine to the actual trading system for executing buy/sell orders.

**Tasks:**
- Integrate with trading functions
- Implement order batching and optimization
- Add slippage and fee considerations
- Create execution confirmation logic

### Phase 4: Testing & Validation

Comprehensive testing of all rebalancing scenarios including edge cases and failure modes.

**Tasks:**
- Test threshold-based rebalancing
- Validate risk limits enforcement
- Test order execution flow
- Verify state persistence

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### UPDATE src/agents/portfolio_rebalancing_agent.py - Replace simulated wallet data

- **IMPLEMENT**: Replace _get_current_strategy_values() with real wallet integration
- **PATTERN**: Use get_portfolio_value() pattern from risk_agent.py:119-165
- **IMPORTS**: Add `from src import nice_funcs as n`
- **GOTCHA**: Handle wallet connection failures gracefully
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); print(agent._get_current_strategy_values())"`

### CREATE src/agents/portfolio_performance_tracker.py

- **IMPLEMENT**: New class to track strategy performance from position history
- **PATTERN**: Follow BaseAgent pattern from base_agent.py
- **IMPORTS**: `import pandas as pd, numpy as np, from datetime import datetime, timedelta`
- **GOTCHA**: Handle missing historical data gracefully
- **VALIDATE**: `python -c "from src.agents.portfolio_performance_tracker import PortfolioPerformanceTracker; tracker = PortfolioPerformanceTracker(); print('Created successfully')"`

### UPDATE src/agents/portfolio_rebalancing_agent.py - Add strategy-token mapping

- **IMPLEMENT**: Create mapping between strategy IDs and token addresses
- **PATTERN**: Use MONITORED_TOKENS pattern from config.py
- **IMPORTS**: `from src.config import MONITORED_TOKENS, USDC_ADDRESS`
- **GOTCHA**: Strategies may trade multiple tokens
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); print(agent.strategy_token_map)"`

### UPDATE src/agents/portfolio_rebalancing_agent.py - Real performance data

- **IMPLEMENT**: Replace _get_strategy_performance() with real calculations
- **PATTERN**: Integrate with PortfolioPerformanceTracker
- **IMPORTS**: `from src.agents.portfolio_performance_tracker import PortfolioPerformanceTracker`
- **GOTCHA**: Need at least 30 days of data for meaningful metrics
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); perf = agent._get_strategy_performance(); print(perf)"`

### UPDATE src/agents/portfolio_rebalancing_agent.py - Real order execution

- **IMPLEMENT**: Replace _execute_orders() simulation with real trades
- **PATTERN**: Use chunk_kill() and market_buy() from nice_funcs.py
- **IMPORTS**: Already imported nice_funcs
- **GOTCHA**: Execute sells before buys to free capital
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); print('Order execution ready')"`

### ADD correlation calculation to PortfolioMonitor

- **IMPLEMENT**: Update calculate_correlations() to use real returns data
- **PATTERN**: Use pandas DataFrame.corr() method
- **IMPORTS**: Already has pandas imported
- **GOTCHA**: Need sufficient data points for meaningful correlations
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioMonitor; monitor = PortfolioMonitor(); print('Correlation calculation ready')"`

### CREATE state persistence mechanism

- **IMPLEMENT**: Add save_state() and load_state() methods
- **PATTERN**: Use JSON for state storage like strategy_registry_agent.py
- **IMPORTS**: `import json`
- **GOTCHA**: Handle file corruption gracefully
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); agent.save_state(); print('State saved')"`

### UPDATE risk validation logic

- **IMPLEMENT**: Enhance validate_allocation() with real risk checks
- **PATTERN**: Follow risk limits from config.py
- **IMPORTS**: `from src.config import MAX_LOSS_USD, MAX_GAIN_USD`
- **GOTCHA**: Consider both portfolio and individual strategy limits
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRiskManager; rm = PortfolioRiskManager(); print(rm.risk_limits)"`

### ADD monitoring and alerting

- **IMPLEMENT**: Create alert system for rebalancing events and risks
- **PATTERN**: Use cprint patterns from existing agents
- **IMPORTS**: `from termcolor import cprint`
- **GOTCHA**: Don't spam alerts, use cooldown periods
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); agent.check_alerts(); print('Alerts configured')"`

### UPDATE main execution loop

- **IMPLEMENT**: Enhance run() method for production use
- **PATTERN**: Follow main loop pattern from risk_agent.py
- **IMPORTS**: `import time`
- **GOTCHA**: Handle keyboard interrupts gracefully
- **VALIDATE**: `python src/agents/portfolio_rebalancing_agent.py`

---

## TESTING STRATEGY

Based on the project's testing patterns discovered during research.

### Unit Tests

Design unit tests with fixtures and assertions following existing testing approaches:

```python
def test_real_wallet_integration():
    """Test fetching real wallet positions"""
    agent = PortfolioRebalancingAgent(test_config)
    positions = agent._get_current_strategy_values()
    assert isinstance(positions, dict)
    assert sum(positions.values()) > 0

def test_rebalancing_triggers():
    """Test various rebalancing trigger scenarios"""
    # Test drift trigger
    # Test calendar trigger
    # Test risk trigger

def test_order_generation():
    """Test rebalancing order generation"""
    # Test buy/sell order creation
    # Test order optimization
    # Test minimum trade size
```

### Integration Tests

```python
def test_full_rebalancing_flow():
    """Test complete rebalancing process"""
    # Create portfolio config
    # Simulate drift
    # Trigger rebalancing
    # Verify execution
    # Check final allocations

def test_strategy_registry_integration():
    """Test loading strategies from registry"""
    # Load active strategies
    # Verify metadata
    # Check performance data
```

### Edge Cases

- Empty wallet (no positions)
- Single strategy portfolio
- Highly correlated strategies
- Market crash scenario (all strategies down)
- Insufficient USDC for rebalancing
- Network failures during execution
- Invalid strategy configurations

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/agents/portfolio_rebalancing_agent.py
python -m py_compile src/agents/portfolio_performance_tracker.py

# Check imports
python -c "import src.agents.portfolio_rebalancing_agent"
python -c "import src.agents.portfolio_performance_tracker"
```

### Level 2: Unit Tests

```bash
# Run portfolio-specific tests
python -m pytest tests/test_portfolio_rebalancing.py -v

# Test individual components
python -c "from src.agents.portfolio_rebalancing_agent import PortfolioMonitor; monitor = PortfolioMonitor(); print('Monitor OK')"
python -c "from src.agents.portfolio_rebalancing_agent import RebalancingEngine; engine = RebalancingEngine(); print('Engine OK')"
python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRiskManager; rm = PortfolioRiskManager(); print('Risk Manager OK')"
```

### Level 3: Integration Tests

```bash
# Test with demo configuration
python src/agents/portfolio_rebalancing_agent.py

# Test strategy loading
python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); strategies = agent.load_strategies(); print(f'Loaded {len(strategies)} strategies')"
```

### Level 4: Manual Validation

1. Configure a test portfolio with 2-3 strategies
2. Monitor for drift over time
3. Verify rebalancing triggers correctly
4. Confirm orders execute as expected
5. Check portfolio metrics update properly

### Level 5: Additional Validation (Optional)

```bash
# Check data persistence
ls -la src/data/portfolio/
cat src/data/portfolio/state.json

# Verify rebalancing history
tail -20 src/data/portfolio/rebalancing_history.csv

# Monitor performance metrics
python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; agent = PortfolioRebalancingAgent(); agent.display_portfolio_dashboard()"
```

---

## ACCEPTANCE CRITERIA

- [ ] Portfolio agent fetches real wallet positions and values
- [ ] Strategy performance is calculated from actual position history
- [ ] Correlation matrices use real returns data
- [ ] Rebalancing orders execute through actual trading functions
- [ ] All rebalancing methods work (threshold, calendar, adaptive, risk parity)
- [ ] Risk limits are enforced at portfolio and strategy level
- [ ] State persists across agent restarts
- [ ] Performance dashboard shows accurate real-time metrics
- [ ] Integration with existing agents (risk, trading, strategy registry)
- [ ] No regressions in existing functionality

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms feature works
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

### Design Decisions
1. **Real-time vs Historical**: Focus on real-time positions with historical tracking for performance metrics
2. **Strategy Identification**: Map strategies to token addresses for position tracking
3. **Order Execution**: Batch orders and execute sells first to optimize capital usage
4. **State Management**: Use JSON for configuration persistence, CSV for history logging

### Trade-offs
1. **Performance Calculation**: Simplified returns calculation may not capture all nuances
2. **Correlation Window**: 30-day default may be too short for some strategies
3. **Rebalancing Frequency**: Daily checks may be excessive for low-activity portfolios

### Future Enhancements
1. Multi-exchange support (currently Solana-only)
2. Tax-aware rebalancing
3. Machine learning for adaptive weight optimization
4. Social portfolio sharing features
5. Advanced risk analytics (VaR, CVaR, stress testing)