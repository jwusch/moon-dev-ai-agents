# Strategy Marketplace Integration Guide

## Integrating with Existing Agents

### 1. RBI Agent Integration

The RBI Agent can automatically submit successful strategies to the marketplace:

```python
# In rbi_agent.py, after successful backtest:

from src.agents.strategy_registry_agent import StrategyRegistryAgent
from src.marketplace.analytics import StrategyAnalytics

# After generating strategy code and running backtest
if backtest_return > SAVE_IF_OVER_RETURN:
    registry = StrategyRegistryAgent()
    analytics = StrategyAnalytics()
    
    # Register strategy
    metadata = registry.register_strategy(
        name=strategy_name,
        description=f"AI-generated strategy from: {source_url}",
        author="rbi_agent",
        code_path=strategy_file_path,
        category=detected_categories,
        timeframes=["15m", "1H"],
        instruments=tested_symbols,
        min_capital=initial_capital,
        risk_level="medium"
    )
    
    # Add performance
    metrics = analytics.calculate_metrics(equity_curve, trades)
    registry.update_performance(metadata['strategy_id'], metrics)
```

### 2. Strategy Agent Enhancement

Modify the Strategy Agent to load strategies from the marketplace:

```python
# In strategy_agent.py

def load_marketplace_strategies(self):
    """Load active strategies from marketplace"""
    registry = StrategyRegistryAgent()
    
    # Get all active strategies
    active_strategies = registry.search_strategies(
        risk_level=self.risk_tolerance,
        min_rating=4.0
    )
    
    for strategy in active_strategies:
        if strategy['status'] == 'active':
            # Load strategy module dynamically
            strategy_module = self._load_strategy_module(
                strategy['file_path']
            )
            self.strategies[strategy['name']] = strategy_module
```

### 3. Swarm Agent Integration

Use swarm consensus to evaluate marketplace strategies:

```python
# In swarm_agent.py

def evaluate_marketplace_strategy(self, strategy_id):
    """Get AI consensus on a marketplace strategy"""
    registry = StrategyRegistryAgent()
    strategy = registry.get_strategy(strategy_id)
    
    # Query multiple models
    prompt = f"""
    Evaluate this trading strategy:
    Name: {strategy['name']}
    Description: {strategy['description']}
    Performance: {strategy['performance_summary']}
    
    Should we use this strategy? Why or why not?
    """
    
    responses = self.query_all_models(prompt)
    consensus = self.get_consensus(responses)
    
    return consensus
```

### 4. Risk Agent Integration

Add marketplace strategy monitoring to risk checks:

```python
# In risk_agent.py

def check_strategy_performance(self):
    """Monitor active marketplace strategies"""
    registry = StrategyRegistryAgent()
    
    # Get strategies we're currently using
    for strategy_id in self.active_strategies:
        strategy = registry.get_strategy(strategy_id)
        
        # Check if strategy has been deprecated
        if strategy['status'] == 'deprecated':
            self.disable_strategy(strategy_id)
            self.log_warning(f"Strategy {strategy['name']} deprecated")
        
        # Check recent performance
        if strategy['performance_summary'].get('max_drawdown', 0) > self.max_allowed_drawdown:
            self.pause_strategy(strategy_id)
            self.log_warning(f"Strategy {strategy['name']} exceeding risk limits")
```

### 5. Main Orchestrator Integration

Update main.py to include marketplace monitoring:

```python
# In main.py

from src.agents.strategy_registry_agent import StrategyRegistryAgent

# Add to ACTIVE_AGENTS
ACTIVE_AGENTS = {
    'risk': True,
    'trading': True,
    'strategy': True,
    'marketplace': True,  # New
    # ... other agents
}

# In run_agents function
if ACTIVE_AGENTS['marketplace']:
    marketplace_agent = StrategyRegistryAgent()
    cprint("\n🏪 Checking Marketplace for New Strategies...", "cyan")
    
    # Check for highly rated new strategies
    new_strategies = marketplace_agent.search_strategies(
        min_rating=4.5
    )
    
    for strategy in new_strategies[:5]:  # Top 5
        cprint(f"📊 New strategy available: {strategy['name']} "
               f"(Return: {strategy['performance_summary'].get('total_return', 'N/A')}%)", 
               "green")
```

## Configuration Updates

### config.py Additions

```python
# Strategy Marketplace Settings
MARKETPLACE_ENABLED = True
MARKETPLACE_AUTO_DOWNLOAD = False  # Auto-download high-rated strategies
MARKETPLACE_MIN_RATING = 4.0      # Minimum rating to consider
MARKETPLACE_MIN_TRADES = 50       # Minimum trades for validity
MARKETPLACE_MAX_DRAWDOWN = 30     # Maximum acceptable drawdown %

# Strategy Selection Criteria
STRATEGY_SELECTION = {
    'prefer_categories': ['momentum', 'mean_reversion'],
    'avoid_categories': ['experimental', 'high_risk'],
    'min_sharpe_ratio': 1.0,
    'min_profit_factor': 1.5,
    'max_strategies': 5  # Maximum concurrent strategies
}
```

## Automated Workflows

### 1. Daily Strategy Discovery

Create a scheduled task to check for new strategies:

```python
# src/scripts/daily_strategy_check.py

import schedule
import time

def check_new_strategies():
    """Daily check for new high-performing strategies"""
    registry = StrategyRegistryAgent()
    analytics = StrategyAnalytics()
    
    # Get strategies added in last 24 hours
    recent_strategies = registry.search_strategies()
    
    for strategy in recent_strategies:
        # Only check strategies with performance data
        if strategy.get('performance_summary'):
            perf = strategy['performance_summary']
            
            # Alert on exceptional strategies
            if (perf.get('total_return', 0) > 100 and 
                perf.get('sharpe_ratio', 0) > 2 and
                perf.get('max_drawdown', -100) > -20):
                
                send_alert(f"🚀 Exceptional strategy found: {strategy['name']}")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(check_new_strategies)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. Automated Performance Tracking

Track performance of marketplace strategies in production:

```python
# In your trading loop

def track_marketplace_performance(self, strategy_id, trade_result):
    """Track real-world performance of marketplace strategies"""
    # Store results
    self.marketplace_results[strategy_id].append({
        'timestamp': datetime.now(),
        'trade_result': trade_result,
        'pnl': trade_result['pnl']
    })
    
    # Update marketplace if significantly different from backtest
    if len(self.marketplace_results[strategy_id]) > 100:
        real_performance = self.calculate_real_metrics(
            self.marketplace_results[strategy_id]
        )
        
        registry = StrategyRegistryAgent()
        strategy = registry.get_strategy(strategy_id)
        
        backtest_return = strategy['performance_summary'].get('total_return', 0)
        real_return = real_performance['total_return']
        
        # Alert if significant deviation
        if abs(real_return - backtest_return) > 20:  # 20% deviation
            self.log_warning(f"Strategy {strategy_id} deviating from backtest")
```

## Security Considerations

### Strategy Validation Pipeline

```python
class StrategyValidator:
    """Enhanced validation for marketplace strategies"""
    
    def validate_comprehensive(self, strategy_path):
        """Comprehensive validation before marketplace use"""
        
        # 1. Static code analysis
        if not self.check_code_safety(strategy_path):
            return False
        
        # 2. Sandbox execution test
        if not self.test_in_sandbox(strategy_path):
            return False
        
        # 3. Performance validation
        if not self.validate_performance_claims(strategy_path):
            return False
        
        # 4. Dependency check
        if not self.check_dependencies(strategy_path):
            return False
        
        return True
    
    def check_code_safety(self, path):
        """Check for malicious code patterns"""
        with open(path, 'r') as f:
            code = f.read()
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'eval(', 'exec(', '__import__',
            'subprocess', 'os.system',
            'requests.post',  # Prevent data exfiltration
            'socket.'         # Prevent network access
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False
        
        return True
```

## Monitoring and Alerts

### Marketplace Health Dashboard

Create a monitoring dashboard for marketplace health:

```python
# src/scripts/marketplace_monitor.py

def generate_health_report():
    """Generate marketplace health metrics"""
    registry = StrategyRegistryAgent()
    analytics = StrategyAnalytics()
    
    report = {
        'total_strategies': len(registry.registry['strategies']),
        'active_strategies': sum(1 for s in registry.registry['strategies'].values() 
                               if s['status'] == 'active'),
        'average_rating': calculate_avg_rating(),
        'total_downloads': sum(s['downloads'] for s in registry.registry['strategies'].values()),
        'new_this_week': count_new_strategies(days=7),
        'top_performers': get_top_performers(limit=5),
        'most_downloaded': get_most_downloaded(limit=5),
        'quality_score': calculate_quality_score()
    }
    
    return report
```

## Best Practices for Integration

1. **Gradual Adoption**
   - Start with paper trading marketplace strategies
   - Monitor for 2-4 weeks before live trading
   - Begin with small position sizes

2. **Diversity Management**
   - Don't run similar strategies simultaneously
   - Balance strategy categories (momentum, mean reversion, etc.)
   - Monitor correlation between strategies

3. **Performance Tracking**
   - Compare real performance to backtest claims
   - Track slippage and execution differences
   - Report significant deviations to marketplace

4. **Community Contribution**
   - Share your strategy improvements back to marketplace
   - Rate strategies you've tested
   - Report issues or suspicious strategies

---

This integration enables the full Moon Dev ecosystem to leverage community-created strategies while maintaining security and performance standards.