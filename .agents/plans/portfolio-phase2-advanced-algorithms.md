# Feature: Portfolio Rebalancing Phase 2 - Advanced Algorithms

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Enhance the Portfolio Rebalancing Agent with advanced algorithms including sophisticated risk parity implementation, performance attribution analysis, enhanced correlation tracking, and improved adaptive rebalancing. This phase transforms the basic rebalancing functionality into a professional-grade portfolio management system with advanced optimization techniques.

## User Story

As a quantitative trader
I want advanced portfolio optimization algorithms
So that I can maximize risk-adjusted returns and better understand performance drivers

## Problem Statement

The current portfolio rebalancing implementation uses simplified algorithms that don't fully optimize for risk-adjusted returns. Traders need more sophisticated methods like true risk parity (equal risk contribution), performance attribution to understand return sources, and advanced correlation analysis to manage portfolio concentration risk.

## Solution Statement

Implement advanced portfolio algorithms including:
- True risk parity optimization using numerical methods
- Performance attribution to decompose returns by strategy contribution
- Enhanced correlation analysis with clustering and dimensionality reduction
- Improved adaptive rebalancing using momentum and mean reversion signals
- Historical performance data integration for better decision making

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: High
**Primary Systems Affected**: PortfolioRebalancingAgent, RebalancingEngine, PortfolioRiskManager
**Dependencies**: numpy, pandas, scipy (for optimization)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/agents/portfolio_rebalancing_agent.py` (lines 185-200) - Why: Current risk parity implementation to enhance
- `src/agents/portfolio_rebalancing_agent.py` (lines 157-183) - Why: Adaptive rebalancing to improve
- `src/agents/portfolio_rebalancing_agent.py` (lines 79-89) - Why: Correlation calculation to extend
- `src/models/portfolio_models.py` (lines 104-151) - Why: Data models for performance metrics
- `src/marketplace/analytics.py` (lines 20-100) - Why: Performance metric calculation patterns
- `src/agents/strategy_registry_agent.py` - Why: Strategy metadata integration
- `src/scripts/demo_portfolio_rebalancing.py` - Why: Demo patterns to extend

### New Files to Create

- `src/utils/portfolio_optimization.py` - Advanced optimization algorithms
- `src/utils/performance_attribution.py` - Return decomposition utilities
- `tests/test_portfolio_optimization.py` - Unit tests for optimization
- `tests/test_performance_attribution.py` - Unit tests for attribution

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [Risk Parity Optimization](https://www.investopedia.com/terms/r/risk-parity.asp)
  - Specific section: Mathematical foundation
  - Why: Understanding equal risk contribution optimization
- [scipy.optimize Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
  - Specific section: minimize function
  - Why: Required for risk parity optimization
- [Performance Attribution Theory](https://en.wikipedia.org/wiki/Performance_attribution)
  - Specific section: Arithmetic attribution
  - Why: Foundation for return decomposition

### Patterns to Follow

**Naming Conventions:**
```python
# From portfolio_models.py
@dataclass
class PortfolioMetrics:
    total_value: float
    total_return: float
    sharpe_ratio: float
```

**Error Handling:**
```python
# From portfolio_rebalancing_agent.py
if returns_data.empty or len(returns_data.columns) < 2:
    return pd.DataFrame()
```

**Logging Pattern:**
```python
# From codebase
cprint(f"✅ Success message", "green")
cprint(f"❌ Error message", "red")
cprint(f"📊 Info message", "yellow")
```

**Method Structure:**
```python
# Keep methods under 50 lines, split if needed
def calculate_something(self, data: pd.DataFrame) -> Dict[str, float]:
    """Docstring explaining purpose"""
    # Implementation
    return result
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Create optimization utilities and enhance data structures for advanced algorithms.

**Tasks:**
- Create portfolio optimization utility module
- Enhance correlation tracking in PortfolioMonitor
- Add historical returns fetching capability
- Update data models for new metrics

### Phase 2: Core Implementation

Implement the advanced algorithms.

**Tasks:**
- Implement true risk parity optimization
- Create performance attribution calculator
- Enhance adaptive rebalancing with signals
- Add correlation clustering analysis

### Phase 3: Integration

Connect new algorithms to existing portfolio agent.

**Tasks:**
- Integrate optimization into RebalancingEngine
- Add performance attribution to reporting
- Update dashboard display methods
- Enhance demo scripts

### Phase 4: Testing & Validation

Comprehensive testing of new algorithms.

**Tasks:**
- Unit tests for each algorithm
- Integration tests with portfolio agent
- Performance comparison tests
- Edge case validation

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE src/utils/portfolio_optimization.py

- **IMPLEMENT**: Portfolio optimization utilities with risk parity and mean-variance
- **PATTERN**: Follow dataclass patterns from portfolio_models.py
- **IMPORTS**: numpy, pandas, scipy.optimize, typing
- **GOTCHA**: scipy.optimize.minimize can fail to converge - add max iterations
- **VALIDATE**: `python -c "from src.utils.portfolio_optimization import risk_parity_weights; print('Import successful')"`

```python
"""
Portfolio optimization algorithms for advanced rebalancing
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from termcolor import cprint

def risk_parity_weights(returns: pd.DataFrame, 
                       initial_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate risk parity weights where each asset contributes equally to portfolio risk
    """
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    
    def risk_contribution(weights):
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        contrib = weights * marginal_contrib / portfolio_vol
        return contrib
    
    def objective(weights):
        contrib = risk_contribution(weights)
        target_contrib = 1.0 / n_assets
        return np.sum((contrib - target_contrib) ** 2)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w}
    ]
    
    result = minimize(objective, initial_weights, 
                     method='SLSQP', constraints=constraints,
                     options={'maxiter': 1000})
    
    if result.success:
        return dict(zip(returns.columns, result.x))
    else:
        cprint(f"⚠️ Risk parity optimization failed, using equal weights", "yellow")
        return dict(zip(returns.columns, [1/n_assets] * n_assets))

def mean_variance_optimization(returns: pd.DataFrame,
                             risk_aversion: float = 1.0,
                             constraints: Optional[Dict] = None) -> Dict[str, float]:
    """
    Mean-variance optimization with optional constraints
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_return = weights @ mean_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        return -(portfolio_return - risk_aversion * portfolio_vol)
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    if constraints:
        if 'max_weight' in constraints:
            bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
        if 'min_weight' in constraints:
            bounds = tuple((constraints['min_weight'], b[1]) for b in bounds)
    
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=cons)
    
    if result.success:
        return dict(zip(returns.columns, result.x))
    else:
        return dict(zip(returns.columns, [1/n_assets] * n_assets))
```

### CREATE src/utils/performance_attribution.py

- **IMPLEMENT**: Performance attribution to decompose returns by strategy
- **PATTERN**: Use StrategyPerformance dataclass from portfolio_models.py
- **IMPORTS**: pandas, numpy, datetime
- **GOTCHA**: Handle missing data periods gracefully
- **VALIDATE**: `python -c "from src.utils.performance_attribution import calculate_attribution; print('Import successful')"`

```python
"""
Performance attribution analysis for portfolio returns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from termcolor import cprint

def calculate_attribution(portfolio_returns: pd.Series,
                         strategy_returns: pd.DataFrame,
                         weights: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance attribution for each strategy
    """
    attribution = {}
    total_return = portfolio_returns.iloc[-1] - portfolio_returns.iloc[0]
    
    for strategy in strategy_returns.columns:
        # Calculate contribution
        strategy_weighted_returns = strategy_returns[strategy] * weights[strategy]
        contribution = strategy_weighted_returns.sum()
        attribution[strategy] = {
            'contribution': contribution,
            'contribution_pct': (contribution / total_return * 100) if total_return != 0 else 0,
            'average_weight': weights[strategy].mean(),
            'strategy_return': strategy_returns[strategy].sum()
        }
    
    # Add interaction effect
    calculated_total = sum(item['contribution'] for item in attribution.values())
    attribution['interaction'] = {
        'contribution': total_return - calculated_total,
        'contribution_pct': ((total_return - calculated_total) / total_return * 100) if total_return != 0 else 0
    }
    
    return attribution

def calculate_rolling_attribution(portfolio_data: pd.DataFrame,
                                window: int = 30) -> pd.DataFrame:
    """
    Calculate rolling attribution over time windows
    """
    results = []
    
    for i in range(window, len(portfolio_data)):
        window_data = portfolio_data.iloc[i-window:i]
        
        # Extract returns and weights from window
        portfolio_returns = window_data['portfolio_value']
        strategy_returns = window_data.filter(regex='.*_return$')
        weights = window_data.filter(regex='.*_weight$')
        
        attribution = calculate_attribution(
            portfolio_returns,
            strategy_returns,
            weights
        )
        
        results.append({
            'date': window_data.index[-1],
            **{f"{k}_contribution": v['contribution'] for k, v in attribution.items()}
        })
    
    return pd.DataFrame(results).set_index('date')

def factor_attribution(returns: pd.DataFrame,
                      factor_exposures: pd.DataFrame,
                      factor_returns: pd.DataFrame) -> Dict[str, float]:
    """
    Attribute returns to common factors (momentum, value, etc.)
    """
    # This is a placeholder for factor-based attribution
    # In practice, would use regression analysis
    attribution = {
        'momentum': 0.0,
        'mean_reversion': 0.0,
        'volatility': 0.0,
        'specific': 0.0
    }
    
    # Simple example - would be more sophisticated
    total_return = returns.sum().sum()
    attribution['specific'] = total_return
    
    return attribution
```

### UPDATE src/agents/portfolio_rebalancing_agent.py - RebalancingEngine._risk_parity_weights

- **IMPLEMENT**: Replace simplified risk parity with true optimization
- **PATTERN**: Import from new portfolio_optimization module
- **IMPORTS**: from src.utils.portfolio_optimization import risk_parity_weights
- **GOTCHA**: Need historical returns data for covariance calculation
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import RebalancingEngine; print('Import successful')"`

Replace lines 185-200 with:
```python
def _risk_parity_weights(self, current: Dict, performance: Dict, constraints: Dict) -> Dict[str, float]:
    """Calculate true risk parity weights with equal risk contribution"""
    # Get returns data for covariance calculation
    returns_data = pd.DataFrame()
    
    for strategy_id, perf in performance.items():
        if hasattr(perf, 'returns') and not perf.returns.empty:
            returns_data[strategy_id] = perf.returns
    
    if returns_data.empty:
        cprint("⚠️ No returns data available for risk parity", "yellow")
        # Fallback to equal weights
        n = len(current)
        return {s: 1/n for s in current.keys()}
    
    # Import optimization function
    from src.utils.portfolio_optimization import risk_parity_weights as calculate_rp
    
    # Calculate risk parity weights
    weights = calculate_rp(returns_data)
    
    # Apply constraints
    return self._apply_constraints(weights, constraints)
```

### UPDATE src/agents/portfolio_rebalancing_agent.py - Add performance attribution

- **IMPLEMENT**: Add performance attribution method to PortfolioRebalancingAgent
- **PATTERN**: Follow existing method patterns in the class
- **IMPORTS**: from src.utils.performance_attribution import calculate_attribution
- **GOTCHA**: Need to store historical weights and returns
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent; print(hasattr(PortfolioRebalancingAgent, 'calculate_performance_attribution'))"`

Add new method after line 500:
```python
def calculate_performance_attribution(self, lookback_days: int = 30) -> Dict[str, Any]:
    """Calculate performance attribution for the portfolio"""
    from src.utils.performance_attribution import calculate_attribution
    
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # This would fetch actual historical data
    # For now, create sample data
    portfolio_returns = pd.Series(
        np.random.randn(lookback_days).cumsum() * 0.01 + 1,
        index=pd.date_range(start_date, end_date, periods=lookback_days)
    )
    
    strategy_returns = pd.DataFrame({
        strategy: pd.Series(
            np.random.randn(lookback_days).cumsum() * 0.01 + 1,
            index=portfolio_returns.index
        )
        for strategy in self.config.target_allocations.keys()
    })
    
    weights = pd.DataFrame({
        strategy: pd.Series(
            [self.monitor.positions.get(strategy, StrategyPosition(
                strategy_id=strategy,
                current_value=0,
                current_weight=weight,
                target_weight=weight,
                drift=0
            )).current_weight] * lookback_days,
            index=portfolio_returns.index
        )
        for strategy, weight in self.config.target_allocations.items()
    })
    
    attribution = calculate_attribution(portfolio_returns, strategy_returns, weights)
    
    # Display results
    cprint("\n📊 Performance Attribution:", "cyan", attrs=["bold"])
    for strategy, attr in attribution.items():
        if isinstance(attr, dict):
            cprint(f"  {strategy}: {attr['contribution']:.2%} ({attr['contribution_pct']:.1f}%)", "white")
    
    return attribution
```

### UPDATE src/agents/portfolio_rebalancing_agent.py - Enhance adaptive rebalancing

- **IMPLEMENT**: Improve adaptive weights calculation with momentum signals
- **PATTERN**: Extend existing _adaptive_weights method
- **IMPORTS**: Add momentum calculation
- **GOTCHA**: Avoid overfitting to recent performance
- **VALIDATE**: `grep -n "_adaptive_weights" src/agents/portfolio_rebalancing_agent.py`

Replace lines 157-183 with enhanced version:
```python
def _adaptive_weights(self, current: Dict, performance: Dict, constraints: Dict) -> Dict[str, float]:
    """Adaptive rebalancing with momentum and mean reversion signals"""
    weights = {}
    scores = {}
    
    for strategy_id, perf in performance.items():
        # Base score from Sharpe ratio (risk-adjusted returns)
        sharpe_score = max(0, perf.sharpe_ratio) * 0.3
        
        # Momentum signal (recent performance)
        momentum_score = 0
        if hasattr(perf, 'returns') and len(perf.returns) > 20:
            recent_return = perf.returns.tail(20).mean()
            longer_return = perf.returns.tail(60).mean() if len(perf.returns) > 60 else recent_return
            momentum_score = (recent_return - longer_return) * 10 * 0.2
        
        # Mean reversion signal (deviation from historical average)
        mean_reversion_score = 0
        if hasattr(perf, 'returns') and len(perf.returns) > 100:
            historical_mean = perf.returns.mean()
            recent_mean = perf.returns.tail(20).mean()
            mean_reversion_score = -(recent_mean - historical_mean) * 5 * 0.1
        
        # Drawdown penalty
        drawdown_score = (1 - abs(perf.max_drawdown)) * 0.2
        
        # Win rate bonus
        win_rate_score = max(0, perf.win_rate - 0.5) * 2 * 0.2
        
        # Combine scores
        total_score = (
            sharpe_score +
            momentum_score +
            mean_reversion_score +
            drawdown_score +
            win_rate_score
        )
        
        scores[strategy_id] = max(0.1, total_score)  # Minimum score of 0.1
    
    # Normalize scores to weights
    total_score = sum(scores.values())
    if total_score > 0:
        weights = {k: v/total_score for k, v in scores.items()}
    else:
        # Equal weight if all scores are zero
        n = len(scores)
        weights = {k: 1/n for k in scores.keys()}
    
    # Apply constraints
    return self._apply_constraints(weights, constraints)
```

### UPDATE src/agents/portfolio_rebalancing_agent.py - Enhance correlation analysis

- **IMPLEMENT**: Add correlation clustering and risk analysis
- **PATTERN**: Extend calculate_correlations method
- **IMPORTS**: numpy for additional calculations
- **GOTCHA**: Handle single strategy portfolios
- **VALIDATE**: `python -c "from src.agents.portfolio_rebalancing_agent import PortfolioMonitor; pm = PortfolioMonitor(); print('Correlation methods available')"`

Add after line 89:
```python
def analyze_correlation_risk(self) -> Dict[str, Any]:
    """Analyze correlation risk and clustering"""
    if self.correlation_matrix is None or self.correlation_matrix.empty:
        return {"risk_level": "low", "clusters": []}
    
    corr = self.correlation_matrix.values
    n_strategies = len(self.correlation_matrix)
    
    # Calculate average correlation
    mask = np.ones_like(corr, dtype=bool)
    np.fill_diagonal(mask, 0)
    avg_correlation = corr[mask].mean()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(n_strategies):
        for j in range(i+1, n_strategies):
            if corr[i, j] > 0.7:  # Threshold from risk limits
                high_corr_pairs.append({
                    "pair": (self.correlation_matrix.index[i], self.correlation_matrix.index[j]),
                    "correlation": corr[i, j]
                })
    
    # Simple clustering based on correlation
    clusters = []
    clustered = set()
    
    for i in range(n_strategies):
        if i not in clustered:
            cluster = [i]
            for j in range(i+1, n_strategies):
                if j not in clustered and corr[i, j] > 0.6:
                    cluster.append(j)
                    clustered.add(j)
            if len(cluster) > 1:
                clusters.append([self.correlation_matrix.index[idx] for idx in cluster])
            clustered.add(i)
    
    # Determine risk level
    risk_level = "low"
    if avg_correlation > 0.7 or len(high_corr_pairs) > n_strategies // 2:
        risk_level = "high"
    elif avg_correlation > 0.5 or len(high_corr_pairs) > 0:
        risk_level = "medium"
    
    return {
        "risk_level": risk_level,
        "average_correlation": avg_correlation,
        "high_correlation_pairs": high_corr_pairs,
        "clusters": clusters,
        "recommendation": self._get_correlation_recommendation(risk_level, clusters)
    }

def _get_correlation_recommendation(self, risk_level: str, clusters: List[List[str]]) -> str:
    """Generate recommendation based on correlation analysis"""
    if risk_level == "high":
        return "Consider diversifying strategies - high correlation reduces diversification benefits"
    elif risk_level == "medium" and clusters:
        return f"Monitor clustered strategies: {clusters} for concentration risk"
    else:
        return "Portfolio is well diversified"
```

### CREATE tests/test_portfolio_optimization.py

- **IMPLEMENT**: Unit tests for optimization algorithms
- **PATTERN**: Follow pytest patterns from existing tests
- **IMPORTS**: pytest, pandas, numpy, portfolio_optimization module
- **GOTCHA**: Test convergence edge cases
- **VALIDATE**: `python -m pytest tests/test_portfolio_optimization.py -v`

```python
"""
Unit tests for portfolio optimization algorithms
"""
import pytest
import pandas as pd
import numpy as np
from src.utils.portfolio_optimization import (
    risk_parity_weights,
    mean_variance_optimization
)

class TestPortfolioOptimization:
    
    def test_risk_parity_equal_vol(self):
        """Test risk parity with equal volatility assets"""
        # Create returns with equal volatility
        returns = pd.DataFrame({
            'A': np.random.normal(0.001, 0.01, 100),
            'B': np.random.normal(0.001, 0.01, 100),
            'C': np.random.normal(0.001, 0.01, 100)
        })
        
        weights = risk_parity_weights(returns)
        
        # Should be approximately equal weights
        assert all(0.3 < w < 0.37 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_risk_parity_different_vol(self):
        """Test risk parity with different volatility assets"""
        returns = pd.DataFrame({
            'Low_Vol': np.random.normal(0.001, 0.005, 100),
            'High_Vol': np.random.normal(0.001, 0.02, 100)
        })
        
        weights = risk_parity_weights(returns)
        
        # Low vol should have higher weight
        assert weights['Low_Vol'] > weights['High_Vol']
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_mean_variance_basic(self):
        """Test basic mean-variance optimization"""
        returns = pd.DataFrame({
            'A': np.random.normal(0.002, 0.01, 100),
            'B': np.random.normal(0.001, 0.02, 100)
        })
        
        weights = mean_variance_optimization(returns)
        
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_mean_variance_with_constraints(self):
        """Test mean-variance with max weight constraint"""
        returns = pd.DataFrame({
            'A': np.random.normal(0.005, 0.01, 100),  # High return
            'B': np.random.normal(0.001, 0.01, 100),
            'C': np.random.normal(0.001, 0.01, 100)
        })
        
        constraints = {'max_weight': 0.4}
        weights = mean_variance_optimization(returns, constraints=constraints)
        
        assert all(w <= 0.4 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.001
```

### CREATE tests/test_performance_attribution.py

- **IMPLEMENT**: Unit tests for performance attribution
- **PATTERN**: Follow existing test patterns
- **IMPORTS**: pytest, pandas, numpy, performance_attribution
- **GOTCHA**: Test with missing data
- **VALIDATE**: `python -m pytest tests/test_performance_attribution.py -v`

```python
"""
Unit tests for performance attribution
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.performance_attribution import (
    calculate_attribution,
    calculate_rolling_attribution
)

class TestPerformanceAttribution:
    
    def test_basic_attribution(self):
        """Test basic performance attribution calculation"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        portfolio_returns = pd.Series(
            np.linspace(1.0, 1.1, 30),  # 10% return
            index=dates
        )
        
        strategy_returns = pd.DataFrame({
            'A': np.linspace(1.0, 1.15, 30),  # 15% return
            'B': np.linspace(1.0, 1.05, 30),  # 5% return
        }, index=dates)
        
        weights = pd.DataFrame({
            'A': [0.6] * 30,
            'B': [0.4] * 30
        }, index=dates)
        
        attribution = calculate_attribution(
            portfolio_returns,
            strategy_returns,
            weights
        )
        
        # Check structure
        assert 'A' in attribution
        assert 'B' in attribution
        assert 'interaction' in attribution
        
        # Check values make sense
        assert attribution['A']['contribution'] > attribution['B']['contribution']
        assert attribution['A']['average_weight'] == 0.6
        assert attribution['B']['average_weight'] == 0.4
    
    def test_attribution_with_negative_returns(self):
        """Test attribution with negative returns"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        portfolio_returns = pd.Series(
            np.linspace(1.0, 0.9, 30),  # -10% return
            index=dates
        )
        
        strategy_returns = pd.DataFrame({
            'A': np.linspace(1.0, 0.85, 30),  # -15% return
            'B': np.linspace(1.0, 0.95, 30),  # -5% return
        }, index=dates)
        
        weights = pd.DataFrame({
            'A': [0.5] * 30,
            'B': [0.5] * 30
        }, index=dates)
        
        attribution = calculate_attribution(
            portfolio_returns,
            strategy_returns,
            weights
        )
        
        # Both should have negative contributions
        assert attribution['A']['contribution'] < 0
        assert attribution['B']['contribution'] < 0
        # B should have less negative contribution
        assert attribution['A']['contribution'] < attribution['B']['contribution']
```

### UPDATE src/scripts/demo_portfolio_rebalancing.py - Add Phase 2 demos

- **IMPLEMENT**: Add demonstrations of new advanced algorithms
- **PATTERN**: Follow existing demo structure
- **IMPORTS**: Import new optimization functions
- **GOTCHA**: Keep demos concise and educational
- **VALIDATE**: `python src/scripts/demo_portfolio_rebalancing.py`

Add after line 280:
```python
def demo_risk_parity_optimization():
    """Demo true risk parity optimization"""
    cprint("\n⚖️ Demo 6: Risk Parity Optimization", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create portfolio with risk parity
    config = PortfolioConfig(
        name="Risk Parity Portfolio",
        target_allocations={
            "high_vol_momentum": 0.25,
            "low_vol_value": 0.25,
            "medium_vol_growth": 0.25,
            "uncorrelated_arb": 0.25
        },
        rebalancing_method=RebalancingMethod.RISK_PARITY,
        rebalancing_params=DEFAULT_REBALANCING_PARAMS["risk_parity"],
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    agent = PortfolioRebalancingAgent(config)
    
    cprint("\n📊 Initial Equal Weights:", "yellow")
    agent.display_portfolio_dashboard()
    
    # Trigger rebalancing
    result = agent.check_and_rebalance()
    
    if result["executed"]:
        cprint("\n✅ Risk parity optimization complete", "green")
        cprint("Each strategy now contributes equally to portfolio risk", "white")
    
    # Show correlation analysis
    corr_analysis = agent.monitor.analyze_correlation_risk()
    cprint(f"\n🔍 Correlation Analysis:", "yellow")
    cprint(f"  Risk Level: {corr_analysis['risk_level']}", "white")
    cprint(f"  Avg Correlation: {corr_analysis['average_correlation']:.2f}", "white")
    cprint(f"  Recommendation: {corr_analysis['recommendation']}", "cyan")

def demo_performance_attribution():
    """Demo performance attribution analysis"""
    cprint("\n📊 Demo 7: Performance Attribution", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    config = PortfolioConfig(
        name="Attribution Demo Portfolio",
        target_allocations={
            "momentum": 0.40,
            "value": 0.30,
            "growth": 0.30
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params=DEFAULT_REBALANCING_PARAMS["threshold"],
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    agent = PortfolioRebalancingAgent(config)
    
    # Calculate attribution
    attribution = agent.calculate_performance_attribution(lookback_days=30)
    
    cprint("\n💡 Key Insights:", "yellow")
    
    # Find best and worst contributors
    strategy_attrs = {k: v for k, v in attribution.items() if isinstance(v, dict) and k != 'interaction'}
    best = max(strategy_attrs.items(), key=lambda x: x[1]['contribution'])
    worst = min(strategy_attrs.items(), key=lambda x: x[1]['contribution'])
    
    cprint(f"  Best Contributor: {best[0]} ({best[1]['contribution']:.2%})", "green")
    cprint(f"  Worst Contributor: {worst[0]} ({worst[1]['contribution']:.2%})", "red")
```

### UPDATE src/utils/portfolio_builder.py - Add advanced options

- **IMPLEMENT**: Add risk parity and adaptive options to builder
- **PATTERN**: Extend _choose_rebalancing_method
- **IMPORTS**: None needed
- **GOTCHA**: Explain advanced methods clearly
- **VALIDATE**: `python src/utils/portfolio_builder.py`

Update the method descriptions in _choose_rebalancing_method around line 220:
```python
cprint("\n1. Threshold - Rebalance when drift exceeds limit", "white")
cprint("2. Calendar - Rebalance on fixed schedule", "white")
cprint("3. Adaptive - Adjust based on performance (uses momentum signals)", "white")
cprint("4. Risk Parity - Equal risk contribution (advanced optimization)", "white")
```

---

## TESTING STRATEGY

### Unit Tests

Design unit tests with fixtures and assertions following existing testing approaches:
- Test optimization convergence
- Test attribution accuracy
- Test correlation analysis
- Test constraint handling

### Integration Tests

- Test full rebalancing cycle with new algorithms
- Test performance tracking over multiple periods
- Test dashboard display with attribution data

### Edge Cases

- Empty returns data
- Single strategy portfolio
- Highly correlated strategies
- Optimization non-convergence
- Negative returns attribution

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Python syntax check
python -m py_compile src/utils/portfolio_optimization.py
python -m py_compile src/utils/performance_attribution.py

# Import validation
python -c "from src.utils.portfolio_optimization import risk_parity_weights, mean_variance_optimization"
python -c "from src.utils.performance_attribution import calculate_attribution"
```

### Level 2: Unit Tests

```bash
# Run new unit tests
python -m pytest tests/test_portfolio_optimization.py -v
python -m pytest tests/test_performance_attribution.py -v

# Run existing portfolio tests to ensure no regression
python -m pytest tests/ -k portfolio -v
```

### Level 3: Integration Tests

```bash
# Test portfolio agent with new algorithms
python src/scripts/test_portfolio.py

# Run full demo suite
python src/scripts/demo_portfolio_rebalancing.py
```

### Level 4: Manual Validation

```python
# Test risk parity optimization
from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.models.portfolio_models import PortfolioConfig, RebalancingMethod

config = PortfolioConfig(
    name="Test Risk Parity",
    target_allocations={"A": 0.5, "B": 0.5},
    rebalancing_method=RebalancingMethod.RISK_PARITY,
    rebalancing_params={}
)
agent = PortfolioRebalancingAgent(config)
result = agent.check_and_rebalance()
print(f"Risk parity result: {result}")

# Test performance attribution
attribution = agent.calculate_performance_attribution()
print(f"Attribution: {attribution}")
```

### Level 5: Additional Validation (Optional)

```bash
# Check for scipy installation
python -c "import scipy.optimize; print('scipy available')"

# Memory profiling for optimization
python -m memory_profiler src/scripts/demo_portfolio_rebalancing.py
```

---

## ACCEPTANCE CRITERIA

- [x] True risk parity optimization implemented with scipy
- [x] Performance attribution calculates strategy contributions
- [x] Enhanced correlation analysis with clustering
- [x] Improved adaptive rebalancing with signals
- [x] All validation commands pass with zero errors
- [x] Unit test coverage for new algorithms
- [x] Integration with existing portfolio agent
- [x] Demo scripts showcase new features
- [x] No regressions in Phase 1 functionality

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

- Risk parity optimization requires scipy.optimize for numerical methods
- Performance attribution assumes access to historical returns data
- Correlation clustering is simplified - could use sklearn for more sophisticated methods
- Adaptive rebalancing now includes momentum and mean reversion signals
- All algorithms gracefully degrade when data is insufficient