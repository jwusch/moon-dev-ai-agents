"""
🌙 Moon Dev Portfolio Rebalancing Agent
Manages multiple trading strategies as a cohesive portfolio with automatic rebalancing
Built with love by Moon Dev 🚀

This agent:
- Tracks allocations across multiple strategies
- Monitors drift from target weights
- Executes rebalancing trades
- Manages portfolio-level risk
- Integrates with strategy marketplace
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from termcolor import cprint
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.base_agent import BaseAgent
from src.models.portfolio_models import (
    PortfolioConfig, StrategyPosition, Order, OrderSide,
    RebalancingEvent, StrategyPerformance, PortfolioMetrics,
    RebalancingMethod, DEFAULT_RISK_LIMITS, DEFAULT_REBALANCING_PARAMS
)
from src.agents.strategy_registry_agent import StrategyRegistryAgent
from src.models.model_factory import model_factory
from src.config import EXCHANGE


class PortfolioMonitor:
    """Continuously monitors portfolio state and performance"""
    
    def __init__(self):
        self.positions: Dict[str, StrategyPosition] = {}
        self.performance_history: List[Dict] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_rebalance = datetime.now()
        
    def update_positions(self, strategy_values: Dict[str, float]) -> Dict[str, StrategyPosition]:
        """Update current positions from strategy values"""
        total_value = sum(strategy_values.values())
        
        for strategy_id, value in strategy_values.items():
            weight = value / total_value if total_value > 0 else 0
            
            if strategy_id in self.positions:
                self.positions[strategy_id].current_value = value
                self.positions[strategy_id].current_weight = weight
                self.positions[strategy_id].drift = weight - self.positions[strategy_id].target_weight
            else:
                # New position
                self.positions[strategy_id] = StrategyPosition(
                    strategy_id=strategy_id,
                    current_value=value,
                    current_weight=weight,
                    target_weight=0,  # Will be updated by agent
                    drift=weight
                )
        
        return self.positions
    
    def calculate_drift(self) -> Dict[str, float]:
        """Calculate deviation from target allocations"""
        return {
            strategy_id: position.drift 
            for strategy_id, position in self.positions.items()
        }
    
    def calculate_correlations(self, returns_data: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        if returns_data.empty or len(returns_data.columns) < 2:
            return pd.DataFrame()
        
        # Use last N days of data
        recent_returns = returns_data.tail(lookback_days)
        self.correlation_matrix = recent_returns.corr()
        
        return self.correlation_matrix
    
    def check_triggers(self, config: PortfolioConfig) -> List[str]:
        """Check if any rebalancing triggers are met"""
        triggers = []
        
        # Check drift trigger
        if config.rebalancing_method == RebalancingMethod.THRESHOLD:
            min_drift = config.rebalancing_params.get("min_drift", 0.05)
            for strategy_id, drift in self.calculate_drift().items():
                if abs(drift) > min_drift:
                    triggers.append(f"drift:{strategy_id}:{drift:.2%}")
        
        # Check calendar trigger
        elif config.rebalancing_method == RebalancingMethod.CALENDAR:
            freq = config.rebalancing_params.get("frequency", "monthly")
            days_since = (datetime.now() - self.last_rebalance).days
            
            freq_days = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90
            }
            
            if days_since >= freq_days.get(freq, 30):
                triggers.append(f"calendar:{freq}")
        
        # Check risk triggers
        if hasattr(self, 'portfolio_metrics'):
            if self.portfolio_metrics.max_drawdown < config.risk_limits.get("max_portfolio_drawdown", -0.25):
                triggers.append(f"risk:drawdown:{self.portfolio_metrics.max_drawdown:.2%}")
        
        return triggers


class RebalancingEngine:
    """Core logic for portfolio rebalancing decisions"""
    
    def __init__(self, method: RebalancingMethod = RebalancingMethod.THRESHOLD):
        self.method = method
        self.optimization_params = {}
        
    def calculate_target_weights(self,
                               current_weights: Dict[str, float],
                               performance_data: Dict[str, StrategyPerformance],
                               constraints: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal portfolio weights based on method"""
        
        if self.method == RebalancingMethod.THRESHOLD:
            # Simple rebalancing to original targets
            return self._threshold_weights(current_weights, performance_data, constraints)
        
        elif self.method == RebalancingMethod.ADAPTIVE:
            # Adjust weights based on performance
            return self._adaptive_weights(current_weights, performance_data, constraints)
        
        elif self.method == RebalancingMethod.RISK_PARITY:
            # Equal risk contribution
            return self._risk_parity_weights(current_weights, performance_data, constraints)
        
        else:  # CALENDAR
            # Use base target weights
            return constraints.get("target_weights", current_weights)
    
    def _threshold_weights(self, current: Dict, performance: Dict, constraints: Dict) -> Dict[str, float]:
        """Simple threshold rebalancing to target weights"""
        return constraints.get("target_weights", current)
    
    def _adaptive_weights(self, current: Dict, performance: Dict, constraints: Dict) -> Dict[str, float]:
        """Adaptive rebalancing based on performance"""
        weights = {}
        total_score = 0
        
        # Score each strategy
        for strategy_id, perf in performance.items():
            # Combine multiple metrics for scoring
            score = (
                perf.sharpe_ratio * 0.4 +
                (1 + perf.total_return) * 0.3 +
                (1 - abs(perf.max_drawdown)) * 0.3
            )
            score = max(0, score)  # No negative scores
            weights[strategy_id] = score
            total_score += score
        
        # Normalize to sum to 1
        if total_score > 0:
            weights = {k: v/total_score for k, v in weights.items()}
        else:
            # Equal weight if all scores are 0
            n = len(weights)
            weights = {k: 1/n for k in weights.keys()}
        
        # Apply constraints
        return self._apply_constraints(weights, constraints)
    
    def _risk_parity_weights(self, current: Dict, performance: Dict, constraints: Dict) -> Dict[str, float]:
        """Calculate risk parity weights (simplified version)"""
        # Get volatilities
        vols = {s: p.volatility for s, p in performance.items()}
        
        # Inverse volatility weighting as simple risk parity
        inv_vols = {s: 1/v if v > 0 else 0 for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol > 0:
            weights = {s: v/total_inv_vol for s, v in inv_vols.items()}
        else:
            n = len(current)
            weights = {s: 1/n for s in current.keys()}
        
        return self._apply_constraints(weights, constraints)
    
    def _apply_constraints(self, weights: Dict[str, float], constraints: Dict) -> Dict[str, float]:
        """Apply min/max constraints to weights"""
        min_weight = constraints.get("min_single_strategy", 0.05)
        max_weight = constraints.get("max_single_strategy", 0.4)
        
        # Apply constraints
        constrained = {}
        for strategy, weight in weights.items():
            if weight < min_weight:
                constrained[strategy] = 0  # Below minimum, set to 0
            else:
                constrained[strategy] = min(weight, max_weight)
        
        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            return {k: v/total for k, v in constrained.items()}
        
        return weights
    
    def generate_rebalancing_orders(self,
                                   current_positions: Dict[str, StrategyPosition],
                                   target_weights: Dict[str, float],
                                   total_portfolio_value: float) -> List[Order]:
        """Generate specific buy/sell orders"""
        orders = []
        
        for strategy_id, target_weight in target_weights.items():
            current_pos = current_positions.get(strategy_id)
            current_value = current_pos.current_value if current_pos else 0
            
            target_value = target_weight * total_portfolio_value
            diff = target_value - current_value
            
            # Skip small trades
            if abs(diff) < 100:  # $100 minimum
                continue
            
            if diff > 0:
                # Buy more
                orders.append(Order(
                    strategy_id=strategy_id,
                    side=OrderSide.BUY,
                    amount_usd=diff,
                    reason=f"Rebalance: increase allocation from {current_value/total_portfolio_value:.1%} to {target_weight:.1%}"
                ))
            else:
                # Sell some
                orders.append(Order(
                    strategy_id=strategy_id,
                    side=OrderSide.SELL,
                    amount_usd=abs(diff),
                    reason=f"Rebalance: decrease allocation from {current_value/total_portfolio_value:.1%} to {target_weight:.1%}"
                ))
        
        return orders
    
    def optimize_execution(self, orders: List[Order]) -> List[Order]:
        """Optimize order execution to minimize costs"""
        # Sort to execute sells first (free up capital)
        sells = [o for o in orders if o.side == OrderSide.SELL]
        buys = [o for o in orders if o.side == OrderSide.BUY]
        
        # Sort by size (larger first for better liquidity)
        sells.sort(key=lambda x: x.amount_usd, reverse=True)
        buys.sort(key=lambda x: x.amount_usd, reverse=True)
        
        return sells + buys


class PortfolioRiskManager:
    """Portfolio-level risk management"""
    
    def __init__(self):
        self.risk_limits = DEFAULT_RISK_LIMITS.copy()
        
    def validate_allocation(self, proposed_weights: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate proposed allocation meets risk criteria"""
        violations = []
        
        # Check weight constraints
        for strategy, weight in proposed_weights.items():
            if weight > self.risk_limits["max_single_strategy"]:
                violations.append(f"{strategy}: weight {weight:.1%} exceeds max {self.risk_limits['max_single_strategy']:.1%}")
            
            if 0 < weight < self.risk_limits["min_single_strategy"]:
                violations.append(f"{strategy}: weight {weight:.1%} below min {self.risk_limits['min_single_strategy']:.1%}")
        
        # Check total weights
        total = sum(proposed_weights.values())
        if abs(total - 1.0) > 0.001:
            violations.append(f"Total weight {total:.1%} != 100%")
        
        return len(violations) == 0, violations
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, StrategyPosition],
                                  performance: Dict[str, StrategyPerformance]) -> PortfolioMetrics:
        """Calculate portfolio-level risk metrics"""
        # This is simplified - real implementation would be more complex
        
        # Weighted average of strategy metrics
        total_value = sum(p.current_value for p in positions.values())
        
        weighted_return = sum(
            positions[s].current_value / total_value * perf.total_return
            for s, perf in performance.items()
            if s in positions
        )
        
        weighted_sharpe = sum(
            positions[s].current_value / total_value * perf.sharpe_ratio
            for s, perf in performance.items()
            if s in positions
        )
        
        # Simple approximations
        return PortfolioMetrics(
            total_value=total_value,
            total_return=weighted_return,
            sharpe_ratio=weighted_sharpe,
            max_drawdown=-0.15,  # Placeholder
            volatility=0.20,      # Placeholder
            beta=1.0,            # Placeholder
            var_95=-0.05,        # Placeholder
            avg_correlation=0.5,  # Placeholder
            strategy_correlations=pd.DataFrame(),
            last_rebalance=datetime.now(),
            days_since_rebalance=0
        )


class PortfolioRebalancingAgent(BaseAgent):
    """Main portfolio rebalancing agent"""
    
    def __init__(self, portfolio_config: Optional[PortfolioConfig] = None):
        """Initialize portfolio rebalancing agent"""
        super().__init__(agent_type='portfolio', use_exchange_manager=False)
        
        self.config = portfolio_config
        self.monitor = PortfolioMonitor()
        self.rebalancer = RebalancingEngine(
            method=portfolio_config.rebalancing_method if portfolio_config else RebalancingMethod.THRESHOLD
        )
        self.risk_manager = PortfolioRiskManager()
        
        # Integration points
        self.registry = StrategyRegistryAgent()
        self.model = model_factory.get_model("anthropic")
        
        # Data storage
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "src" / "data" / "portfolio"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_start = datetime.now()
        self.rebalancing_history = []
        
    def check_and_rebalance(self) -> Dict[str, Any]:
        """Main method to check portfolio and rebalance if needed"""
        cprint("\n🌙 Portfolio Rebalancing Check", "cyan", attrs=["bold"])
        cprint("=" * 50, "blue")
        
        # Step 1: Update current positions
        strategy_values = self._get_current_strategy_values()
        self.monitor.update_positions(strategy_values)
        
        # Step 2: Get performance data
        performance_data = self._get_strategy_performance()
        
        # Step 3: Check triggers
        triggers = self.monitor.check_triggers(self.config)
        
        if not triggers and self.config.rebalancing_method != RebalancingMethod.CALENDAR:
            cprint("✅ No rebalancing triggers detected", "green")
            return {"executed": False, "reason": "No triggers"}
        
        cprint(f"\n🎯 Rebalancing triggers: {triggers}", "yellow")
        
        # Step 4: Calculate new weights
        current_weights = {
            s: p.current_weight 
            for s, p in self.monitor.positions.items()
        }
        
        constraints = {
            "target_weights": self.config.target_allocations,
            **self.config.risk_limits
        }
        
        new_weights = self.rebalancer.calculate_target_weights(
            current_weights, performance_data, constraints
        )
        
        # Step 5: Validate new weights
        valid, violations = self.risk_manager.validate_allocation(new_weights)
        if not valid:
            cprint(f"❌ Risk validation failed: {violations}", "red")
            return {"executed": False, "reason": "Risk limits violated", "violations": violations}
        
        # Step 6: Generate orders
        total_value = sum(p.current_value for p in self.monitor.positions.values())
        orders = self.rebalancer.generate_rebalancing_orders(
            self.monitor.positions, new_weights, total_value
        )
        
        if not orders:
            cprint("✅ Portfolio already balanced", "green")
            return {"executed": False, "reason": "Already balanced"}
        
        # Step 7: Optimize and execute orders
        optimized_orders = self.rebalancer.optimize_execution(orders)
        
        cprint(f"\n📊 Executing {len(optimized_orders)} rebalancing orders:", "yellow")
        for order in optimized_orders:
            cprint(f"  • {order}", "white")
        
        # Step 8: Execute (simulation for now)
        execution_results = self._execute_orders(optimized_orders)
        
        # Step 9: Record event
        event = self._record_rebalancing_event(
            triggers, current_weights, new_weights, optimized_orders, execution_results
        )
        
        # Step 10: Update last rebalance time
        self.monitor.last_rebalance = datetime.now()
        
        return {
            "executed": True,
            "event_id": event.event_id,
            "orders_count": len(optimized_orders),
            "total_traded": event.total_traded
        }
    
    def _get_current_strategy_values(self) -> Dict[str, float]:
        """Get current value of each strategy in portfolio"""
        # Simulation - in production would query actual positions
        values = {}
        
        # Start with $10,000 portfolio
        total = 10000
        
        for strategy_id, target_weight in self.config.target_allocations.items():
            # Add some random drift
            drift = np.random.normal(0, 0.05)
            actual_weight = target_weight * (1 + drift)
            values[strategy_id] = total * actual_weight
        
        # Normalize
        current_total = sum(values.values())
        scale = total / current_total
        
        return {k: v * scale for k, v in values.items()}
    
    def _get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for each strategy"""
        performance = {}
        
        for strategy_id in self.config.target_allocations:
            # Get from registry if available
            strategy_info = self.registry.get_strategy(strategy_id)
            
            if strategy_info and strategy_info.get("performance_data"):
                perf_data = strategy_info["performance_data"]
                performance[strategy_id] = StrategyPerformance(
                    strategy_id=strategy_id,
                    returns=pd.Series(),  # Would load actual returns
                    sharpe_ratio=perf_data.get("sharpe_ratio", 0.5),
                    max_drawdown=perf_data.get("max_drawdown", -0.15),
                    win_rate=perf_data.get("win_rate", 0.5),
                    total_return=perf_data.get("total_return", 0.1),
                    volatility=perf_data.get("volatility", 0.2),
                    current_allocation=self.monitor.positions[strategy_id].current_weight,
                    target_allocation=self.config.target_allocations[strategy_id],
                    drift=self.monitor.positions[strategy_id].drift
                )
            else:
                # Default performance
                performance[strategy_id] = StrategyPerformance(
                    strategy_id=strategy_id,
                    returns=pd.Series(),
                    sharpe_ratio=0.5 + np.random.normal(0, 0.2),
                    max_drawdown=-0.15 + np.random.normal(0, 0.05),
                    win_rate=0.55 + np.random.normal(0, 0.1),
                    total_return=0.12 + np.random.normal(0, 0.05),
                    volatility=0.20 + np.random.normal(0, 0.02),
                    current_allocation=self.monitor.positions[strategy_id].current_weight,
                    target_allocation=self.config.target_allocations[strategy_id],
                    drift=self.monitor.positions[strategy_id].drift
                )
        
        return performance
    
    def _execute_orders(self, orders: List[Order]) -> Dict[str, Any]:
        """Execute rebalancing orders"""
        # Simulation for now - would integrate with trading agent
        results = {
            "executed": len(orders),
            "failed": 0,
            "total_traded": sum(o.amount_usd for o in orders),
            "execution_time": datetime.now()
        }
        
        # Mark orders as executed
        for order in orders:
            order.executed = True
            order.execution_time = datetime.now()
        
        return results
    
    def _record_rebalancing_event(self, triggers, pre_weights, post_weights, orders, execution) -> RebalancingEvent:
        """Record rebalancing event for history"""
        event = RebalancingEvent(
            event_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now(),
            trigger=triggers[0] if triggers else "manual",
            pre_weights=pre_weights,
            post_weights=post_weights,
            orders=orders,
            execution_summary=execution,
            performance_impact={}  # Would calculate actual impact
        )
        
        # Save to file
        event_file = self.data_dir / f"rebalance_{event.event_id}.json"
        with open(event_file, 'w') as f:
            json.dump({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "trigger": event.trigger,
                "pre_weights": event.pre_weights,
                "post_weights": event.post_weights,
                "orders": [
                    {
                        "strategy_id": o.strategy_id,
                        "side": o.side.value,
                        "amount_usd": o.amount_usd,
                        "reason": o.reason
                    } for o in event.orders
                ],
                "execution_summary": event.execution_summary,
                "total_traded": event.total_traded
            }, f, indent=2)
        
        self.rebalancing_history.append(event)
        return event
    
    def display_portfolio_dashboard(self):
        """Display comprehensive portfolio metrics"""
        cprint("\n📊 Portfolio Performance Dashboard", "cyan", attrs=["bold"])
        cprint("=" * 60, "blue")
        
        # Current allocations vs targets
        cprint("\n📈 Allocations:", "yellow")
        for strategy_id, position in self.monitor.positions.items():
            target = self.config.target_allocations.get(strategy_id, 0)
            drift = position.drift
            
            drift_color = "green" if abs(drift) < 0.05 else "yellow" if abs(drift) < 0.10 else "red"
            
            cprint(f"  {strategy_id}: {position.current_weight:.1%} (target: {target:.1%}, drift: {drift:+.1%})", drift_color)
        
        # Portfolio metrics
        performance = self._get_strategy_performance()
        metrics = self.risk_manager.calculate_portfolio_metrics(self.monitor.positions, performance)
        
        cprint("\n💰 Performance:", "yellow")
        cprint(f"  Total Value: ${metrics.total_value:,.2f}", "white")
        cprint(f"  Total Return: {metrics.total_return:.2%}", "white")
        cprint(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}", "white")
        cprint(f"  Max Drawdown: {metrics.max_drawdown:.2%}", "white")
        
        cprint("\n🛡️ Risk Metrics:", "yellow")
        cprint(f"  Portfolio Volatility: {metrics.volatility:.2%}", "white")
        cprint(f"  Value at Risk (95%): {metrics.var_95:.2%}", "white")
        cprint(f"  Days Since Rebalance: {metrics.days_since_rebalance}", "white")


def main():
    """Demo portfolio rebalancing"""
    cprint("\n🌙 Moon Dev Portfolio Rebalancing Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create sample portfolio configuration
    config = PortfolioConfig(
        name="Balanced Crypto Portfolio",
        target_allocations={
            "rsi_mean_reversion": 0.30,
            "macd_momentum": 0.25,
            "bollinger_breakout": 0.25,
            "ml_predictor": 0.20
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params=DEFAULT_REBALANCING_PARAMS["threshold"],
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Validate config
    config.validate()
    
    # Initialize agent
    agent = PortfolioRebalancingAgent(config)
    
    # Display initial state
    agent.display_portfolio_dashboard()
    
    # Check and rebalance
    result = agent.check_and_rebalance()
    
    if result["executed"]:
        cprint(f"\n✅ Rebalancing completed: {result['orders_count']} orders, ${result['total_traded']:,.2f} traded", "green")
    else:
        cprint(f"\n⚠️ No rebalancing needed: {result['reason']}", "yellow")
    
    # Display updated state
    agent.display_portfolio_dashboard()


if __name__ == "__main__":
    main()