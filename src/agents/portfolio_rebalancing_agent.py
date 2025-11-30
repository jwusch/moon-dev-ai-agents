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
from src.config import EXCHANGE, MONITORED_TOKENS, USDC_ADDRESS, EXCLUDED_TOKENS, address, SOL_ADDRESS
from src import nice_funcs as n


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
        """Calculate correlation matrix between strategies using real returns"""
        # If returns_data is provided, use it
        if returns_data is not None and not returns_data.empty and len(returns_data.columns) >= 2:
            # Use last N days of data
            recent_returns = returns_data.tail(lookback_days)
            self.correlation_matrix = recent_returns.corr()
            return self.correlation_matrix
        
        # Otherwise, try to get real returns from performance tracker
        try:
            if not hasattr(self, 'performance_tracker'):
                from src.agents.portfolio_performance_tracker import PortfolioPerformanceTracker
                self.performance_tracker = PortfolioPerformanceTracker()
            
            # Get list of strategies from positions
            strategy_ids = list(self.positions.keys())
            
            if len(strategy_ids) < 2:
                cprint("⚠️ Need at least 2 strategies for correlation analysis", "yellow")
                return pd.DataFrame()
            
            # Get correlation matrix from performance tracker
            self.correlation_matrix = self.performance_tracker.get_correlation_matrix(
                strategy_ids, lookback_days
            )
            
            if not self.correlation_matrix.empty:
                cprint(f"📊 Calculated correlations for {len(strategy_ids)} strategies", "green")
            else:
                cprint("⚠️ No correlation data available", "yellow")
            
            return self.correlation_matrix
            
        except Exception as e:
            cprint(f"❌ Error calculating correlations: {str(e)}", "red")
            return pd.DataFrame()
    
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
        
        # Import additional risk limits from config
        from src.config import MAX_LOSS_USD, MAX_GAIN_USD, MINIMUM_BALANCE_USD, MAX_POSITION_PERCENTAGE
        
        # Update risk limits with config values
        self.risk_limits.update({
            "max_loss_usd": MAX_LOSS_USD,
            "max_gain_usd": MAX_GAIN_USD,
            "minimum_balance_usd": MINIMUM_BALANCE_USD,
            "max_position_percentage": MAX_POSITION_PERCENTAGE / 100.0  # Convert to decimal
        })
        
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
    
    def validate_portfolio_risk(self, portfolio_value: float, positions: Dict[str, StrategyPosition]) -> Tuple[bool, List[str]]:
        """Validate portfolio-level risk constraints"""
        violations = []
        
        # Check minimum balance
        if portfolio_value < self.risk_limits["minimum_balance_usd"]:
            violations.append(f"Portfolio value ${portfolio_value:.2f} below minimum ${self.risk_limits['minimum_balance_usd']}")
        
        # Check individual position sizes
        for strategy_id, position in positions.items():
            position_pct = position.current_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.risk_limits.get("max_position_percentage", 0.3):
                violations.append(f"{strategy_id}: Position {position_pct:.1%} exceeds max {self.risk_limits['max_position_percentage']:.1%}")
        
        return len(violations) == 0, violations
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, StrategyPosition],
                                  performance: Dict[str, StrategyPerformance]) -> PortfolioMetrics:
        """Calculate portfolio-level risk metrics"""
        # This is simplified - real implementation would be more complex
        
        # Weighted average of strategy metrics
        total_value = sum(p.current_value for p in positions.values())
        
        if total_value == 0:
            return PortfolioMetrics(
                total_value=0,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                volatility=0,
                beta=0,
                var_95=0,
                avg_correlation=0,
                strategy_correlations=pd.DataFrame(),
                last_rebalance=datetime.now(),
                days_since_rebalance=0
            )
        
        weighted_return = sum(
            positions[s].current_value / total_value * perf.total_return
            for s, perf in performance.items()
            if s in positions and positions[s].current_value > 0
        )
        
        weighted_sharpe = sum(
            positions[s].current_value / total_value * perf.sharpe_ratio
            for s, perf in performance.items()
            if s in positions and positions[s].current_value > 0
        )
        
        weighted_drawdown = sum(
            positions[s].current_value / total_value * perf.max_drawdown
            for s, perf in performance.items()
            if s in positions and positions[s].current_value > 0
        )
        
        weighted_volatility = sum(
            positions[s].current_value / total_value * perf.volatility
            for s, perf in performance.items()
            if s in positions and positions[s].current_value > 0
        )
        
        # Calculate average correlation (if available)
        avg_correlation = 0.5  # Default
        if hasattr(self, 'monitor') and hasattr(self.monitor, 'correlation_matrix') and not self.monitor.correlation_matrix.empty:
            corr_matrix = self.monitor.correlation_matrix.values
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, 0)
            avg_correlation = corr_matrix[mask].mean()
        
        return PortfolioMetrics(
            total_value=total_value,
            total_return=weighted_return,
            sharpe_ratio=weighted_sharpe,
            max_drawdown=weighted_drawdown,
            volatility=weighted_volatility,
            beta=1.0,  # Would need market data to calculate
            var_95=-weighted_volatility * 1.65,  # Simplified VaR
            avg_correlation=avg_correlation,
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
        
        # Strategy to token mapping
        # This maps strategy IDs to the tokens they trade
        # Can be loaded from config or strategy registry
        self.strategy_token_map = self._load_strategy_token_map()
        
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
        """Get current value of each strategy in portfolio from real wallet"""
        values = {}
        
        try:
            cprint("🔍 Fetching real wallet positions...", "cyan")
            
            # Get USDC balance first
            usdc_value = n.get_token_balance_usd(USDC_ADDRESS)
            cprint(f"💵 USDC Balance: ${usdc_value:.2f}", "green")
            
            # Get all wallet holdings
            holdings_df = n.fetch_wallet_holdings_og(address)
            
            if holdings_df.empty:
                cprint("⚠️ No holdings found in wallet", "yellow")
                # Return equal distribution of USDC among strategies
                strategy_count = len(self.config.target_allocations)
                if strategy_count > 0:
                    value_per_strategy = usdc_value / strategy_count
                    return {strategy_id: value_per_strategy 
                            for strategy_id in self.config.target_allocations.keys()}
                return {}
            
            # Map strategies to their token holdings
            # For now, we'll distribute the total portfolio value according to actual holdings
            # This will be enhanced when we add strategy-token mapping
            total_portfolio_value = holdings_df['USD Value'].sum()
            
            # If we have a strategy-token map, use it
            if hasattr(self, 'strategy_token_map') and self.strategy_token_map:
                for strategy_id, token_list in self.strategy_token_map.items():
                    strategy_value = 0
                    for token in token_list:
                        token_holdings = holdings_df[holdings_df['Mint Address'] == token]
                        if not token_holdings.empty:
                            strategy_value += token_holdings['USD Value'].iloc[0]
                    values[strategy_id] = strategy_value
            else:
                # Fallback: distribute based on target allocations
                # This is temporary until we implement strategy-token mapping
                for strategy_id, target_weight in self.config.target_allocations.items():
                    values[strategy_id] = total_portfolio_value * target_weight
            
            cprint(f"💎 Total Portfolio Value: ${total_portfolio_value:.2f}", "green")
            return values
            
        except Exception as e:
            cprint(f"❌ Error getting wallet positions: {str(e)}", "red")
            # Fallback to simulation if real data fails
            cprint("⚠️ Falling back to simulated values", "yellow")
            total = 10000
            for strategy_id, target_weight in self.config.target_allocations.items():
                drift = np.random.normal(0, 0.05)
                actual_weight = target_weight * (1 + drift)
                values[strategy_id] = total * actual_weight
            
            current_total = sum(values.values())
            scale = total / current_total
            return {k: v * scale for k, v in values.items()}
    
    def _get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for each strategy using real data"""
        performance = {}
        
        # Initialize performance tracker if not already done
        if not hasattr(self, 'performance_tracker'):
            from src.agents.portfolio_performance_tracker import PortfolioPerformanceTracker
            self.performance_tracker = PortfolioPerformanceTracker()
        
        for strategy_id in self.config.target_allocations:
            try:
                # First, try to get real performance from tracker
                perf_metrics = self.performance_tracker.get_strategy_performance(
                    strategy_id, lookback_days=30
                )
                
                # Check if we have meaningful data
                if perf_metrics['data_points'] >= 5:  # Need at least 5 days of data
                    performance[strategy_id] = StrategyPerformance(
                        strategy_id=strategy_id,
                        returns=perf_metrics['returns'],
                        sharpe_ratio=perf_metrics['sharpe_ratio'],
                        max_drawdown=perf_metrics['max_drawdown'],
                        win_rate=perf_metrics['win_rate'],
                        total_return=perf_metrics['total_return'],
                        volatility=perf_metrics['volatility'],
                        current_allocation=self.monitor.positions.get(
                            strategy_id, 
                            StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                           current_weight=0, target_weight=0, drift=0)
                        ).current_weight,
                        target_allocation=self.config.target_allocations[strategy_id],
                        drift=self.monitor.positions.get(
                            strategy_id,
                            StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                           current_weight=0, target_weight=0, drift=0)
                        ).drift
                    )
                    cprint(f"✅ Loaded real performance for {strategy_id}", "green")
                    continue
                
            except Exception as e:
                cprint(f"⚠️ Could not get real performance for {strategy_id}: {str(e)}", "yellow")
            
            # Fallback: Check registry for performance data
            strategy_info = self.registry.get_strategy(strategy_id)
            
            if strategy_info and strategy_info.get("performance_data"):
                perf_data = strategy_info["performance_data"]
                performance[strategy_id] = StrategyPerformance(
                    strategy_id=strategy_id,
                    returns=pd.Series(),  # Empty series if no real data
                    sharpe_ratio=perf_data.get("sharpe_ratio", 0.5),
                    max_drawdown=perf_data.get("max_drawdown", -0.15),
                    win_rate=perf_data.get("win_rate", 0.5),
                    total_return=perf_data.get("total_return", 0.1),
                    volatility=perf_data.get("volatility", 0.2),
                    current_allocation=self.monitor.positions.get(
                        strategy_id,
                        StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                       current_weight=0, target_weight=0, drift=0)
                    ).current_weight,
                    target_allocation=self.config.target_allocations[strategy_id],
                    drift=self.monitor.positions.get(
                        strategy_id,
                        StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                       current_weight=0, target_weight=0, drift=0)
                    ).drift
                )
                cprint(f"📊 Loaded registry performance for {strategy_id}", "cyan")
            else:
                # Default performance if no real data available
                performance[strategy_id] = StrategyPerformance(
                    strategy_id=strategy_id,
                    returns=pd.Series(),
                    sharpe_ratio=0.5,
                    max_drawdown=-0.15,
                    win_rate=0.55,
                    total_return=0.10,
                    volatility=0.20,
                    current_allocation=self.monitor.positions.get(
                        strategy_id,
                        StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                       current_weight=0, target_weight=0, drift=0)
                    ).current_weight,
                    target_allocation=self.config.target_allocations[strategy_id],
                    drift=self.monitor.positions.get(
                        strategy_id,
                        StrategyPosition(strategy_id=strategy_id, current_value=0, 
                                       current_weight=0, target_weight=0, drift=0)
                    ).drift
                )
                cprint(f"ℹ️ Using default performance for {strategy_id}", "white")
        
        return performance
    
    def _execute_orders(self, orders: List[Order]) -> Dict[str, Any]:
        """Execute rebalancing orders with real trades"""
        results = {
            "executed": 0,
            "failed": 0,
            "total_traded": 0,
            "execution_time": datetime.now(),
            "execution_details": []
        }
        
        try:
            # Check if we're in dry run mode (safety feature)
            dry_run = getattr(self.config, 'dry_run', True)
            
            if dry_run:
                cprint("🏃 DRY RUN MODE - No real trades will be executed", "yellow")
            
            # Execute sells first to free up capital
            sells = [o for o in orders if o.side == OrderSide.SELL]
            buys = [o for o in orders if o.side == OrderSide.BUY]
            
            # Process sell orders
            for order in sells:
                try:
                    cprint(f"\n💰 Executing SELL: {order.strategy_id} - ${order.amount_usd:.2f}", "yellow")
                    
                    if not dry_run:
                        # Get tokens for this strategy
                        tokens_to_sell = self.strategy_token_map.get(order.strategy_id, [])
                        
                        if not tokens_to_sell:
                            cprint(f"⚠️ No tokens mapped for strategy {order.strategy_id}", "yellow")
                            order.execution_error = "No token mapping"
                            results["failed"] += 1
                            continue
                        
                        # Calculate amount per token if multiple tokens
                        amount_per_token = order.amount_usd / len(tokens_to_sell)
                        
                        for token in tokens_to_sell:
                            # Use chunk_kill for safety (handles partial sells)
                            tx_hash = n.chunk_kill(
                                token_mint_address=token,
                                max_usd_order_size=amount_per_token,
                                slippage=199  # Use config slippage
                            )
                            
                            if tx_hash:
                                cprint(f"✅ Sold ${amount_per_token:.2f} of {token[:8]}... TX: {tx_hash}", "green")
                            else:
                                cprint(f"❌ Failed to sell {token[:8]}...", "red")
                    
                    order.executed = True
                    order.execution_time = datetime.now()
                    results["executed"] += 1
                    results["total_traded"] += order.amount_usd
                    
                except Exception as e:
                    cprint(f"❌ Error executing sell order: {str(e)}", "red")
                    order.execution_error = str(e)
                    results["failed"] += 1
                    results["execution_details"].append({
                        "order": order.strategy_id,
                        "side": "SELL",
                        "error": str(e)
                    })
            
            # Wait between sells and buys
            if sells and buys and not dry_run:
                cprint("\n⏳ Waiting 5 seconds between sells and buys...", "cyan")
                time.sleep(5)
            
            # Process buy orders
            for order in buys:
                try:
                    cprint(f"\n🛒 Executing BUY: {order.strategy_id} - ${order.amount_usd:.2f}", "green")
                    
                    if not dry_run:
                        # Get tokens for this strategy
                        tokens_to_buy = self.strategy_token_map.get(order.strategy_id, [])
                        
                        if not tokens_to_buy:
                            cprint(f"⚠️ No tokens mapped for strategy {order.strategy_id}", "yellow")
                            order.execution_error = "No token mapping"
                            results["failed"] += 1
                            continue
                        
                        # Calculate amount per token if multiple tokens
                        amount_per_token = order.amount_usd / len(tokens_to_buy)
                        
                        for token in tokens_to_buy:
                            # Use market_buy for purchases
                            tx_hash = n.market_buy(
                                token=token,
                                amount=amount_per_token,
                                slippage=199  # Use config slippage
                            )
                            
                            if tx_hash:
                                cprint(f"✅ Bought ${amount_per_token:.2f} of {token[:8]}... TX: {tx_hash}", "green")
                            else:
                                cprint(f"❌ Failed to buy {token[:8]}...", "red")
                    
                    order.executed = True
                    order.execution_time = datetime.now()
                    results["executed"] += 1
                    results["total_traded"] += order.amount_usd
                    
                except Exception as e:
                    cprint(f"❌ Error executing buy order: {str(e)}", "red")
                    order.execution_error = str(e)
                    results["failed"] += 1
                    results["execution_details"].append({
                        "order": order.strategy_id,
                        "side": "BUY",
                        "error": str(e)
                    })
            
            # Summary
            cprint(f"\n📊 Execution Summary:", "cyan")
            cprint(f"  Executed: {results['executed']}/{len(orders)}", "white")
            cprint(f"  Failed: {results['failed']}", "red" if results['failed'] > 0 else "white")
            cprint(f"  Total Traded: ${results['total_traded']:.2f}", "green")
            
        except Exception as e:
            cprint(f"❌ Critical error in order execution: {str(e)}", "red")
            results["execution_details"].append({
                "error": str(e),
                "type": "critical"
            })
        
        return results
    
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
    
    def _load_strategy_token_map(self) -> Dict[str, List[str]]:
        """Load or create strategy to token mapping"""
        try:
            # Try to load from saved mapping file
            mapping_file = self.data_dir / "strategy_token_map.json"
            
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                cprint(f"📋 Loaded strategy-token mapping for {len(mapping)} strategies", "green")
                return mapping
            
            # Otherwise, try to infer from strategy registry
            mapping = {}
            
            if self.config and self.config.target_allocations:
                for strategy_id in self.config.target_allocations:
                    strategy_info = self.registry.get_strategy(strategy_id)
                    
                    if strategy_info and "traded_tokens" in strategy_info:
                        mapping[strategy_id] = strategy_info["traded_tokens"]
                    else:
                        # Default mapping based on strategy name hints
                        if "btc" in strategy_id.lower():
                            # This would be a BTC token address on Solana
                            mapping[strategy_id] = ["3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh"]  # Wrapped BTC
                        elif "eth" in strategy_id.lower():
                            mapping[strategy_id] = ["7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"]  # Wrapped ETH
                        elif "sol" in strategy_id.lower():
                            mapping[strategy_id] = [SOL_ADDRESS]
                        else:
                            # Default to monitored tokens
                            if MONITORED_TOKENS:
                                # Distribute monitored tokens among strategies
                                mapping[strategy_id] = MONITORED_TOKENS
                            else:
                                cprint(f"⚠️ No token mapping found for {strategy_id}", "yellow")
            
            # Save the mapping
            if mapping:
                with open(mapping_file, 'w') as f:
                    json.dump(mapping, f, indent=2)
                cprint(f"💾 Saved strategy-token mapping for {len(mapping)} strategies", "green")
            
            return mapping
            
        except Exception as e:
            cprint(f"❌ Error loading strategy-token mapping: {str(e)}", "red")
            return {}
    
    def save_state(self) -> None:
        """Save current portfolio state and configuration"""
        try:
            state_file = self.data_dir / "state.json"
            
            # Prepare state data
            state = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "name": self.config.name,
                    "target_allocations": self.config.target_allocations,
                    "rebalancing_method": self.config.rebalancing_method.value,
                    "rebalancing_params": self.config.rebalancing_params,
                    "risk_limits": self.config.risk_limits
                },
                "positions": {
                    strategy_id: {
                        "current_value": pos.current_value,
                        "current_weight": pos.current_weight,
                        "target_weight": pos.target_weight,
                        "drift": pos.drift
                    }
                    for strategy_id, pos in self.monitor.positions.items()
                },
                "strategy_token_map": self.strategy_token_map,
                "last_rebalance": self.monitor.last_rebalance.isoformat() if self.monitor.last_rebalance else None,
                "rebalancing_history_count": len(self.rebalancing_history),
                "portfolio_metrics": {
                    "total_value": sum(p.current_value for p in self.monitor.positions.values()),
                    "position_count": len(self.monitor.positions)
                }
            }
            
            # Save state
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            cprint(f"💾 Portfolio state saved to: {state_file}", "green")
            
            # Also save rebalancing history
            if self.rebalancing_history:
                history_file = self.data_dir / "rebalancing_history.csv"
                history_data = []
                
                for event in self.rebalancing_history:
                    history_data.append({
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "trigger": event.trigger,
                        "orders_count": len(event.orders),
                        "total_traded": event.total_traded,
                        "execution_status": event.execution_summary.get("executed", 0)
                    })
                
                df = pd.DataFrame(history_data)
                df.to_csv(history_file, index=False)
                cprint(f"📊 Rebalancing history saved ({len(history_data)} events)", "green")
            
        except Exception as e:
            cprint(f"❌ Error saving state: {str(e)}", "red")
    
    def load_state(self) -> bool:
        """Load saved portfolio state"""
        try:
            state_file = self.data_dir / "state.json"
            
            if not state_file.exists():
                cprint("ℹ️ No saved state found, starting fresh", "white")
                return False
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore strategy token map
            if "strategy_token_map" in state:
                self.strategy_token_map = state["strategy_token_map"]
                cprint(f"✅ Loaded strategy-token mapping", "green")
            
            # Restore last rebalance time
            if state.get("last_rebalance"):
                self.monitor.last_rebalance = datetime.fromisoformat(state["last_rebalance"])
                days_since = (datetime.now() - self.monitor.last_rebalance).days
                cprint(f"⏰ Last rebalance: {days_since} days ago", "cyan")
            
            # Load rebalancing history
            history_file = self.data_dir / "rebalancing_history.csv"
            if history_file.exists():
                history_df = pd.read_csv(history_file)
                cprint(f"📜 Loaded {len(history_df)} rebalancing events", "green")
            
            cprint(f"✅ Portfolio state loaded from: {state_file}", "green")
            return True
            
        except Exception as e:
            cprint(f"❌ Error loading state: {str(e)}", "red")
            return False
    
    def check_alerts(self) -> None:
        """Check for portfolio alerts and warnings"""
        try:
            # Get current portfolio value
            total_value = sum(p.current_value for p in self.monitor.positions.values())
            
            # Check portfolio-level alerts
            alerts = []
            
            # 1. Check minimum balance
            if total_value < self.risk_manager.risk_limits["minimum_balance_usd"]:
                alerts.append({
                    "level": "critical",
                    "message": f"⚠️ Portfolio value ${total_value:.2f} below minimum ${self.risk_manager.risk_limits['minimum_balance_usd']}",
                    "action": "Consider adding funds or closing positions"
                })
            
            # 2. Check drift alerts
            for strategy_id, position in self.monitor.positions.items():
                if abs(position.drift) > 0.15:  # 15% drift
                    alerts.append({
                        "level": "warning",
                        "message": f"📊 {strategy_id} drift: {position.drift:+.1%}",
                        "action": "Rebalancing recommended"
                    })
            
            # 3. Check correlation alerts
            corr_analysis = self.monitor.analyze_correlation_risk()
            if corr_analysis["risk_level"] == "high":
                alerts.append({
                    "level": "warning",
                    "message": "🔗 High strategy correlation detected",
                    "action": corr_analysis["recommendation"]
                })
            
            # 4. Check time since last rebalance
            days_since = (datetime.now() - self.monitor.last_rebalance).days
            if days_since > 30:
                alerts.append({
                    "level": "info",
                    "message": f"📅 {days_since} days since last rebalance",
                    "action": "Consider reviewing portfolio allocations"
                })
            
            # 5. Check performance alerts
            performance = self._get_strategy_performance()
            for strategy_id, perf in performance.items():
                if perf.max_drawdown < -0.25:  # 25% drawdown
                    alerts.append({
                        "level": "warning",
                        "message": f"📉 {strategy_id} drawdown: {perf.max_drawdown:.1%}",
                        "action": "Review strategy performance"
                    })
            
            # Display alerts
            if alerts:
                cprint("\n🚨 Portfolio Alerts:", "yellow", attrs=["bold"])
                cprint("=" * 50, "yellow")
                
                for alert in alerts:
                    color = "red" if alert["level"] == "critical" else "yellow" if alert["level"] == "warning" else "cyan"
                    cprint(f"\n{alert['message']}", color)
                    cprint(f"   → {alert['action']}", "white")
            else:
                cprint("\n✅ No portfolio alerts", "green")
            
            # Record alert check
            self.last_alert_check = datetime.now()
            
        except Exception as e:
            cprint(f"❌ Error checking alerts: {str(e)}", "red")
    
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
    
    def run(self) -> None:
        """Main execution loop for production use"""
        cprint("\n🌙 Moon Dev Portfolio Rebalancing Agent", "cyan", attrs=["bold"])
        cprint("=" * 60, "blue")
        cprint("Starting automated portfolio management...", "white")
        
        try:
            # Load saved state
            self.load_state()
            
            # Main loop
            while True:
                try:
                    cprint(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white")
                    
                    # Display dashboard
                    self.display_portfolio_dashboard()
                    
                    # Check alerts
                    self.check_alerts()
                    
                    # Check and rebalance if needed
                    result = self.check_and_rebalance()
                    
                    # Save state after each run
                    self.save_state()
                    
                    # Save position snapshot for performance tracking
                    if hasattr(self, 'performance_tracker'):
                        strategy_positions = {}
                        for strategy_id, position in self.monitor.positions.items():
                            tokens = self.strategy_token_map.get(strategy_id, [])
                            strategy_positions[strategy_id] = {
                                "tokens": {
                                    token: {
                                        "amount": 0,  # Would get actual amount
                                        "usd_value": position.current_value / len(tokens) if tokens else 0,
                                        "price": 0  # Would get actual price
                                    }
                                    for token in tokens
                                }
                            }
                        self.performance_tracker.save_position_snapshot(strategy_positions)
                    
                    # Sleep between runs
                    sleep_minutes = 15  # Default, or use config.SLEEP_BETWEEN_RUNS_MINUTES
                    cprint(f"\n💤 Sleeping for {sleep_minutes} minutes...", "cyan")
                    cprint("Press Ctrl+C to exit gracefully\n", "white")
                    
                    time.sleep(sleep_minutes * 60)
                    
                except KeyboardInterrupt:
                    cprint("\n⚠️ Keyboard interrupt detected", "yellow")
                    break
                except Exception as e:
                    cprint(f"\n❌ Error in main loop: {str(e)}", "red")
                    cprint("Continuing after 60 second pause...", "yellow")
                    time.sleep(60)
            
            # Graceful shutdown
            cprint("\n🛑 Shutting down portfolio rebalancing agent...", "yellow")
            self.save_state()
            cprint("✅ State saved. Goodbye! 🌙", "green")
            
        except Exception as e:
            cprint(f"❌ Critical error: {str(e)}", "red")
            raise


def main():
    """Main entry point for portfolio rebalancing agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Moon Dev Portfolio Rebalancing Agent 🌙')
    parser.add_argument('--mode', choices=['demo', 'production'], default='demo',
                      help='Run mode: demo for testing, production for live trading')
    parser.add_argument('--config-file', type=str, help='Path to portfolio config JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Run without executing real trades')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Demo mode
        cprint("\n🌙 Moon Dev Portfolio Rebalancing Demo", "cyan", attrs=["bold"])
        cprint("=" * 50, "blue")
        
        # Create sample portfolio configuration
        config = PortfolioConfig(
            name="Demo Balanced Portfolio",
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
        
        # Force dry run in demo mode
        config.dry_run = True
        
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
        
    else:
        # Production mode
        cprint("\n🚀 Moon Dev Portfolio Rebalancing Agent - PRODUCTION MODE", "red", attrs=["bold"])
        cprint("=" * 60, "red")
        
        # Load config from file or use default
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            
            config = PortfolioConfig(
                name=config_data["name"],
                target_allocations=config_data["target_allocations"],
                rebalancing_method=RebalancingMethod(config_data["rebalancing_method"]),
                rebalancing_params=config_data.get("rebalancing_params", DEFAULT_REBALANCING_PARAMS["threshold"]),
                risk_limits=config_data.get("risk_limits", DEFAULT_RISK_LIMITS)
            )
        else:
            # Default production config
            config = PortfolioConfig(
                name="Moon Dev Production Portfolio",
                target_allocations={
                    "momentum_btc": 0.40,
                    "mean_reversion_eth": 0.30,
                    "arbitrage_sol": 0.30
                },
                rebalancing_method=RebalancingMethod.THRESHOLD,
                rebalancing_params={
                    **DEFAULT_REBALANCING_PARAMS["threshold"],
                    "min_drift": 0.05  # 5% drift threshold
                },
                risk_limits=DEFAULT_RISK_LIMITS
            )
        
        # Set dry run mode
        config.dry_run = args.dry_run
        
        if args.dry_run:
            cprint("\n🏃 DRY RUN MODE - No real trades will be executed", "yellow", attrs=["bold"])
        else:
            cprint("\n💰 LIVE TRADING MODE - Real trades will be executed!", "red", attrs=["bold"])
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                cprint("Exiting...", "yellow")
                return
        
        # Validate config
        config.validate()
        
        # Initialize agent
        agent = PortfolioRebalancingAgent(config)
        
        # Run main loop
        agent.run()


if __name__ == "__main__":
    main()