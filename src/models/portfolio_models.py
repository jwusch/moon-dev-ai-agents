"""
🌙 Portfolio Data Models for Moon Dev Trading System
Defines core data structures for portfolio management and rebalancing
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import pandas as pd


class RebalancingMethod(Enum):
    """Available rebalancing methods"""
    THRESHOLD = "threshold"
    CALENDAR = "calendar"
    ADAPTIVE = "adaptive"
    RISK_PARITY = "risk_parity"


class OrderSide(Enum):
    """Order side for rebalancing trades"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class PortfolioConfig:
    """Configuration for a portfolio"""
    name: str
    target_allocations: Dict[str, float]  # strategy_id -> weight (0-1)
    rebalancing_method: RebalancingMethod
    rebalancing_params: Dict[str, Any]
    risk_limits: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        """Validate portfolio configuration"""
        # Check weights sum to 1.0 (with small tolerance)
        weight_sum = sum(self.target_allocations.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        # Check all weights are non-negative
        for strategy, weight in self.target_allocations.items():
            if weight < 0:
                raise ValueError(f"Negative weight {weight} for {strategy}")
        
        return True


@dataclass
class StrategyPosition:
    """Current position for a strategy"""
    strategy_id: str
    current_value: float
    current_weight: float
    target_weight: float
    drift: float  # current_weight - target_weight
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Order:
    """Rebalancing order"""
    strategy_id: str
    side: OrderSide
    amount_usd: float
    reason: str
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_time: Optional[datetime] = None
    execution_price: Optional[float] = None
    
    def __str__(self):
        return f"{self.side.value.upper()} ${self.amount_usd:.2f} of {self.strategy_id}"


@dataclass
class RebalancingEvent:
    """Record of a rebalancing event"""
    event_id: str
    timestamp: datetime
    trigger: str  # "drift", "calendar", "risk", "manual"
    pre_weights: Dict[str, float]
    post_weights: Dict[str, float]
    orders: List[Order]
    execution_summary: Dict[str, Any]
    performance_impact: Dict[str, float]
    
    @property
    def total_traded(self) -> float:
        """Total USD traded in rebalancing"""
        return sum(order.amount_usd for order in self.orders)
    
    @property
    def strategies_affected(self) -> List[str]:
        """List of strategies affected"""
        return list(set(order.strategy_id for order in self.orders))


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_id: str
    returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    volatility: float
    current_allocation: float
    target_allocation: float
    drift: float
    
    @property
    def needs_rebalance(self) -> bool:
        """Check if strategy needs rebalancing based on drift"""
        return abs(self.drift) > 0.05  # 5% default threshold


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics"""
    total_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    var_95: float  # Value at Risk (95% confidence)
    avg_correlation: float
    strategy_correlations: pd.DataFrame
    last_rebalance: datetime
    days_since_rebalance: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display"""
        return {
            "total_value": self.total_value,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "beta": self.beta,
            "var_95": self.var_95,
            "avg_correlation": self.avg_correlation,
            "last_rebalance": self.last_rebalance.strftime("%Y-%m-%d"),
            "days_since_rebalance": self.days_since_rebalance
        }


# Risk Limits Configuration
DEFAULT_RISK_LIMITS = {
    "max_single_strategy": 0.4,      # 40% max allocation
    "min_single_strategy": 0.05,     # 5% minimum (or 0)
    "max_correlation_pair": 0.7,     # Max correlation between any two
    "max_avg_correlation": 0.5,      # Max average correlation
    "min_strategy_sharpe": 0.0,      # Remove negative Sharpe strategies
    "max_strategy_drawdown": 0.3,    # Remove if drawdown > 30%
    "max_portfolio_leverage": 1.0,   # No leverage initially
    "min_portfolio_sharpe": 0.5,     # Target portfolio Sharpe
    "max_portfolio_drawdown": 0.25   # 25% max portfolio DD
}

# Rebalancing Parameters
DEFAULT_REBALANCING_PARAMS = {
    "threshold": {
        "min_drift": 0.05,           # 5% drift triggers rebalancing
        "min_trade_size": 100,       # Minimum $100 trade
        "check_frequency": "daily"    # How often to check
    },
    "calendar": {
        "frequency": "monthly",       # monthly, weekly, quarterly
        "rebalance_day": 1,          # Day of month/week
        "force_rebalance": True      # Rebalance even if no drift
    },
    "adaptive": {
        "volatility_threshold": 0.3,  # High vol threshold
        "correlation_threshold": 0.8, # High correlation threshold
        "performance_window": 30,     # Days for performance calc
        "min_improvement": 0.02      # 2% improvement needed
    },
    "risk_parity": {
        "target_risk_contribution": None,  # Equal if None
        "use_leverage": False,
        "rebalance_frequency": "weekly"
    }
}