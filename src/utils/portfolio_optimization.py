"""
Portfolio optimization algorithms for advanced rebalancing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from termcolor import cprint

# Try to import scipy, but provide fallback
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cprint("⚠️ scipy not installed - using simplified optimization methods", "yellow")

def risk_parity_weights(returns: pd.DataFrame, 
                       initial_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate risk parity weights where each asset contributes equally to portfolio risk
    """
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    
    if not SCIPY_AVAILABLE:
        # Simplified inverse volatility weighting as fallback
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        return dict(zip(returns.columns, weights))
    
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
    
    if not SCIPY_AVAILABLE:
        # Simplified Sharpe ratio weighting as fallback
        sharpe_ratios = mean_returns / np.sqrt(np.diag(cov_matrix))
        sharpe_ratios = np.maximum(sharpe_ratios, 0)  # No negative weights
        if sharpe_ratios.sum() > 0:
            weights = sharpe_ratios / sharpe_ratios.sum()
        else:
            weights = np.ones(n_assets) / n_assets
        return dict(zip(returns.columns, weights))
    
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