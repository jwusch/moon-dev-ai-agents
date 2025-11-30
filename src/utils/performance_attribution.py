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