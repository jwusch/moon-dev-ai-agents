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