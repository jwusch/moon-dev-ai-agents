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