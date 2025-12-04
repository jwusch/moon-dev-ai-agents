"""
🌊 Multifractal Analysis Tests
Tests for multifractal and scaling analysis indicators
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from ..indicators.multifractal.hurst_exponent import HurstExponentIndicator
from ..indicators.multifractal.dfa import DFAIndicator
from ..base.types import SignalType
from .fixtures import SyntheticDataGenerator, get_test_data_for_scenario


class TestHurstExponentIndicator:
    """Test Hurst Exponent calculation"""
    
    def test_hurst_random_walk(self):
        """Test Hurst exponent for random walk should be ~0.5"""
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_random_walk(1000, volatility=0.01)
        
        indicator = HurstExponentIndicator(lookback=500, min_periods=100)
        result = indicator.calculate(data, "RANDOM_WALK")
        
        # Random walk should have Hurst ~0.5 (within tolerance)
        assert abs(result.value - 50) < 15  # Allow 15% tolerance
        assert 'random' in result.metadata['market_regime'].lower()
        
    def test_hurst_trending_data(self):
        """Test Hurst exponent for trending data should be > 0.5"""
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_trending_data(1000, trend_strength=0.002)
        
        indicator = HurstExponentIndicator(lookback=500)
        result = indicator.calculate(data, "TRENDING")
        
        # Trending data should have Hurst > 0.5
        assert result.value > 50
        assert 'trend' in result.metadata['market_regime'].lower()
        
    def test_hurst_mean_reverting_data(self):
        """Test Hurst exponent for mean-reverting data should be < 0.5"""
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_mean_reverting_data(1000, reversion_strength=0.1)
        
        indicator = HurstExponentIndicator(lookback=500)
        result = indicator.calculate(data, "MEAN_REVERTING")
        
        # Mean-reverting data should have Hurst < 0.5
        assert result.value < 50
        assert 'revert' in result.metadata['market_regime'].lower()
        
    def test_hurst_calculation_methods(self):
        """Test different Hurst calculation methods"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_random_walk(500)
        
        methods = ['rs', 'dfa', 'rescaled_variance']
        
        for method in methods:
            indicator = HurstExponentIndicator(method=method, lookback=400)
            result = indicator.calculate(data, f"TEST_{method}")
            
            # Should return valid Hurst values
            assert 0 <= result.value <= 100
            assert result.metadata['calculation_method'] == method
            
            # For random walk, all methods should give similar results
            hurst_actual = result.value / 100
            assert 0.3 <= hurst_actual <= 0.7  # Reasonable range for random walk
            
    def test_hurst_confidence_calculation(self):
        """Test confidence calculation based on data quality"""
        
        generator = SyntheticDataGenerator()
        
        # High quality data (long series, clear pattern)
        long_trending = generator.generate_trending_data(2000, trend_strength=0.003)
        
        # Low quality data (short series, noisy)
        short_noisy = generator.generate_random_walk(100, volatility=0.05)
        
        indicator = HurstExponentIndicator()
        
        result_high = indicator.calculate(long_trending, "HIGH_QUALITY")
        result_low = indicator.calculate(short_noisy, "LOW_QUALITY")
        
        # High quality data should have higher confidence
        assert result_high.confidence > result_low.confidence
        
    def test_hurst_rolling_window(self):
        """Test rolling window functionality"""
        
        generator = SyntheticDataGenerator()
        
        # Create regime-changing data
        regime_data = generator.generate_regime_change_data(1500)
        
        indicator = HurstExponentIndicator(lookback=300, rolling=True)
        result = indicator.calculate(regime_data, "REGIME_CHANGE")
        
        # Should detect regime changes
        assert 'stability_score' in result.metadata
        assert 'regime_changes_detected' in result.metadata
        
        # Stability should be low for regime-changing data
        assert result.metadata['stability_score'] < 0.8
        
    def test_hurst_signal_generation(self):
        """Test signal generation logic"""
        
        generator = SyntheticDataGenerator()
        
        # Strong trending regime
        trending = generator.generate_trending_data(500, trend_strength=0.004)
        
        # Strong mean-reverting regime  
        reverting = generator.generate_mean_reverting_data(500, reversion_strength=0.2)
        
        indicator = HurstExponentIndicator()
        
        trend_result = indicator.calculate(trending, "TREND_TEST")
        revert_result = indicator.calculate(reverting, "REVERT_TEST")
        
        # Trending should suggest trend-following
        if trend_result.confidence > 60:
            assert 'trend' in trend_result.metadata['recommended_strategy']
            
        # Mean-reverting should suggest mean-reversion
        if revert_result.confidence > 60:
            assert 'reversion' in revert_result.metadata['recommended_strategy']


class TestDFAIndicator:
    """Test Detrended Fluctuation Analysis"""
    
    def test_dfa_scaling_exponents(self):
        """Test DFA scaling exponent calculation"""
        
        generator = SyntheticDataGenerator(seed=42)
        
        # Test known data patterns
        random_walk = generator.generate_random_walk(1000)
        trending = generator.generate_trending_data(1000, trend_strength=0.002)
        mean_reverting = generator.generate_mean_reverting_data(1000, reversion_strength=0.1)
        
        indicator = DFAIndicator(min_scale=10, max_scale=100, n_scales=15)
        
        # Random walk: α ≈ 0.5
        rw_result = indicator.calculate(random_walk, "RW_TEST")
        assert 0.4 <= rw_result.metadata['scaling_exponent'] <= 0.6
        
        # Trending: α > 0.5
        trend_result = indicator.calculate(trending, "TREND_TEST")
        assert trend_result.metadata['scaling_exponent'] > 0.5
        
        # Mean-reverting: α < 0.5
        mr_result = indicator.calculate(mean_reverting, "MR_TEST")
        assert mr_result.metadata['scaling_exponent'] < 0.5
        
    def test_dfa_multifractal_analysis(self):
        """Test multifractal DFA analysis"""
        
        generator = SyntheticDataGenerator()
        
        # Generate complex multifractal-like data
        data = generator.generate_regime_change_data(1200)
        
        indicator = DFAIndicator(
            min_scale=8,
            max_scale=150,
            n_scales=20,
            multi_scale=True
        )
        
        result = indicator.calculate(data, "MULTIFRACTAL_TEST")
        
        # Should have multifractal analysis
        if 'multifractal_width' in result.metadata:
            assert result.metadata['multifractal'] == True
            assert 'q_values' in result.metadata
            assert 'hq_values' in result.metadata
            assert 'spectrum_width' in result.metadata
            
            # Width should be positive
            assert result.metadata['multifractal_width'] >= 0
            
    def test_dfa_crossover_detection(self):
        """Test crossover scale detection"""
        
        generator = SyntheticDataGenerator()
        
        # Create data with different behavior at different scales
        data = generator.generate_regime_change_data(1000)
        
        indicator = DFAIndicator(min_scale=5, max_scale=200, n_scales=25)
        result = indicator.calculate(data, "CROSSOVER_TEST")
        
        # Should analyze different scales
        assert 'n_scales_analyzed' in result.metadata
        assert result.metadata['n_scales_analyzed'] > 10
        
        # May detect crossovers
        if 'crossover_scales' in result.metadata:
            crossovers = result.metadata['crossover_scales']
            assert isinstance(crossovers, list)
            
    def test_dfa_r_squared_quality(self):
        """Test R-squared quality metric"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_random_walk(800)
        
        indicator = DFAIndicator(min_scale=10, max_scale=100)
        result = indicator.calculate(data, "R_SQUARED_TEST")
        
        # Should have R-squared metric
        assert 'r_squared' in result.metadata
        r_squared = result.metadata['r_squared']
        
        # R-squared should be reasonable (good fit)
        assert 0 <= r_squared <= 1
        
        # For clean synthetic data, should be high
        assert r_squared > 0.7
        
    def test_dfa_detrending_orders(self):
        """Test different polynomial detrending orders"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_trending_data(600)  # With trend
        
        # Test different detrending orders
        for order in [1, 2, 3]:
            indicator = DFAIndicator(
                min_scale=8,
                max_scale=80,
                detrend_order=order
            )
            result = indicator.calculate(data, f"DETREND_{order}")
            
            # Should successfully calculate
            assert 'scaling_exponent' in result.metadata
            assert np.isfinite(result.metadata['scaling_exponent'])
            
    def test_dfa_signal_confidence(self):
        """Test signal confidence based on statistical quality"""
        
        generator = SyntheticDataGenerator()
        
        # High quality data (clear pattern, long series)
        clear_trend = generator.generate_trending_data(1500, trend_strength=0.005)
        
        # Noisy data (short series, high noise)
        noisy_data = generator.generate_random_walk(200, volatility=0.1)
        
        indicator = DFAIndicator()
        
        clear_result = indicator.calculate(clear_trend, "CLEAR")
        noisy_result = indicator.calculate(noisy_data, "NOISY")
        
        # Clear pattern should have higher confidence
        assert clear_result.confidence >= noisy_result.confidence
        
        # Clear trend should be detected
        assert clear_result.metadata['market_behavior'] in [
            'trending', 'strong_trending', 'persistent'
        ]
        
    def test_dfa_trading_strategy_recommendation(self):
        """Test trading strategy recommendation"""
        
        generator = SyntheticDataGenerator()
        
        # Strong mean-reverting data
        mr_data = generator.generate_mean_reverting_data(800, reversion_strength=0.2)
        
        # Strong trending data
        trend_data = generator.generate_trending_data(800, trend_strength=0.003)
        
        indicator = DFAIndicator()
        
        mr_result = indicator.calculate(mr_data, "MR_STRATEGY")
        trend_result = indicator.calculate(trend_data, "TREND_STRATEGY")
        
        # Should recommend appropriate strategies
        assert mr_result.metadata['trading_strategy'] == 'mean_reversion'
        assert trend_result.metadata['trading_strategy'] == 'trend_following'
        
        # Holding periods should be different
        assert mr_result.metadata['optimal_holding_period'] == 'short'
        assert trend_result.metadata['optimal_holding_period'] in ['medium_to_long', 'medium', 'long']
        
    def test_dfa_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        
        generator = SyntheticDataGenerator()
        
        # Very short data series
        short_data = generator.generate_random_walk(20)
        
        indicator = DFAIndicator(min_scale=10, max_scale=50)
        result = indicator.calculate(short_data, "SHORT")
        
        # Should handle gracefully
        assert result.confidence == 0.0
        assert result.signal == SignalType.HOLD
        assert 'error' in result.metadata
        
    def test_dfa_regime_classification(self):
        """Test market regime classification accuracy"""
        
        generator = SyntheticDataGenerator()
        
        # Test all regime types
        regimes = {
            'random': generator.generate_random_walk(600),
            'trending': generator.generate_trending_data(600, trend_strength=0.003),
            'mean_reverting': generator.generate_mean_reverting_data(600, reversion_strength=0.15)
        }
        
        indicator = DFAIndicator()
        
        for regime_name, data in regimes.items():
            result = indicator.calculate(data, f"{regime_name.upper()}_REGIME")
            
            # Check regime detection
            detected_behavior = result.metadata['market_behavior']
            
            if regime_name == 'random':
                assert 'efficient' in detected_behavior
            elif regime_name == 'trending':
                assert 'trend' in detected_behavior
            elif regime_name == 'mean_reverting':
                assert 'revert' in detected_behavior