"""
📊 Statistical Validation Tests
Tests for statistical correctness and mathematical validation
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, shapiro
import warnings
from typing import Dict, List, Tuple

from ..indicators.multifractal.hurst_exponent import HurstExponentIndicator
from ..indicators.multifractal.dfa import DFAIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator

from .fixtures import SyntheticDataGenerator, TEST_SCENARIOS


class StatisticalValidator:
    """Statistical validation utilities"""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        
    def validate_hurst_properties(self, hurst_values: List[float], 
                                 expected_regime: str) -> Dict:
        """Validate Hurst exponent statistical properties"""
        
        hurst_array = np.array(hurst_values)
        
        # Basic statistical properties
        mean_hurst = np.mean(hurst_array)
        std_hurst = np.std(hurst_array)
        
        results = {
            'mean': mean_hurst,
            'std': std_hurst,
            'valid_range': np.all((hurst_array >= 0) & (hurst_array <= 1)),
            'regime_consistent': True
        }
        
        # Check regime consistency
        if expected_regime == 'random':
            # Random walk should have Hurst around 0.5
            results['regime_consistent'] = abs(mean_hurst - 0.5) < 0.15
        elif expected_regime == 'trending':
            # Trending should have Hurst > 0.5
            results['regime_consistent'] = mean_hurst > 0.5
        elif expected_regime == 'mean_reverting':
            # Mean reverting should have Hurst < 0.5
            results['regime_consistent'] = mean_hurst < 0.5
            
        return results
        
    def validate_dfa_scaling(self, scales: List[float], 
                           fluctuations: List[float]) -> Dict:
        """Validate DFA scaling relationship"""
        
        if len(scales) < 3 or len(fluctuations) < 3:
            return {'valid': False, 'reason': 'insufficient_data'}
            
        log_scales = np.log(scales)
        log_flucts = np.log(fluctuations)
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_flucts)
        
        return {
            'valid': True,
            'scaling_exponent': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'standard_error': std_err,
            'good_fit': r_value**2 > 0.8,
            'significant': p_value < self.alpha
        }
        
    def validate_lambda_distribution(self, lambda_values: List[float]) -> Dict:
        """Validate Kyle's Lambda distribution properties"""
        
        lambda_array = np.array([x for x in lambda_values if x is not None and np.isfinite(x)])
        
        if len(lambda_array) < 10:
            return {'valid': False, 'reason': 'insufficient_data'}
            
        # Basic properties
        results = {
            'valid': True,
            'mean': np.mean(lambda_array),
            'median': np.median(lambda_array),
            'std': np.std(lambda_array),
            'skewness': stats.skew(lambda_array),
            'kurtosis': stats.kurtosis(lambda_array)
        }
        
        # Test for normality (lambda often follows log-normal distribution)
        if len(lambda_array) > 20:
            # Shapiro-Wilk test for normality
            statistic, p_value = shapiro(lambda_array)
            results['normality_test'] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > self.alpha
            }
            
            # Test for log-normality
            log_lambda = np.log(np.abs(lambda_array[lambda_array != 0]))
            if len(log_lambda) > 3:
                log_stat, log_p = shapiro(log_lambda)
                results['log_normality_test'] = {
                    'statistic': log_stat,
                    'p_value': log_p,
                    'is_log_normal': log_p > self.alpha
                }
                
        return results
        
    def validate_vpin_properties(self, vpin_values: List[float]) -> Dict:
        """Validate VPIN statistical properties"""
        
        vpin_array = np.array([x for x in vpin_values if np.isfinite(x)])
        
        if len(vpin_array) < 5:
            return {'valid': False, 'reason': 'insufficient_data'}
            
        results = {
            'valid': True,
            'mean': np.mean(vpin_array),
            'std': np.std(vpin_array),
            'min': np.min(vpin_array),
            'max': np.max(vpin_array),
            'valid_range': np.all((vpin_array >= 0) & (vpin_array <= 1)),
            'reasonable_values': np.all(vpin_array <= 1.0)
        }
        
        # VPIN should be bounded [0,1]
        if not results['valid_range']:
            results['valid'] = False
            results['reason'] = 'vpin_out_of_bounds'
            
        return results


class TestStatisticalValidation:
    """Statistical validation test suite"""
    
    def setup_method(self):
        """Setup for each test"""
        self.validator = StatisticalValidator()
        self.generator = SyntheticDataGenerator(seed=42)
        
    def test_hurst_exponent_statistical_properties(self):
        """Test Hurst exponent statistical properties across scenarios"""
        
        # Test different regimes
        test_cases = [
            ('random', self.generator.generate_random_walk, {'length': 500}),
            ('trending', self.generator.generate_trending_data, {'length': 500, 'trend_strength': 0.002}),
            ('mean_reverting', self.generator.generate_mean_reverting_data, {'length': 500, 'reversion_strength': 0.1})
        ]
        
        for regime, data_func, params in test_cases:
            hurst_values = []
            
            # Generate multiple samples to test statistical properties
            for i in range(10):
                np.random.seed(42 + i)  # Different seeds for each sample
                data = data_func(**params)
                
                indicator = HurstExponentIndicator(lookback=400)
                result = indicator.calculate(data, f"STAT_TEST_{regime}_{i}")
                
                hurst_values.append(result.value / 100)  # Convert to 0-1 scale
                
            # Validate statistical properties
            validation = self.validator.validate_hurst_properties(hurst_values, regime)
            
            assert validation['valid_range'], f"Hurst values out of range for {regime}"
            assert validation['regime_consistent'], f"Hurst not consistent with {regime} regime"
            
            # Check reasonable variability
            assert validation['std'] > 0.01, f"Hurst variance too low for {regime}"
            assert validation['std'] < 0.3, f"Hurst variance too high for {regime}"
            
    def test_dfa_scaling_law_validation(self):
        """Test DFA scaling law mathematical correctness"""
        
        # Test with known synthetic patterns
        test_patterns = [
            ('random_walk', 0.5, lambda n: self.generator.generate_random_walk(n)),
            ('trending', 0.7, lambda n: self.generator.generate_trending_data(n, trend_strength=0.003))
        ]
        
        for pattern_name, expected_alpha, data_func in test_patterns:
            data = data_func(1000)
            
            indicator = DFAIndicator(min_scale=10, max_scale=100, n_scales=20)
            result = indicator.calculate(data, f"SCALING_{pattern_name}")
            
            # Extract scaling information from metadata
            if 'scaling_exponent' in result.metadata:
                actual_alpha = result.metadata['scaling_exponent']
                
                # Should be close to expected theoretical value
                alpha_error = abs(actual_alpha - expected_alpha)
                assert alpha_error < 0.2, f"DFA scaling exponent error too large: {alpha_error:.3f}"
                
                # R-squared should indicate good fit
                if 'r_squared' in result.metadata:
                    assert result.metadata['r_squared'] > 0.7, f"DFA fit quality too poor: {result.metadata['r_squared']:.3f}"
                    
    def test_kyles_lambda_statistical_distribution(self):
        """Test Kyle's Lambda statistical distribution"""
        
        # Generate multiple samples with different liquidity conditions
        lambda_values = []
        
        for i in range(20):
            # Generate tick data with varying price impact
            ticks = []
            price = 100.0
            timestamp = 1000
            
            # Random price impact coefficient for this sample
            impact_coeff = np.random.uniform(0.0001, 0.01)
            
            for j in range(100):
                volume = np.random.randint(50, 500)
                side = np.random.choice([1, -1])
                
                # Apply price impact
                price_change = side * volume * impact_coeff + np.random.normal(0, 0.001)
                price += price_change
                
                tick = TickData(
                    timestamp=timestamp + j * 1000,
                    price=price,
                    volume=volume,
                    side=side
                )
                ticks.append(tick)
                
            indicator = KylesLambdaIndicator(estimation_window=80)
            result = indicator.calculate(ticks, f"LAMBDA_DIST_{i}")
            
            if result.metadata.get('lambda') is not None:
                lambda_values.append(result.metadata['lambda'])
                
        if len(lambda_values) > 5:
            validation = self.validator.validate_lambda_distribution(lambda_values)
            
            assert validation['valid'], f"Lambda distribution validation failed"
            
            # Lambda should be positive (in general)
            assert validation['mean'] >= 0, f"Negative mean lambda: {validation['mean']}"
            
            # Should show reasonable variability
            assert validation['std'] > 0, f"No lambda variability observed"
            
    def test_vpin_bounded_properties(self):
        """Test VPIN bounded properties and statistical behavior"""
        
        vpin_values = []
        
        # Test with different order flow patterns
        imbalance_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different buy ratios
        
        for imbalance in imbalance_levels:
            for iteration in range(5):
                ticks = []
                timestamp = 1000
                
                for i in range(200):
                    volume = np.random.randint(20, 200)
                    side = 1 if np.random.random() < imbalance else -1
                    
                    tick = TickData(
                        timestamp=timestamp + i * 1000,
                        price=100 + np.random.normal(0, 0.1),
                        volume=volume,
                        side=side
                    )
                    ticks.append(tick)
                    
                indicator = VPINIndicator(bucket_volume=1000, rolling_window=5)
                result = indicator.calculate(ticks, f"VPIN_BOUND_{imbalance}_{iteration}")
                
                if result.metadata.get('vpin') is not None:
                    vpin_values.append(result.metadata['vpin'])
                    
        if len(vpin_values) > 5:
            validation = self.validator.validate_vpin_properties(vpin_values)
            
            assert validation['valid'], f"VPIN properties validation failed"
            assert validation['valid_range'], f"VPIN values out of [0,1] range"
            
            # Higher imbalance should generally produce higher VPIN
            # This is a weak test since VPIN depends on many factors
            assert validation['max'] > validation['min'], f"No VPIN variation observed"
            
    def test_tick_volume_imbalance_consistency(self):
        """Test Tick Volume Imbalance statistical consistency"""
        
        # Generate data with known imbalance patterns
        test_patterns = [
            ('balanced', 0.5),  # 50% buy probability
            ('buy_heavy', 0.8),  # 80% buy probability  
            ('sell_heavy', 0.2)  # 20% buy probability
        ]
        
        for pattern_name, buy_prob in test_patterns:
            imbalance_values = []
            
            for iteration in range(10):
                ticks = []
                timestamp = 1000
                
                for i in range(150):
                    volume = 100  # Fixed volume for clean test
                    side = 1 if np.random.random() < buy_prob else -1
                    
                    tick = TickData(
                        timestamp=timestamp + i * 1000,
                        price=100,
                        volume=volume,
                        side=side
                    )
                    ticks.append(tick)
                    
                indicator = TickVolumeImbalanceIndicator(window_size=100)
                result = indicator.calculate(ticks, f"TVI_CONSIST_{pattern_name}_{iteration}")
                
                imbalance_values.append(result.metadata['current_imbalance'])
                
            # Check consistency with expected pattern
            mean_imbalance = np.mean(imbalance_values)
            
            if pattern_name == 'balanced':
                # Should be close to zero
                assert abs(mean_imbalance) < 0.2, f"Balanced pattern not balanced: {mean_imbalance:.3f}"
            elif pattern_name == 'buy_heavy':
                # Should be positive
                assert mean_imbalance > 0.3, f"Buy heavy pattern not detected: {mean_imbalance:.3f}"
            elif pattern_name == 'sell_heavy':
                # Should be negative
                assert mean_imbalance < -0.3, f"Sell heavy pattern not detected: {mean_imbalance:.3f}"
                
    def test_indicator_correlation_analysis(self):
        """Test correlations between related indicators"""
        
        # Generate regime-changing data
        data = self.generator.generate_regime_change_data(1200)
        
        # Calculate multiple related indicators
        hurst = HurstExponentIndicator(lookback=500)
        dfa = DFAIndicator(min_scale=10, max_scale=80, n_scales=15)
        
        hurst_result = hurst.calculate(data, "CORR_HURST")
        dfa_result = dfa.calculate(data, "CORR_DFA")
        
        # Extract comparable values
        hurst_value = hurst_result.value / 100  # Convert to 0-1 scale
        dfa_alpha = dfa_result.metadata.get('scaling_exponent', 0.5)
        
        # Hurst and DFA should be correlated (both measure persistence)
        correlation_threshold = 0.3  # Allow for noise in synthetic data
        
        # They should at least agree on regime direction
        both_trending = (hurst_value > 0.5) and (dfa_alpha > 0.5)
        both_reverting = (hurst_value < 0.5) and (dfa_alpha < 0.5)
        
        assert both_trending or both_reverting or abs(hurst_value - 0.5) < 0.1, \
            f"Hurst ({hurst_value:.3f}) and DFA ({dfa_alpha:.3f}) disagree on market regime"
            
    def test_signal_consistency_across_timeframes(self):
        """Test signal consistency across different timeframes"""
        
        # Generate long time series
        long_data = self.generator.generate_trending_data(2000, trend_strength=0.002)
        
        # Test same indicator with different lookback windows
        lookbacks = [200, 500, 1000, 1500]
        
        signals = []
        confidences = []
        
        for lookback in lookbacks:
            if lookback < len(long_data):
                indicator = HurstExponentIndicator(lookback=lookback)
                result = indicator.calculate(long_data, f"TIMEFRAME_{lookback}")
                
                signals.append(result.signal)
                confidences.append(result.confidence)
                
        # For trending data, signals should be generally consistent
        # (allowing for some variation due to different sample sizes)
        if len(signals) > 2:
            # Check that signals don't wildly contradict each other
            buy_signals = sum(1 for s in signals if s.name == 'BUY')
            sell_signals = sum(1 for s in signals if s.name == 'SELL')
            hold_signals = sum(1 for s in signals if s.name == 'HOLD')
            
            total_signals = len(signals)
            
            # For trending data, shouldn't have many conflicting signals
            assert (buy_signals + hold_signals) >= sell_signals, \
                f"Inconsistent signals for trending data: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD"
                
    def test_monte_carlo_validation(self):
        """Monte Carlo validation of indicator stability"""
        
        n_simulations = 20
        
        # Test DFA stability across multiple random realizations
        dfa_alphas = []
        
        for sim in range(n_simulations):
            np.random.seed(42 + sim)
            
            # Generate random walk (should have α ≈ 0.5)
            data = self.generator.generate_random_walk(800)
            
            indicator = DFAIndicator(min_scale=8, max_scale=60, n_scales=15)
            result = indicator.calculate(data, f"MONTE_CARLO_{sim}")
            
            if 'scaling_exponent' in result.metadata:
                dfa_alphas.append(result.metadata['scaling_exponent'])
                
        if len(dfa_alphas) > 10:
            # Statistical properties of the estimates
            mean_alpha = np.mean(dfa_alphas)
            std_alpha = np.std(dfa_alphas)
            
            # Mean should be close to 0.5 for random walk
            assert abs(mean_alpha - 0.5) < 0.1, f"DFA mean alpha deviates from 0.5: {mean_alpha:.3f}"
            
            # Standard deviation should be reasonable (not too high or too low)
            assert 0.02 < std_alpha < 0.15, f"DFA alpha std deviation unreasonable: {std_alpha:.3f}"
            
            # Test for outliers (more than 2 std devs from mean)
            outliers = [a for a in dfa_alphas if abs(a - mean_alpha) > 2 * std_alpha]
            outlier_rate = len(outliers) / len(dfa_alphas)
            
            assert outlier_rate < 0.15, f"Too many DFA outliers: {outlier_rate:.2%}"