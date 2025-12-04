"""
🧪 Base Indicator Tests
Tests for common indicator functionality and interface compliance
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Type
from datetime import datetime

from ..base.indicator import BaseIndicator, IndicatorResult
from ..base.types import SignalType, TimeFrame
from ..indicators.multifractal.hurst_exponent import HurstExponentIndicator
from ..indicators.multifractal.dfa import DFAIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator
from ..indicators.time_patterns.volume_bars import VolumeBarAggregator

from .fixtures import SyntheticDataGenerator


class TestBaseIndicator:
    """Test base indicator functionality"""
    
    def test_indicator_interface_compliance(self):
        """Test that all indicators implement required interface"""
        
        # All indicator classes to test
        indicator_classes = [
            HurstExponentIndicator,
            DFAIndicator,
            KylesLambdaIndicator,
            VPINIndicator,
            TickVolumeImbalanceIndicator,
            VolumeBarAggregator
        ]
        
        for indicator_class in indicator_classes:
            # Should be able to instantiate
            indicator = indicator_class()
            
            # Should have required attributes
            assert hasattr(indicator, 'name')
            assert hasattr(indicator, 'timeframe')
            assert hasattr(indicator, 'lookback_periods')
            assert hasattr(indicator, 'params')
            
            # Should have required methods
            assert hasattr(indicator, 'calculate')
            assert hasattr(indicator, 'validate_data')
            assert callable(indicator.calculate)
            assert callable(indicator.validate_data)
            
            # Name should be non-empty string
            assert isinstance(indicator.name, str)
            assert len(indicator.name) > 0
            
            # Timeframe should be valid
            assert isinstance(indicator.timeframe, TimeFrame)
            
            # Params should be dict
            assert isinstance(indicator.params, dict)
            
    def test_indicator_result_structure(self):
        """Test that indicators return properly structured results"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_random_walk(100)
        
        # Test with DFA indicator
        indicator = DFAIndicator(min_scale=5, max_scale=50, n_scales=10)
        result = indicator.calculate(data, "TEST")
        
        # Check result structure
        assert isinstance(result, IndicatorResult)
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'timeframe')
        assert hasattr(result, 'metadata')
        
        # Check data types
        assert isinstance(result.timestamp, int)
        assert isinstance(result.symbol, str)
        assert isinstance(result.indicator_name, str)
        assert isinstance(result.value, (int, float))
        assert isinstance(result.signal, SignalType)
        assert isinstance(result.confidence, (int, float))
        assert isinstance(result.timeframe, TimeFrame)
        assert isinstance(result.metadata, dict)
        
        # Check value ranges
        assert 0 <= result.confidence <= 100
        assert result.symbol == "TEST"
        assert result.indicator_name == indicator.name
        
    def test_data_validation(self):
        """Test data validation across indicators"""
        
        generator = SyntheticDataGenerator()
        
        # Valid data
        valid_data = generator.generate_random_walk(100)
        
        # Invalid data
        empty_data = pd.DataFrame()
        small_data = generator.generate_random_walk(5)
        
        indicator = HurstExponentIndicator()
        
        # Valid data should pass
        assert indicator.validate_data(valid_data) == True
        
        # Invalid data should fail
        assert indicator.validate_data(empty_data) == False
        assert indicator.validate_data(small_data) == False
        
        # Test with tick data
        tick_indicator = VPINIndicator()
        tick_data = generator.generate_tick_data(100)
        
        assert tick_indicator.validate_data(tick_data) == True
        assert tick_indicator.validate_data([]) == False
        
    def test_error_handling(self):
        """Test error handling with malformed data"""
        
        indicator = HurstExponentIndicator()
        
        # Test with None
        with pytest.raises((TypeError, AttributeError)):
            indicator.calculate(None, "TEST")
            
        # Test with empty data (should return empty result, not crash)
        empty_data = pd.DataFrame()
        result = indicator.calculate(empty_data, "TEST")
        assert isinstance(result, IndicatorResult)
        assert result.confidence == 0.0
        
    def test_parameter_validation(self):
        """Test parameter validation during initialization"""
        
        # Valid parameters
        indicator = DFAIndicator(min_scale=10, max_scale=100)
        assert indicator.min_scale == 10
        assert indicator.max_scale == 100
        
        # Invalid parameters should either raise error or use defaults
        with pytest.raises((ValueError, TypeError)):
            DFAIndicator(min_scale=-1)
            
        with pytest.raises((ValueError, TypeError)):
            DFAIndicator(max_scale=5, min_scale=10)  # max < min
            
    def test_reproducibility(self):
        """Test that indicators produce consistent results"""
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_random_walk(200)
        
        indicator = HurstExponentIndicator(lookback=100)
        
        # Calculate multiple times
        result1 = indicator.calculate(data, "TEST")
        result2 = indicator.calculate(data, "TEST")
        
        # Results should be identical
        assert result1.value == result2.value
        assert result1.confidence == result2.confidence
        assert result1.signal == result2.signal
        
    def test_metadata_completeness(self):
        """Test that metadata contains expected information"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_trending_data(150)
        
        # Test DFA metadata
        dfa = DFAIndicator()
        result = dfa.calculate(data, "TEST")
        
        expected_keys = [
            'scaling_exponent',
            'r_squared',
            'persistence_type',
            'market_behavior'
        ]
        
        for key in expected_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"
            
        # Test Kyle's Lambda metadata
        tick_data = generator.generate_tick_data(200)
        kyle = KylesLambdaIndicator()
        result = kyle.calculate(tick_data, "TEST")
        
        if result.metadata.get('lambda') is not None:
            assert 'liquidity_score' in result.metadata
            assert 'trade_difficulty' in result.metadata
            
    def test_signal_generation_logic(self):
        """Test signal generation makes sense"""
        
        generator = SyntheticDataGenerator()
        
        # Strong trending data should generate trending signals
        trending_data = generator.generate_trending_data(200, trend_strength=0.005)
        indicator = DFAIndicator()
        result = indicator.calculate(trending_data, "TREND_TEST")
        
        # Should detect persistence
        assert result.metadata['scaling_exponent'] > 0.5
        assert 'trending' in result.metadata['market_behavior'].lower()
        
        # Mean reverting data should generate mean reversion signals  
        mean_rev_data = generator.generate_mean_reverting_data(200)
        result2 = indicator.calculate(mean_rev_data, "MR_TEST")
        
        # Should detect anti-persistence
        assert result2.metadata['scaling_exponent'] < 0.5
        assert 'revert' in result2.metadata['market_behavior'].lower()
        
    def test_timeframe_consistency(self):
        """Test that timeframe is handled consistently"""
        
        # Different indicators should handle timeframe appropriately
        bar_indicator = DFAIndicator(timeframe=TimeFrame.HOUR_1)
        tick_indicator = VPINIndicator(timeframe=TimeFrame.TICK)
        
        assert bar_indicator.timeframe == TimeFrame.HOUR_1
        assert tick_indicator.timeframe == TimeFrame.TICK
        
        # Result should preserve timeframe
        generator = SyntheticDataGenerator()
        data = generator.generate_random_walk(100)
        
        result = bar_indicator.calculate(data, "TEST")
        assert result.timeframe == TimeFrame.HOUR_1
        
    def test_confidence_bounds(self):
        """Test that confidence values are within valid bounds"""
        
        generator = SyntheticDataGenerator()
        
        # Test multiple scenarios
        scenarios = [
            generator.generate_random_walk(100),
            generator.generate_trending_data(100),
            generator.generate_mean_reverting_data(100)
        ]
        
        indicators = [
            HurstExponentIndicator(),
            DFAIndicator(n_scales=10),
        ]
        
        for data in scenarios:
            for indicator in indicators:
                result = indicator.calculate(data, "TEST")
                
                # Confidence should be between 0 and 100
                assert 0 <= result.confidence <= 100
                
                # If confidence > 0, signal should not be HOLD (generally)
                if result.confidence > 50:
                    # High confidence should have definitive signal
                    assert result.signal != SignalType.HOLD or result.confidence < 20