"""
🔗 Integration Tests
Tests for indicator integration, workflow, and end-to-end functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ..fractal_indicators import FractalIndicatorFactory, create_market_regime_ensemble, create_microstructure_ensemble
from ..indicators.multifractal.hurst_exponent import HurstExponentIndicator
from ..indicators.multifractal.dfa import DFAIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator
from ..indicators.time_patterns.volume_bars import VolumeBarAggregator
from ..base.types import TimeFrame, SignalType, TickData

from .fixtures import SyntheticDataGenerator


class TestFractalIndicatorFactory:
    """Test the unified fractal indicator factory"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = SyntheticDataGenerator()
        
    def test_factory_indicator_creation(self):
        """Test factory methods create proper indicators"""
        
        # Test all factory methods
        indicators = [
            FractalIndicatorFactory.create_hurst_exponent(),
            FractalIndicatorFactory.create_dfa(),
            FractalIndicatorFactory.create_kyles_lambda(),
            FractalIndicatorFactory.create_vpin(),
            FractalIndicatorFactory.create_tick_volume_imbalance(),
            FractalIndicatorFactory.create_volume_bars()
        ]
        
        # Check each indicator is properly configured
        for indicator in indicators:
            assert hasattr(indicator, 'name')
            assert hasattr(indicator, 'calculate')
            assert hasattr(indicator, 'validate_data')
            assert indicator.name is not None
            assert len(indicator.name) > 0
            
    def test_factory_custom_parameters(self):
        """Test factory methods with custom parameters"""
        
        # Test custom Hurst parameters
        custom_hurst = FractalIndicatorFactory.create_hurst_exponent(
            timeframe=TimeFrame.MINUTE_5,
            lookback=200,
            method='dfa'
        )
        
        assert custom_hurst.timeframe == TimeFrame.MINUTE_5
        assert custom_hurst.lookback_periods == 200
        assert custom_hurst.method == 'dfa'
        
        # Test custom DFA parameters
        custom_dfa = FractalIndicatorFactory.create_dfa(
            min_scale=5,
            max_scale=50,
            multi_scale=False
        )
        
        assert custom_dfa.min_scale == 5
        assert custom_dfa.max_scale == 50
        assert custom_dfa.multi_scale == False
        
        # Test custom VPIN parameters
        custom_vpin = FractalIndicatorFactory.create_vpin(
            bucket_volume=5000,
            n_buckets=20
        )
        
        assert custom_vpin.bucket_volume == 5000
        assert custom_vpin.n_buckets == 20


class TestEnsembleIntegration:
    """Test ensemble indicator functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = SyntheticDataGenerator()
        
    def test_market_regime_ensemble(self):
        """Test market regime detection ensemble"""
        
        # Test with different market regimes
        test_data = [
            ('trending', self.generator.generate_trending_data(800, trend_strength=0.003)),
            ('mean_reverting', self.generator.generate_mean_reverting_data(800, reversion_strength=0.1)),
            ('random', self.generator.generate_random_walk(800))
        ]
        
        for regime_name, data in test_data:
            ensemble = create_market_regime_ensemble()
            results = ensemble.analyze(data, f"ENSEMBLE_{regime_name}")
            
            # Should return results from multiple indicators
            assert len(results) >= 2  # At least Hurst and DFA
            
            # All results should be valid
            for result in results:
                assert result.symbol == f"ENSEMBLE_{regime_name}"
                assert 0 <= result.confidence <= 100
                assert result.signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
                assert isinstance(result.metadata, dict)
                
            # Check regime consensus
            consensus = ensemble.get_consensus(results)
            
            assert 'consensus_signal' in consensus
            assert 'consensus_confidence' in consensus
            assert 'agreement_score' in consensus
            assert 'regime_classification' in consensus
            
            # Verify regime classification makes sense
            if regime_name == 'trending':
                assert 'trend' in consensus['regime_classification'].lower()
            elif regime_name == 'mean_reverting':
                assert 'revert' in consensus['regime_classification'].lower()
                
    def test_microstructure_ensemble(self):
        """Test microstructure analysis ensemble"""
        
        # Generate tick data for microstructure analysis
        tick_data = self.generator.generate_informed_trading_ticks(1000, informed_ratio=0.4)
        
        ensemble = create_microstructure_ensemble()
        results = ensemble.analyze_ticks(tick_data, "MICROSTRUCTURE_TEST")
        
        # Should have results from multiple microstructure indicators
        assert len(results) >= 2
        
        # Verify microstructure-specific metrics
        for result in results:
            assert result.symbol == "MICROSTRUCTURE_TEST"
            
            if result.indicator_name == "VPIN":
                assert 'vpin' in result.metadata
                assert 'order_flow_toxicity' in result.metadata
            elif result.indicator_name == "KylesLambda":
                assert 'liquidity_score' in result.metadata or 'lambda' in result.metadata
            elif result.indicator_name == "TickVolumeImbalance":
                assert 'current_imbalance' in result.metadata
                
        # Get microstructure consensus
        consensus = ensemble.get_consensus(results)
        
        assert 'liquidity_assessment' in consensus
        assert 'trading_difficulty' in consensus
        assert 'informed_trading_probability' in consensus
        
    def test_ensemble_consensus_calculation(self):
        """Test ensemble consensus calculation logic"""
        
        data = self.generator.generate_trending_data(600)
        ensemble = create_market_regime_ensemble()
        
        results = ensemble.analyze(data, "CONSENSUS_TEST")
        consensus = ensemble.get_consensus(results)
        
        # Consensus should aggregate individual results sensibly
        individual_confidences = [r.confidence for r in results]
        consensus_confidence = consensus['consensus_confidence']
        
        # Consensus confidence should be related to individual confidences
        if len(individual_confidences) > 1:
            mean_confidence = np.mean(individual_confidences)
            assert abs(consensus_confidence - mean_confidence) < 30  # Reasonable aggregation
            
        # Agreement score should reflect signal consistency
        signals = [r.signal for r in results]
        unique_signals = set(signals)
        
        if len(unique_signals) == 1:
            # Perfect agreement
            assert consensus['agreement_score'] > 0.8
        elif len(unique_signals) == len(signals):
            # No agreement
            assert consensus['agreement_score'] < 0.5
            
    def test_cross_timeframe_analysis(self):
        """Test analysis across multiple timeframes"""
        
        # Generate data for different timeframes
        long_data = self.generator.generate_trending_data(2000, trend_strength=0.002)
        
        timeframes = [
            (TimeFrame.MINUTE_1, 100),
            (TimeFrame.MINUTE_5, 300),
            (TimeFrame.HOUR_1, 1000)
        ]
        
        results_by_timeframe = {}
        
        for tf, lookback in timeframes:
            if lookback <= len(long_data):
                indicator = FractalIndicatorFactory.create_hurst_exponent(
                    timeframe=tf,
                    lookback=lookback
                )
                
                data_subset = long_data.iloc[-lookback:]
                result = indicator.calculate(data_subset, f"TIMEFRAME_{tf.value}")
                
                results_by_timeframe[tf] = result
                
        # Cross-timeframe analysis
        if len(results_by_timeframe) >= 2:
            # Check for consistency across timeframes
            regime_classifications = []
            
            for tf, result in results_by_timeframe.items():
                if 'market_regime' in result.metadata:
                    regime_classifications.append(result.metadata['market_regime'])
                    
            # For trending data, most timeframes should detect trending
            if len(regime_classifications) >= 2:
                trending_count = sum(1 for regime in regime_classifications if 'trend' in regime.lower())
                # At least half should detect trending
                assert trending_count >= len(regime_classifications) // 2


class TestWorkflowIntegration:
    """Test complete trading workflow integration"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = SyntheticDataGenerator()
        
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow from data to signals"""
        
        # Step 1: Generate market data
        bar_data = self.generator.generate_regime_change_data(1000)
        tick_data = self.generator.generate_informed_trading_ticks(500)
        
        # Step 2: Market regime analysis
        regime_ensemble = create_market_regime_ensemble()
        regime_results = regime_ensemble.analyze(bar_data, "WORKFLOW_TEST")
        regime_consensus = regime_ensemble.get_consensus(regime_results)
        
        # Step 3: Microstructure analysis
        micro_ensemble = create_microstructure_ensemble()
        micro_results = micro_ensemble.analyze_ticks(tick_data, "WORKFLOW_TEST")
        micro_consensus = micro_ensemble.get_consensus(micro_results)
        
        # Step 4: Combined decision making
        combined_analysis = self._combine_analysis(regime_consensus, micro_consensus)
        
        # Verify complete workflow
        assert 'regime_classification' in regime_consensus
        assert 'liquidity_assessment' in micro_consensus
        assert 'final_signal' in combined_analysis
        assert 'risk_assessment' in combined_analysis
        
        # Final signal should be valid
        assert combined_analysis['final_signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= combined_analysis['confidence'] <= 100
        
    def _combine_analysis(self, regime_consensus: Dict, micro_consensus: Dict) -> Dict:
        """Combine regime and microstructure analysis"""
        
        # Simple combination logic for testing
        regime_signal = regime_consensus.get('consensus_signal', 'HOLD')
        regime_confidence = regime_consensus.get('consensus_confidence', 0)
        
        liquidity_score = micro_consensus.get('liquidity_assessment', {}).get('score', 50)
        trading_difficulty = micro_consensus.get('trading_difficulty', 'moderate')
        
        # Adjust confidence based on liquidity
        if trading_difficulty == 'very_difficult':
            final_confidence = regime_confidence * 0.5
        elif trading_difficulty == 'easy':
            final_confidence = regime_confidence * 1.2
        else:
            final_confidence = regime_confidence
            
        final_confidence = min(final_confidence, 100)
        
        # Risk assessment
        if liquidity_score < 30:
            risk_level = 'high'
        elif liquidity_score > 70:
            risk_level = 'low'
        else:
            risk_level = 'medium'
            
        return {
            'final_signal': regime_signal,
            'confidence': final_confidence,
            'risk_assessment': {
                'liquidity_risk': risk_level,
                'regime_stability': regime_consensus.get('agreement_score', 0),
                'execution_difficulty': trading_difficulty
            }
        }
        
    def test_error_handling_in_workflow(self):
        """Test error handling throughout the workflow"""
        
        # Test with problematic data
        problematic_data = [
            pd.DataFrame(),  # Empty data
            pd.DataFrame({'close': [100, np.nan, 102]}),  # NaN values
            pd.DataFrame({'close': [100] * 10})  # Very short data
        ]
        
        ensemble = create_market_regime_ensemble()
        
        for i, data in enumerate(problematic_data):
            try:
                results = ensemble.analyze(data, f"ERROR_TEST_{i}")
                
                # Should handle gracefully, not crash
                assert isinstance(results, list)
                
                # Results should indicate low confidence or errors
                for result in results:
                    if 'error' not in result.metadata:
                        assert result.confidence <= 20  # Low confidence for bad data
                        
            except Exception as e:
                # If it raises an exception, it should be a specific, expected one
                assert isinstance(e, (ValueError, TypeError))
                
    def test_real_time_update_simulation(self):
        """Test real-time update capabilities"""
        
        # Simulate streaming data updates
        full_data = self.generator.generate_trending_data(1000)
        
        # Initialize indicators
        hurst = FractalIndicatorFactory.create_hurst_exponent(lookback=200)
        dfa = FractalIndicatorFactory.create_dfa(min_scale=10, max_scale=50)
        
        # Simulate incremental updates
        update_points = [300, 500, 700, 900]
        
        previous_results = {}
        
        for update_point in update_points:
            current_data = full_data.iloc[:update_point]
            
            # Calculate current indicators
            hurst_result = hurst.calculate(current_data, f"REALTIME_{update_point}")
            dfa_result = dfa.calculate(current_data, f"REALTIME_{update_point}")
            
            current_results = {
                'hurst': hurst_result,
                'dfa': dfa_result,
                'timestamp': update_point
            }
            
            # Check for consistency with previous results
            if previous_results:
                # Values shouldn't change dramatically between updates
                prev_hurst = previous_results['hurst'].value
                curr_hurst = current_results['hurst'].value
                
                hurst_change = abs(curr_hurst - prev_hurst) / prev_hurst if prev_hurst != 0 else 0
                
                # For trending data, Hurst shouldn't change dramatically
                assert hurst_change < 0.5, f"Hurst changed too much: {hurst_change:.2%}"
                
            previous_results = current_results
            
    def test_performance_under_load(self):
        """Test performance with realistic market data loads"""
        
        # Generate realistic sized datasets
        large_bar_data = self.generator.generate_regime_change_data(5000)
        large_tick_data = self.generator.generate_tick_data(2000)
        
        # Test ensemble performance
        regime_ensemble = create_market_regime_ensemble()
        micro_ensemble = create_microstructure_ensemble()
        
        # Time the analysis
        import time
        
        start_time = time.perf_counter()
        regime_results = regime_ensemble.analyze(large_bar_data, "LOAD_TEST_BAR")
        mid_time = time.perf_counter()
        micro_results = micro_ensemble.analyze_ticks(large_tick_data, "LOAD_TEST_TICK")
        end_time = time.perf_counter()
        
        regime_time = mid_time - start_time
        micro_time = end_time - mid_time
        total_time = end_time - start_time
        
        # Performance assertions
        assert regime_time < 30.0, f"Regime analysis too slow: {regime_time:.2f}s"
        assert micro_time < 10.0, f"Microstructure analysis too slow: {micro_time:.2f}s"
        assert total_time < 40.0, f"Total analysis too slow: {total_time:.2f}s"
        
        # Results should still be valid
        assert len(regime_results) > 0
        assert len(micro_results) > 0
        
        for result in regime_results + micro_results:
            assert result.confidence >= 0
            assert isinstance(result.metadata, dict)
            
    def test_memory_management_in_integration(self):
        """Test memory management during long-running operations"""
        
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate long-running analysis with multiple datasets
        ensemble = create_market_regime_ensemble()
        
        max_memory = initial_memory
        
        for i in range(10):
            # Generate new data each iteration
            data = self.generator.generate_regime_change_data(1000)
            
            results = ensemble.analyze(data, f"MEMORY_TEST_{i}")
            
            # Process results
            consensus = ensemble.get_consensus(results)
            
            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            
            # Clean up explicitly
            del data, results, consensus
            
            if i % 3 == 0:
                gc.collect()
                
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 200, f"Excessive memory growth in integration: {memory_growth:.1f}MB"
        assert max_memory - initial_memory < 300, f"Peak memory usage too high: {max_memory - initial_memory:.1f}MB"