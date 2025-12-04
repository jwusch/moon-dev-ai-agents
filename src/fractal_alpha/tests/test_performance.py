"""
⚡ Performance and Benchmark Tests
Tests for indicator performance, speed, and resource usage
"""

import pytest
import time
import numpy as np
import pandas as pd
from memory_profiler import profile
import psutil
import gc
from typing import Dict, List, Callable

from ..indicators.multifractal.hurst_exponent import HurstExponentIndicator
from ..indicators.multifractal.dfa import DFAIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator
from ..indicators.time_patterns.volume_bars import VolumeBarAggregator

from .fixtures import SyntheticDataGenerator


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_indicator(self, indicator_class, data_sizes: List[int], 
                          iterations: int = 10, **kwargs) -> Dict:
        """Benchmark an indicator across different data sizes"""
        
        results = {
            'indicator': indicator_class.__name__,
            'data_sizes': data_sizes,
            'execution_times': [],
            'memory_usage': [],
            'throughput': []  # items per second
        }
        
        generator = SyntheticDataGenerator()
        
        for size in data_sizes:
            times = []
            memories = []
            
            for _ in range(iterations):
                # Generate test data
                if 'tick' in indicator_class.__name__.lower():
                    data = generator.generate_tick_data(size)
                else:
                    data = generator.generate_random_walk(size)
                    
                # Create indicator instance
                indicator = indicator_class(**kwargs)
                
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Benchmark execution time
                start_time = time.perf_counter()
                result = indicator.calculate(data, f"PERF_TEST_{size}")
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                memories.append(memory_used)
                
                # Clean up
                del data, indicator, result
                gc.collect()
                
            # Calculate statistics
            avg_time = np.mean(times)
            avg_memory = np.mean(memories)
            throughput = size / avg_time if avg_time > 0 else 0
            
            results['execution_times'].append({
                'size': size,
                'mean': avg_time,
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            })
            
            results['memory_usage'].append({
                'size': size,
                'mean_mb': avg_memory,
                'std_mb': np.std(memories)
            })
            
            results['throughput'].append({
                'size': size,
                'items_per_second': throughput
            })
            
        return results
        
    def analyze_scaling(self, results: Dict) -> Dict:
        """Analyze algorithmic scaling behavior"""
        
        sizes = [r['size'] for r in results['execution_times']]
        times = [r['mean'] for r in results['execution_times']]
        
        # Fit different complexity models
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        
        # Linear fit in log space: log(time) = a*log(size) + b
        # This gives us time = C * size^a
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exponent = coeffs[0]
        
        # Determine complexity class
        if scaling_exponent < 1.1:
            complexity_class = "O(n)"
        elif scaling_exponent < 1.5:
            complexity_class = "O(n log n)"
        elif scaling_exponent < 2.1:
            complexity_class = "O(n²)"
        else:
            complexity_class = "O(n^{:.1f})".format(scaling_exponent)
            
        return {
            'scaling_exponent': scaling_exponent,
            'complexity_class': complexity_class,
            'r_squared': 1 - np.sum((log_times - np.polyval(coeffs, log_sizes))**2) / np.sum((log_times - np.mean(log_times))**2)
        }


class TestIndicatorPerformance:
    """Test suite for indicator performance"""
    
    def setup_method(self):
        """Setup for each test"""
        self.benchmark = PerformanceBenchmark()
        self.data_sizes = [100, 500, 1000, 2000]  # Reduced for CI
        self.iterations = 3  # Reduced for CI
        
    def test_hurst_exponent_performance(self):
        """Test Hurst Exponent indicator performance"""
        
        results = self.benchmark.benchmark_indicator(
            HurstExponentIndicator,
            self.data_sizes,
            self.iterations,
            lookback=min(self.data_sizes)  # Ensure we can handle smallest size
        )
        
        # Check performance characteristics
        max_time = max(r['mean'] for r in results['execution_times'])
        
        # Should complete within reasonable time (10 seconds for largest dataset)
        assert max_time < 10.0, f"Hurst calculation too slow: {max_time:.2f}s"
        
        # Memory usage should be reasonable
        max_memory = max(r['mean_mb'] for r in results['memory_usage'])
        assert max_memory < 500, f"Hurst uses too much memory: {max_memory:.1f}MB"
        
        # Analyze scaling
        scaling = self.benchmark.analyze_scaling(results)
        
        # Should scale reasonably (not worse than O(n²))
        assert scaling['scaling_exponent'] < 2.5, f"Hurst scaling too poor: {scaling['complexity_class']}"
        
    def test_dfa_performance(self):
        """Test DFA indicator performance"""
        
        results = self.benchmark.benchmark_indicator(
            DFAIndicator,
            self.data_sizes,
            self.iterations,
            min_scale=5,
            max_scale=min(50, min(self.data_sizes) // 4),  # Ensure valid parameters
            n_scales=10
        )
        
        # DFA should complete within reasonable time
        max_time = max(r['mean'] for r in results['execution_times'])
        assert max_time < 15.0, f"DFA calculation too slow: {max_time:.2f}s"
        
        # Memory usage check
        max_memory = max(r['mean_mb'] for r in results['memory_usage'])
        assert max_memory < 500, f"DFA uses too much memory: {max_memory:.1f}MB"
        
        # Scaling analysis
        scaling = self.benchmark.analyze_scaling(results)
        assert scaling['scaling_exponent'] < 3.0, f"DFA scaling too poor: {scaling['complexity_class']}"
        
    def test_kyles_lambda_performance(self):
        """Test Kyle's Lambda performance"""
        
        # Kyle's Lambda needs tick data
        results = self.benchmark.benchmark_indicator(
            KylesLambdaIndicator,
            [size // 2 for size in self.data_sizes],  # Smaller sizes for tick data
            self.iterations,
            estimation_window=50
        )
        
        max_time = max(r['mean'] for r in results['execution_times'])
        assert max_time < 5.0, f"Kyle's Lambda too slow: {max_time:.2f}s"
        
        scaling = self.benchmark.analyze_scaling(results)
        assert scaling['scaling_exponent'] < 2.0, f"Kyle's Lambda scaling poor: {scaling['complexity_class']}"
        
    def test_vpin_performance(self):
        """Test VPIN performance"""
        
        results = self.benchmark.benchmark_indicator(
            VPINIndicator,
            [size // 2 for size in self.data_sizes],
            self.iterations,
            bucket_volume=100,  # Small buckets for test data
            n_buckets=20
        )
        
        max_time = max(r['mean'] for r in results['execution_times'])
        assert max_time < 3.0, f"VPIN too slow: {max_time:.2f}s"
        
        # VPIN should be linear or near-linear
        scaling = self.benchmark.analyze_scaling(results)
        assert scaling['scaling_exponent'] < 1.5, f"VPIN scaling poor: {scaling['complexity_class']}"
        
    def test_tick_volume_imbalance_performance(self):
        """Test Tick Volume Imbalance performance"""
        
        results = self.benchmark.benchmark_indicator(
            TickVolumeImbalanceIndicator,
            [size // 2 for size in self.data_sizes],
            self.iterations,
            window_size=50
        )
        
        max_time = max(r['mean'] for r in results['execution_times'])
        assert max_time < 2.0, f"Tick Volume Imbalance too slow: {max_time:.2f}s"
        
        # Should be very efficient (near linear)
        scaling = self.benchmark.analyze_scaling(results)
        assert scaling['scaling_exponent'] < 1.3, f"TVI scaling poor: {scaling['complexity_class']}"
        
    def test_volume_bar_aggregator_performance(self):
        """Test Volume Bar Aggregator performance"""
        
        results = self.benchmark.benchmark_indicator(
            VolumeBarAggregator,
            [size // 2 for size in self.data_sizes],
            self.iterations,
            volume_threshold=500
        )
        
        max_time = max(r['mean'] for r in results['execution_times'])
        assert max_time < 2.0, f"Volume Bar Aggregator too slow: {max_time:.2f}s"
        
        # Should be linear
        scaling = self.benchmark.analyze_scaling(results)
        assert scaling['scaling_exponent'] < 1.2, f"Volume bars scaling poor: {scaling['complexity_class']}"
        
    def test_concurrent_indicator_usage(self):
        """Test performance when multiple indicators run simultaneously"""
        
        generator = SyntheticDataGenerator()
        data = generator.generate_random_walk(1000)
        tick_data = generator.generate_tick_data(500)
        
        indicators = [
            HurstExponentIndicator(lookback=800),
            DFAIndicator(min_scale=10, max_scale=100, n_scales=15),
        ]
        
        tick_indicators = [
            VPINIndicator(bucket_volume=100, n_buckets=10),
            TickVolumeImbalanceIndicator(window_size=100)
        ]
        
        # Measure concurrent execution time
        start_time = time.perf_counter()
        
        results = []
        for indicator in indicators:
            result = indicator.calculate(data, f"CONCURRENT_{indicator.name}")
            results.append(result)
            
        for indicator in tick_indicators:
            result = indicator.calculate(tick_data, f"CONCURRENT_{indicator.name}")
            results.append(result)
            
        total_time = time.perf_counter() - start_time
        
        # Should complete within reasonable time
        assert total_time < 30.0, f"Concurrent execution too slow: {total_time:.2f}s"
        
        # All indicators should return valid results
        assert len(results) == len(indicators) + len(tick_indicators)
        for result in results:
            assert result.confidence >= 0
            assert result.value is not None
            
    def test_memory_efficiency(self):
        """Test memory efficiency and garbage collection"""
        
        generator = SyntheticDataGenerator()
        
        # Monitor memory usage over multiple calculations
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        indicator = DFAIndicator()
        
        max_memory = initial_memory
        for i in range(10):
            # Generate fresh data each time
            data = generator.generate_random_walk(1000)
            result = indicator.calculate(data, f"MEMORY_TEST_{i}")
            
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            
            # Force cleanup
            del data, result
            if i % 3 == 0:
                gc.collect()
                
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (< 50MB)
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.1f}MB"
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        
        generator = SyntheticDataGenerator()
        
        # Test with very small values
        small_data = generator.generate_random_walk(500, volatility=1e-8)
        small_data['close'] = small_data['close'] * 1e-6  # Very small prices
        
        # Test with very large values
        large_data = generator.generate_random_walk(500, volatility=0.1)
        large_data['close'] = large_data['close'] * 1e6  # Very large prices
        
        # Test with high volatility
        volatile_data = generator.generate_random_walk(500, volatility=0.5)
        
        indicators = [
            HurstExponentIndicator(lookback=400),
            DFAIndicator(min_scale=10, max_scale=50)
        ]
        
        test_data = [
            ("small", small_data),
            ("large", large_data), 
            ("volatile", volatile_data)
        ]
        
        for indicator in indicators:
            for name, data in test_data:
                result = indicator.calculate(data, f"STABILITY_{name}")
                
                # Results should be finite and reasonable
                assert np.isfinite(result.value), f"{indicator.name} produced non-finite value for {name} data"
                assert 0 <= result.confidence <= 100, f"{indicator.name} confidence out of bounds for {name} data"
                
                # Metadata should not contain NaN or inf
                for key, value in result.metadata.items():
                    if isinstance(value, (int, float)):
                        assert np.isfinite(value), f"{indicator.name} metadata '{key}' is not finite for {name} data"