"""
🔧 Pytest Configuration and Shared Fixtures
Global configuration and fixtures for the fractal alpha test suite
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.fractal_alpha.tests.fixtures import SyntheticDataGenerator, TEST_SCENARIOS


# Configure pytest
def pytest_configure(config):
    """Configure pytest settings"""
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    
    # Add markers based on test file names
    for item in items:
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "statistical" in item.nodeid:
            item.add_marker(pytest.mark.statistical)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# Global fixtures
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        'random_seed': 42,
        'sample_size': 1000,
        'statistical_alpha': 0.05,
        'performance_threshold': 10.0,  # seconds
        'memory_threshold': 500,  # MB
    }


@pytest.fixture(scope="session") 
def data_generator_session():
    """Session-scoped data generator for expensive operations"""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def data_generator():
    """Function-scoped data generator for most tests"""
    return SyntheticDataGenerator()


# Data fixtures
@pytest.fixture
def small_random_walk(data_generator):
    """Small random walk dataset for quick tests"""
    return data_generator.generate_random_walk(100)


@pytest.fixture
def medium_trending_data(data_generator):
    """Medium trending dataset"""
    return data_generator.generate_trending_data(500, trend_strength=0.002)


@pytest.fixture
def large_mean_reverting_data(data_generator):
    """Large mean-reverting dataset"""
    return data_generator.generate_mean_reverting_data(1000, reversion_strength=0.1)


@pytest.fixture
def regime_change_data(data_generator):
    """Dataset with multiple regime changes"""
    return data_generator.generate_regime_change_data(1500)


@pytest.fixture
def tick_data_sample(data_generator):
    """Sample tick data"""
    return data_generator.generate_tick_data(200)


@pytest.fixture
def informed_tick_data(data_generator):
    """Tick data with informed trading patterns"""
    return data_generator.generate_informed_trading_ticks(300, informed_ratio=0.4)


# Scenario-based fixtures
@pytest.fixture(params=TEST_SCENARIOS)
def test_scenario(request, data_generator):
    """Parametrized fixture for all test scenarios"""
    scenario = request.param
    
    if scenario.data_type == "random_walk":
        data = data_generator.generate_random_walk(scenario.length)
    elif scenario.data_type == "trending":
        data = data_generator.generate_trending_data(scenario.length)
    elif scenario.data_type == "mean_reverting":
        data = data_generator.generate_mean_reverting_data(scenario.length)
    elif scenario.data_type == "regime_change":
        data = data_generator.generate_regime_change_data(scenario.length)
    else:
        raise ValueError(f"Unknown scenario data type: {scenario.data_type}")
        
    return {
        'scenario': scenario,
        'data': data
    }


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance"""
    import time
    import psutil
    
    process = psutil.Process()
    
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield {
        'start_time': start_time,
        'start_memory': start_memory,
        'process': process
    }
    
    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    # Store results for potential reporting
    pytest.current_test_performance = {
        'execution_time': execution_time,
        'memory_used': memory_used,
        'peak_memory': end_memory
    }


# Validation helpers
@pytest.fixture
def validation_helpers():
    """Helper functions for test validation"""
    
    def assert_valid_indicator_result(result):
        """Assert that an indicator result is valid"""
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'metadata')
        
        assert 0 <= result.confidence <= 100
        assert isinstance(result.metadata, dict)
        
    def assert_reasonable_performance(execution_time, memory_used, max_time=10, max_memory=100):
        """Assert reasonable performance characteristics"""
        assert execution_time < max_time, f"Execution too slow: {execution_time:.2f}s"
        assert memory_used < max_memory, f"Memory usage too high: {memory_used:.1f}MB"
        
    def assert_statistical_significance(p_value, alpha=0.05):
        """Assert statistical significance"""
        assert p_value < alpha, f"Result not statistically significant: p={p_value:.4f}"
        
    return {
        'assert_valid_result': assert_valid_indicator_result,
        'assert_performance': assert_reasonable_performance,
        'assert_significance': assert_statistical_significance
    }


# Mock data fixtures for edge cases
@pytest.fixture
def edge_case_data():
    """Edge case datasets for robustness testing"""
    
    # Very small dataset
    tiny_data = pd.DataFrame({
        'close': [100, 101, 99],
        'volume': [1000, 1100, 900]
    })
    
    # Dataset with NaN values
    nan_data = pd.DataFrame({
        'close': [100, np.nan, 102, 103, np.nan],
        'volume': [1000, 1100, np.nan, 1200, 1000]
    })
    
    # Zero variance dataset
    constant_data = pd.DataFrame({
        'close': [100] * 50,
        'volume': [1000] * 50
    })
    
    # High volatility dataset
    volatile_data = pd.DataFrame({
        'close': 100 + np.random.normal(0, 10, 100),
        'volume': np.random.lognormal(7, 2, 100)
    })
    
    return {
        'tiny': tiny_data,
        'nan_values': nan_data,
        'constant': constant_data,
        'volatile': volatile_data
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_numpy_warnings():
    """Automatically clean up numpy warnings"""
    yield
    # Reset numpy error handling after each test
    np.seterr(all='warn')


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state after each test for reproducibility"""
    yield
    np.random.seed(None)  # Reset to system entropy


# Custom markers for test categorization
pytest.mark.slow = pytest.mark.slow
pytest.mark.statistical = pytest.mark.statistical  
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance


# Test reporting hooks
def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test"""
    import gc
    
    # Force garbage collection to prevent memory accumulation
    gc.collect()
    
    # Clear any matplotlib figures if present
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup and reporting"""
    
    # Print performance summary if available
    if hasattr(pytest, 'performance_results'):
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        for test_name, perf in pytest.performance_results.items():
            print(f"{test_name}: {perf['execution_time']:.2f}s, {perf['memory_used']:.1f}MB")


# Skip conditions for different environments
def pytest_configure(config):
    """Configure test skipping based on environment"""
    
    # Skip slow tests if --fast flag is used
    if config.getoption('--fast', default=False):
        pytest.mark.skip_slow = pytest.mark.skipif(
            True, reason="Skipping slow tests in fast mode"
        )
    else:
        pytest.mark.skip_slow = pytest.mark.skipif(
            False, reason="Not skipping slow tests"
        )


def pytest_addoption(parser):
    """Add custom command line options"""
    
    parser.addoption(
        "--fast", 
        action="store_true", 
        default=False, 
        help="Run only fast tests"
    )
    
    parser.addoption(
        "--statistical-only",
        action="store_true", 
        default=False,
        help="Run only statistical validation tests"
    )
    
    parser.addoption(
        "--performance-only",
        action="store_true",
        default=False, 
        help="Run only performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options"""
    
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in fast mode")
        for item in items:
            if "slow" in item.keywords or "performance" in item.keywords:
                item.add_marker(skip_slow)
                
    if config.getoption("--statistical-only"):
        skip_non_stat = pytest.mark.skip(reason="Running only statistical tests")
        for item in items:
            if "statistical" not in item.keywords:
                item.add_marker(skip_non_stat)
                
    if config.getoption("--performance-only"):
        skip_non_perf = pytest.mark.skip(reason="Running only performance tests")
        for item in items:
            if "performance" not in item.keywords:
                item.add_marker(skip_non_perf)