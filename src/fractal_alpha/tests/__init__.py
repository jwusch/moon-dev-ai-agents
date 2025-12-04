"""
🧪 Fractal Alpha Testing Framework
Comprehensive test suite for fractal market analysis indicators
"""

from .test_indicator_base import *
from .test_microstructure import *
from .test_multifractal import *
from .test_time_patterns import *
from .test_performance import *
from .test_statistical_validation import *
from .test_integration import *

__version__ = "1.0.0"
__author__ = "Moon Dev AI Agents"

# Test configuration
TEST_CONFIG = {
    'sample_data_size': 1000,
    'benchmark_iterations': 100,
    'statistical_significance': 0.05,
    'performance_tolerance': 0.1,  # 10% performance tolerance
    'random_seed': 42
}

# Test data generators
from .fixtures import *