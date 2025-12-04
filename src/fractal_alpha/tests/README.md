# 🧪 Fractal Alpha Testing Framework

Comprehensive test suite for fractal market analysis indicators, ensuring reliability, performance, and statistical correctness.

## 📋 Test Categories

### 1. **Unit Tests** (`test_indicator_base.py`)
- Interface compliance testing
- Parameter validation
- Error handling
- Data validation
- Result structure verification

### 2. **Microstructure Tests** (`test_microstructure.py`)
- Kyle's Lambda price impact calculation
- VPIN bucket formation and calculation
- Tick Volume Imbalance detection
- Order flow analysis validation

### 3. **Multifractal Tests** (`test_multifractal.py`)
- Hurst Exponent calculation accuracy
- DFA scaling law validation
- Regime detection consistency
- Signal generation logic

### 4. **Time Pattern Tests** (`test_time_patterns.py`)
- Volume bar aggregation
- OHLC calculation accuracy
- VWAP and imbalance metrics
- Auto-threshold adjustment

### 5. **Performance Tests** (`test_performance.py`)
- Execution speed benchmarks
- Memory usage monitoring
- Algorithmic complexity analysis
- Scalability testing

### 6. **Statistical Validation** (`test_statistical_validation.py`)
- Mathematical correctness
- Distribution properties
- Monte Carlo validation
- Cross-correlation analysis

### 7. **Integration Tests** (`test_integration.py`)
- End-to-end workflows
- Ensemble indicator coordination
- Real-time simulation
- Error propagation handling

## 🚀 Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest src/fractal_alpha/tests/

# Run specific test category
pytest src/fractal_alpha/tests/test_microstructure.py

# Run with coverage
pytest --cov=src/fractal_alpha src/fractal_alpha/tests/
```

### Test Selection Options
```bash
# Fast tests only (skip performance benchmarks)
pytest --fast src/fractal_alpha/tests/

# Statistical validation only
pytest --statistical-only src/fractal_alpha/tests/

# Performance tests only  
pytest --performance-only src/fractal_alpha/tests/

# Run tests by marker
pytest -m "not slow" src/fractal_alpha/tests/
pytest -m "statistical" src/fractal_alpha/tests/
pytest -m "integration" src/fractal_alpha/tests/
```

### Verbose Output
```bash
# Detailed output
pytest -v src/fractal_alpha/tests/

# Show test setup/teardown
pytest -s src/fractal_alpha/tests/

# Capture print statements
pytest --capture=no src/fractal_alpha/tests/
```

## 📊 Test Fixtures and Data

### Synthetic Data Generation
The test suite uses `SyntheticDataGenerator` to create controlled market data:

- **Random Walk**: Efficient market simulation (H ≈ 0.5)
- **Trending Data**: Persistent market with momentum (H > 0.5)  
- **Mean Reverting**: Anti-persistent market (H < 0.5)
- **Regime Changes**: Multi-regime transitions
- **Informed Trading**: Tick data with order flow patterns

### Test Scenarios
Pre-configured test scenarios validate indicator behavior:

```python
TEST_SCENARIOS = [
    TestScenario("random_walk", expected_hurst=0.5),
    TestScenario("strong_trend", expected_hurst=0.7), 
    TestScenario("mean_reverting", expected_hurst=0.3),
    TestScenario("regime_change", regime_changes=2)
]
```

### Shared Fixtures
- `data_generator`: Fresh generator instance per test
- `test_scenarios`: Parametrized scenarios for comprehensive testing
- `performance_monitor`: Automatic performance tracking
- `validation_helpers`: Common assertion utilities

## 🔍 Statistical Validation

### Hurst Exponent Validation
- Range validation: 0 ≤ H ≤ 1
- Regime consistency: H < 0.5 (mean-reverting), H > 0.5 (trending)
- Statistical stability across multiple realizations

### DFA Scaling Law Validation  
- Power law relationship: F(n) ∝ n^α
- R-squared goodness of fit > 0.8
- Scaling exponent consistency with known patterns

### Kyle's Lambda Distribution
- Positive values (price impact should be non-negative)
- Log-normal distribution characteristics
- Correlation with market liquidity conditions

### VPIN Properties
- Bounded range: 0 ≤ VPIN ≤ 1
- Monotonic relationship with order imbalance
- Stability across different bucket sizes

## ⚡ Performance Benchmarks

### Speed Requirements
- **Hurst Exponent**: < 10s for 2000 data points
- **DFA**: < 15s for 2000 data points  
- **Kyle's Lambda**: < 5s for 1000 ticks
- **VPIN**: < 3s for 1000 ticks
- **Volume Bars**: < 2s for 1000 ticks

### Memory Constraints
- Maximum memory usage: < 500MB
- Memory growth: < 50MB over 10 iterations
- Proper cleanup and garbage collection

### Algorithmic Complexity
- Most indicators should scale O(n) or O(n log n)
- DFA allowed up to O(n²) due to multi-scale analysis
- No indicator should exceed O(n³)

## 🔧 Test Configuration

### Pytest Markers
- `@pytest.mark.slow`: Performance-intensive tests
- `@pytest.mark.statistical`: Statistical validation tests
- `@pytest.mark.integration`: End-to-end workflow tests
- `@pytest.mark.performance`: Speed and memory benchmarks

### Configuration Options
```python
TEST_CONFIG = {
    'sample_data_size': 1000,
    'benchmark_iterations': 100,
    'statistical_significance': 0.05,
    'performance_tolerance': 0.1,
    'random_seed': 42
}
```

### Command Line Options
- `--fast`: Skip slow performance tests
- `--statistical-only`: Run only statistical validation
- `--performance-only`: Run only performance benchmarks

## 📈 Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Fast Tests
  run: pytest --fast src/fractal_alpha/tests/
  
- name: Run Statistical Validation  
  run: pytest --statistical-only src/fractal_alpha/tests/
  
- name: Performance Regression Check
  run: pytest -m performance src/fractal_alpha/tests/
```

### Test Coverage Goals
- Unit tests: > 95% line coverage
- Integration tests: > 80% scenario coverage
- Performance tests: All indicators benchmarked
- Statistical tests: Key mathematical properties validated

## 🔧 Adding New Tests

### Test Structure Template
```python
class TestNewIndicator:
    def test_basic_functionality(self):
        # Test core algorithm
        pass
        
    def test_edge_cases(self):
        # Test boundary conditions
        pass
        
    def test_performance(self):
        # Benchmark speed and memory
        pass
        
    def test_statistical_properties(self):
        # Validate mathematical correctness
        pass
```

### Fixture Usage
```python
def test_with_synthetic_data(test_scenario, validation_helpers):
    scenario = test_scenario['scenario']
    data = test_scenario['data']
    
    indicator = create_indicator()
    result = indicator.calculate(data, scenario.name)
    
    validation_helpers['assert_valid_result'](result)
```

### Performance Testing
```python
def test_indicator_performance(performance_monitor):
    # Test implementation
    
    perf = performance_monitor
    execution_time = time.perf_counter() - perf['start_time']
    
    assert execution_time < 10.0  # Performance requirement
```

## 🎯 Testing Best Practices

1. **Reproducibility**: Use fixed seeds for deterministic results
2. **Isolation**: Each test should be independent
3. **Clarity**: Test names should describe expected behavior
4. **Coverage**: Test both happy paths and edge cases
5. **Performance**: Monitor resource usage in CI/CD
6. **Documentation**: Comment complex test logic
7. **Maintenance**: Regular review and cleanup of test code

## 🐛 Debugging Failed Tests

### Common Issues
- **Floating Point Precision**: Use `np.isclose()` for comparisons
- **Random Variation**: Increase sample sizes or use fixed seeds
- **Platform Differences**: Account for OS-specific numerical differences
- **Memory Leaks**: Ensure proper cleanup in teardown

### Debug Commands
```bash
# Run single test with full output
pytest -xvs src/fractal_alpha/tests/test_file.py::test_function

# Debug with PDB
pytest --pdb src/fractal_alpha/tests/test_file.py::test_function

# Show test durations
pytest --durations=10 src/fractal_alpha/tests/
```

This comprehensive testing framework ensures the fractal alpha indicators are reliable, performant, and mathematically sound for production trading applications.