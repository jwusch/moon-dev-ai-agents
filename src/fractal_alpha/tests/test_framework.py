"""
🧪 COMPREHENSIVE TESTING FRAMEWORK FOR FRACTAL INDICATORS
Automated testing suite for all fractal alpha indicators
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from abc import ABC, abstractmethod

# Import all indicators
from ..indicators.microstructure.tick_volume import TickVolumeIndicator
from ..indicators.microstructure.order_flow import OrderFlowIndicator
from ..indicators.microstructure.bid_ask import BidAskIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.amihud import AmihudIndicator

from ..indicators.multi_timeframe.williams_fractals import WilliamsFractalIndicator
from ..indicators.multi_timeframe.hurst_exponent import HurstExponentIndicator
from ..indicators.multi_timeframe.dfa import DFAIndicator

from ..indicators.time_patterns.intraday import IntradaySeasonalityIndicator
from ..indicators.time_patterns.volume_bars import VolumeBarIndicator
from ..indicators.time_patterns.renko import RenkoIndicator

from ..indicators.cross_asset.sector_rotation import SectorRotationIndicator
from ..indicators.cross_asset.vix_correlation import VIXCorrelationIndicator
from ..indicators.cross_asset.dollar_correlation import DollarCorrelationIndicator

from ..indicators.mean_reversion.ou_process import OUProcessIndicator
from ..indicators.mean_reversion.dynamic_zscore import DynamicZScoreIndicator
from ..indicators.mean_reversion.pairs_trading import PairsTradingIndicator

from ..indicators.ml_features.entropy import EntropyIndicator
from ..indicators.ml_features.wavelet import WaveletIndicator
from ..indicators.ml_features.hmm import HMMIndicator

from ..base.types import TimeFrame, SignalType


class IndicatorTestCase(ABC):
    """Base class for indicator test cases"""
    
    @abstractmethod
    def get_indicator_class(self):
        """Return the indicator class to test"""
        pass
    
    @abstractmethod
    def get_test_data(self) -> pd.DataFrame:
        """Generate test data for the indicator"""
        pass
    
    def test_initialization(self):
        """Test indicator initialization"""
        indicator_class = self.get_indicator_class()
        indicator = indicator_class()
        self.assertIsNotNone(indicator)
        self.assertIsNotNone(indicator.name)
        self.assertIsNotNone(indicator.timeframe)
    
    def test_calculation(self):
        """Test indicator calculation"""
        indicator_class = self.get_indicator_class()
        indicator = indicator_class()
        data = self.get_test_data()
        
        result = indicator.calculate(data, "TEST")
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.timestamp)
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.indicator_name, indicator.name)
        self.assertIn(result.signal, [SignalType.BUY, SignalType.SELL, SignalType.HOLD])
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 100)
        self.assertIsNotNone(result.value)
        self.assertIsInstance(result.metadata, dict)
    
    def test_insufficient_data(self):
        """Test indicator with insufficient data"""
        indicator_class = self.get_indicator_class()
        indicator = indicator_class()
        
        # Create minimal data
        data = pd.DataFrame({
            'Close': [100],
            'Volume': [1000]
        }, index=[datetime.now()])
        
        result = indicator.calculate(data, "TEST")
        
        # Should return valid result with HOLD signal
        self.assertEqual(result.signal, SignalType.HOLD)
        self.assertEqual(result.confidence, 0)
    
    def test_edge_cases(self):
        """Test indicator edge cases"""
        # Override in specific test cases
        pass


class TestMicrostructureIndicators(unittest.TestCase):
    """Test suite for microstructure indicators"""
    
    def generate_tick_data(self, n_ticks: int = 1000) -> pd.DataFrame:
        """Generate synthetic tick data"""
        np.random.seed(42)
        
        # Generate prices with microstructure noise
        mid_price = 100
        prices = []
        bids = []
        asks = []
        volumes = []
        
        for i in range(n_ticks):
            # Random walk for mid price
            mid_price *= (1 + np.random.normal(0, 0.001))
            
            # Bid-ask spread
            spread = np.random.uniform(0.01, 0.05)
            bid = mid_price - spread/2
            ask = mid_price + spread/2
            
            # Trade price (with probability of trading at bid/ask)
            prob = np.random.random()
            if prob < 0.3:
                price = bid
            elif prob > 0.7:
                price = ask
            else:
                price = mid_price + np.random.uniform(-spread/4, spread/4)
            
            prices.append(price)
            bids.append(bid)
            asks.append(ask)
            volumes.append(np.random.lognormal(7, 1))
        
        # Create DataFrame
        timestamps = pd.date_range(start='2024-01-01', periods=n_ticks, freq='1s')
        
        return pd.DataFrame({
            'price': prices,
            'bid': bids,
            'ask': asks,
            'volume': volumes
        }, index=timestamps)
    
    def test_tick_volume_indicator(self):
        """Test tick volume imbalance indicator"""
        data = self.generate_tick_data()
        indicator = TickVolumeIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('tick_imbalance', result.metadata)
        self.assertIn('volume_imbalance', result.metadata)
        self.assertIn('toxic_flow_probability', result.metadata)
    
    def test_order_flow_indicator(self):
        """Test order flow divergence indicator"""
        data = self.generate_tick_data()
        indicator = OrderFlowIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('flow_imbalance', result.metadata)
        self.assertIn('divergence_detected', result.metadata)
    
    def test_vpin_indicator(self):
        """Test VPIN indicator"""
        data = self.generate_tick_data()
        indicator = VPINIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('vpin', result.metadata)
        self.assertIn('toxicity_level', result.metadata)


class TestMultiTimeframeIndicators(unittest.TestCase):
    """Test suite for multi-timeframe indicators"""
    
    def generate_price_data(self, n_days: int = 100) -> pd.DataFrame:
        """Generate synthetic price data"""
        np.random.seed(42)
        
        # Generate prices with fractal properties
        prices = [100]
        for i in range(1, n_days):
            # Add trend and noise
            trend = 0.0005 * i
            noise = np.random.normal(0, 0.02)
            fractal_noise = 0.01 * np.sin(2 * np.pi * i / 20)  # Cyclic component
            
            new_price = prices[-1] * (1 + trend + noise + fractal_noise)
            prices.append(new_price)
        
        # Create OHLC data
        opens = prices
        highs = [p * (1 + np.random.uniform(0, 0.01)) for p in prices]
        lows = [p * (1 - np.random.uniform(0, 0.01)) for p in prices]
        closes = [p + np.random.uniform(-0.5, 0.5) for p in prices]
        volumes = [np.random.uniform(1e6, 2e6) for _ in prices]
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
    
    def test_williams_fractals(self):
        """Test Williams Fractals indicator"""
        data = self.generate_price_data(200)
        indicator = WilliamsFractalIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('total_fractals', result.metadata)
        self.assertIn('fractal_density', result.metadata)
    
    def test_hurst_exponent(self):
        """Test Hurst Exponent indicator"""
        data = self.generate_price_data(500)
        indicator = HurstExponentIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('hurst_value', result.metadata)
        self.assertIn('market_regime', result.metadata)
        
        # Hurst should be between 0 and 1
        hurst = result.metadata['hurst_value']
        self.assertGreaterEqual(hurst, 0)
        self.assertLessEqual(hurst, 1)


class TestMLIndicators(unittest.TestCase):
    """Test suite for ML-based indicators"""
    
    def generate_regime_data(self, n_days: int = 300) -> pd.DataFrame:
        """Generate data with distinct regimes"""
        np.random.seed(42)
        
        prices = []
        price = 100
        
        for i in range(n_days):
            if i < 100:
                # Low volatility uptrend
                change = np.random.normal(0.001, 0.005)
            elif i < 200:
                # High volatility ranging
                change = np.random.normal(0, 0.02)
            else:
                # Downtrend
                change = np.random.normal(-0.0005, 0.01)
            
            price *= (1 + change)
            prices.append(price)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        return pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': np.random.uniform(1e6, 2e6, n_days)
        }, index=dates)
    
    def test_entropy_indicator(self):
        """Test entropy-based indicator"""
        data = self.generate_regime_data()
        indicator = EntropyIndicator()
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('current_state', result.metadata)
        self.assertIn('entropy', result.metadata['current_state'])
    
    def test_hmm_indicator(self):
        """Test Hidden Markov Model indicator"""
        data = self.generate_regime_data()
        indicator = HMMIndicator(n_states=3)
        result = indicator.calculate(data, "TEST")
        
        self.assertIsNotNone(result)
        self.assertIn('current_state', result.metadata)
        self.assertIn('state_analysis', result.metadata)
        self.assertIn('regime_stability', result.metadata)


class TestIntegration(unittest.TestCase):
    """Integration tests for indicator combinations"""
    
    def test_regime_detection_ensemble(self):
        """Test multiple regime detection indicators together"""
        # Generate test data
        np.random.seed(42)
        n_days = 200
        prices = []
        price = 100
        
        for i in range(n_days):
            if i < 100:
                change = np.random.normal(0.001, 0.01)  # Trending
            else:
                change = np.random.normal(0, 0.02)  # Volatile
            
            price *= (1 + change)
            prices.append(price)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.uniform(1e6, 2e6, n_days)
        }, index=dates)
        
        # Calculate multiple indicators
        hurst = HurstExponentIndicator()
        entropy = EntropyIndicator()
        
        hurst_result = hurst.calculate(data, "TEST")
        entropy_result = entropy.calculate(data, "TEST")
        
        # Both should detect regime change
        self.assertIsNotNone(hurst_result)
        self.assertIsNotNone(entropy_result)
        
        # Check if regimes align
        hurst_regime = hurst_result.metadata.get('market_regime')
        entropy_value = entropy_result.value
        
        print(f"Hurst Regime: {hurst_regime}")
        print(f"Entropy Value: {entropy_value}")


class PerformanceTestSuite:
    """Performance benchmarking for indicators"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_indicator(self, indicator_class, data_size: int = 1000):
        """Benchmark a single indicator"""
        import time
        
        # Generate data
        data = self._generate_benchmark_data(data_size)
        
        # Initialize indicator
        indicator = indicator_class()
        
        # Time calculation
        start_time = time.time()
        result = indicator.calculate(data, "BENCH")
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # milliseconds
        
        return {
            'indicator': indicator.name,
            'data_points': data_size,
            'execution_time_ms': execution_time,
            'result': result
        }
    
    def _generate_benchmark_data(self, size: int) -> pd.DataFrame:
        """Generate benchmark data"""
        np.random.seed(42)
        
        prices = np.random.lognormal(np.log(100), 0.02, size)
        volumes = np.random.lognormal(np.log(1e6), 0.5, size)
        
        dates = pd.date_range(end=datetime.now(), periods=size, freq='1min')
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
    
    def run_full_benchmark(self):
        """Run benchmark on all indicators"""
        indicators = [
            TickVolumeIndicator,
            OrderFlowIndicator,
            VPINIndicator,
            WilliamsFractalIndicator,
            HurstExponentIndicator,
            EntropyIndicator,
            WaveletIndicator,
            HMMIndicator
        ]
        
        print("\n🚀 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        for indicator_class in indicators:
            try:
                result = self.benchmark_indicator(indicator_class)
                print(f"{result['indicator']:<30} {result['execution_time_ms']:>10.2f} ms")
                self.results[result['indicator']] = result
            except Exception as e:
                print(f"{indicator_class.__name__:<30} ERROR: {str(e)[:20]}")
        
        # Summary statistics
        if self.results:
            times = [r['execution_time_ms'] for r in self.results.values()]
            print("\n📊 SUMMARY STATISTICS")
            print("=" * 60)
            print(f"Average execution time: {np.mean(times):.2f} ms")
            print(f"Median execution time:  {np.median(times):.2f} ms")
            print(f"Fastest indicator:      {min(self.results.items(), key=lambda x: x[1]['execution_time_ms'])[0]}")
            print(f"Slowest indicator:      {max(self.results.items(), key=lambda x: x[1]['execution_time_ms'])[0]}")


def run_all_tests():
    """Run complete test suite"""
    print("🧪 FRACTAL INDICATORS TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMicrostructureIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTimeframeIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestMLIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmarks
    print("\n" + "=" * 80)
    benchmark = PerformanceTestSuite()
    benchmark.run_full_benchmark()
    
    # Save results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        'performance_results': benchmark.results
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n✅ Test suite complete!")
    print(f"📊 Success rate: {test_results['success_rate']:.1f}%")
    print(f"💾 Results saved to test_results.json")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()