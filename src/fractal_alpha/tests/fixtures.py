"""
🔧 Test Fixtures and Data Generators
Provides standardized test data for fractal indicator testing
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from datetime import datetime, timedelta
import pytest
from dataclasses import dataclass

from ..base.types import TickData, BarData, TimeFrame, SignalType


@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    data_type: str  # 'trending', 'mean_reverting', 'random_walk', 'volatile', 'regime_change'
    length: int
    expected_characteristics: Dict
    noise_level: float = 0.01


class SyntheticDataGenerator:
    """Generates synthetic market data with known characteristics for testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.base_price = 100.0
        
    def generate_random_walk(self, length: int, volatility: float = 0.01) -> pd.DataFrame:
        """Generate random walk data (efficient market)"""
        
        timestamps = pd.date_range(
            start=datetime.now(), 
            periods=length, 
            freq='1min'
        )
        
        prices = [self.base_price]
        volumes = np.random.lognormal(mean=7, sigma=1, size=length)  # ~1000 avg volume
        
        for i in range(length - 1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
            
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('timestamp')
        
    def generate_trending_data(self, length: int, trend_strength: float = 0.001, 
                              volatility: float = 0.01) -> pd.DataFrame:
        """Generate persistent trending data"""
        
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=length,
            freq='1min'
        )
        
        prices = [self.base_price]
        volumes = np.random.lognormal(mean=7, sigma=1, size=length)
        
        # Add momentum to create persistence
        momentum = 0
        for i in range(length - 1):
            # Trending component
            trend = trend_strength
            
            # Random component
            random_change = np.random.normal(0, volatility)
            
            # Momentum component (creates persistence)
            momentum = 0.8 * momentum + 0.2 * random_change
            
            total_change = trend + random_change + momentum * 0.5
            prices.append(prices[-1] * (1 + total_change))
            
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Fix OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('timestamp')
        
    def generate_mean_reverting_data(self, length: int, reversion_strength: float = 0.05,
                                   volatility: float = 0.01) -> pd.DataFrame:
        """Generate anti-persistent mean-reverting data"""
        
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=length,
            freq='1min'
        )
        
        prices = [self.base_price]
        volumes = np.random.lognormal(mean=7, sigma=1, size=length)
        
        for i in range(length - 1):
            # Mean reversion force
            deviation = (prices[-1] - self.base_price) / self.base_price
            reversion = -reversion_strength * deviation
            
            # Random component
            random_change = np.random.normal(0, volatility)
            
            total_change = reversion + random_change
            prices.append(prices[-1] * (1 + total_change))
            
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Fix OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data.set_index('timestamp')
        
    def generate_regime_change_data(self, length: int) -> pd.DataFrame:
        """Generate data with regime changes"""
        
        # Split into three regimes
        regime1_length = length // 3
        regime2_length = length // 3
        regime3_length = length - regime1_length - regime2_length
        
        # Regime 1: Mean reverting
        regime1 = self.generate_mean_reverting_data(regime1_length)
        
        # Regime 2: Trending
        self.base_price = regime1['close'].iloc[-1]
        regime2 = self.generate_trending_data(regime2_length)
        
        # Regime 3: High volatility random walk
        self.base_price = regime2['close'].iloc[-1]
        regime3 = self.generate_random_walk(regime3_length, volatility=0.03)
        
        # Combine regimes
        combined = pd.concat([regime1, regime2, regime3])
        
        # Reset timestamps to be continuous
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=length,
            freq='1min'
        )
        combined.index = timestamps
        
        return combined
        
    def generate_tick_data(self, n_ticks: int, base_price: float = 100.0) -> List[TickData]:
        """Generate realistic tick data"""
        
        ticks = []
        current_price = base_price
        timestamp = int(datetime.now().timestamp() * 1000)
        
        for i in range(n_ticks):
            # Price movement
            price_change = np.random.normal(0, 0.001)
            current_price *= (1 + price_change)
            
            # Volume
            volume = int(np.random.lognormal(4, 1))  # ~50-500 range
            
            # Side (buy/sell)
            side = np.random.choice([1, -1])
            
            # Create tick
            tick = TickData(
                timestamp=timestamp + i * 100,  # 100ms between ticks
                price=current_price,
                volume=volume,
                side=side
            )
            
            ticks.append(tick)
            
        return ticks
        
    def generate_informed_trading_ticks(self, n_ticks: int, 
                                      informed_ratio: float = 0.3) -> List[TickData]:
        """Generate tick data with informed trading patterns"""
        
        ticks = []
        current_price = 100.0
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Informed traders have directional bias
        informed_direction = np.random.choice([1, -1])
        
        for i in range(n_ticks):
            # Determine if this is an informed trade
            is_informed = np.random.random() < informed_ratio
            
            if is_informed:
                # Informed trades
                side = informed_direction
                volume = int(np.random.lognormal(6, 0.5))  # Larger sizes
                price_impact = side * 0.0001  # Small price impact
            else:
                # Uninformed trades
                side = np.random.choice([1, -1])
                volume = int(np.random.lognormal(4, 1))  # Normal sizes
                price_impact = np.random.normal(0, 0.0005)  # Random impact
                
            current_price += price_impact
            
            tick = TickData(
                timestamp=timestamp + i * 50,  # 50ms between ticks
                price=current_price,
                volume=volume,
                side=side
            )
            
            ticks.append(tick)
            
        return ticks


# Test scenarios
TEST_SCENARIOS = [
    TestScenario(
        name="random_walk",
        description="Efficient market random walk",
        data_type="random_walk",
        length=1000,
        expected_characteristics={
            'hurst_exponent': 0.5,
            'dfa_scaling': 0.5,
            'trend_strength': 0.0,
            'mean_reversion': False
        }
    ),
    TestScenario(
        name="strong_trend",
        description="Persistent trending market",
        data_type="trending",
        length=1000,
        expected_characteristics={
            'hurst_exponent': 0.7,
            'dfa_scaling': 0.7,
            'trend_strength': 0.8,
            'mean_reversion': False
        }
    ),
    TestScenario(
        name="mean_reverting",
        description="Anti-persistent mean-reverting market",
        data_type="mean_reverting",
        length=1000,
        expected_characteristics={
            'hurst_exponent': 0.3,
            'dfa_scaling': 0.3,
            'trend_strength': 0.0,
            'mean_reversion': True
        }
    ),
    TestScenario(
        name="regime_change",
        description="Multiple regime changes",
        data_type="regime_change",
        length=1500,
        expected_characteristics={
            'hurst_exponent': 0.5,  # Average across regimes
            'regime_changes': 2,
            'stability': False
        }
    )
]


# Pytest fixtures
@pytest.fixture
def data_generator():
    """Fixture providing data generator"""
    return SyntheticDataGenerator()


@pytest.fixture
def random_walk_data(data_generator):
    """Random walk test data"""
    return data_generator.generate_random_walk(1000)


@pytest.fixture
def trending_data(data_generator):
    """Trending test data"""
    return data_generator.generate_trending_data(1000)


@pytest.fixture
def mean_reverting_data(data_generator):
    """Mean-reverting test data"""
    return data_generator.generate_mean_reverting_data(1000)


@pytest.fixture
def regime_change_data(data_generator):
    """Regime change test data"""
    return data_generator.generate_regime_change_data(1500)


@pytest.fixture
def tick_data(data_generator):
    """Standard tick data"""
    return data_generator.generate_tick_data(500)


@pytest.fixture
def informed_tick_data(data_generator):
    """Tick data with informed trading"""
    return data_generator.generate_informed_trading_ticks(500)


@pytest.fixture
def test_scenarios():
    """All test scenarios"""
    return TEST_SCENARIOS


def get_test_data_for_scenario(scenario_name: str) -> pd.DataFrame:
    """Get test data for a specific scenario"""
    generator = SyntheticDataGenerator()
    
    scenario = next((s for s in TEST_SCENARIOS if s.name == scenario_name), None)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}")
        
    if scenario.data_type == "random_walk":
        return generator.generate_random_walk(scenario.length)
    elif scenario.data_type == "trending":
        return generator.generate_trending_data(scenario.length)
    elif scenario.data_type == "mean_reverting":
        return generator.generate_mean_reverting_data(scenario.length)
    elif scenario.data_type == "regime_change":
        return generator.generate_regime_change_data(scenario.length)
    else:
        raise ValueError(f"Unknown data type: {scenario.data_type}")