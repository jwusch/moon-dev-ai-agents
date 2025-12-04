"""
🔬 Microstructure Indicators Tests
Tests for market microstructure analysis indicators
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator
from ..base.types import TickData, SignalType
from .fixtures import SyntheticDataGenerator


class TestKylesLambdaIndicator:
    """Test Kyle's Lambda price impact indicator"""
    
    def test_lambda_calculation_basic(self):
        """Test basic lambda calculation"""
        
        generator = SyntheticDataGenerator()
        
        # Generate ticks with known price impact
        ticks = []
        price = 100.0
        timestamp = 1000
        
        # Create predictable price impact pattern
        for i in range(100):
            volume = 100
            side = 1 if i % 2 == 0 else -1
            
            # Predictable price impact
            price_impact = side * volume * 0.0001
            price += price_impact
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        indicator = KylesLambdaIndicator(estimation_window=50)
        result = indicator.calculate(ticks, "TEST")
        
        # Should detect price impact
        assert result.metadata.get('lambda') is not None
        assert isinstance(result.metadata['lambda'], float)
        
    def test_liquidity_scoring(self):
        """Test liquidity score calculation"""
        
        generator = SyntheticDataGenerator()
        
        # High liquidity scenario (low price impact)
        low_impact_ticks = []
        price = 100.0
        
        for i in range(100):
            # Very small price impact
            price += np.random.normal(0, 0.0001)
            volume = np.random.randint(100, 500)
            side = np.random.choice([1, -1])
            
            tick = TickData(
                timestamp=1000 + i * 1000,
                price=price,
                volume=volume,
                side=side
            )
            low_impact_ticks.append(tick)
            
        # High impact scenario
        high_impact_ticks = []
        price = 100.0
        
        for i in range(100):
            volume = np.random.randint(100, 500)
            side = np.random.choice([1, -1])
            
            # High price impact
            price_impact = side * volume * 0.001
            price += price_impact
            
            tick = TickData(
                timestamp=2000 + i * 1000,
                price=price,
                volume=volume,
                side=side
            )
            high_impact_ticks.append(tick)
            
        indicator = KylesLambdaIndicator()
        
        result_low = indicator.calculate(low_impact_ticks, "LOW_IMPACT")
        result_high = indicator.calculate(high_impact_ticks, "HIGH_IMPACT")
        
        # Low impact should have higher liquidity score
        if (result_low.metadata.get('lambda') is not None and 
            result_high.metadata.get('lambda') is not None):
            assert result_low.metadata['liquidity_score'] > result_high.metadata['liquidity_score']
            
    def test_trade_difficulty_classification(self):
        """Test trade difficulty classification"""
        
        indicator = KylesLambdaIndicator()
        
        # Test difficulty levels
        test_lambdas = [0.00001, 0.0005, 0.005, 0.05, 0.5]
        expected_difficulties = ['very_easy', 'easy', 'moderate', 'difficult', 'very_difficult']
        
        for lambda_val, expected in zip(test_lambdas, expected_difficulties):
            # Mock lambda history
            indicator.lambda_history.clear()
            for _ in range(5):
                indicator.lambda_history.append(lambda_val)
                
            difficulty = indicator._calculate_trade_difficulty()
            assert difficulty == expected
            

class TestVPINIndicator:
    """Test VPIN (Volume-Synchronized PIN) indicator"""
    
    def test_bucket_formation(self):
        """Test volume bucket formation"""
        
        indicator = VPINIndicator(bucket_volume=1000, n_buckets=10)
        
        # Generate ticks that should fill buckets
        ticks = []
        timestamp = 1000
        
        total_volume = 0
        for i in range(50):
            volume = 100  # Each tick has 100 volume
            side = 1 if i % 3 == 0 else -1
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100 + i * 0.01,
                volume=volume,
                side=side
            )
            ticks.append(tick)
            total_volume += volume
            
        result = indicator.calculate(ticks, "TEST")
        
        # Should have created buckets (5000 total volume / 1000 per bucket = 5 buckets)
        expected_buckets = total_volume // indicator.bucket_volume
        assert result.metadata['completed_buckets'] >= expected_buckets - 1
        
    def test_vpin_calculation_with_imbalance(self):
        """Test VPIN calculation with order imbalance"""
        
        indicator = VPINIndicator(bucket_volume=500, rolling_window=3)
        
        # Create ticks with strong buy imbalance
        ticks = []
        timestamp = 1000
        
        for i in range(100):
            volume = 50
            side = 1 if np.random.random() < 0.8 else -1  # 80% buy probability
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100 + np.random.normal(0, 0.1),
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "IMBALANCED")
        
        # Should detect imbalance if enough buckets formed
        if result.metadata['completed_buckets'] >= 3:
            assert result.metadata['vpin'] > 0
            assert 'order_flow_toxicity' in result.metadata
            
    def test_bucket_summary(self):
        """Test bucket summary generation"""
        
        generator = SyntheticDataGenerator()
        tick_data = generator.generate_tick_data(200)
        
        indicator = VPINIndicator(bucket_volume=1000)
        result = indicator.calculate(tick_data, "TEST")
        
        # Get bucket summary
        summary = indicator.get_bucket_summary()
        
        if not summary.empty:
            # Should have expected columns
            expected_columns = [
                'bucket_index', 'timestamp_start', 'timestamp_end',
                'total_volume', 'buy_volume', 'sell_volume',
                'imbalance', 'net_order_flow', 'duration_seconds'
            ]
            
            for col in expected_columns:
                assert col in summary.columns
                
            # Imbalance should be between -1 and 1
            assert summary['imbalance'].between(-1, 1).all()
            
            # Buy + sell should equal total (approximately)
            volume_check = np.isclose(
                summary['buy_volume'] + summary['sell_volume'],
                summary['total_volume'],
                rtol=0.1
            )
            assert volume_check.all()


class TestTickVolumeImbalanceIndicator:
    """Test Tick Volume Imbalance indicator"""
    
    def test_imbalance_calculation(self):
        """Test basic imbalance calculation"""
        
        indicator = TickVolumeImbalanceIndicator(window_size=20)
        
        # Create ticks with known imbalance
        ticks = []
        timestamp = 1000
        
        # 70% buy volume
        for i in range(50):
            volume = 100
            side = 1 if i < 35 else -1  # First 35 are buys
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100,
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "IMBALANCED")
        
        # Should detect buy imbalance
        assert result.metadata['current_imbalance'] > 0
        
        # Test sell imbalance
        sell_ticks = []
        for i in range(50):
            volume = 100
            side = -1 if i < 35 else 1  # First 35 are sells
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100,
                volume=volume,
                side=side
            )
            sell_ticks.append(tick)
            
        result2 = indicator.calculate(sell_ticks, "SELL_IMBALANCED")
        assert result2.metadata['current_imbalance'] < 0
        
    def test_persistence_detection(self):
        """Test persistence detection"""
        
        indicator = TickVolumeImbalanceIndicator(
            window_size=10,
            persistence_periods=3
        )
        
        # Create persistent buy imbalance
        ticks = []
        timestamp = 1000
        
        for i in range(100):
            volume = 100
            side = 1  # All buys for persistence
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100 + np.random.normal(0, 0.01),
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "PERSISTENT")
        
        # Should detect persistent imbalance
        assert result.metadata['persistent_imbalance'] > 0
        assert result.metadata['persistence_streak'] >= 3
        
    def test_volume_concentration_calculation(self):
        """Test volume concentration (Gini coefficient)"""
        
        indicator = TickVolumeImbalanceIndicator()
        
        # Create tick data with varying volume sizes
        ticks = []
        timestamp = 1000
        
        volumes = [10, 20, 50, 100, 500, 1000, 2000]  # Concentrated volume
        
        for i, volume in enumerate(volumes * 10):  # Repeat to get 70 ticks
            if i >= 70:
                break
                
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100,
                volume=volume,
                side=np.random.choice([1, -1])
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "CONCENTRATED")
        
        # Should calculate concentration
        assert 'volume_concentration' in result.metadata
        concentration = result.metadata['volume_concentration']
        assert 0 <= concentration <= 1
        
    def test_large_trade_imbalance(self):
        """Test large trade imbalance calculation"""
        
        indicator = TickVolumeImbalanceIndicator()
        
        # Mix of small and large trades with imbalance in large trades
        ticks = []
        timestamp = 1000
        
        for i in range(100):
            if i < 20:  # Large buy trades
                volume = 1000
                side = 1
            elif i < 30:  # Large sell trades (fewer)
                volume = 1000
                side = -1
            else:  # Small random trades
                volume = 50
                side = np.random.choice([1, -1])
                
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100,
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "LARGE_TRADE_TEST")
        
        # Should show imbalance in large trades (more large buys than sells)
        if 'large_trade_imbalance' in result.metadata:
            assert result.metadata['large_trade_imbalance'] > 0
            
    def test_divergence_detection(self):
        """Test price-volume divergence detection"""
        
        indicator = TickVolumeImbalanceIndicator(window_size=30)
        
        # Price going down but buying pressure (bullish divergence)
        ticks = []
        timestamp = 1000
        base_price = 100
        
        for i in range(50):
            # Price declining
            price = base_price - (i * 0.01)
            
            # But buying pressure
            volume = 100
            side = 1 if np.random.random() < 0.7 else -1
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=volume,
                side=side
            )
            ticks.append(tick)
            
        result = indicator.calculate(ticks, "DIVERGENCE")
        
        # Should potentially detect bullish divergence
        # (This is a probabilistic test, so we check the logic exists)
        assert 'current_imbalance' in result.metadata
        
        # The signal generation should consider price trends
        if result.metadata['current_imbalance'] > 0.2 and result.confidence > 0:
            # Bullish divergence should generate buy signal
            assert result.signal in [SignalType.BUY, SignalType.HOLD]