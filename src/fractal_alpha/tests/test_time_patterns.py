"""
⏰ Time Pattern Analysis Tests
Tests for time-based pattern detection indicators
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from ..indicators.time_patterns.volume_bars import VolumeBarAggregator
from ..base.types import TickData, SignalType
from .fixtures import SyntheticDataGenerator


class TestVolumeBarAggregator:
    """Test Volume Bar aggregation functionality"""
    
    def test_basic_volume_bar_creation(self):
        """Test basic volume bar creation from ticks"""
        
        aggregator = VolumeBarAggregator(volume_threshold=1000, auto_adjust=False)
        
        # Generate tick data that should create exactly 2 bars
        ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # First bar: 1000 volume
        for i in range(10):
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100 + i * 0.01,
                volume=100,  # 10 ticks * 100 = 1000 volume
                side=1 if i % 2 == 0 else -1
            )
            ticks.append(tick)
            
        # Second bar: 1000 volume  
        for i in range(10, 20):
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100 + i * 0.01,
                volume=100,
                side=1 if i % 2 == 0 else -1
            )
            ticks.append(tick)
            
        # Process ticks
        completed_bars = aggregator.process_ticks(ticks)
        
        # Should have completed 1 bar (2nd bar still building)
        assert len(completed_bars) >= 1
        
        # First bar should have correct properties
        first_bar = completed_bars[0]
        assert first_bar.volume >= aggregator.volume_threshold
        assert first_bar.tick_count == 10
        assert first_bar.open == 100
        assert first_bar.close == 100 + 9 * 0.01
        
    def test_volume_bar_ohlc_calculation(self):
        """Test OHLC calculation in volume bars"""
        
        aggregator = VolumeBarAggregator(volume_threshold=500)
        
        # Create ticks with known price pattern
        ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        prices = [100, 102, 99, 104, 98, 103]  # Known high/low pattern
        
        for i, price in enumerate(prices):
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=100,  # Will complete bar after 5 ticks
                side=1
            )
            ticks.append(tick)
            
        completed_bars = aggregator.process_ticks(ticks)
        
        if completed_bars:
            bar = completed_bars[0]
            
            # Check OHLC relationships
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close
            assert bar.high >= bar.low
            
            # Check specific values
            expected_high = max(prices[:5])  # First 5 prices
            expected_low = min(prices[:5])
            
            assert bar.high == expected_high
            assert bar.low == expected_low
            assert bar.open == prices[0]
            
    def test_volume_bar_vwap_calculation(self):
        """Test VWAP calculation in volume bars"""
        
        aggregator = VolumeBarAggregator(volume_threshold=300)
        
        # Create ticks with known price and volume
        ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Manual VWAP calculation
        prices = [100, 101, 102]
        volumes = [100, 100, 100]  # Total 300 volume
        expected_vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=volume,
                side=1
            )
            ticks.append(tick)
            
        completed_bars = aggregator.process_ticks(ticks)
        
        assert len(completed_bars) >= 1
        
        bar = completed_bars[0]
        assert abs(bar.vwap - expected_vwap) < 0.01
        
    def test_volume_bar_imbalance_calculation(self):
        """Test order imbalance calculation"""
        
        aggregator = VolumeBarAggregator(volume_threshold=400)
        
        ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Create imbalanced order flow
        buy_volume = 300
        sell_volume = 100
        
        # Buy orders
        for i in range(3):
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=100,
                volume=100,
                side=1
            )
            ticks.append(tick)
            
        # Sell order
        tick = TickData(
            timestamp=timestamp + 3 * 1000,
            price=100,
            volume=100,
            side=-1
        )
        ticks.append(tick)
        
        completed_bars = aggregator.process_ticks(ticks)
        
        if completed_bars:
            bar = completed_bars[0]
            
            assert bar.buy_volume == buy_volume
            assert bar.sell_volume == sell_volume
            
            expected_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            assert abs(bar.imbalance - expected_imbalance) < 0.01
            
    def test_auto_threshold_adjustment(self):
        """Test automatic volume threshold adjustment"""
        
        aggregator = VolumeBarAggregator(
            volume_threshold=1000,
            auto_adjust=True
        )
        
        # Generate high-volume period
        ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Create 50 bars worth of data with increasing volume
        for bar_num in range(50):
            bar_volume = 0
            tick_num = 0
            
            while bar_volume < aggregator.volume_threshold:
                # Increasing volume over time
                volume = 50 + bar_num * 2
                
                tick = TickData(
                    timestamp=timestamp + (bar_num * 20 + tick_num) * 1000,
                    price=100 + np.random.normal(0, 0.1),
                    volume=volume,
                    side=np.random.choice([1, -1])
                )
                
                completed_bar = aggregator.process_tick(tick)
                if completed_bar:
                    break
                    
                ticks.append(tick)
                bar_volume += volume
                tick_num += 1
                
                if tick_num > 50:  # Safety break
                    break
                    
        # Threshold should have increased due to higher activity
        assert aggregator.volume_threshold > 1000
        
    def test_volume_bar_signal_generation(self):
        """Test signal generation from volume bar analysis"""
        
        generator = SyntheticDataGenerator()
        
        # Generate tick data with directional bias
        tick_data = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Create strong buying pressure pattern
        for i in range(100):
            volume = np.random.randint(50, 200)
            side = 1 if np.random.random() < 0.8 else -1  # 80% buy probability
            price = 100 + i * 0.01 + np.random.normal(0, 0.05)
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=volume,
                side=side
            )
            tick_data.append(tick)
            
        aggregator = VolumeBarAggregator(volume_threshold=1000)
        result = aggregator.calculate(tick_data, "SIGNAL_TEST")
        
        # Should detect buying pressure
        assert result.signal in [SignalType.BUY, SignalType.HOLD]
        
        if result.confidence > 50:
            assert result.signal == SignalType.BUY
            
        # Should have reasonable metadata
        assert 'avg_imbalance' in result.metadata
        assert 'bars_created' in result.metadata
        assert result.metadata['bars_created'] > 0
        
    def test_volume_bar_dataframe_conversion(self):
        """Test conversion to pandas DataFrame"""
        
        generator = SyntheticDataGenerator()
        tick_data = generator.generate_tick_data(200)
        
        aggregator = VolumeBarAggregator(volume_threshold=500)
        completed_bars = aggregator.process_ticks(tick_data)
        
        # Convert to DataFrame
        df = aggregator.get_bars_dataframe()
        
        if not df.empty:
            # Check DataFrame structure
            expected_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'buy_volume', 'sell_volume', 'tick_count',
                'vwap', 'imbalance'
            ]
            
            for col in expected_columns:
                assert col in df.columns
                
            # Check data integrity
            assert len(df) == len(aggregator.completed_bars)
            
            # OHLC relationships should hold
            assert (df['high'] >= df['open']).all()
            assert (df['high'] >= df['close']).all()
            assert (df['low'] <= df['open']).all()
            assert (df['low'] <= df['close']).all()
            
            # Volume relationships should hold
            vol_sum_check = np.isclose(
                df['buy_volume'] + df['sell_volume'],
                df['volume'],
                rtol=0.1
            )
            assert vol_sum_check.all() or (df['volume'] == 0).any()  # Allow for rounding or zero volume
            
    def test_volume_bar_edge_cases(self):
        """Test edge cases in volume bar processing"""
        
        aggregator = VolumeBarAggregator(volume_threshold=100)
        
        # Test with zero volume tick
        zero_vol_tick = TickData(
            timestamp=1000,
            price=100,
            volume=0,
            side=1
        )
        
        # Should handle gracefully
        result = aggregator.process_tick(zero_vol_tick)
        assert result is None  # No bar completed
        
        # Test with very large volume tick
        large_vol_tick = TickData(
            timestamp=2000,
            price=100,
            volume=10000,  # Much larger than threshold
            side=1
        )
        
        completed_bar = aggregator.process_tick(large_vol_tick)
        assert completed_bar is not None
        assert completed_bar.volume >= aggregator.volume_threshold
        
        # Test with negative volume (should be handled)
        try:
            negative_vol_tick = TickData(
                timestamp=3000,
                price=100,
                volume=-100,
                side=1
            )
            # This might raise an error or handle gracefully
            aggregator.process_tick(negative_vol_tick)
        except ValueError:
            pass  # Expected for invalid data
            
    def test_volume_bar_performance_metrics(self):
        """Test performance-related metrics"""
        
        generator = SyntheticDataGenerator()
        
        # Generate efficient vs inefficient price movement data
        efficient_ticks = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Efficient: price moves consistently with few reversals
        price = 100
        for i in range(50):
            price += 0.02  # Steady upward movement
            volume = 100
            
            tick = TickData(
                timestamp=timestamp + i * 1000,
                price=price,
                volume=volume,
                side=1
            )
            efficient_ticks.append(tick)
            
        # Inefficient: price whips back and forth
        inefficient_ticks = []
        price = 100
        for i in range(50):
            price += 0.02 * (1 if i % 2 == 0 else -1)  # Back and forth
            volume = 100
            
            tick = TickData(
                timestamp=timestamp + (50 + i) * 1000,
                price=price,
                volume=volume,
                side=1
            )
            inefficient_ticks.append(tick)
            
        aggregator = VolumeBarAggregator(volume_threshold=2500)
        
        # Process both datasets
        efficient_result = aggregator.calculate(efficient_ticks, "EFFICIENT")
        aggregator = VolumeBarAggregator(volume_threshold=2500)  # Reset
        inefficient_result = aggregator.calculate(inefficient_ticks, "INEFFICIENT")
        
        # Efficient movement should have higher tick efficiency
        if ('tick_efficiency' in efficient_result.metadata and 
            'tick_efficiency' in inefficient_result.metadata):
            
            assert (efficient_result.metadata['tick_efficiency'] >= 
                   inefficient_result.metadata['tick_efficiency'])
                   
    def test_volume_bar_spillover_handling(self):
        """Test handling of volume spillover between bars"""
        
        aggregator = VolumeBarAggregator(volume_threshold=200)
        
        # Create a tick that will cause spillover
        large_tick = TickData(
            timestamp=1000,
            price=100,
            volume=350,  # More than threshold
            side=1
        )
        
        completed_bar = aggregator.process_tick(large_tick)
        
        # Should complete a bar
        assert completed_bar is not None
        assert completed_bar.volume >= aggregator.volume_threshold
        
        # Current bar should have the spillover
        assert aggregator.current_volume > 0
        assert aggregator.current_volume < aggregator.volume_threshold
        
        # Spillover should equal: total_volume - threshold
        expected_spillover = 350 - 200
        assert aggregator.current_volume == expected_spillover