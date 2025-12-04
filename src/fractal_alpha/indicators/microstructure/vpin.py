"""
🎯 VPIN (Volume-Synchronized Probability of Informed Trading)
Advanced market microstructure indicator for detecting informed trading
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
from dataclasses import dataclass

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


@dataclass
class VolumeBucket:
    """Volume bucket for VPIN calculation"""
    timestamp_start: int
    timestamp_end: int
    total_volume: int
    buy_volume: int
    sell_volume: int
    imbalance: float  # |buy_volume - sell_volume|
    bucket_index: int


class VPINIndicator(BaseIndicator):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN)
    
    VPIN measures the probability that the next trade will be informed
    by synchronizing on volume rather than time. This removes the noise
    from varying trade frequencies and focuses on actual information flow.
    
    Key concepts:
    - Volume buckets of fixed size
    - Order imbalance within each bucket  
    - Rolling average of imbalances
    - Higher VPIN = Higher probability of informed trading
    """
    
    def __init__(self,
                 bucket_volume: int = 10000,
                 n_buckets: int = 50,
                 rolling_window: int = 20,
                 imbalance_threshold: float = 0.3):
        """
        Initialize VPIN indicator
        
        Args:
            bucket_volume: Volume per bucket (default: 10,000 shares)
            n_buckets: Number of buckets to maintain
            rolling_window: Window for VPIN calculation
            imbalance_threshold: Threshold for high VPIN signals
        """
        super().__init__(
            name="VPIN",
            timeframe=TimeFrame.TICK,
            lookback_periods=n_buckets,
            params={
                'bucket_volume': bucket_volume,
                'n_buckets': n_buckets,
                'rolling_window': rolling_window,
                'imbalance_threshold': imbalance_threshold
            }
        )
        
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        self.rolling_window = rolling_window
        self.imbalance_threshold = imbalance_threshold
        
        # Volume bucket storage
        self.buckets: Deque[VolumeBucket] = deque(maxlen=n_buckets)
        
        # Current bucket being filled
        self.current_bucket = {
            'timestamp_start': None,
            'total_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'start_time': None
        }
        
        self.bucket_counter = 0
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate VPIN from tick data
        
        Args:
            data: Tick data or OHLCV bars
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with VPIN analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Generate synthetic ticks from bars
            from ...utils.synthetic_ticks import SyntheticTickGenerator
            generator = SyntheticTickGenerator(method='volume_weighted')
            
            ticks = []
            for _, bar in data.iterrows():
                bar_ticks = generator.generate_ticks(bar, n_ticks=20)
                ticks.extend(bar_ticks)
        else:
            ticks = data
            
        if not ticks:
            return self._empty_result(symbol)
            
        # Process all ticks to build volume buckets
        for tick in ticks:
            self._process_tick(tick)
            
        if len(self.buckets) < self.rolling_window:
            return self._empty_result(symbol)
            
        # Calculate VPIN
        vpin = self._calculate_vpin()
        
        # Calculate additional metrics
        order_flow_toxicity = self._calculate_order_flow_toxicity()
        volume_intensity = self._calculate_volume_intensity()
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            vpin, order_flow_toxicity, volume_intensity
        )
        
        # Create metadata
        metadata = self._create_metadata(vpin, order_flow_toxicity, volume_intensity)
        
        # Get latest timestamp
        timestamp = ticks[-1].timestamp if ticks else int(datetime.now().timestamp() * 1000)
        
        return IndicatorResult(
            timestamp=timestamp,
            symbol=symbol,
            indicator_name=self.name,
            value=value,
            signal=signal,
            confidence=confidence,
            timeframe=self.timeframe,
            metadata=metadata,
            calculation_time_ms=0
        )
    
    def _process_tick(self, tick: TickData):
        """Process a single tick and update volume buckets"""
        
        # Initialize first bucket
        if self.current_bucket['timestamp_start'] is None:
            self.current_bucket['timestamp_start'] = tick.timestamp
            
        # Add to current bucket
        self.current_bucket['total_volume'] += tick.volume
        
        if tick.side == 1:  # Buy
            self.current_bucket['buy_volume'] += tick.volume
        elif tick.side == -1:  # Sell
            self.current_bucket['sell_volume'] += tick.volume
            
        # Check if bucket is full
        if self.current_bucket['total_volume'] >= self.bucket_volume:
            self._complete_bucket(tick.timestamp)
            self._start_new_bucket(tick.timestamp)
    
    def _complete_bucket(self, end_timestamp: int):
        """Complete the current bucket and add to collection"""
        
        buy_vol = self.current_bucket['buy_volume']
        sell_vol = self.current_bucket['sell_volume']
        total_vol = self.current_bucket['total_volume']
        
        # Calculate imbalance
        imbalance = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        
        # Create completed bucket
        bucket = VolumeBucket(
            timestamp_start=self.current_bucket['timestamp_start'],
            timestamp_end=end_timestamp,
            total_volume=total_vol,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            imbalance=imbalance,
            bucket_index=self.bucket_counter
        )
        
        self.buckets.append(bucket)
        self.bucket_counter += 1
    
    def _start_new_bucket(self, timestamp: int):
        """Start a new volume bucket"""
        
        self.current_bucket = {
            'timestamp_start': timestamp,
            'total_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0
        }
    
    def _calculate_vpin(self) -> float:
        """Calculate the VPIN metric"""
        
        if len(self.buckets) < self.rolling_window:
            return 0.0
            
        # Get recent buckets
        recent_buckets = list(self.buckets)[-self.rolling_window:]
        
        # Calculate average absolute order imbalance
        total_imbalance = sum(bucket.imbalance for bucket in recent_buckets)
        total_volume = sum(bucket.total_volume for bucket in recent_buckets)
        
        if total_volume == 0:
            return 0.0
            
        # VPIN is the average order imbalance
        vpin = total_imbalance / len(recent_buckets)
        
        return vpin
    
    def _calculate_order_flow_toxicity(self) -> float:
        """Calculate order flow toxicity (alternative VPIN formulation)"""
        
        if len(self.buckets) < self.rolling_window:
            return 0.0
            
        recent_buckets = list(self.buckets)[-self.rolling_window:]
        
        # Calculate signed order imbalances
        signed_imbalances = []
        for bucket in recent_buckets:
            if bucket.total_volume > 0:
                signed_oi = (bucket.buy_volume - bucket.sell_volume) / bucket.total_volume
                signed_imbalances.append(signed_oi)
        
        if not signed_imbalances:
            return 0.0
            
        # Toxicity is the absolute value of average signed imbalance
        avg_signed_imbalance = np.mean(signed_imbalances)
        return abs(avg_signed_imbalance)
    
    def _calculate_volume_intensity(self) -> float:
        """Calculate volume intensity (buckets completed per unit time)"""
        
        if len(self.buckets) < 5:
            return 0.0
            
        recent_buckets = list(self.buckets)[-5:]
        
        # Calculate time span
        time_span = (recent_buckets[-1].timestamp_end - 
                    recent_buckets[0].timestamp_start) / 1000  # Convert to seconds
        
        if time_span <= 0:
            return 0.0
            
        # Buckets per minute
        buckets_per_minute = (len(recent_buckets) - 1) * 60 / time_span
        
        return buckets_per_minute
    
    def _generate_signal(self,
                        vpin: float,
                        order_flow_toxicity: float,
                        volume_intensity: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from VPIN analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # High VPIN suggests informed trading - be cautious
        if vpin > 0.4:
            # Very high informed trading probability
            if order_flow_toxicity > 0.3:
                # High toxicity + high VPIN = avoid trading
                signal = SignalType.HOLD
                confidence = 0  # No confidence in either direction
            else:
                # High VPIN but low toxicity might indicate one-sided informed trading
                # Check recent bucket direction
                if len(self.buckets) >= 3:
                    recent_net = sum((b.buy_volume - b.sell_volume) 
                                   for b in list(self.buckets)[-3:])
                    if recent_net > 0:
                        signal = SignalType.BUY
                        confidence = min(vpin * 100, 70)
                    elif recent_net < 0:
                        signal = SignalType.SELL
                        confidence = min(vpin * 100, 70)
                        
        elif vpin < 0.15:
            # Low VPIN suggests uninformed trading - good for mean reversion
            signal = SignalType.HOLD  # Wait for other signals
            confidence = 20
            
        # Adjust confidence based on volume intensity
        if volume_intensity > 10:  # High trading activity
            confidence *= 1.2
        elif volume_intensity < 2:  # Low activity
            confidence *= 0.8
            
        confidence = min(confidence, 90)
        
        # Value represents VPIN level (0-100)
        value = vpin * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        vpin: float,
                        order_flow_toxicity: float,
                        volume_intensity: float) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'vpin': vpin,
            'order_flow_toxicity': order_flow_toxicity,
            'volume_intensity': volume_intensity,
            'completed_buckets': len(self.buckets),
            'current_bucket_fill': (self.current_bucket['total_volume'] / 
                                  self.bucket_volume) if self.bucket_volume > 0 else 0
        }
        
        # Add bucket statistics
        if self.buckets:
            recent_buckets = list(self.buckets)[-10:]  # Last 10 buckets
            
            imbalances = [b.imbalance for b in recent_buckets]
            durations = []
            
            for i in range(1, len(recent_buckets)):
                duration = (recent_buckets[i].timestamp_end - 
                          recent_buckets[i-1].timestamp_end) / 1000
                durations.append(duration)
            
            metadata.update({
                'avg_bucket_imbalance': np.mean(imbalances),
                'max_bucket_imbalance': np.max(imbalances),
                'avg_bucket_duration': np.mean(durations) if durations else 0,
                'bucket_consistency': 1 - (np.std(imbalances) / np.mean(imbalances)) 
                                    if np.mean(imbalances) > 0 else 0
            })
            
            # Calculate informed trading indicators
            high_imbalance_count = sum(1 for i in imbalances if i > 0.3)
            metadata['high_imbalance_ratio'] = high_imbalance_count / len(imbalances)
            
            # Recent trend
            if len(recent_buckets) >= 5:
                early_avg = np.mean([b.imbalance for b in recent_buckets[:5]])
                late_avg = np.mean([b.imbalance for b in recent_buckets[-5:]])
                metadata['imbalance_trend'] = 'increasing' if late_avg > early_avg else 'decreasing'
            
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=0.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient buckets for VPIN calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= self.bucket_volume // 100  # Reasonable minimum
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 10
        return False
    
    def get_bucket_summary(self) -> pd.DataFrame:
        """Get summary of completed volume buckets"""
        
        if not self.buckets:
            return pd.DataFrame()
            
        data = []
        for bucket in self.buckets:
            data.append({
                'bucket_index': bucket.bucket_index,
                'timestamp_start': bucket.timestamp_start,
                'timestamp_end': bucket.timestamp_end,
                'total_volume': bucket.total_volume,
                'buy_volume': bucket.buy_volume,
                'sell_volume': bucket.sell_volume,
                'imbalance': bucket.imbalance,
                'net_order_flow': bucket.buy_volume - bucket.sell_volume,
                'duration_seconds': (bucket.timestamp_end - bucket.timestamp_start) / 1000
            })
            
        df = pd.DataFrame(data)
        df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='ms')
        df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], unit='ms')
        
        return df


def demonstrate_vpin():
    """Demonstration of VPIN indicator"""
    
    print("🎯 VPIN (Volume-Synchronized PIN) Demonstration\n")
    
    # Generate synthetic tick data with informed trading pattern
    np.random.seed(42)
    ticks = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    print("Generating synthetic tick data with informed trading pattern...")
    
    # Phase 1: Normal trading (50 ticks)
    for i in range(50):
        price = 100 + np.random.normal(0, 0.1)
        volume = np.random.randint(50, 200)
        side = np.random.choice([1, -1])  # Random buy/sell
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Phase 2: Informed buying pattern (100 ticks)
    print("Simulating informed buying pattern...")
    for i in range(50, 150):
        price = 100.1 + (i - 50) * 0.01 + np.random.normal(0, 0.05)
        
        # Informed traders use larger sizes and buy more frequently
        if np.random.random() < 0.7:  # 70% buy probability
            volume = np.random.randint(200, 500)  # Larger sizes
            side = 1
        else:
            volume = np.random.randint(30, 100)   # Smaller sells
            side = -1
            
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Phase 3: Return to normal (50 ticks)
    print("Returning to normal trading...")
    for i in range(150, 200):
        price = 101.5 + np.random.normal(0, 0.1)
        volume = np.random.randint(50, 200)
        side = np.random.choice([1, -1])
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Create VPIN indicator
    vpin_indicator = VPINIndicator(
        bucket_volume=2000,  # Smaller buckets for demo
        n_buckets=30,
        rolling_window=10,
        imbalance_threshold=0.3
    )
    
    # Process ticks and analyze evolution
    print("\nProcessing ticks and monitoring VPIN evolution...\n")
    
    results = []
    for i in range(50, len(ticks), 20):  # Every 20 ticks after initial 50
        batch = ticks[:i]
        result = vpin_indicator.calculate(batch, "TEST")
        results.append(result)
        
        if result.value > 0:  # Only show when we have VPIN calculation
            phase = "Normal" if i < 100 else "Informed" if i < 180 else "Post-Informed"
            
            print(f"After {i} ticks ({phase} trading):")
            print(f"  VPIN: {result.metadata['vpin']:.3f}")
            print(f"  Order Flow Toxicity: {result.metadata['order_flow_toxicity']:.3f}")
            print(f"  Volume Intensity: {result.metadata['volume_intensity']:.1f} buckets/min")
            print(f"  Signal: {result.signal.value}")
            print(f"  Confidence: {result.confidence:.1f}%")
            print(f"  Completed Buckets: {result.metadata['completed_buckets']}")
            print()
    
    # Final comprehensive analysis
    if results:
        final_result = results[-1]
        print("\n" + "="*60)
        print("FINAL VPIN ANALYSIS:")
        print("="*60)
        
        print(f"Final VPIN: {final_result.metadata['vpin']:.3f}")
        print(f"Order Flow Toxicity: {final_result.metadata['order_flow_toxicity']:.3f}")
        print(f"High Imbalance Ratio: {final_result.metadata.get('high_imbalance_ratio', 0):.2%}")
        print(f"Imbalance Trend: {final_result.metadata.get('imbalance_trend', 'stable')}")
        
        print(f"\nTrading Signal: {final_result.signal.value}")
        print(f"Confidence: {final_result.confidence:.1f}%")
        
        # Get bucket summary
        bucket_summary = vpin_indicator.get_bucket_summary()
        if not bucket_summary.empty:
            print(f"\nBucket Statistics:")
            print(f"  Total Buckets: {len(bucket_summary)}")
            print(f"  Avg Imbalance: {bucket_summary['imbalance'].mean():.3f}")
            print(f"  Max Imbalance: {bucket_summary['imbalance'].max():.3f}")
            print(f"  Avg Duration: {bucket_summary['duration_seconds'].mean():.1f}s")
    
    print("\n💡 VPIN Interpretation:")
    print("- VPIN > 0.4: High probability of informed trading (be cautious)")
    print("- VPIN < 0.2: Low informed trading (good for mean reversion)")
    print("- Rising VPIN: Increasing information asymmetry")
    print("- Volume intensity shows market activity level")
    print("- Order flow toxicity indicates trading difficulty")


if __name__ == "__main__":
    demonstrate_vpin()