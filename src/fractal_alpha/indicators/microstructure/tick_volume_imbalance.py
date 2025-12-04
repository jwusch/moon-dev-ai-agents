"""
🔍 Tick Volume Imbalance Indicator
Measures buy/sell pressure at tick level to detect accumulation/distribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
from datetime import datetime, timedelta

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class TickVolumeImbalanceIndicator(BaseIndicator):
    """
    Analyzes tick-level volume imbalance to detect hidden accumulation/distribution
    
    Key concepts:
    - Buy pressure: Volume on upticks
    - Sell pressure: Volume on downticks
    - Imbalance: (Buy Volume - Sell Volume) / Total Volume
    - Persistence: How long imbalance maintains direction
    """
    
    def __init__(self,
                 window_size: int = 100,
                 imbalance_threshold: float = 0.3,
                 persistence_periods: int = 5,
                 use_volume_weighted: bool = True):
        """
        Initialize Tick Volume Imbalance indicator
        
        Args:
            window_size: Number of ticks to analyze
            imbalance_threshold: Threshold for significant imbalance (0.3 = 30%)
            persistence_periods: Consecutive periods needed for signal
            use_volume_weighted: Weight by volume size
        """
        super().__init__(
            name="TickVolumeImbalance",
            timeframe=TimeFrame.TICK,
            lookback_periods=window_size,
            params={
                'window_size': window_size,
                'imbalance_threshold': imbalance_threshold,
                'persistence_periods': persistence_periods,
                'use_volume_weighted': use_volume_weighted
            }
        )
        
        self.window_size = window_size
        self.imbalance_threshold = imbalance_threshold
        self.persistence_periods = persistence_periods
        self.use_volume_weighted = use_volume_weighted
        
        # Rolling windows
        self.tick_window = deque(maxlen=window_size)
        self.imbalance_history = deque(maxlen=persistence_periods)
        
        # Tracking
        self.total_buy_volume = 0
        self.total_sell_volume = 0
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate tick volume imbalance
        
        Args:
            data: Tick data or OHLCV bars
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with imbalance analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Generate synthetic ticks from bars
            from ...utils.synthetic_ticks import SyntheticTickGenerator
            generator = SyntheticTickGenerator(method='adaptive')
            
            ticks = []
            for _, bar in data.iterrows():
                bar_ticks = generator.generate_ticks(bar, n_ticks=10)
                ticks.extend(bar_ticks)
        else:
            ticks = data
            
        if not ticks:
            return self._empty_result(symbol)
            
        # Process all ticks
        for tick in ticks:
            self._process_tick(tick)
            
        # Calculate current imbalance
        current_imbalance = self._calculate_current_imbalance()
        
        # Check persistence
        persistent_imbalance = self._check_persistence()
        
        # Analyze for signals
        signal, confidence, value = self._analyze_imbalance(
            current_imbalance, 
            persistent_imbalance
        )
        
        # Create metadata
        metadata = self._create_metadata(current_imbalance, persistent_imbalance)
        
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
        """Process a single tick"""
        
        self.tick_window.append(tick)
        
        # Update volume counters
        if tick.side == 1:
            self.total_buy_volume += tick.volume
        elif tick.side == -1:
            self.total_sell_volume += tick.volume
            
        # Remove old tick if window is full
        if len(self.tick_window) == self.window_size:
            old_tick = self.tick_window[0]
            if old_tick.side == 1:
                self.total_buy_volume -= old_tick.volume
            elif old_tick.side == -1:
                self.total_sell_volume -= old_tick.volume
    
    def _calculate_current_imbalance(self) -> float:
        """Calculate current volume imbalance"""
        
        if not self.tick_window:
            return 0.0
            
        if self.use_volume_weighted:
            # Volume-weighted imbalance
            buy_volume = sum(t.volume for t in self.tick_window if t.side == 1)
            sell_volume = sum(t.volume for t in self.tick_window if t.side == -1)
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
                
            imbalance = (buy_volume - sell_volume) / total_volume
            
        else:
            # Count-based imbalance
            buy_count = sum(1 for t in self.tick_window if t.side == 1)
            sell_count = sum(1 for t in self.tick_window if t.side == -1)
            total_count = len(self.tick_window)
            
            if total_count == 0:
                return 0.0
                
            imbalance = (buy_count - sell_count) / total_count
            
        # Add to history
        self.imbalance_history.append(imbalance)
        
        return imbalance
    
    def _check_persistence(self) -> float:
        """Check if imbalance persists over multiple periods"""
        
        if len(self.imbalance_history) < self.persistence_periods:
            return 0.0
            
        # All imbalances in same direction?
        all_positive = all(i > 0 for i in self.imbalance_history)
        all_negative = all(i < 0 for i in self.imbalance_history)
        
        if all_positive or all_negative:
            # Return average persistent imbalance
            return np.mean(list(self.imbalance_history))
        else:
            return 0.0
    
    def _analyze_imbalance(self, 
                          current_imbalance: float,
                          persistent_imbalance: float) -> Tuple[SignalType, float, float]:
        """Analyze imbalance patterns for trading signals"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Strong persistent buying
        if persistent_imbalance > self.imbalance_threshold:
            signal = SignalType.BUY
            confidence = min(persistent_imbalance * 150, 90)
            
        # Strong persistent selling
        elif persistent_imbalance < -self.imbalance_threshold:
            signal = SignalType.SELL
            confidence = min(abs(persistent_imbalance) * 150, 90)
            
        # Extreme current imbalance (shorter term)
        elif abs(current_imbalance) > self.imbalance_threshold * 1.5:
            if current_imbalance > 0:
                signal = SignalType.BUY
                confidence = min(current_imbalance * 100, 70)
            else:
                signal = SignalType.SELL
                confidence = min(abs(current_imbalance) * 100, 70)
        
        # Additional analysis: Volume divergence
        if self.tick_window and len(self.tick_window) >= 20:
            # Check if price and volume imbalance diverge
            prices = [t.price for t in self.tick_window]
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # Bullish divergence: price down but buying pressure
            if price_trend < 0 and current_imbalance > 0.2:
                signal = SignalType.BUY
                confidence = max(confidence, 65)
                
            # Bearish divergence: price up but selling pressure
            elif price_trend > 0 and current_imbalance < -0.2:
                signal = SignalType.SELL
                confidence = max(confidence, 65)
        
        # Value is the persistent imbalance strength
        value = persistent_imbalance * 100 if persistent_imbalance != 0 else current_imbalance * 100
        
        return signal, confidence, value
    
    def _create_metadata(self, current_imbalance: float, persistent_imbalance: float) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'current_imbalance': current_imbalance,
            'persistent_imbalance': persistent_imbalance,
            'buy_volume': self.total_buy_volume,
            'sell_volume': self.total_sell_volume,
            'tick_count': len(self.tick_window),
            'persistence_streak': self._calculate_persistence_streak(),
            'volume_concentration': self._calculate_volume_concentration(),
            'large_trade_imbalance': self._calculate_large_trade_imbalance()
        }
        
        # Add tick statistics
        if self.tick_window:
            volumes = [t.volume for t in self.tick_window]
            metadata['avg_tick_size'] = np.mean(volumes)
            metadata['max_tick_size'] = np.max(volumes)
            metadata['volume_std'] = np.std(volumes)
            
        return metadata
    
    def _calculate_persistence_streak(self) -> int:
        """Calculate how many periods imbalance has persisted"""
        
        if not self.imbalance_history:
            return 0
            
        streak = 1
        last_sign = np.sign(self.imbalance_history[-1])
        
        for i in range(len(self.imbalance_history) - 2, -1, -1):
            if np.sign(self.imbalance_history[i]) == last_sign:
                streak += 1
            else:
                break
                
        return streak
    
    def _calculate_volume_concentration(self) -> float:
        """Calculate how concentrated volume is (Gini coefficient)"""
        
        if not self.tick_window:
            return 0.0
            
        volumes = sorted([t.volume for t in self.tick_window])
        n = len(volumes)
        
        if n == 0 or sum(volumes) == 0:
            return 0.0
            
        # Gini coefficient calculation
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * volumes)) / (n * np.sum(volumes)) - (n + 1) / n
    
    def _calculate_large_trade_imbalance(self) -> float:
        """Calculate imbalance for large trades only"""
        
        if not self.tick_window:
            return 0.0
            
        volumes = [t.volume for t in self.tick_window]
        threshold = np.percentile(volumes, 80)  # Top 20% of trades
        
        large_buy_volume = sum(t.volume for t in self.tick_window 
                              if t.side == 1 and t.volume >= threshold)
        large_sell_volume = sum(t.volume for t in self.tick_window 
                               if t.side == -1 and t.volume >= threshold)
        
        total_large_volume = large_buy_volume + large_sell_volume
        
        if total_large_volume == 0:
            return 0.0
            
        return (large_buy_volume - large_sell_volume) / total_large_volume
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when no data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=0.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'No data available'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) > 0
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) > 0
        return False


def demonstrate_tick_volume_imbalance():
    """Demonstration of Tick Volume Imbalance indicator"""
    
    print("🔍 Tick Volume Imbalance Demonstration\n")
    
    # Create sample tick data with clear accumulation pattern
    ticks = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Phase 1: Normal trading (balanced)
    print("Phase 1: Normal balanced trading...")
    for i in range(50):
        price = 100 + np.random.normal(0, 0.1)
        volume = np.random.randint(10, 100)
        side = np.random.choice([1, -1])
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Phase 2: Hidden accumulation (more buy volume)
    print("Phase 2: Hidden accumulation pattern...")
    for i in range(100):
        price = 100 - 0.001 * i + np.random.normal(0, 0.1)  # Slight downtrend
        
        # But buyers are accumulating
        if np.random.random() < 0.7:  # 70% buy probability
            volume = np.random.randint(50, 200)  # Larger buys
            side = 1
        else:
            volume = np.random.randint(10, 50)   # Smaller sells
            side = -1
            
        ticks.append(TickData(
            timestamp=timestamp + (50 + i) * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Create indicator
    indicator = TickVolumeImbalanceIndicator(
        window_size=50,
        imbalance_threshold=0.3,
        persistence_periods=5
    )
    
    # Process ticks incrementally to show evolution
    print("\nProcessing ticks and monitoring imbalance...\n")
    
    results = []
    for i in range(0, len(ticks), 10):
        batch = ticks[:i+10]
        result = indicator.calculate(batch, "TEST")
        results.append(result)
        
        if i > 50 and i % 30 == 0:
            print(f"After {i+10} ticks:")
            print(f"  Imbalance: {result.metadata['current_imbalance']:+.3f}")
            print(f"  Signal: {result.signal.value}")
            print(f"  Confidence: {result.confidence:.1f}%")
            print(f"  Buy Volume: {result.metadata['buy_volume']:,}")
            print(f"  Sell Volume: {result.metadata['sell_volume']:,}")
            print()
    
    # Final analysis
    final_result = results[-1]
    print("\n" + "="*50)
    print("FINAL ANALYSIS:")
    print("="*50)
    print(f"Signal: {final_result.signal.value}")
    print(f"Confidence: {final_result.confidence:.1f}%")
    print(f"Imbalance Value: {final_result.value:+.1f}")
    
    print("\nDetailed Metrics:")
    for key, value in final_result.metadata.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n💡 Interpretation:")
    print("- Positive imbalance indicates net buying pressure")
    print("- Persistence shows sustained accumulation/distribution")
    print("- Large trade imbalance reveals institutional activity")
    print("- Volume concentration shows if few large traders dominate")


if __name__ == "__main__":
    demonstrate_tick_volume_imbalance()