"""
📊 Volume-Based Bar Aggregator
Creates bars based on fixed volume intervals rather than time
Provides more consistent statistical properties for analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio

from ...base.types import TickData, BarData, TimeFrame
from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import SignalType


@dataclass
class VolumeBar:
    """Represents a volume-based bar"""
    timestamp_start: int
    timestamp_end: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_volume: int
    sell_volume: int
    tick_count: int
    vwap: float
    imbalance: float  # (buy_volume - sell_volume) / total_volume
    

class VolumeBarAggregator(BaseIndicator):
    """
    Aggregates tick data into volume-based bars
    
    Volume bars have several advantages:
    1. More consistent statistical properties
    2. Better for detecting institutional activity
    3. Naturally adapt to market activity
    4. Reduce noise in low-activity periods
    """
    
    def __init__(self,
                 volume_threshold: int = 10000,
                 lookback_bars: int = 100,
                 auto_adjust: bool = True):
        """
        Initialize Volume Bar Aggregator
        
        Args:
            volume_threshold: Volume required to complete a bar
            lookback_bars: Number of completed bars to keep
            auto_adjust: Automatically adjust threshold based on activity
        """
        super().__init__(
            name="VolumeBarAggregator",
            timeframe=TimeFrame.TICK,  # Works on tick data
            lookback_periods=lookback_bars,
            params={
                'volume_threshold': volume_threshold,
                'auto_adjust': auto_adjust
            }
        )
        
        self.volume_threshold = volume_threshold
        self.auto_adjust = auto_adjust
        
        # Current bar being built
        self.current_bar = None
        self.current_ticks = []
        self.current_volume = 0
        
        # Completed bars
        self.completed_bars = []
        
        # For auto-adjustment
        self.volume_history = []
        
    def process_tick(self, tick: TickData) -> Optional[VolumeBar]:
        """
        Process a single tick, returns completed bar if threshold reached
        
        Args:
            tick: Incoming tick data
            
        Returns:
            Completed VolumeBar if threshold reached, None otherwise
        """
        # Initialize first bar
        if self.current_bar is None:
            self.current_bar = {
                'timestamp_start': tick.timestamp,
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'volume': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'tick_count': 0,
                'volume_price_sum': 0  # For VWAP calculation
            }
            
        # Update current bar
        self.current_bar['high'] = max(self.current_bar['high'], tick.price)
        self.current_bar['low'] = min(self.current_bar['low'], tick.price)
        self.current_bar['close'] = tick.price
        self.current_bar['volume'] += tick.volume
        self.current_bar['tick_count'] += 1
        self.current_bar['volume_price_sum'] += tick.price * tick.volume
        
        # Track buy/sell volume
        if tick.side == 1:
            self.current_bar['buy_volume'] += tick.volume
        elif tick.side == -1:
            self.current_bar['sell_volume'] += tick.volume
            
        self.current_ticks.append(tick)
        self.current_volume += tick.volume
        
        # Check if bar is complete
        if self.current_volume >= self.volume_threshold:
            completed_bar = self._complete_current_bar(tick.timestamp)
            self._start_new_bar(tick)
            
            # Auto-adjust threshold if enabled
            if self.auto_adjust:
                self._adjust_threshold()
                
            return completed_bar
            
        return None
    
    def process_ticks(self, ticks: List[TickData]) -> List[VolumeBar]:
        """Process multiple ticks, returns all completed bars"""
        
        completed_bars = []
        
        for tick in ticks:
            bar = self.process_tick(tick)
            if bar:
                completed_bars.append(bar)
                
        return completed_bars
    
    def _complete_current_bar(self, end_timestamp: int) -> VolumeBar:
        """Complete the current bar and return it"""
        
        # Calculate VWAP
        vwap = (self.current_bar['volume_price_sum'] / 
                self.current_bar['volume'] if self.current_bar['volume'] > 0 else 
                self.current_bar['close'])
        
        # Calculate imbalance
        total_directional = (self.current_bar['buy_volume'] + 
                           self.current_bar['sell_volume'])
        
        if total_directional > 0:
            imbalance = ((self.current_bar['buy_volume'] - 
                         self.current_bar['sell_volume']) / total_directional)
        else:
            imbalance = 0.0
            
        # Create completed bar
        bar = VolumeBar(
            timestamp_start=self.current_bar['timestamp_start'],
            timestamp_end=end_timestamp,
            open=self.current_bar['open'],
            high=self.current_bar['high'],
            low=self.current_bar['low'],
            close=self.current_bar['close'],
            volume=self.current_bar['volume'],
            buy_volume=self.current_bar['buy_volume'],
            sell_volume=self.current_bar['sell_volume'],
            tick_count=self.current_bar['tick_count'],
            vwap=vwap,
            imbalance=imbalance
        )
        
        # Store in history
        self.completed_bars.append(bar)
        if len(self.completed_bars) > self.lookback_periods:
            self.completed_bars.pop(0)
            
        # Track volume for auto-adjustment
        self.volume_history.append(self.current_bar['volume'])
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)
            
        return bar
    
    def _start_new_bar(self, tick: TickData):
        """Start a new bar with remaining volume from triggering tick"""
        
        # Calculate spillover volume
        spillover = self.current_volume - self.volume_threshold
        
        # Reset current bar
        self.current_bar = {
            'timestamp_start': tick.timestamp,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': spillover,
            'buy_volume': spillover if tick.side == 1 else 0,
            'sell_volume': spillover if tick.side == -1 else 0,
            'tick_count': 1 if spillover > 0 else 0,
            'volume_price_sum': tick.price * spillover
        }
        
        self.current_ticks = [tick] if spillover > 0 else []
        self.current_volume = spillover
    
    def _adjust_threshold(self):
        """Automatically adjust volume threshold based on recent activity"""
        
        if len(self.volume_history) < 20:
            return
            
        # Calculate average bar volume
        avg_volume = np.mean(self.volume_history[-20:])
        
        # Adjust threshold to target ~50 bars per hour
        # This is a simple heuristic that can be improved
        if avg_volume > self.volume_threshold * 1.5:
            # Market is more active, increase threshold
            self.volume_threshold = int(self.volume_threshold * 1.1)
        elif avg_volume < self.volume_threshold * 0.7:
            # Market is less active, decrease threshold
            self.volume_threshold = int(self.volume_threshold * 0.9)
            
        # Keep within reasonable bounds
        self.volume_threshold = max(1000, min(100000, self.volume_threshold))
    
    def calculate(self, 
                  data: List[TickData], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate signals from volume bars
        
        Args:
            data: List of tick data
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with volume bar analysis
        """
        # Process all ticks
        bars = self.process_ticks(data)
        
        if len(bars) < 3:
            return IndicatorResult(
                timestamp=int(datetime.now().timestamp() * 1000),
                symbol=symbol,
                indicator_name=self.name,
                value=0.0,
                signal=SignalType.HOLD,
                confidence=0.0,
                timeframe=TimeFrame.TICK,
                metadata={'bars_created': len(bars)},
                calculation_time_ms=0
            )
            
        # Analyze recent bars for signals
        signal, confidence, value = self._analyze_volume_bars(bars)
        
        # Create metadata
        metadata = {
            'bars_created': len(bars),
            'current_threshold': self.volume_threshold,
            'avg_bar_volume': np.mean([b.volume for b in bars]) if bars else 0,
            'avg_imbalance': np.mean([b.imbalance for b in bars[-10:]]) if len(bars) >= 10 else 0,
            'tick_efficiency': self._calculate_tick_efficiency(bars)
        }
        
        return IndicatorResult(
            timestamp=bars[-1].timestamp_end if bars else int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=value,
            signal=signal,
            confidence=confidence,
            timeframe=TimeFrame.TICK,
            metadata=metadata,
            calculation_time_ms=0
        )
    
    def _analyze_volume_bars(self, bars: List[VolumeBar]) -> Tuple[SignalType, float, float]:
        """Analyze volume bars to generate trading signals"""
        
        if len(bars) < 3:
            return SignalType.HOLD, 0.0, 0.0
            
        recent_bars = bars[-10:] if len(bars) >= 10 else bars
        
        # Calculate metrics
        imbalances = [b.imbalance for b in recent_bars]
        avg_imbalance = np.mean(imbalances)
        imbalance_trend = np.polyfit(range(len(imbalances)), imbalances, 1)[0]
        
        # Price trend
        closes = [b.close for b in recent_bars]
        price_trend = np.polyfit(range(len(closes)), closes, 1)[0]
        
        # Volume profile
        volumes = [b.volume for b in recent_bars]
        volume_increasing = np.polyfit(range(len(volumes)), volumes, 1)[0] > 0
        
        # VWAP deviation
        last_bar = bars[-1]
        vwap_deviation = (last_bar.close - last_bar.vwap) / last_bar.vwap
        
        # Generate signals
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Strong buying pressure
        if avg_imbalance > 0.3 and imbalance_trend > 0 and price_trend > 0:
            signal = SignalType.BUY
            confidence = min(avg_imbalance * 100 + 50, 90)
            
        # Strong selling pressure
        elif avg_imbalance < -0.3 and imbalance_trend < 0 and price_trend < 0:
            signal = SignalType.SELL
            confidence = min(abs(avg_imbalance) * 100 + 50, 90)
            
        # Volume expansion with directional imbalance
        elif volume_increasing and abs(avg_imbalance) > 0.2:
            if avg_imbalance > 0:
                signal = SignalType.BUY
            else:
                signal = SignalType.SELL
            confidence = abs(avg_imbalance) * 150
            
        # VWAP reversion signals
        elif abs(vwap_deviation) > 0.01:  # 1% from VWAP
            if vwap_deviation < -0.01:  # Below VWAP
                signal = SignalType.BUY
                confidence = min(abs(vwap_deviation) * 2000, 70)
            else:  # Above VWAP
                signal = SignalType.SELL
                confidence = min(abs(vwap_deviation) * 2000, 70)
        
        # Calculate indicator value (normalized imbalance)
        value = avg_imbalance * 100
        
        return signal, confidence, value
    
    def _calculate_tick_efficiency(self, bars: List[VolumeBar]) -> float:
        """Calculate how efficiently price moves per tick"""
        
        if not bars:
            return 0.0
            
        efficiencies = []
        
        for bar in bars[-10:]:  # Last 10 bars
            if bar.tick_count > 0:
                price_range = bar.high - bar.low
                efficiency = price_range / bar.tick_count
                efficiencies.append(efficiency)
                
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def validate_data(self, data: List[TickData]) -> bool:
        """Validate tick data"""
        return isinstance(data, list) and len(data) > 0
    
    def get_bars_dataframe(self) -> pd.DataFrame:
        """Convert completed bars to DataFrame for analysis"""
        
        if not self.completed_bars:
            return pd.DataFrame()
            
        data = []
        for bar in self.completed_bars:
            data.append({
                'timestamp': bar.timestamp_end,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'buy_volume': bar.buy_volume,
                'sell_volume': bar.sell_volume,
                'tick_count': bar.tick_count,
                'vwap': bar.vwap,
                'imbalance': bar.imbalance
            })
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df


def demonstrate_volume_bars():
    """Demonstration of Volume Bar aggregation"""
    
    print("📊 Volume Bar Aggregator Demonstration\n")
    
    # Create sample tick data
    np.random.seed(42)
    ticks = []
    
    base_price = 100.0
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Generate realistic tick sequence
    for i in range(1000):
        # Random walk
        price_change = np.random.normal(0, 0.01)
        base_price *= (1 + price_change)
        
        # Volume varies
        volume = int(np.random.lognormal(3, 1))
        
        # Buy/sell pressure
        if price_change > 0:
            side = 1 if np.random.random() < 0.7 else -1
        else:
            side = -1 if np.random.random() < 0.7 else 1
            
        tick = TickData(
            timestamp=timestamp + i * 100,  # 100ms between ticks
            price=base_price,
            volume=volume,
            side=side
        )
        
        ticks.append(tick)
    
    # Create aggregator
    aggregator = VolumeBarAggregator(
        volume_threshold=5000,
        auto_adjust=True
    )
    
    # Process ticks
    print(f"Processing {len(ticks)} ticks...")
    completed_bars = aggregator.process_ticks(ticks)
    
    print(f"\nCreated {len(completed_bars)} volume bars")
    
    # Show some bars
    print("\nFirst 5 Volume Bars:")
    print("-" * 80)
    
    for i, bar in enumerate(completed_bars[:5]):
        duration = (bar.timestamp_end - bar.timestamp_start) / 1000  # seconds
        print(f"Bar {i+1}:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  OHLC: ${bar.open:.2f} / ${bar.high:.2f} / ${bar.low:.2f} / ${bar.close:.2f}")
        print(f"  Volume: {bar.volume:,} ({bar.tick_count} ticks)")
        print(f"  Imbalance: {bar.imbalance:+.2f} (Buy: {bar.buy_volume}, Sell: {bar.sell_volume})")
        print(f"  VWAP: ${bar.vwap:.2f}")
        print()
    
    # Analyze patterns
    if len(completed_bars) >= 10:
        # Calculate signals
        result = aggregator.calculate(ticks, "TEST")
        
        print(f"\nSignal Analysis:")
        print(f"Signal: {result.signal.value}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Value: {result.value:.2f}")
        
        print(f"\nMetadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
    
    # Show advantages of volume bars
    print("\n📊 Volume Bar Advantages:")
    print("1. Consistent information content per bar")
    print("2. Adapts to market activity automatically")
    print("3. Better for detecting institutional flow")
    print("4. More stable statistical properties")


if __name__ == "__main__":
    demonstrate_volume_bars()