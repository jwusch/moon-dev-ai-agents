"""
🎯 Order Flow Divergence Indicator
Detects when price and order flow are diverging, signaling potential reversals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class OrderFlowDivergenceIndicator(BaseIndicator):
    """
    Analyzes divergences between price movement and order flow
    
    Key divergences:
    - Bullish: Price declining but net order flow positive (accumulation)
    - Bearish: Price rising but net order flow negative (distribution)
    
    Components:
    1. Order Flow: Cumulative volume delta
    2. Price Action: Direction and momentum
    3. Divergence Score: Magnitude and persistence of divergence
    """
    
    def __init__(self,
                 lookback_window: int = 50,
                 divergence_threshold: float = 0.2,
                 momentum_window: int = 20,
                 volume_normalization: bool = True):
        """
        Initialize Order Flow Divergence indicator
        
        Args:
            lookback_window: Ticks to analyze for divergence
            divergence_threshold: Minimum divergence score to signal
            momentum_window: Window for price momentum calculation
            volume_normalization: Normalize flow by average volume
        """
        super().__init__(
            name="OrderFlowDivergence",
            timeframe=TimeFrame.TICK,
            lookback_periods=lookback_window,
            params={
                'lookback_window': lookback_window,
                'divergence_threshold': divergence_threshold,
                'momentum_window': momentum_window,
                'volume_normalization': volume_normalization
            }
        )
        
        self.lookback_window = lookback_window
        self.divergence_threshold = divergence_threshold
        self.momentum_window = momentum_window
        self.volume_normalization = volume_normalization
        
        # Tick storage
        self.tick_window = deque(maxlen=lookback_window)
        
        # Order flow tracking
        self.cumulative_delta = 0
        self.delta_history = deque(maxlen=lookback_window)
        
        # Price tracking
        self.price_history = deque(maxlen=lookback_window)
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate order flow divergence
        
        Args:
            data: Tick data or OHLCV bars
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with divergence analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Generate synthetic ticks from bars
            from ...utils.synthetic_ticks import SyntheticTickGenerator
            generator = SyntheticTickGenerator(method='vwap_weighted')
            
            ticks = []
            for _, bar in data.iterrows():
                bar_ticks = generator.generate_ticks(bar, n_ticks=15)
                ticks.extend(bar_ticks)
        else:
            ticks = data
            
        if not ticks:
            return self._empty_result(symbol)
            
        # Process all ticks
        for tick in ticks:
            self._process_tick(tick)
            
        if len(self.tick_window) < min(20, self.lookback_window // 2):
            return self._empty_result(symbol)
            
        # Calculate components
        order_flow = self._calculate_order_flow()
        price_momentum = self._calculate_price_momentum()
        
        # Detect divergences
        divergence_score = self._calculate_divergence_score(order_flow, price_momentum)
        divergence_type = self._classify_divergence(divergence_score, order_flow, price_momentum)
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            divergence_score, divergence_type, order_flow, price_momentum
        )
        
        # Create metadata
        metadata = self._create_metadata(
            order_flow, price_momentum, divergence_score, divergence_type
        )
        
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
        self.price_history.append(tick.price)
        
        # Calculate volume delta
        volume_delta = tick.volume * tick.side  # Positive for buys, negative for sells
        self.cumulative_delta += volume_delta
        self.delta_history.append(self.cumulative_delta)
        
    def _calculate_order_flow(self) -> Dict[str, float]:
        """Calculate order flow metrics"""
        
        if len(self.delta_history) < 2:
            return {'flow': 0, 'momentum': 0, 'strength': 0}
            
        # Recent flow
        recent_delta = list(self.delta_history)[-self.momentum_window:]
        
        # Flow direction and momentum
        flow_change = recent_delta[-1] - recent_delta[0]
        
        # Normalize by volume if requested
        if self.volume_normalization and self.tick_window:
            total_volume = sum(abs(t.volume) for t in list(self.tick_window)[-self.momentum_window:])
            if total_volume > 0:
                flow_change = flow_change / total_volume
                
        # Flow momentum (rate of change)
        if len(recent_delta) >= 3:
            flow_momentum = np.polyfit(range(len(recent_delta)), recent_delta, 1)[0]
        else:
            flow_momentum = 0
            
        # Flow strength (consistency)
        flow_values = np.diff(recent_delta)
        if len(flow_values) > 0:
            positive_flows = sum(1 for f in flow_values if f > 0)
            flow_strength = abs(positive_flows / len(flow_values) - 0.5) * 2  # 0-1 scale
        else:
            flow_strength = 0
            
        return {
            'flow': flow_change,
            'momentum': flow_momentum,
            'strength': flow_strength
        }
    
    def _calculate_price_momentum(self) -> Dict[str, float]:
        """Calculate price momentum metrics"""
        
        if len(self.price_history) < 2:
            return {'momentum': 0, 'velocity': 0, 'acceleration': 0}
            
        recent_prices = list(self.price_history)[-self.momentum_window:]
        
        # Simple return
        price_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Linear regression for trend
        if len(recent_prices) >= 3:
            x = np.arange(len(recent_prices))
            momentum = np.polyfit(x, recent_prices, 1)[0]
            
            # Normalize momentum
            momentum = momentum / recent_prices[0]
        else:
            momentum = price_return / len(recent_prices)
            
        # Calculate velocity (rate of change)
        if len(recent_prices) >= 5:
            mid = len(recent_prices) // 2
            first_half_return = (recent_prices[mid] - recent_prices[0]) / recent_prices[0]
            second_half_return = (recent_prices[-1] - recent_prices[mid]) / recent_prices[mid]
            acceleration = second_half_return - first_half_return
        else:
            acceleration = 0
            
        return {
            'momentum': momentum,
            'velocity': price_return,
            'acceleration': acceleration
        }
    
    def _calculate_divergence_score(self, 
                                   order_flow: Dict[str, float],
                                   price_momentum: Dict[str, float]) -> float:
        """
        Calculate divergence score between price and order flow
        
        Positive score: Bullish divergence (price down, flow up)
        Negative score: Bearish divergence (price up, flow down)
        """
        
        # Normalize components
        price_direction = np.sign(price_momentum['momentum'])
        flow_direction = np.sign(order_flow['flow'])
        
        # Basic divergence exists when signs differ
        if price_direction * flow_direction < 0:
            # Calculate magnitude
            price_magnitude = abs(price_momentum['momentum'])
            flow_magnitude = abs(order_flow['flow'])
            
            # Divergence score incorporates both magnitude and consistency
            base_score = (price_magnitude + flow_magnitude) / 2
            
            # Adjust for flow strength (consistency)
            score = base_score * (1 + order_flow['strength'])
            
            # Sign indicates type: positive for bullish, negative for bearish
            if flow_direction > 0:  # Bullish divergence
                return score
            else:  # Bearish divergence
                return -score
        else:
            # No divergence
            return 0
    
    def _classify_divergence(self,
                            divergence_score: float,
                            order_flow: Dict[str, float],
                            price_momentum: Dict[str, float]) -> str:
        """Classify type of divergence"""
        
        if abs(divergence_score) < self.divergence_threshold:
            return "none"
            
        if divergence_score > 0:
            # Further classify bullish divergence
            if order_flow['momentum'] > 0 and price_momentum['acceleration'] > 0:
                return "hidden_bullish"  # Price starting to turn
            else:
                return "regular_bullish"
        else:
            # Further classify bearish divergence
            if order_flow['momentum'] < 0 and price_momentum['acceleration'] < 0:
                return "hidden_bearish"  # Price starting to turn
            else:
                return "regular_bearish"
    
    def _generate_signal(self,
                        divergence_score: float,
                        divergence_type: str,
                        order_flow: Dict[str, float],
                        price_momentum: Dict[str, float]) -> Tuple[SignalType, float, float]:
        """Generate trading signal from divergence analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        if divergence_type == "none":
            # Check for aligned strong flows
            if order_flow['flow'] > 0 and price_momentum['momentum'] > 0:
                if order_flow['strength'] > 0.7:
                    signal = SignalType.BUY
                    confidence = min(order_flow['strength'] * 50, 60)
            elif order_flow['flow'] < 0 and price_momentum['momentum'] < 0:
                if order_flow['strength'] > 0.7:
                    signal = SignalType.SELL
                    confidence = min(order_flow['strength'] * 50, 60)
                    
        elif "bullish" in divergence_type:
            signal = SignalType.BUY
            
            # Hidden divergences are stronger signals
            if "hidden" in divergence_type:
                confidence = min(abs(divergence_score) * 200, 90)
            else:
                confidence = min(abs(divergence_score) * 150, 80)
                
        elif "bearish" in divergence_type:
            signal = SignalType.SELL
            
            if "hidden" in divergence_type:
                confidence = min(abs(divergence_score) * 200, 90)
            else:
                confidence = min(abs(divergence_score) * 150, 80)
        
        # Additional filters
        if signal != SignalType.HOLD:
            # Check flow consistency
            if order_flow['strength'] < 0.3:
                confidence *= 0.7
                
            # Check if divergence is fresh
            if self._is_divergence_fresh():
                confidence *= 1.2
                
        confidence = min(confidence, 95)
        
        # Value represents divergence strength
        value = abs(divergence_score) * 100
        
        return signal, confidence, value
    
    def _is_divergence_fresh(self) -> bool:
        """Check if divergence recently appeared"""
        
        if len(self.delta_history) < 10:
            return True
            
        # Check if divergence appeared in last 20% of window
        recent_start = int(len(self.delta_history) * 0.8)
        
        # Compare early vs late order flow
        early_delta = list(self.delta_history)[:recent_start]
        late_delta = list(self.delta_history)[recent_start:]
        
        early_trend = np.mean(np.diff(early_delta)) if len(early_delta) > 1 else 0
        late_trend = np.mean(np.diff(late_delta)) if len(late_delta) > 1 else 0
        
        # Fresh if trend changed sign
        return np.sign(early_trend) != np.sign(late_trend)
    
    def _create_metadata(self,
                        order_flow: Dict[str, float],
                        price_momentum: Dict[str, float],
                        divergence_score: float,
                        divergence_type: str) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'divergence_score': divergence_score,
            'divergence_type': divergence_type,
            'order_flow': order_flow['flow'],
            'flow_momentum': order_flow['momentum'],
            'flow_strength': order_flow['strength'],
            'price_momentum': price_momentum['momentum'],
            'price_velocity': price_momentum['velocity'],
            'price_acceleration': price_momentum['acceleration'],
            'cumulative_delta': self.cumulative_delta,
            'tick_count': len(self.tick_window)
        }
        
        # Add tick-level statistics
        if self.tick_window:
            recent_ticks = list(self.tick_window)[-20:]
            buy_ticks = sum(1 for t in recent_ticks if t.side == 1)
            sell_ticks = sum(1 for t in recent_ticks if t.side == -1)
            
            metadata['recent_buy_ratio'] = buy_ticks / len(recent_ticks) if recent_ticks else 0
            metadata['avg_tick_size'] = np.mean([t.volume for t in recent_ticks])
            
            # Large order detection
            volumes = [t.volume for t in self.tick_window]
            if volumes:
                large_threshold = np.percentile(volumes, 90)
                large_orders = [t for t in recent_ticks if t.volume >= large_threshold]
                
                large_buy_vol = sum(t.volume for t in large_orders if t.side == 1)
                large_sell_vol = sum(t.volume for t in large_orders if t.side == -1)
                
                metadata['large_order_imbalance'] = (
                    (large_buy_vol - large_sell_vol) / (large_buy_vol + large_sell_vol)
                    if (large_buy_vol + large_sell_vol) > 0 else 0
                )
                
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
            metadata={'error': 'Insufficient data'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= self.momentum_window
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 5
        return False


def demonstrate_order_flow_divergence():
    """Demonstration of Order Flow Divergence indicator"""
    
    print("🎯 Order Flow Divergence Demonstration\n")
    
    # Generate synthetic market scenarios
    np.random.seed(42)
    ticks = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Scenario 1: Normal aligned movement (no divergence)
    print("Phase 1: Aligned price and flow (normal market)...")
    base_price = 100.0
    for i in range(50):
        # Price trending up
        price = base_price + i * 0.05 + np.random.normal(0, 0.1)
        
        # Order flow aligned (more buys)
        if np.random.random() < 0.65:  # 65% buy probability
            volume = np.random.randint(50, 150)
            side = 1
        else:
            volume = np.random.randint(30, 80)
            side = -1
            
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Scenario 2: Bearish divergence (price up, flow down)
    print("Phase 2: Bearish divergence (distribution)...")
    for i in range(50, 100):
        # Price still going up (but slowing)
        price = base_price + 2.5 + (i-50) * 0.02 + np.random.normal(0, 0.1)
        
        # But selling pressure increasing (distribution)
        if np.random.random() < 0.3:  # Only 30% buy probability
            volume = np.random.randint(20, 60)
            side = 1
        else:
            volume = np.random.randint(80, 200)  # Larger sells
            side = -1
            
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Scenario 3: Price reversal after divergence
    print("Phase 3: Price reversal following divergence...")
    for i in range(100, 150):
        # Price starting to fall
        price = base_price + 3.5 - (i-100) * 0.08 + np.random.normal(0, 0.1)
        
        # Heavy selling continues
        if np.random.random() < 0.2:
            volume = np.random.randint(10, 40)
            side = 1
        else:
            volume = np.random.randint(100, 250)
            side = -1
            
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
    
    # Create indicator
    indicator = OrderFlowDivergenceIndicator(
        lookback_window=40,
        divergence_threshold=0.15,
        momentum_window=20
    )
    
    # Process ticks incrementally
    print("\nAnalyzing order flow divergence...\n")
    
    results = []
    for i in range(20, len(ticks), 10):
        batch = ticks[:i+1]
        result = indicator.calculate(batch, "TEST")
        results.append(result)
        
        if i in [45, 95, 145]:  # Key points
            print(f"After {i} ticks:")
            print(f"  Price: ${ticks[i].price:.2f}")
            print(f"  Cumulative Delta: {result.metadata['cumulative_delta']:+,}")
            print(f"  Divergence Type: {result.metadata['divergence_type']}")
            print(f"  Divergence Score: {result.metadata['divergence_score']:+.3f}")
            print(f"  Signal: {result.signal.value}")
            print(f"  Confidence: {result.confidence:.1f}%")
            print(f"  Order Flow: {result.metadata['order_flow']:+.3f}")
            print(f"  Price Momentum: {result.metadata['price_momentum']:+.3f}")
            print()
    
    # Final analysis
    final_result = results[-1]
    print("\n" + "="*50)
    print("FINAL ANALYSIS:")
    print("="*50)
    print(f"Signal: {final_result.signal.value}")
    print(f"Confidence: {final_result.confidence:.1f}%")
    print(f"Divergence Strength: {final_result.value:.1f}")
    
    print("\nKey Insights:")
    print(f"- Divergence detected: {final_result.metadata['divergence_type']}")
    print(f"- Order flow strength: {final_result.metadata['flow_strength']:.2%}")
    print(f"- Large order imbalance: {final_result.metadata.get('large_order_imbalance', 0):+.2%}")
    
    print("\n💡 Interpretation:")
    print("- Divergences signal potential reversals")
    print("- Hidden divergences are stronger continuation signals")
    print("- Flow strength indicates conviction behind the move")
    print("- Large order imbalance reveals institutional activity")


if __name__ == "__main__":
    demonstrate_order_flow_divergence()