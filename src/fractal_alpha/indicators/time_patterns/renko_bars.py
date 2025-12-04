"""
🧱 Renko Bars - Price-Based Time Alternative
Creates fixed price movement bars that filter out time and noise
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class RenkoBarsIndicator(BaseIndicator):
    """
    Renko Bars indicator for noise-filtered price analysis
    
    Renko bars are price-based charts that:
    - Only form new bars when price moves by a fixed amount
    - Completely ignore time and volume
    - Filter out market noise and show pure price action
    - Reveal trends more clearly than time-based charts
    
    Key advantages:
    - Removes time distortion (weekend gaps, overnight moves)
    - Filters minor price fluctuations
    - Makes support/resistance levels clearer
    - Simplifies trend identification
    - Reduces false signals
    """
    
    def __init__(self,
                 brick_size: Optional[float] = None,
                 atr_period: int = 14,
                 atr_multiplier: float = 1.0,
                 min_bricks: int = 10,
                 reversal_bricks: int = 2,
                 detect_patterns: bool = True):
        """
        Initialize Renko Bars indicator
        
        Args:
            brick_size: Fixed brick size (if None, uses ATR-based)
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR-based brick size
            min_bricks: Minimum bricks for pattern detection
            reversal_bricks: Bricks needed for reversal (typically 1 or 2)
            detect_patterns: Whether to detect Renko patterns
        """
        super().__init__(
            name="RenkoBars",
            timeframe=TimeFrame.MINUTE_5,
            lookback_periods=atr_period * 10,
            params={
                'brick_size': brick_size,
                'atr_period': atr_period,
                'atr_multiplier': atr_multiplier,
                'min_bricks': min_bricks,
                'reversal_bricks': reversal_bricks,
                'detect_patterns': detect_patterns
            }
        )
        
        self.brick_size = brick_size
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_bricks = min_bricks
        self.reversal_bricks = reversal_bricks
        self.detect_patterns = detect_patterns
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Renko bars and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with Renko analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.atr_period * 2:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
            highs = data['High'].values if 'High' in data else data['high'].values
            lows = data['Low'].values if 'Low' in data else data['low'].values
        else:
            if len(data) < self.atr_period * 2:
                return self._empty_result(symbol)
            prices = np.array(data)
            highs = prices
            lows = prices
            
        # Calculate brick size
        if self.brick_size is None:
            brick_size = self._calculate_dynamic_brick_size(highs, lows, prices)
        else:
            brick_size = self.brick_size
            
        # Build Renko bricks
        renko_bricks = self._build_renko_bricks(prices, brick_size)
        
        if len(renko_bricks) < self.min_bricks:
            return self._empty_result(symbol)
            
        # Analyze Renko structure
        renko_analysis = self._analyze_renko_structure(renko_bricks)
        
        # Detect patterns if enabled
        patterns = []
        if self.detect_patterns:
            patterns = self._detect_renko_patterns(renko_bricks)
            
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(renko_bricks)
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            renko_bricks, renko_analysis, patterns, trend_strength
        )
        
        # Create metadata
        metadata = self._create_metadata(
            renko_bricks, renko_analysis, patterns, trend_strength,
            brick_size, len(prices)
        )
        
        # Get timestamp
        if isinstance(data, pd.DataFrame):
            timestamp = int(data.index[-1].timestamp() * 1000) if hasattr(data.index[-1], 'timestamp') else int(datetime.now().timestamp() * 1000)
        else:
            timestamp = int(datetime.now().timestamp() * 1000)
            
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
    
    def _calculate_dynamic_brick_size(self, 
                                     highs: np.ndarray, 
                                     lows: np.ndarray,
                                     closes: np.ndarray) -> float:
        """Calculate dynamic brick size using ATR"""
        
        # Calculate true range
        tr = np.zeros(len(closes))
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr[i] = max(high_low, high_close, low_close)
        
        # Calculate ATR
        atr = np.zeros(len(closes))
        atr[self.atr_period] = np.mean(tr[1:self.atr_period+1])
        
        for i in range(self.atr_period + 1, len(closes)):
            atr[i] = (atr[i-1] * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        # Use recent ATR for brick size
        current_atr = atr[-1] if atr[-1] > 0 else np.mean(tr[1:])
        
        return current_atr * self.atr_multiplier
    
    def _build_renko_bricks(self, prices: np.ndarray, brick_size: float) -> List[Dict]:
        """Build Renko bricks from price series"""
        
        bricks = []
        
        if len(prices) == 0:
            return bricks
            
        # Initialize first brick
        current_price = prices[0]
        current_direction = None
        brick_high = current_price
        brick_low = current_price
        
        for i, price in enumerate(prices):
            # Upward movement
            if price >= brick_high + brick_size:
                # Calculate how many bricks to add
                num_bricks = int((price - brick_high) / brick_size)
                
                for j in range(num_bricks):
                    new_high = brick_high + brick_size
                    new_low = brick_high
                    
                    # Check for reversal
                    if current_direction == 'down' and self.reversal_bricks > 1:
                        # Need more bricks for reversal
                        if j < self.reversal_bricks - 1:
                            continue
                    
                    bricks.append({
                        'index': i,
                        'price': price,
                        'open': new_low,
                        'close': new_high,
                        'high': new_high,
                        'low': new_low,
                        'direction': 'up',
                        'brick_num': len(bricks)
                    })
                    
                    brick_high = new_high
                    brick_low = new_low
                    current_direction = 'up'
                    
            # Downward movement
            elif price <= brick_low - brick_size:
                # Calculate how many bricks to add
                num_bricks = int((brick_low - price) / brick_size)
                
                for j in range(num_bricks):
                    new_low = brick_low - brick_size
                    new_high = brick_low
                    
                    # Check for reversal
                    if current_direction == 'up' and self.reversal_bricks > 1:
                        # Need more bricks for reversal
                        if j < self.reversal_bricks - 1:
                            continue
                    
                    bricks.append({
                        'index': i,
                        'price': price,
                        'open': new_high,
                        'close': new_low,
                        'high': new_high,
                        'low': new_low,
                        'direction': 'down',
                        'brick_num': len(bricks)
                    })
                    
                    brick_high = new_high
                    brick_low = new_low
                    current_direction = 'down'
        
        return bricks
    
    def _analyze_renko_structure(self, bricks: List[Dict]) -> Dict:
        """Analyze Renko brick structure"""
        
        if not bricks:
            return {}
            
        # Count consecutive bricks
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        
        current_direction = bricks[0]['direction']
        current_consecutive = 1
        
        for i in range(1, len(bricks)):
            if bricks[i]['direction'] == current_direction:
                current_consecutive += 1
            else:
                if current_direction == 'up':
                    max_consecutive_up = max(max_consecutive_up, current_consecutive)
                else:
                    max_consecutive_down = max(max_consecutive_down, current_consecutive)
                    
                current_direction = bricks[i]['direction']
                current_consecutive = 1
        
        # Update final sequence
        if current_direction == 'up':
            consecutive_up = current_consecutive
            max_consecutive_up = max(max_consecutive_up, current_consecutive)
        else:
            consecutive_down = current_consecutive
            max_consecutive_down = max(max_consecutive_down, current_consecutive)
        
        # Calculate trend metrics
        up_bricks = sum(1 for b in bricks if b['direction'] == 'up')
        down_bricks = len(bricks) - up_bricks
        up_ratio = up_bricks / len(bricks) if len(bricks) > 0 else 0.5
        
        # Recent trend (last 10 bricks)
        recent_bricks = bricks[-10:] if len(bricks) >= 10 else bricks
        recent_up = sum(1 for b in recent_bricks if b['direction'] == 'up')
        recent_trend = 'up' if recent_up > len(recent_bricks) / 2 else 'down'
        
        return {
            'total_bricks': len(bricks),
            'up_bricks': up_bricks,
            'down_bricks': down_bricks,
            'up_ratio': up_ratio,
            'consecutive_up': consecutive_up,
            'consecutive_down': consecutive_down,
            'max_consecutive_up': max_consecutive_up,
            'max_consecutive_down': max_consecutive_down,
            'current_direction': bricks[-1]['direction'] if bricks else None,
            'recent_trend': recent_trend,
            'last_reversal': self._find_last_reversal(bricks)
        }
    
    def _find_last_reversal(self, bricks: List[Dict]) -> Optional[int]:
        """Find the last reversal point in bricks"""
        
        if len(bricks) < 2:
            return None
            
        for i in range(len(bricks) - 1, 0, -1):
            if bricks[i]['direction'] != bricks[i-1]['direction']:
                return i
                
        return None
    
    def _detect_renko_patterns(self, bricks: List[Dict]) -> List[Dict]:
        """Detect patterns in Renko bricks"""
        
        patterns = []
        
        if len(bricks) < 3:
            return patterns
            
        # Double bottom/top
        for i in range(2, len(bricks)):
            # Double bottom: down-up-down-up
            if (i >= 3 and 
                bricks[i-3]['direction'] == 'down' and
                bricks[i-2]['direction'] == 'up' and
                bricks[i-1]['direction'] == 'down' and
                bricks[i]['direction'] == 'up' and
                abs(bricks[i-3]['low'] - bricks[i-1]['low']) < bricks[i]['close'] - bricks[i]['open']):
                
                patterns.append({
                    'type': 'double_bottom',
                    'position': i,
                    'strength': 'strong' if abs(bricks[i-3]['low'] - bricks[i-1]['low']) < 0.5 * (bricks[i]['close'] - bricks[i]['open']) else 'moderate'
                })
            
            # Double top: up-down-up-down
            elif (i >= 3 and
                  bricks[i-3]['direction'] == 'up' and
                  bricks[i-2]['direction'] == 'down' and
                  bricks[i-1]['direction'] == 'up' and
                  bricks[i]['direction'] == 'down' and
                  abs(bricks[i-3]['high'] - bricks[i-1]['high']) < bricks[i]['open'] - bricks[i]['close']):
                
                patterns.append({
                    'type': 'double_top',
                    'position': i,
                    'strength': 'strong' if abs(bricks[i-3]['high'] - bricks[i-1]['high']) < 0.5 * (bricks[i]['open'] - bricks[i]['close']) else 'moderate'
                })
        
        # Trend continuation patterns (3+ bricks in same direction)
        consecutive_count = 1
        current_direction = bricks[0]['direction']
        
        for i in range(1, len(bricks)):
            if bricks[i]['direction'] == current_direction:
                consecutive_count += 1
                
                if consecutive_count == 3:
                    patterns.append({
                        'type': f'trend_continuation_{current_direction}',
                        'position': i,
                        'strength': 'moderate'
                    })
                elif consecutive_count == 5:
                    patterns.append({
                        'type': f'strong_trend_{current_direction}',
                        'position': i,
                        'strength': 'strong'
                    })
            else:
                current_direction = bricks[i]['direction']
                consecutive_count = 1
        
        return patterns
    
    def _calculate_trend_strength(self, bricks: List[Dict]) -> float:
        """Calculate trend strength from Renko bricks"""
        
        if len(bricks) < 3:
            return 0.0
            
        # Use recent bricks for trend strength
        lookback = min(20, len(bricks))
        recent_bricks = bricks[-lookback:]
        
        # Calculate directional movement
        up_moves = sum(1 for b in recent_bricks if b['direction'] == 'up')
        down_moves = lookback - up_moves
        
        # Calculate trend strength (0-100)
        strength = abs(up_moves - down_moves) / lookback * 100
        
        # Adjust for consecutive moves
        max_consecutive = 0
        current_consecutive = 1
        current_dir = recent_bricks[0]['direction']
        
        for i in range(1, len(recent_bricks)):
            if recent_bricks[i]['direction'] == current_dir:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_dir = recent_bricks[i]['direction']
                current_consecutive = 1
        
        # Boost strength for consecutive moves
        if max_consecutive >= 5:
            strength = min(strength * 1.5, 100)
        elif max_consecutive >= 3:
            strength = min(strength * 1.2, 100)
            
        return strength
    
    def _generate_signal(self,
                        bricks: List[Dict],
                        analysis: Dict,
                        patterns: List[Dict],
                        trend_strength: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from Renko analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        if not bricks or not analysis:
            return signal, confidence, 50.0
            
        # Current direction signal
        if analysis['current_direction'] == 'up':
            signal = SignalType.BUY
            confidence = 50.0
        elif analysis['current_direction'] == 'down':
            signal = SignalType.SELL
            confidence = 50.0
            
        # Adjust for consecutive bricks
        if analysis['consecutive_up'] >= 3:
            signal = SignalType.BUY
            confidence = min(60 + analysis['consecutive_up'] * 5, 80)
        elif analysis['consecutive_down'] >= 3:
            signal = SignalType.SELL  
            confidence = min(60 + analysis['consecutive_down'] * 5, 80)
            
        # Pattern-based adjustments
        if patterns:
            recent_pattern = patterns[-1]
            
            if recent_pattern['type'] == 'double_bottom':
                signal = SignalType.BUY
                confidence = 70 if recent_pattern['strength'] == 'strong' else 60
                
            elif recent_pattern['type'] == 'double_top':
                signal = SignalType.SELL
                confidence = 70 if recent_pattern['strength'] == 'strong' else 60
                
            elif recent_pattern['type'].startswith('strong_trend'):
                if 'up' in recent_pattern['type']:
                    signal = SignalType.BUY
                    confidence = 75
                else:
                    signal = SignalType.SELL
                    confidence = 75
        
        # Reversal detection
        if analysis.get('last_reversal') is not None:
            bricks_since_reversal = len(bricks) - analysis['last_reversal']
            
            if bricks_since_reversal <= 2:
                # Recent reversal - early signal
                confidence = min(confidence * 1.2, 85)
            elif bricks_since_reversal > 10:
                # Extended trend - potential exhaustion
                confidence *= 0.8
        
        # Trend strength adjustment
        if trend_strength > 70:
            confidence = min(confidence * 1.1, 85)
        elif trend_strength < 30:
            confidence *= 0.8
            
        # Value represents Renko clarity (0-100)
        value = trend_strength
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        bricks: List[Dict],
                        analysis: Dict,
                        patterns: List[Dict],
                        trend_strength: float,
                        brick_size: float,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'brick_size': brick_size,
            'total_bricks': len(bricks),
            'data_points': data_points,
            'compression_ratio': data_points / len(bricks) if len(bricks) > 0 else 0,
            'trend_strength': trend_strength
        }
        
        # Add analysis details
        if analysis:
            metadata.update({
                'current_direction': analysis['current_direction'],
                'up_ratio': analysis['up_ratio'],
                'consecutive_up': analysis['consecutive_up'],
                'consecutive_down': analysis['consecutive_down'],
                'max_consecutive_up': analysis['max_consecutive_up'],
                'max_consecutive_down': analysis['max_consecutive_down'],
                'recent_trend': analysis['recent_trend']
            })
            
            if analysis.get('last_reversal') is not None:
                metadata['bricks_since_reversal'] = len(bricks) - analysis['last_reversal']
        
        # Add pattern information
        if patterns:
            metadata['patterns_detected'] = len(patterns)
            metadata['recent_patterns'] = [
                {
                    'type': p['type'],
                    'strength': p['strength'],
                    'bricks_ago': len(bricks) - p['position']
                }
                for p in patterns[-3:]  # Last 3 patterns
            ]
        
        # Add current brick details
        if bricks:
            current_brick = bricks[-1]
            metadata['current_brick'] = {
                'direction': current_brick['direction'],
                'open': current_brick['open'],
                'close': current_brick['close'],
                'brick_number': current_brick['brick_num']
            }
            
            # Price levels
            up_bricks = [b for b in bricks if b['direction'] == 'up']
            down_bricks = [b for b in bricks if b['direction'] == 'down']
            
            if up_bricks:
                metadata['resistance_level'] = max(b['high'] for b in up_bricks[-5:])
            if down_bricks:
                metadata['support_level'] = min(b['low'] for b in down_bricks[-5:])
        
        # Trading recommendations
        if trend_strength > 60:
            metadata['recommendation'] = "Strong trend - follow momentum"
        elif trend_strength < 30:
            metadata['recommendation'] = "Choppy market - wait for clarity"
        else:
            metadata['recommendation'] = "Moderate trend - use tight stops"
            
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for Renko analysis'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            required = ['close'] if 'close' in data.columns else ['Close']
            if self.brick_size is None:  # Need OHLC for ATR
                required.extend(['high', 'low'] if 'close' in data.columns else ['High', 'Low'])
            has_required = all(col in data.columns or col.lower() in data.columns for col in required)
            return has_required and len(data) >= self.atr_period * 2
        else:
            return len(data) >= self.atr_period * 2


def demonstrate_renko_bars():
    """Demonstration of Renko Bars indicator"""
    
    print("🧱 Renko Bars Demonstration\n")
    
    # Generate synthetic trending data
    print("Generating synthetic price data with trends and reversals...\n")
    
    np.random.seed(42)
    n_points = 500
    
    prices = []
    price = 100
    
    # Create distinct price movements
    for i in range(n_points):
        if i < 100:
            # Uptrend
            price += np.random.uniform(0, 0.5) if np.random.rand() > 0.3 else np.random.uniform(-0.3, 0)
        elif i < 200:
            # Sideways
            price += np.random.uniform(-0.2, 0.2)
        elif i < 300:
            # Downtrend
            price -= np.random.uniform(0, 0.4) if np.random.rand() > 0.3 else np.random.uniform(-0.3, 0)
        elif i < 400:
            # Sharp reversal up
            price += np.random.uniform(0.1, 0.6) if np.random.rand() > 0.2 else np.random.uniform(-0.2, 0)
        else:
            # Volatile period
            price += np.random.uniform(-0.5, 0.5)
            
        prices.append(price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p + np.random.uniform(0, 0.3) for p in prices],
        'Low': [p - np.random.uniform(0, 0.3) for p in prices],
        'Close': prices
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='5min'))
    
    # Create Renko indicator with fixed brick size
    print("Creating Renko bars with fixed brick size...\n")
    renko_fixed = RenkoBarsIndicator(
        brick_size=1.0,  # $1 bricks
        reversal_bricks=1,
        detect_patterns=True
    )
    
    result_fixed = renko_fixed.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("FIXED BRICK SIZE RENKO ANALYSIS:")
    print("=" * 60)
    print(f"Brick Size: ${result_fixed.metadata['brick_size']:.2f}")
    print(f"Total Bricks: {result_fixed.metadata['total_bricks']}")
    print(f"Compression Ratio: {result_fixed.metadata['compression_ratio']:.1f}:1")
    print(f"Current Direction: {result_fixed.metadata.get('current_direction', 'Unknown')}")
    print(f"Trend Strength: {result_fixed.metadata['trend_strength']:.1f}/100")
    print(f"\nSignal: {result_fixed.signal.value}")
    print(f"Confidence: {result_fixed.confidence:.1f}%")
    
    if 'consecutive_up' in result_fixed.metadata:
        print(f"\nConsecutive Up Bricks: {result_fixed.metadata['consecutive_up']}")
        print(f"Consecutive Down Bricks: {result_fixed.metadata['consecutive_down']}")
        print(f"Max Consecutive Up: {result_fixed.metadata['max_consecutive_up']}")
        print(f"Max Consecutive Down: {result_fixed.metadata['max_consecutive_down']}")
    
    # Create ATR-based Renko
    print("\n" + "=" * 60)
    print("ATR-BASED RENKO ANALYSIS:")
    print("=" * 60)
    
    renko_atr = RenkoBarsIndicator(
        brick_size=None,  # Use ATR
        atr_period=14,
        atr_multiplier=1.5,
        reversal_bricks=2,  # Require 2 bricks for reversal
        detect_patterns=True
    )
    
    result_atr = renko_atr.calculate(data, "SYNTHETIC")
    
    print(f"Dynamic Brick Size: ${result_atr.metadata['brick_size']:.2f}")
    print(f"Total Bricks: {result_atr.metadata['total_bricks']}")
    print(f"Compression Ratio: {result_atr.metadata['compression_ratio']:.1f}:1")
    print(f"\nUp/Down Ratio: {result_atr.metadata.get('up_ratio', 0.5):.2%}")
    
    # Show patterns
    if 'recent_patterns' in result_atr.metadata:
        print("\nRecent Patterns Detected:")
        for pattern in result_atr.metadata['recent_patterns']:
            print(f"  {pattern['type']} ({pattern['strength']}) - {pattern['bricks_ago']} bricks ago")
    
    # Show support/resistance
    if 'resistance_level' in result_atr.metadata:
        print(f"\nResistance Level: ${result_atr.metadata['resistance_level']:.2f}")
    if 'support_level' in result_atr.metadata:
        print(f"Support Level: ${result_atr.metadata['support_level']:.2f}")
    
    # Trading recommendation
    print(f"\nRecommendation: {result_atr.metadata.get('recommendation', 'No recommendation')}")
    
    print("\n💡 Renko Trading Tips:")
    print("- Fixed brick size: Better for stable volatility markets")
    print("- ATR-based: Adapts to changing volatility")
    print("- Consecutive bricks indicate trend strength")
    print("- Reversals after extended trends are high probability")
    print("- Combine with volume for confirmation")
    print("- Renko removes time distortion - focus on price alone")


if __name__ == "__main__":
    demonstrate_renko_bars()