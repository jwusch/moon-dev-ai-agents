"""
🔺 Williams Fractals Indicator
Identifies fractal patterns across multiple timeframes for reversal points
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType, BarData


@dataclass
class FractalPoint:
    """Represents a detected fractal point"""
    timestamp: int
    price: float
    fractal_type: str  # 'high' or 'low'
    strength: int  # Number of bars used (3, 5, 7, etc.)
    

class WilliamsFractalIndicator(BaseIndicator):
    """
    Williams Fractals - identifies turning points in price
    
    A fractal high occurs when:
    - Middle bar has highest high
    - Surrounded by lower highs
    
    A fractal low occurs when:
    - Middle bar has lowest low
    - Surrounded by higher lows
    """
    
    def __init__(self, 
                 timeframe: TimeFrame = TimeFrame.FIVE_MIN,
                 lookback_periods: int = 100,
                 fractal_periods: int = 5,
                 multi_timeframe: bool = True):
        """
        Initialize Williams Fractal indicator
        
        Args:
            timeframe: Primary timeframe
            lookback_periods: Number of periods to analyze
            fractal_periods: Bars for fractal (3, 5, 7, etc. Must be odd)
            multi_timeframe: Enable multi-timeframe confirmation
        """
        super().__init__(
            name="WilliamsFractals",
            timeframe=timeframe,
            lookback_periods=lookback_periods,
            params={
                'fractal_periods': fractal_periods,
                'multi_timeframe': multi_timeframe
            }
        )
        
        if fractal_periods % 2 == 0:
            raise ValueError("fractal_periods must be odd (3, 5, 7, etc.)")
            
        self.fractal_periods = fractal_periods
        self.multi_timeframe = multi_timeframe
        
    def calculate(self, 
                  data: Union[pd.DataFrame, np.ndarray], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Williams Fractals and generate signals
        
        Args:
            data: OHLCV data
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with fractal signals
        """
        if isinstance(data, np.ndarray):
            # Convert to DataFrame for easier handling
            data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            
        # Find fractals
        high_fractals = self._find_fractal_highs(data)
        low_fractals = self._find_fractal_lows(data)
        
        # Get current price
        current_price = data['close'].iloc[-1]
        current_time = int(data.index[-1].timestamp() * 1000) if hasattr(data.index[-1], 'timestamp') else 0
        
        # Analyze fractal patterns
        signal, confidence, value = self._analyze_fractals(
            data, high_fractals, low_fractals, current_price
        )
        
        # Multi-timeframe confirmation if enabled
        if self.multi_timeframe and len(data) > 200:
            mtf_signal = self._multi_timeframe_analysis(data)
            if mtf_signal != signal and mtf_signal != SignalType.HOLD:
                # Reduce confidence if MTF doesn't confirm
                confidence *= 0.7
        
        # Create metadata
        metadata = {
            'high_fractals': len(high_fractals),
            'low_fractals': len(low_fractals),
            'last_high_fractal': high_fractals[-1].price if high_fractals else None,
            'last_low_fractal': low_fractals[-1].price if low_fractals else None,
            'fractal_trend': self._determine_fractal_trend(high_fractals, low_fractals)
        }
        
        return IndicatorResult(
            timestamp=current_time,
            symbol=symbol,
            indicator_name=self.name,
            value=value,
            signal=signal,
            confidence=confidence,
            timeframe=self.timeframe,
            metadata=metadata,
            calculation_time_ms=0
        )
    
    def _find_fractal_highs(self, data: pd.DataFrame) -> List[FractalPoint]:
        """Find all fractal high points"""
        
        fractals = []
        half_period = self.fractal_periods // 2
        
        # Need enough data
        if len(data) < self.fractal_periods:
            return fractals
            
        highs = data['high'].values
        
        for i in range(half_period, len(data) - half_period):
            # Check if this is a fractal high
            is_fractal = True
            center_high = highs[i]
            
            # Check bars before
            for j in range(1, half_period + 1):
                if highs[i - j] >= center_high:
                    is_fractal = False
                    break
                    
            # Check bars after
            if is_fractal:
                for j in range(1, half_period + 1):
                    if highs[i + j] >= center_high:
                        is_fractal = False
                        break
                        
            if is_fractal:
                timestamp = int(data.index[i].timestamp() * 1000) if hasattr(data.index[i], 'timestamp') else i
                fractals.append(FractalPoint(
                    timestamp=timestamp,
                    price=center_high,
                    fractal_type='high',
                    strength=self.fractal_periods
                ))
                
        return fractals
    
    def _find_fractal_lows(self, data: pd.DataFrame) -> List[FractalPoint]:
        """Find all fractal low points"""
        
        fractals = []
        half_period = self.fractal_periods // 2
        
        if len(data) < self.fractal_periods:
            return fractals
            
        lows = data['low'].values
        
        for i in range(half_period, len(data) - half_period):
            # Check if this is a fractal low
            is_fractal = True
            center_low = lows[i]
            
            # Check bars before
            for j in range(1, half_period + 1):
                if lows[i - j] <= center_low:
                    is_fractal = False
                    break
                    
            # Check bars after
            if is_fractal:
                for j in range(1, half_period + 1):
                    if lows[i + j] <= center_low:
                        is_fractal = False
                        break
                        
            if is_fractal:
                timestamp = int(data.index[i].timestamp() * 1000) if hasattr(data.index[i], 'timestamp') else i
                fractals.append(FractalPoint(
                    timestamp=timestamp,
                    price=center_low,
                    fractal_type='low',
                    strength=self.fractal_periods
                ))
                
        return fractals
    
    def _analyze_fractals(self, 
                         data: pd.DataFrame,
                         high_fractals: List[FractalPoint],
                         low_fractals: List[FractalPoint],
                         current_price: float) -> Tuple[SignalType, float, float]:
        """
        Analyze fractal patterns to generate signals
        
        Returns:
            (signal, confidence, value)
        """
        if not high_fractals or not low_fractals:
            return SignalType.HOLD, 0.0, 0.0
            
        # Get recent fractals
        last_high = high_fractals[-1]
        last_low = low_fractals[-1]
        
        # Calculate distances
        dist_to_high = (last_high.price - current_price) / current_price
        dist_to_low = (current_price - last_low.price) / current_price
        
        # Fractal breakout signals
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Recent price action
        recent_high = data['high'].iloc[-5:].max()
        recent_low = data['low'].iloc[-5:].min()
        
        # Bullish fractal breakout
        if current_price > last_high.price and recent_low > last_low.price:
            signal = SignalType.BUY
            confidence = min(80 + dist_to_high * 100, 95)
            
        # Bearish fractal breakdown
        elif current_price < last_low.price and recent_high < last_high.price:
            signal = SignalType.SELL
            confidence = min(80 + dist_to_low * 100, 95)
            
        # Near support (fractal low)
        elif dist_to_low < 0.02 and current_price > last_low.price:
            signal = SignalType.BUY
            confidence = 60 + (0.02 - dist_to_low) * 1000
            
        # Near resistance (fractal high)
        elif dist_to_high < 0.02 and current_price < last_high.price:
            signal = SignalType.SELL
            confidence = 60 + (0.02 - dist_to_high) * 1000
        
        # Calculate indicator value (normalized distance)
        value = (dist_to_high - dist_to_low) * 100
        
        return signal, confidence, value
    
    def _determine_fractal_trend(self, 
                               high_fractals: List[FractalPoint],
                               low_fractals: List[FractalPoint]) -> str:
        """Determine trend based on fractal progression"""
        
        if len(high_fractals) < 2 or len(low_fractals) < 2:
            return "unknown"
            
        # Check if fractals are making higher highs/lows or lower highs/lows
        hh = high_fractals[-1].price > high_fractals[-2].price
        hl = low_fractals[-1].price > low_fractals[-2].price
        
        if hh and hl:
            return "uptrend"
        elif not hh and not hl:
            return "downtrend"
        else:
            return "sideways"
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> SignalType:
        """Analyze fractals on higher timeframe"""
        
        # Resample to higher timeframe (e.g., 5min to 15min)
        resampled = data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(resampled) < self.fractal_periods:
            return SignalType.HOLD
            
        # Find fractals on higher timeframe
        high_fractals = self._find_fractal_highs(resampled)
        low_fractals = self._find_fractal_lows(resampled)
        
        if not high_fractals or not low_fractals:
            return SignalType.HOLD
            
        # Simple trend determination
        current_price = resampled['close'].iloc[-1]
        last_high = high_fractals[-1].price
        last_low = low_fractals[-1].price
        
        if current_price > last_high:
            return SignalType.BUY
        elif current_price < last_low:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate input data has required columns and length"""
        
        if isinstance(data, pd.DataFrame):
            required = ['high', 'low', 'close']
            return all(col in data.columns for col in required) and len(data) >= self.fractal_periods
        else:
            # Assume proper structure for numpy array
            return len(data) >= self.fractal_periods
    
    def get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ['high', 'low', 'close']


def demonstrate_williams_fractals():
    """Demonstration of Williams Fractals"""
    
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("🔺 Williams Fractals Demonstration\n")
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    end = datetime.now()
    start = end - timedelta(days=5)
    
    df = ticker.history(start=start, end=end, interval='5m')
    
    if df.empty:
        print("❌ Could not fetch data")
        return
        
    print(f"Analyzing {len(df)} bars of SPY data...\n")
    
    # Create indicator
    indicator = WilliamsFractalIndicator(
        timeframe=TimeFrame.FIVE_MIN,
        fractal_periods=5,
        multi_timeframe=True
    )
    
    # Calculate
    result = indicator.calculate(df, "SPY")
    
    # Display results
    print(f"Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Value: {result.value:.2f}")
    
    print(f"\nFractal Analysis:")
    print(f"  High Fractals Found: {result.metadata['high_fractals']}")
    print(f"  Low Fractals Found: {result.metadata['low_fractals']}")
    print(f"  Fractal Trend: {result.metadata['fractal_trend']}")
    
    if result.metadata['last_high_fractal']:
        print(f"  Last High Fractal: ${result.metadata['last_high_fractal']:.2f}")
    if result.metadata['last_low_fractal']:
        print(f"  Last Low Fractal: ${result.metadata['last_low_fractal']:.2f}")
    
    print(f"\nCurrent Price: ${df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    demonstrate_williams_fractals()