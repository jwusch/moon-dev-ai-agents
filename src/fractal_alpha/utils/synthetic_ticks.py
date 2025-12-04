"""
🎯 Synthetic Tick Generator
Generates realistic tick data from OHLCV bars for microstructure analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from ..base.types import TickData, BarData

class SyntheticTickGenerator:
    """
    Generates synthetic tick data from OHLCV bars using various methods
    """
    
    def __init__(self, method: str = 'adaptive'):
        """
        Initialize generator
        
        Args:
            method: Generation method ('simple', 'brownian', 'vwap', 'adaptive')
        """
        self.method = method
        self.volume_profiles = {
            'u_shape': self._u_shape_volume,
            'linear': self._linear_volume,
            'momentum': self._momentum_volume
        }
        
    def generate_ticks(self, 
                      bar: Union[BarData, pd.Series], 
                      n_ticks: int = 10,
                      context: Optional[pd.DataFrame] = None) -> List[TickData]:
        """
        Generate synthetic ticks from a single bar
        
        Args:
            bar: OHLCV bar data
            n_ticks: Number of ticks to generate
            context: Previous bars for context
            
        Returns:
            List of synthetic ticks
        """
        if self.method == 'simple':
            return self._generate_simple(bar)
        elif self.method == 'brownian':
            return self._generate_brownian(bar, n_ticks)
        elif self.method == 'vwap':
            return self._generate_vwap_weighted(bar, n_ticks)
        elif self.method == 'adaptive':
            return self._generate_adaptive(bar, n_ticks, context)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_simple(self, bar: Union[BarData, pd.Series]) -> List[TickData]:
        """Simple 4-tick generation (Open, Low, High, Close)"""
        
        # Convert pandas Series to BarData if needed
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        timestamp = bar.timestamp
        ticks = []
        
        # Opening tick - 25% volume
        ticks.append(TickData(
            timestamp=timestamp,
            price=bar.open,
            volume=int(bar.volume * 0.25),
            side=0  # Neutral at open
        ))
        
        # Determine path based on bar characteristics
        if bar.close > bar.open:  # Bullish bar
            # Path: Open -> Low -> High -> Close
            
            # Low tick - 25% volume
            ticks.append(TickData(
                timestamp=timestamp + 15000,  # +15 seconds
                price=bar.low,
                volume=int(bar.volume * 0.25),
                side=-1  # Selling to reach low
            ))
            
            # High tick - 25% volume
            ticks.append(TickData(
                timestamp=timestamp + 30000,  # +30 seconds
                price=bar.high,
                volume=int(bar.volume * 0.25),
                side=1  # Buying to reach high
            ))
            
        else:  # Bearish bar
            # Path: Open -> High -> Low -> Close
            
            # High tick - 25% volume
            ticks.append(TickData(
                timestamp=timestamp + 15000,
                price=bar.high,
                volume=int(bar.volume * 0.25),
                side=1  # Buying to reach high
            ))
            
            # Low tick - 25% volume
            ticks.append(TickData(
                timestamp=timestamp + 30000,
                price=bar.low,
                volume=int(bar.volume * 0.25),
                side=-1  # Selling to reach low
            ))
        
        # Close tick - 25% volume
        close_side = 1 if bar.close > bar.open else -1
        ticks.append(TickData(
            timestamp=timestamp + 59000,  # +59 seconds
            price=bar.close,
            volume=int(bar.volume * 0.25),
            side=close_side
        ))
        
        return ticks
    
    def _generate_brownian(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate using Brownian bridge interpolation"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        # Key points that must be hit
        is_bullish = bar.close > bar.open
        
        if is_bullish:
            # Bullish path: Open -> Low -> High -> Close
            key_points = [
                (0.0, bar.open),
                (0.3, bar.low),
                (0.7, bar.high),
                (1.0, bar.close)
            ]
        else:
            # Bearish path: Open -> High -> Low -> Close
            key_points = [
                (0.0, bar.open),
                (0.3, bar.high),
                (0.7, bar.low),
                (1.0, bar.close)
            ]
        
        # Generate smooth path
        ticks = []
        volatility = (bar.high - bar.low) / bar.open  # Normalized volatility
        
        for i in range(n_ticks):
            t = i / (n_ticks - 1)  # Progress through bar [0, 1]
            
            # Find surrounding key points
            prev_point = max([p for p in key_points if p[0] <= t], key=lambda x: x[0])
            next_point = min([p for p in key_points if p[0] >= t], key=lambda x: x[0])
            
            if prev_point[0] == next_point[0]:
                # Exactly at key point
                price = prev_point[1]
            else:
                # Interpolate between points
                alpha = (t - prev_point[0]) / (next_point[0] - prev_point[0])
                base_price = prev_point[1] + alpha * (next_point[1] - prev_point[1])
                
                # Add Brownian noise
                noise = np.random.normal(0, volatility * bar.open * 0.0001)
                price = base_price + noise
                
                # Constrain to bar range
                price = max(bar.low, min(bar.high, price))
            
            # Determine side based on price movement
            if i == 0:
                side = 0  # Neutral at open
            else:
                side = 1 if price > ticks[-1].price else -1
            
            # Volume with U-shape distribution
            volume_weight = self._u_shape_volume(t)
            volume = int((bar.volume / n_ticks) * volume_weight)
            
            # Create tick
            timestamp = bar.timestamp + int(60000 * t)  # Milliseconds
            
            ticks.append(TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side
            ))
        
        return ticks
    
    def _generate_vwap_weighted(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate with volume-weighted price distribution"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        # Analyze bar characteristics
        range_size = bar.high - bar.low
        body_size = abs(bar.close - bar.open)
        is_trending = body_size > 0.6 * range_size
        is_doji = body_size < 0.1 * range_size
        is_reversal = self._is_reversal_bar(bar)
        
        ticks = []
        
        if is_trending:
            # Trending bar: 70% volume in trend direction
            trend_up = bar.close > bar.open
            
            # Generate directional ticks
            for i in range(n_ticks):
                t = i / (n_ticks - 1)
                
                if trend_up:
                    # Uptrend: more buying pressure
                    if t < 0.7:  # First 70% of time
                        # Moving up
                        price = bar.open + (bar.close - bar.open) * (t / 0.7)
                        side = 1 if np.random.random() < 0.8 else -1  # 80% buys
                        volume_mult = 1.5  # Higher volume
                    else:
                        # Minor pullback
                        price = bar.close - (bar.close - bar.low) * ((t - 0.7) / 0.3)
                        side = -1 if np.random.random() < 0.7 else 1  # 70% sells
                        volume_mult = 0.5  # Lower volume
                else:
                    # Downtrend: more selling pressure
                    if t < 0.7:
                        # Moving down
                        price = bar.open - (bar.open - bar.close) * (t / 0.7)
                        side = -1 if np.random.random() < 0.8 else 1  # 80% sells
                        volume_mult = 1.5
                    else:
                        # Minor bounce
                        price = bar.close + (bar.high - bar.close) * ((t - 0.7) / 0.3)
                        side = 1 if np.random.random() < 0.7 else -1  # 70% buys
                        volume_mult = 0.5
                
                # Apply volume
                base_volume = bar.volume / n_ticks
                volume = int(base_volume * volume_mult)
                
                timestamp = bar.timestamp + int(60000 * t)
                
                ticks.append(TickData(
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    side=side
                ))
                
        elif is_reversal:
            # Reversal bar: Heavy volume at extremes
            ticks = self._generate_reversal_ticks(bar, n_ticks)
            
        else:
            # Range-bound: Volume at boundaries
            ticks = self._generate_range_ticks(bar, n_ticks)
        
        return ticks
    
    def _generate_adaptive(self, 
                          bar: Union[BarData, pd.Series], 
                          n_ticks: int,
                          context: Optional[pd.DataFrame] = None) -> List[TickData]:
        """Adaptive generation based on bar characteristics and context"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        # Analyze bar in context
        bar_type = self._classify_bar(bar, context)
        
        # Choose appropriate method
        if bar_type == 'high_volume_breakout':
            ticks = self._generate_breakout_ticks(bar, n_ticks)
        elif bar_type == 'low_volume_drift':
            ticks = self._generate_drift_ticks(bar, n_ticks)
        elif bar_type == 'volatile_reversal':
            ticks = self._generate_reversal_ticks(bar, n_ticks)
        elif bar_type == 'trend_continuation':
            ticks = self._generate_vwap_weighted(bar, n_ticks)
        else:
            # Default to brownian
            ticks = self._generate_brownian(bar, n_ticks)
        
        # Add microstructure noise
        ticks = self._add_microstructure_effects(ticks, bar)
        
        return ticks
    
    def _generate_reversal_ticks(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate ticks for reversal bars"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        ticks = []
        
        # Determine which extreme was hit first
        high_first = (bar.high - bar.open) > (bar.open - bar.low)
        
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            
            if high_first:
                if t < 0.5:
                    # Move to high with increasing volume
                    price = bar.open + (bar.high - bar.open) * (t / 0.5)
                    side = 1
                    volume_mult = 1 + t  # Volume builds
                else:
                    # Reversal from high to close
                    price = bar.high - (bar.high - bar.close) * ((t - 0.5) / 0.5)
                    side = -1
                    volume_mult = 2 - t  # Volume peaks at reversal
            else:
                if t < 0.5:
                    # Move to low with increasing volume
                    price = bar.open - (bar.open - bar.low) * (t / 0.5)
                    side = -1
                    volume_mult = 1 + t
                else:
                    # Reversal from low to close
                    price = bar.low + (bar.close - bar.low) * ((t - 0.5) / 0.5)
                    side = 1
                    volume_mult = 2 - t
            
            volume = int((bar.volume / n_ticks) * volume_mult)
            timestamp = bar.timestamp + int(60000 * t)
            
            ticks.append(TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side
            ))
        
        return ticks
    
    def _generate_range_ticks(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate ticks for range-bound bars"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        ticks = []
        
        # Price oscillates between support and resistance
        mid_price = (bar.high + bar.low) / 2
        amplitude = (bar.high - bar.low) / 2
        
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            
            # Sine wave pattern
            phase = 2 * math.pi * t * 2  # 2 full cycles
            price = mid_price + amplitude * 0.8 * math.sin(phase)
            
            # Add trend component to reach close
            trend = (bar.close - bar.open) * t
            price += trend
            
            # Constrain to range
            price = max(bar.low, min(bar.high, price))
            
            # Volume higher at extremes
            distance_from_mid = abs(price - mid_price) / amplitude
            volume_mult = 1 + distance_from_mid * 0.5
            
            # Side based on position in cycle
            side = 1 if math.cos(phase) > 0 else -1
            
            volume = int((bar.volume / n_ticks) * volume_mult)
            timestamp = bar.timestamp + int(60000 * t)
            
            ticks.append(TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side
            ))
        
        return ticks
    
    def _generate_breakout_ticks(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate ticks for breakout bars"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        ticks = []
        is_bullish = bar.close > bar.open
        
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            
            if is_bullish:
                # Bullish breakout
                if t < 0.3:
                    # Consolidation phase
                    price = bar.open + (bar.low - bar.open) * (t / 0.3)
                    side = np.random.choice([1, -1])  # Mixed
                    volume_mult = 0.5  # Low volume
                else:
                    # Breakout phase
                    price = bar.low + (bar.close - bar.low) * ((t - 0.3) / 0.7)
                    side = 1 if np.random.random() < 0.9 else -1  # 90% buys
                    volume_mult = 2.0  # High volume
            else:
                # Bearish breakout
                if t < 0.3:
                    # Failed rally
                    price = bar.open + (bar.high - bar.open) * (t / 0.3)
                    side = np.random.choice([1, -1])
                    volume_mult = 0.5
                else:
                    # Breakdown phase
                    price = bar.high - (bar.high - bar.close) * ((t - 0.3) / 0.7)
                    side = -1 if np.random.random() < 0.9 else 1  # 90% sells
                    volume_mult = 2.0
            
            volume = int((bar.volume / n_ticks) * volume_mult)
            timestamp = bar.timestamp + int(60000 * t)
            
            ticks.append(TickData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side
            ))
        
        return ticks
    
    def _generate_drift_ticks(self, bar: Union[BarData, pd.Series], n_ticks: int) -> List[TickData]:
        """Generate ticks for low volume drift bars"""
        
        if isinstance(bar, pd.Series):
            bar = self._series_to_bardata(bar)
            
        ticks = []
        
        # Sparse, random walk-like movement
        current_price = bar.open
        
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            
            # Target price at this point
            target = bar.open + (bar.close - bar.open) * t
            
            # Random walk toward target
            step_size = (bar.high - bar.low) * 0.1
            direction = 1 if target > current_price else -1
            
            # Take step with some randomness
            if np.random.random() < 0.7:  # 70% chance to move toward target
                current_price += direction * step_size * np.random.uniform(0.5, 1.5)
            else:
                # Random movement
                current_price += np.random.uniform(-step_size, step_size)
            
            # Constrain to range
            current_price = max(bar.low, min(bar.high, current_price))
            
            # Low, sporadic volume
            volume = int((bar.volume / n_ticks) * np.random.uniform(0.1, 0.5))
            side = np.random.choice([1, -1])  # Random side
            
            timestamp = bar.timestamp + int(60000 * t)
            
            ticks.append(TickData(
                timestamp=timestamp,
                price=current_price,
                volume=volume,
                side=side
            ))
        
        # Ensure we end at close
        ticks[-1].price = bar.close
        
        return ticks
    
    def _add_microstructure_effects(self, ticks: List[TickData], bar: BarData) -> List[TickData]:
        """Add realistic market microstructure noise"""
        
        # Estimate spread
        avg_price = (bar.high + bar.low) / 2
        volatility = (bar.high - bar.low) / avg_price
        typical_spread = max(0.0001, min(0.001, volatility * 0.1))  # 0.01% to 0.1%
        
        enhanced_ticks = []
        
        for i, tick in enumerate(ticks):
            # Add bid-ask bounce
            half_spread = tick.price * typical_spread / 2
            
            if tick.side == 1:
                # Buy at ask
                tick.price += half_spread
            elif tick.side == -1:
                # Sell at bid
                tick.price -= half_spread
            
            enhanced_ticks.append(tick)
            
            # Occasionally add rapid tick clusters
            if np.random.random() < 0.1:  # 10% chance
                # Generate 2-5 rapid ticks
                n_burst = np.random.randint(2, 6)
                burst_ticks = []
                
                for j in range(n_burst):
                    burst_tick = TickData(
                        timestamp=tick.timestamp + j * 100,  # 100ms apart
                        price=tick.price + np.random.uniform(-half_spread, half_spread),
                        volume=int(tick.volume * np.random.uniform(0.1, 0.3)),
                        side=tick.side
                    )
                    burst_ticks.append(burst_tick)
                
                enhanced_ticks.extend(burst_ticks)
        
        return enhanced_ticks
    
    # Helper methods
    
    def _series_to_bardata(self, series: pd.Series) -> BarData:
        """Convert pandas Series to BarData"""
        
        # Handle both lowercase and capitalized column names
        open_price = series.get('open', series.get('Open', 0))
        high_price = series.get('high', series.get('High', 0))
        low_price = series.get('low', series.get('Low', 0))
        close_price = series.get('close', series.get('Close', 0))
        volume = series.get('volume', series.get('Volume', 0))
        
        # Convert timestamp
        if hasattr(series, 'name') and isinstance(series.name, (datetime, pd.Timestamp)):
            timestamp = int(series.name.timestamp() * 1000)
        else:
            timestamp = int(datetime.now().timestamp() * 1000)
        
        return BarData(
            timestamp=timestamp,
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=int(volume)
        )
    
    def _is_reversal_bar(self, bar: BarData) -> bool:
        """Check if bar shows reversal characteristics"""
        
        # Long wicks indicate reversal
        body_size = abs(bar.close - bar.open)
        upper_wick = bar.high - max(bar.open, bar.close)
        lower_wick = min(bar.open, bar.close) - bar.low
        
        total_range = bar.high - bar.low
        if total_range == 0:
            return False
        
        # Reversal if wicks are much larger than body
        return (upper_wick + lower_wick) > body_size * 2
    
    def _classify_bar(self, bar: BarData, context: Optional[pd.DataFrame] = None) -> str:
        """Classify bar type based on characteristics"""
        
        # Volume analysis
        if context is not None and len(context) > 0:
            avg_volume = context['volume'].mean()
            volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Price movement analysis
        range_size = bar.high - bar.low
        body_size = abs(bar.close - bar.open)
        
        if range_size > 0:
            body_ratio = body_size / range_size
        else:
            body_ratio = 0
        
        # Classification logic
        if volume_ratio > 2 and body_ratio > 0.7:
            return 'high_volume_breakout'
        elif volume_ratio < 0.5:
            return 'low_volume_drift'
        elif self._is_reversal_bar(bar) and volume_ratio > 1.5:
            return 'volatile_reversal'
        elif body_ratio > 0.6:
            return 'trend_continuation'
        else:
            return 'range_bound'
    
    # Volume profile methods
    
    def _u_shape_volume(self, t: float) -> float:
        """U-shaped volume distribution (high at open/close)"""
        return 1.5 - math.cos(2 * math.pi * t)
    
    def _linear_volume(self, t: float) -> float:
        """Linear volume distribution"""
        return 1.0
    
    def _momentum_volume(self, t: float) -> float:
        """Volume builds with momentum"""
        return 0.5 + 1.5 * t
    
    def validate_ticks(self, bar: BarData, ticks: List[TickData]) -> Tuple[bool, Dict[str, bool]]:
        """Validate that synthetic ticks could produce the original bar"""
        
        if not ticks:
            return False, {'no_ticks': False}
        
        prices = [tick.price for tick in ticks]
        total_volume = sum(tick.volume for tick in ticks)
        
        tests = {
            'price_range': (
                min(prices) >= bar.low * 0.9999 and 
                max(prices) <= bar.high * 1.0001
            ),
            'volume_match': abs(total_volume - bar.volume) / bar.volume < 0.01,
            'first_near_open': abs(ticks[0].price - bar.open) / bar.open < 0.001,
            'last_near_close': abs(ticks[-1].price - bar.close) / bar.close < 0.001,
            'time_ordered': all(
                ticks[i].timestamp <= ticks[i+1].timestamp 
                for i in range(len(ticks)-1)
            ),
            'has_extremes': (
                any(abs(t.price - bar.high) / bar.high < 0.001 for t in ticks) and
                any(abs(t.price - bar.low) / bar.low < 0.001 for t in ticks)
            )
        }
        
        return all(tests.values()), tests


def demonstrate_generator():
    """Quick demonstration of the synthetic tick generator"""
    
    # Create sample bar
    bar = BarData(
        timestamp=int(datetime.now().timestamp() * 1000),
        open=100.0,
        high=102.0,
        low=99.5,
        close=101.5,
        volume=10000
    )
    
    # Generate ticks using different methods
    generator = SyntheticTickGenerator()
    
    print("🎯 Synthetic Tick Generation Demo\n")
    print(f"Original Bar: O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}\n")
    
    for method in ['simple', 'brownian', 'vwap', 'adaptive']:
        generator.method = method
        ticks = generator.generate_ticks(bar, n_ticks=10)
        
        print(f"\n{method.upper()} Method ({len(ticks)} ticks):")
        for i, tick in enumerate(ticks[:5]):  # Show first 5
            side_str = "BUY " if tick.side == 1 else "SELL" if tick.side == -1 else "NEUT"
            print(f"  {i+1}: ${tick.price:.2f} | {tick.volume:4d} | {side_str}")
        if len(ticks) > 5:
            print(f"  ... ({len(ticks)-5} more ticks)")
        
        # Validate
        is_valid, tests = generator.validate_ticks(bar, ticks)
        print(f"  Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        if not is_valid:
            failed = [k for k, v in tests.items() if not v]
            print(f"  Failed tests: {failed}")


if __name__ == "__main__":
    demonstrate_generator()