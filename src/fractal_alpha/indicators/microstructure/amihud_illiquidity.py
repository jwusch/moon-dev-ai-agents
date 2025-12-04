"""
💰 Amihud Illiquidity Ratio - Market Liquidity Measurement
Measures price impact per dollar of trading volume - fundamental liquidity metric
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class AmihudIlliquidityIndicator(BaseIndicator):
    """
    Amihud Illiquidity Ratio measures price impact of volume
    
    ILLIQ = |Return| / Volume (in millions)
    
    Higher ILLIQ indicates:
    - Lower liquidity
    - Higher trading costs
    - More price impact per dollar traded
    - Difficult execution environment
    
    Lower ILLIQ indicates:
    - Higher liquidity
    - Lower trading costs
    - Easier execution
    - More efficient market
    
    Originally developed by Yakov Amihud (2002) and widely used by
    institutional investors for liquidity assessment.
    """
    
    def __init__(self,
                 estimation_window: int = 252,  # 1 year of daily data
                 rolling_window: int = 60,      # 3 months rolling
                 volume_threshold: float = 1000, # Minimum volume filter
                 return_threshold: float = 0.15,  # Maximum return filter (15%)
                 use_log_returns: bool = True):
        """
        Initialize Amihud Illiquidity indicator
        
        Args:
            estimation_window: Window for illiquidity estimation
            rolling_window: Rolling window for trend analysis
            volume_threshold: Minimum volume to include in calculation
            return_threshold: Maximum absolute return to include (outlier filter)
            use_log_returns: Use log returns instead of simple returns
        """
        super().__init__(
            name="AmihudIlliquidity",
            timeframe=TimeFrame.DAILY,  # Originally designed for daily data
            lookback_periods=estimation_window,
            params={
                'estimation_window': estimation_window,
                'rolling_window': rolling_window,
                'volume_threshold': volume_threshold,
                'return_threshold': return_threshold,
                'use_log_returns': use_log_returns
            }
        )
        
        self.estimation_window = estimation_window
        self.rolling_window = rolling_window
        self.volume_threshold = volume_threshold
        self.return_threshold = return_threshold
        self.use_log_returns = use_log_returns
        
        # Storage for illiquidity analysis
        self.price_history: Deque[float] = deque(maxlen=estimation_window)
        self.volume_history: Deque[float] = deque(maxlen=estimation_window)
        self.illiq_history: Deque[float] = deque(maxlen=rolling_window)
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Amihud Illiquidity Ratio
        
        Args:
            data: Price/volume data (preferably daily bars)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with illiquidity analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Use OHLCV data directly for daily calculation
            if len(data) < 20:
                return self._empty_result(symbol)
                
            # Calculate daily returns and use daily volume
            returns_data = []
            volume_data = []
            
            for i in range(1, len(data)):
                if self.use_log_returns:
                    ret = np.log(data.iloc[i]['close'] / data.iloc[i-1]['close'])
                else:
                    ret = (data.iloc[i]['close'] / data.iloc[i-1]['close']) - 1
                    
                vol = data.iloc[i]['volume']
                
                # Apply filters
                if (vol >= self.volume_threshold and 
                    abs(ret) <= self.return_threshold and
                    not np.isnan(ret) and not np.isnan(vol)):
                    returns_data.append(ret)
                    volume_data.append(vol)
                    
        else:
            # Convert tick data to synthetic daily data
            if len(data) < 100:
                return self._empty_result(symbol)
                
            # Aggregate ticks into daily periods
            daily_data = self._aggregate_ticks_to_daily(data)
            returns_data = daily_data['returns']
            volume_data = daily_data['volumes']
            
        if len(returns_data) < 10:
            return self._empty_result(symbol)
            
        # Calculate Amihud illiquidity ratios
        illiq_values = self._calculate_illiquidity_ratios(returns_data, volume_data)
        
        if not illiq_values:
            return self._empty_result(symbol)
            
        # Calculate current and average illiquidity
        current_illiq = illiq_values[-1] if illiq_values else 0
        avg_illiq = np.mean(illiq_values)
        
        # Store in history
        for illiq in illiq_values[-min(len(illiq_values), 5):]:  # Add last few values
            self.illiq_history.append(illiq)
            
        # Calculate additional metrics
        illiq_volatility = self._calculate_illiquidity_volatility()
        liquidity_score = self._calculate_liquidity_score()
        market_efficiency = self._calculate_market_efficiency()
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            current_illiq, avg_illiq, illiq_volatility, liquidity_score
        )
        
        # Create metadata
        metadata = self._create_metadata(
            current_illiq, avg_illiq, illiq_volatility, 
            liquidity_score, market_efficiency, len(returns_data)
        )
        
        # Get latest timestamp
        if isinstance(data, pd.DataFrame):
            timestamp = int(data.index[-1].timestamp() * 1000)
        else:
            timestamp = data[-1].timestamp if data else int(datetime.now().timestamp() * 1000)
        
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
    
    def _aggregate_ticks_to_daily(self, ticks: List[TickData]) -> Dict:
        """Aggregate tick data into daily periods"""
        
        daily_periods = {}
        
        for tick in ticks:
            # Get day key (timestamp in days)
            day_key = tick.timestamp // (24 * 60 * 60 * 1000)
            
            if day_key not in daily_periods:
                daily_periods[day_key] = {
                    'prices': [],
                    'volumes': []
                }
                
            daily_periods[day_key]['prices'].append(tick.price)
            daily_periods[day_key]['volumes'].append(tick.volume)
        
        # Calculate daily data
        daily_closes = []
        daily_volumes = []
        
        for day_key in sorted(daily_periods.keys()):
            day_data = daily_periods[day_key]
            
            # Use last price as close, sum volume
            daily_close = day_data['prices'][-1]
            daily_volume = sum(day_data['volumes'])
            
            daily_closes.append(daily_close)
            daily_volumes.append(daily_volume)
        
        # Calculate returns
        returns = []
        volumes = []
        
        for i in range(1, len(daily_closes)):
            if self.use_log_returns:
                ret = np.log(daily_closes[i] / daily_closes[i-1])
            else:
                ret = (daily_closes[i] / daily_closes[i-1]) - 1
                
            vol = daily_volumes[i]
            
            if (vol >= self.volume_threshold and 
                abs(ret) <= self.return_threshold):
                returns.append(ret)
                volumes.append(vol)
                
        return {'returns': returns, 'volumes': volumes}
    
    def _calculate_illiquidity_ratios(self, 
                                    returns: List[float], 
                                    volumes: List[float]) -> List[float]:
        """Calculate Amihud illiquidity ratios"""
        
        illiq_values = []
        
        for i in range(len(returns)):
            ret = returns[i]
            vol = volumes[i]
            
            if vol > 0:
                # ILLIQ = |Return| / (Volume in millions)
                # Convert volume to millions (assuming volume is in shares)
                volume_millions = vol / 1_000_000
                illiq = abs(ret) / volume_millions
                
                # Cap extreme values
                if illiq < 1000:  # Reasonable upper bound
                    illiq_values.append(illiq)
                    
        return illiq_values
    
    def _calculate_illiquidity_volatility(self) -> float:
        """Calculate volatility of illiquidity ratios"""
        
        if len(self.illiq_history) < 5:
            return 0.0
            
        illiq_values = list(self.illiq_history)
        return np.std(illiq_values)
    
    def _calculate_liquidity_score(self) -> float:
        """Calculate overall liquidity score (0-100)"""
        
        if len(self.illiq_history) < 3:
            return 50.0  # Neutral
            
        # Lower illiquidity = higher liquidity
        avg_illiq = np.mean(list(self.illiq_history))
        
        # Normalize to 0-100 scale (approximate)
        # This would need market-specific calibration in practice
        if avg_illiq < 0.001:
            score = 95  # Very liquid
        elif avg_illiq < 0.01:
            score = 80
        elif avg_illiq < 0.1:
            score = 60
        elif avg_illiq < 1:
            score = 40
        elif avg_illiq < 10:
            score = 20
        else:
            score = 5  # Very illiquid
            
        return float(score)
    
    def _calculate_market_efficiency(self) -> str:
        """Classify market efficiency based on illiquidity"""
        
        if len(self.illiq_history) < 3:
            return "unknown"
            
        avg_illiq = np.mean(list(self.illiq_history))
        
        if avg_illiq < 0.001:
            return "highly_efficient"
        elif avg_illiq < 0.01:
            return "efficient"
        elif avg_illiq < 0.1:
            return "moderately_efficient"
        elif avg_illiq < 1:
            return "inefficient"
        else:
            return "highly_inefficient"
    
    def _generate_signal(self,
                        current_illiq: float,
                        avg_illiq: float,
                        illiq_volatility: float,
                        liquidity_score: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from illiquidity analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        if len(self.illiq_history) < 5:
            return signal, confidence, liquidity_score
            
        # High liquidity (low illiquidity) = good for trading
        if liquidity_score > 70 and current_illiq < avg_illiq * 0.8:
            # Improving liquidity conditions
            signal = SignalType.BUY  # Good environment for trading
            confidence = min(liquidity_score * 0.8, 70)
            
        elif liquidity_score < 30 and current_illiq > avg_illiq * 1.5:
            # Deteriorating liquidity
            signal = SignalType.SELL  # Exit positions, poor trading environment
            confidence = min((100 - liquidity_score) * 0.6, 50)
            
        # Adjust for illiquidity trend
        if len(self.illiq_history) >= 10:
            recent_illiq = list(self.illiq_history)[-10:]
            early_avg = np.mean(recent_illiq[:5])
            late_avg = np.mean(recent_illiq[-5:])
            
            if late_avg > early_avg * 1.2:  # Illiquidity increasing
                confidence *= 0.7  # Less confident in any signals
            elif late_avg < early_avg * 0.8:  # Illiquidity decreasing
                confidence *= 1.2  # More confident
                
        confidence = min(confidence, 85)
        
        # Value represents liquidity quality
        value = liquidity_score
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        current_illiq: float,
                        avg_illiq: float,
                        illiq_volatility: float,
                        liquidity_score: float,
                        market_efficiency: str,
                        data_points: int) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'current_illiquidity': current_illiq,
            'average_illiquidity': avg_illiq,
            'illiquidity_volatility': illiq_volatility,
            'liquidity_score': liquidity_score,
            'market_efficiency': market_efficiency,
            'data_points': data_points,
            'history_length': len(self.illiq_history)
        }
        
        if self.illiq_history:
            illiq_values = list(self.illiq_history)
            metadata.update({
                'min_illiquidity': np.min(illiq_values),
                'max_illiquidity': np.max(illiq_values),
                'median_illiquidity': np.median(illiq_values),
                'percentile_90': np.percentile(illiq_values, 90),
                'percentile_10': np.percentile(illiq_values, 10)
            })
            
            # Illiquidity trend
            if len(illiq_values) >= 10:
                early_avg = np.mean(illiq_values[:len(illiq_values)//2])
                late_avg = np.mean(illiq_values[len(illiq_values)//2:])
                trend_direction = 'increasing' if late_avg > early_avg else 'decreasing'
                trend_magnitude = abs((late_avg - early_avg) / early_avg) if early_avg > 0 else 0
                
                metadata.update({
                    'illiquidity_trend': trend_direction,
                    'trend_magnitude': trend_magnitude
                })
        
        # Trading cost implications
        if current_illiq < 0.001:
            cost_assessment = "very_low"
        elif current_illiq < 0.01:
            cost_assessment = "low"
        elif current_illiq < 0.1:
            cost_assessment = "moderate"
        elif current_illiq < 1:
            cost_assessment = "high"
        else:
            cost_assessment = "very_high"
            
        metadata['trading_cost_assessment'] = cost_assessment
        
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,  # Neutral liquidity
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for Amihud illiquidity calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= 100  # Need enough ticks for daily aggregation
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 20
        return False


def demonstrate_amihud_illiquidity():
    """Demonstration of Amihud Illiquidity indicator"""
    
    print("💰 Amihud Illiquidity Ratio Demonstration\n")
    
    # Generate synthetic daily data with varying liquidity conditions
    np.random.seed(42)
    
    print("Generating synthetic market data with liquidity regimes...\n")
    
    # Create synthetic daily data
    n_days = 100
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Phase 1: High liquidity (first 30 days)
    print("Phase 1: High liquidity market...")
    returns_1 = np.random.normal(0.001, 0.015, 30)  # Low volatility
    volumes_1 = np.random.normal(5_000_000, 1_000_000, 30)  # High volume
    
    # Phase 2: Normal liquidity (middle 40 days)
    print("Phase 2: Normal liquidity...")
    returns_2 = np.random.normal(0.002, 0.025, 40)  # Medium volatility
    volumes_2 = np.random.normal(3_000_000, 800_000, 40)  # Medium volume
    
    # Phase 3: Low liquidity crisis (last 30 days)
    print("Phase 3: Liquidity crisis...")
    returns_3 = np.random.normal(0.003, 0.045, 30)  # High volatility
    volumes_3 = np.random.normal(1_500_000, 500_000, 30)  # Low volume
    
    # Combine phases
    returns = np.concatenate([returns_1, returns_2, returns_3])
    volumes = np.concatenate([volumes_1, volumes_2, volumes_3])
    
    # Generate price series
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices[:-1]],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    }, index=dates)
    
    # Create Amihud indicator
    amihud_indicator = AmihudIlliquidityIndicator(
        estimation_window=60,
        rolling_window=20,
        volume_threshold=500_000
    )
    
    # Process data incrementally
    print("Analyzing illiquidity evolution...\n")
    
    results = []
    for i in range(30, len(data), 10):
        batch = data.iloc[:i]
        result = amihud_indicator.calculate(batch, "TEST")
        results.append(result)
        
        phase = "High Liquidity" if i < 50 else "Normal" if i < 80 else "Crisis"
        
        print(f"Day {i} ({phase}):")
        print(f"  Current Illiquidity: {result.metadata['current_illiquidity']:.6f}")
        print(f"  Average Illiquidity: {result.metadata['average_illiquidity']:.6f}")
        print(f"  Liquidity Score: {result.metadata['liquidity_score']:.1f}/100")
        print(f"  Market Efficiency: {result.metadata['market_efficiency']}")
        print(f"  Trading Cost: {result.metadata['trading_cost_assessment']}")
        print(f"  Signal: {result.signal.value}")
        print(f"  Confidence: {result.confidence:.1f}%")
        print()
    
    # Final comprehensive analysis
    if results:
        final_result = results[-1]
        print("=" * 60)
        print("FINAL AMIHUD ILLIQUIDITY ANALYSIS:")
        print("=" * 60)
        
        print(f"Current Illiquidity: {final_result.metadata['current_illiquidity']:.6f}")
        print(f"Average Illiquidity: {final_result.metadata['average_illiquidity']:.6f}")
        print(f"Liquidity Score: {final_result.metadata['liquidity_score']:.1f}/100")
        print(f"Market Efficiency: {final_result.metadata['market_efficiency']}")
        print(f"Trading Cost Assessment: {final_result.metadata['trading_cost_assessment']}")
        
        if 'illiquidity_trend' in final_result.metadata:
            print(f"Illiquidity Trend: {final_result.metadata['illiquidity_trend']}")
            print(f"Trend Magnitude: {final_result.metadata['trend_magnitude']:.2%}")
        
        print(f"\nIlliquidity Statistics:")
        print(f"  Minimum: {final_result.metadata.get('min_illiquidity', 0):.6f}")
        print(f"  Maximum: {final_result.metadata.get('max_illiquidity', 0):.6f}")
        print(f"  Median: {final_result.metadata.get('median_illiquidity', 0):.6f}")
        print(f"  90th Percentile: {final_result.metadata.get('percentile_90', 0):.6f}")
        
        print(f"\nTrading Recommendation: {final_result.signal.value}")
        print(f"Confidence: {final_result.confidence:.1f}%")
    
    print("\n💡 Amihud Illiquidity Interpretation:")
    print("- Lower illiquidity = Better liquidity, easier trading")
    print("- Higher illiquidity = Poor liquidity, higher costs")
    print("- Illiquidity > 1.0 indicates very poor liquidity")
    print("- Rising illiquidity trend suggests deteriorating conditions")
    print("- Used by institutions for execution cost estimation")
    print("- Originally developed for daily frequency data")


if __name__ == "__main__":
    demonstrate_amihud_illiquidity()