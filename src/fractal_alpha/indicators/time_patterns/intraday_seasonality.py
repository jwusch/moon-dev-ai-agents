"""
🌞 Intraday Seasonality Pattern - Time-Based Market Rhythms
Detects recurring patterns and anomalies based on time of day
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, time
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class IntradaySeasonalityIndicator(BaseIndicator):
    """
    Intraday Seasonality Pattern detector for market rhythms
    
    This indicator identifies:
    - Volume patterns by hour/minute
    - Volatility cycles throughout the day
    - Return patterns at specific times
    - Opening/closing auction effects
    - Lunch hour dynamics
    - Option expiry patterns
    
    Key patterns detected:
    - Opening volatility (9:30-10:00 AM)
    - Mid-morning reversal (10:30 AM)
    - Lunch lull (12:00-1:00 PM)
    - Afternoon trend (2:00-3:30 PM)
    - Closing imbalances (3:30-4:00 PM)
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 time_buckets: int = 48,  # 30-minute buckets
                 min_samples: int = 5,
                 volume_threshold: float = 1.5,
                 volatility_threshold: float = 2.0,
                 detect_anomalies: bool = True):
        """
        Initialize Intraday Seasonality indicator
        
        Args:
            lookback_days: Days of history to analyze patterns
            time_buckets: Number of time buckets per day (48 = 30min)
            min_samples: Minimum samples per bucket for significance
            volume_threshold: Volume spike threshold (std devs)
            volatility_threshold: Volatility spike threshold
            detect_anomalies: Whether to detect unusual patterns
        """
        super().__init__(
            name="IntradaySeasonality",
            timeframe=TimeFrame.MINUTE_5,  # Need intraday data
            lookback_periods=lookback_days * 24 * 12,  # 5-min bars
            params={
                'lookback_days': lookback_days,
                'time_buckets': time_buckets,
                'min_samples': min_samples,
                'volume_threshold': volume_threshold,
                'volatility_threshold': volatility_threshold,
                'detect_anomalies': detect_anomalies
            }
        )
        
        self.lookback_days = lookback_days
        self.time_buckets = time_buckets
        self.min_samples = min_samples
        self.volume_threshold = volume_threshold
        self.volatility_threshold = volatility_threshold
        self.detect_anomalies = detect_anomalies
        
        # Trading session times
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate intraday seasonality patterns
        
        Args:
            data: Price/volume data (DataFrame with OHLCV)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with seasonality analysis
        """
        # Convert to DataFrame if needed
        if isinstance(data, list):
            return self._empty_result(symbol)
            
        if len(data) < self.min_samples * self.time_buckets:
            return self._empty_result(symbol)
            
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            return self._empty_result(symbol)
            
        # Calculate time-based features
        data = data.copy()
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['time_bucket'] = (data['hour'] * 60 + data['minute']) // (24 * 60 // self.time_buckets)
        
        # Calculate returns and volatility
        data['returns'] = data['Close'].pct_change() if 'Close' in data else data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(12).std() * np.sqrt(252 * 78)  # Annualized
        
        # Analyze patterns by time bucket
        seasonality_patterns = self._analyze_time_patterns(data)
        
        # Detect current time anomalies
        current_anomalies = self._detect_current_anomalies(data, seasonality_patterns)
        
        # Identify key time zones
        time_zones = self._identify_time_zones(seasonality_patterns)
        
        # Calculate trading opportunity
        opportunity_score = self._calculate_opportunity(
            seasonality_patterns, current_anomalies, time_zones
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            seasonality_patterns, current_anomalies, time_zones, data
        )
        
        # Create metadata
        metadata = self._create_metadata(
            seasonality_patterns, current_anomalies, time_zones, 
            opportunity_score, len(data)
        )
        
        # Get timestamp
        timestamp = int(data.index[-1].timestamp() * 1000)
        
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
    
    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze patterns by time bucket"""
        
        patterns = {}
        
        # Group by time bucket
        time_groups = data.groupby('time_bucket')
        
        for bucket in range(self.time_buckets):
            if bucket not in time_groups.groups:
                continue
                
            bucket_data = time_groups.get_group(bucket)
            
            if len(bucket_data) < self.min_samples:
                continue
                
            # Volume analysis
            volume_col = 'Volume' if 'Volume' in data else 'volume'
            avg_volume = bucket_data[volume_col].mean()
            std_volume = bucket_data[volume_col].std()
            
            # Return analysis
            avg_return = bucket_data['returns'].mean()
            std_return = bucket_data['returns'].std()
            skew_return = bucket_data['returns'].skew()
            
            # Volatility analysis
            avg_volatility = bucket_data['volatility'].mean()
            std_volatility = bucket_data['volatility'].std()
            
            # Win rate (positive returns)
            win_rate = (bucket_data['returns'] > 0).mean()
            
            # Store pattern
            patterns[bucket] = {
                'time_start': self._bucket_to_time(bucket)[0],
                'time_end': self._bucket_to_time(bucket)[1],
                'avg_volume': avg_volume,
                'std_volume': std_volume,
                'avg_return': avg_return,
                'std_return': std_return,
                'skew_return': skew_return,
                'avg_volatility': avg_volatility,
                'std_volatility': std_volatility,
                'win_rate': win_rate,
                'sample_size': len(bucket_data)
            }
        
        return patterns
    
    def _detect_current_anomalies(self, data: pd.DataFrame, patterns: Dict) -> Dict:
        """Detect anomalies in current time period"""
        
        if len(data) == 0:
            return {}
            
        # Get current time bucket
        current_time = data.index[-1]
        current_bucket = (current_time.hour * 60 + current_time.minute) // (24 * 60 // self.time_buckets)
        
        if current_bucket not in patterns:
            return {}
            
        pattern = patterns[current_bucket]
        
        # Get recent data for this bucket (last few occurrences)
        bucket_mask = data['time_bucket'] == current_bucket
        recent_bucket_data = data[bucket_mask].tail(5)
        
        if len(recent_bucket_data) == 0:
            return {}
            
        anomalies = {}
        
        # Volume anomaly
        volume_col = 'Volume' if 'Volume' in data else 'volume'
        current_volume = recent_bucket_data[volume_col].iloc[-1]
        volume_zscore = (current_volume - pattern['avg_volume']) / (pattern['std_volume'] + 1e-8)
        
        if abs(volume_zscore) > self.volume_threshold:
            anomalies['volume_anomaly'] = {
                'current': current_volume,
                'expected': pattern['avg_volume'],
                'zscore': volume_zscore,
                'type': 'high' if volume_zscore > 0 else 'low'
            }
        
        # Volatility anomaly
        current_volatility = recent_bucket_data['volatility'].iloc[-1]
        if not np.isnan(current_volatility):
            vol_zscore = (current_volatility - pattern['avg_volatility']) / (pattern['std_volatility'] + 1e-8)
            
            if abs(vol_zscore) > self.volatility_threshold:
                anomalies['volatility_anomaly'] = {
                    'current': current_volatility,
                    'expected': pattern['avg_volatility'],
                    'zscore': vol_zscore,
                    'type': 'high' if vol_zscore > 0 else 'low'
                }
        
        # Return pattern anomaly
        recent_returns = recent_bucket_data['returns'].dropna()
        if len(recent_returns) > 0:
            current_return = recent_returns.iloc[-1]
            return_zscore = (current_return - pattern['avg_return']) / (pattern['std_return'] + 1e-8)
            
            if abs(return_zscore) > 2:
                anomalies['return_anomaly'] = {
                    'current': current_return,
                    'expected': pattern['avg_return'],
                    'zscore': return_zscore,
                    'type': 'unusual_gain' if return_zscore > 0 else 'unusual_loss'
                }
        
        return anomalies
    
    def _identify_time_zones(self, patterns: Dict) -> Dict:
        """Identify key trading time zones"""
        
        zones = {
            'opening': {'buckets': [], 'characteristics': {}},
            'mid_morning': {'buckets': [], 'characteristics': {}},
            'lunch': {'buckets': [], 'characteristics': {}},
            'afternoon': {'buckets': [], 'characteristics': {}},
            'closing': {'buckets': [], 'characteristics': {}}
        }
        
        # Classify buckets into zones
        for bucket, pattern in patterns.items():
            hour = pattern['time_start'].hour
            minute = pattern['time_start'].minute
            
            if hour == 9 and minute >= 30 or hour == 10 and minute < 30:
                zones['opening']['buckets'].append(bucket)
            elif hour == 10 and minute >= 30 or hour == 11:
                zones['mid_morning']['buckets'].append(bucket)
            elif hour == 12 or hour == 13 and minute < 30:
                zones['lunch']['buckets'].append(bucket)
            elif hour >= 14 and hour < 15 or hour == 15 and minute < 30:
                zones['afternoon']['buckets'].append(bucket)
            elif hour == 15 and minute >= 30 or hour == 16:
                zones['closing']['buckets'].append(bucket)
        
        # Analyze zone characteristics
        for zone_name, zone_data in zones.items():
            if not zone_data['buckets']:
                continue
                
            # Aggregate zone patterns
            zone_patterns = [patterns[b] for b in zone_data['buckets'] if b in patterns]
            
            if zone_patterns:
                zones[zone_name]['characteristics'] = {
                    'avg_volatility': np.mean([p['avg_volatility'] for p in zone_patterns]),
                    'avg_volume': np.mean([p['avg_volume'] for p in zone_patterns]),
                    'avg_return': np.mean([p['avg_return'] for p in zone_patterns]),
                    'win_rate': np.mean([p['win_rate'] for p in zone_patterns]),
                    'typical_behavior': self._classify_zone_behavior(zone_patterns)
                }
        
        return zones
    
    def _classify_zone_behavior(self, zone_patterns: List[Dict]) -> str:
        """Classify typical behavior of a time zone"""
        
        avg_return = np.mean([p['avg_return'] for p in zone_patterns])
        avg_volatility = np.mean([p['avg_volatility'] for p in zone_patterns])
        win_rate = np.mean([p['win_rate'] for p in zone_patterns])
        
        if avg_volatility > 0.02:  # 2% annualized
            if win_rate > 0.55:
                return "high_vol_bullish"
            elif win_rate < 0.45:
                return "high_vol_bearish"
            else:
                return "high_vol_neutral"
        else:
            if abs(avg_return) > 0.0005:  # 0.05% average move
                return "directional_" + ("up" if avg_return > 0 else "down")
            else:
                return "low_activity"
    
    def _calculate_opportunity(self, 
                              patterns: Dict,
                              anomalies: Dict,
                              zones: Dict) -> float:
        """Calculate trading opportunity score"""
        
        score = 50.0  # Base score
        
        # Current time analysis
        if patterns:
            current_bucket = max(patterns.keys())
            if current_bucket in patterns:
                pattern = patterns[current_bucket]
                
                # High win rate periods
                if pattern['win_rate'] > 0.6:
                    score += 20
                elif pattern['win_rate'] < 0.4:
                    score += 15  # Contrarian opportunity
                
                # High volatility periods (opportunity)
                if pattern['avg_volatility'] > 0.015:
                    score += 10
        
        # Anomaly bonus
        if anomalies:
            if 'volume_anomaly' in anomalies and anomalies['volume_anomaly']['zscore'] > 2:
                score += 15  # High volume = opportunity
            
            if 'volatility_anomaly' in anomalies:
                score += 10  # Volatility spike = opportunity
        
        # Zone-based adjustments
        current_hour = datetime.now().hour
        if 9 <= current_hour < 10:
            score += 10  # Opening opportunities
        elif 15 <= current_hour < 16:
            score += 10  # Closing opportunities
        
        return min(score, 100)
    
    def _generate_signal(self,
                        patterns: Dict,
                        anomalies: Dict,
                        zones: Dict,
                        data: pd.DataFrame) -> Tuple[SignalType, float, float]:
        """Generate trading signal from seasonality analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        if not patterns:
            return signal, confidence, 50.0
            
        # Get current time bucket
        current_time = data.index[-1]
        current_bucket = (current_time.hour * 60 + current_time.minute) // (24 * 60 // self.time_buckets)
        
        if current_bucket not in patterns:
            return signal, confidence, 50.0
            
        pattern = patterns[current_bucket]
        
        # Base signal on historical patterns
        if pattern['avg_return'] > 0.0002 and pattern['win_rate'] > 0.55:
            signal = SignalType.BUY
            confidence = (pattern['win_rate'] - 0.5) * 200  # Scale win rate to confidence
            
        elif pattern['avg_return'] < -0.0002 and pattern['win_rate'] < 0.45:
            signal = SignalType.SELL
            confidence = (0.5 - pattern['win_rate']) * 200
            
        # Adjust for anomalies
        if anomalies:
            if 'volume_anomaly' in anomalies:
                vol_anomaly = anomalies['volume_anomaly']
                if vol_anomaly['type'] == 'high':
                    confidence *= 1.2  # Higher confidence with volume
                    
            if 'return_anomaly' in anomalies:
                ret_anomaly = anomalies['return_anomaly']
                # Potential reversal after extreme move
                if ret_anomaly['type'] == 'unusual_gain' and signal != SignalType.SELL:
                    signal = SignalType.SELL
                    confidence = 60
                elif ret_anomaly['type'] == 'unusual_loss' and signal != SignalType.BUY:
                    signal = SignalType.BUY
                    confidence = 60
        
        # Zone-specific adjustments
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Opening volatility (9:30-10:00)
        if current_hour == 9 and current_minute >= 30 or current_hour == 10 and current_minute < 15:
            confidence *= 0.8  # Lower confidence during volatile open
            
        # Lunch lull (12:00-13:00)
        elif current_hour == 12:
            confidence *= 0.7  # Lower confidence during lunch
            
        # Power hour (15:00-16:00)
        elif current_hour == 15:
            confidence *= 1.1  # Higher confidence during power hour
        
        confidence = min(confidence, 85)
        
        # Value represents pattern strength (0-100)
        pattern_strength = abs(pattern['avg_return']) / (pattern['std_return'] + 1e-8) * 50
        value = min(pattern_strength + pattern['win_rate'] * 50, 100)
        
        return signal, confidence, value
    
    def _bucket_to_time(self, bucket: int) -> Tuple[time, time]:
        """Convert bucket number to time range"""
        
        minutes_per_bucket = 24 * 60 // self.time_buckets
        start_minutes = bucket * minutes_per_bucket
        end_minutes = (bucket + 1) * minutes_per_bucket
        
        start_hour = start_minutes // 60
        start_minute = start_minutes % 60
        end_hour = end_minutes // 60
        end_minute = end_minutes % 60
        
        return (
            time(start_hour, start_minute),
            time(min(end_hour, 23), min(end_minute, 59))
        )
    
    def _create_metadata(self,
                        patterns: Dict,
                        anomalies: Dict,
                        zones: Dict,
                        opportunity_score: float,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'opportunity_score': opportunity_score,
            'data_points': data_points,
            'patterns_found': len(patterns),
            'anomalies_detected': len(anomalies)
        }
        
        # Add current time analysis
        current_time = datetime.now()
        current_bucket = (current_time.hour * 60 + current_time.minute) // (24 * 60 // self.time_buckets)
        
        if current_bucket in patterns:
            pattern = patterns[current_bucket]
            metadata['current_pattern'] = {
                'time_range': f"{pattern['time_start'].strftime('%H:%M')}-{pattern['time_end'].strftime('%H:%M')}",
                'expected_return': pattern['avg_return'],
                'expected_volatility': pattern['avg_volatility'],
                'historical_win_rate': pattern['win_rate'],
                'sample_size': pattern['sample_size']
            }
        
        # Add anomaly details
        if anomalies:
            metadata['anomalies'] = anomalies
        
        # Add best/worst time periods
        if patterns:
            sorted_by_return = sorted(patterns.items(), key=lambda x: x[1]['avg_return'])
            
            if len(sorted_by_return) >= 3:
                metadata['best_times'] = [
                    {
                        'time': f"{p[1]['time_start'].strftime('%H:%M')}-{p[1]['time_end'].strftime('%H:%M')}",
                        'avg_return': p[1]['avg_return'],
                        'win_rate': p[1]['win_rate']
                    }
                    for p in sorted_by_return[-3:]
                ]
                
                metadata['worst_times'] = [
                    {
                        'time': f"{p[1]['time_start'].strftime('%H:%M')}-{p[1]['time_end'].strftime('%H:%M')}",
                        'avg_return': p[1]['avg_return'],
                        'win_rate': p[1]['win_rate']
                    }
                    for p in sorted_by_return[:3]
                ]
        
        # Add zone summary
        zone_summary = {}
        for zone_name, zone_data in zones.items():
            if zone_data['characteristics']:
                zone_summary[zone_name] = {
                    'behavior': zone_data['characteristics'].get('typical_behavior', 'unknown'),
                    'avg_volatility': zone_data['characteristics'].get('avg_volatility', 0),
                    'win_rate': zone_data['characteristics'].get('win_rate', 0.5)
                }
        
        if zone_summary:
            metadata['zone_analysis'] = zone_summary
        
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
            metadata={'error': 'Insufficient intraday data'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return False
            
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
            
        required = ['close', 'volume'] if 'close' in data.columns else ['Close', 'Volume']
        has_required = all(col in data.columns or col.lower() in data.columns for col in required)
        
        return has_required and len(data) >= self.min_samples * self.time_buckets


def demonstrate_intraday_seasonality():
    """Demonstration of Intraday Seasonality indicator"""
    
    print("🌞 Intraday Seasonality Pattern Demonstration\n")
    
    # Generate synthetic intraday data
    print("Generating synthetic intraday market data...\n")
    
    # Create 20 days of 5-minute data
    dates = pd.date_range('2024-01-01 09:30', '2024-01-20 16:00', freq='5min')
    
    # Filter for market hours only
    market_dates = []
    for d in dates:
        if d.hour >= 9 and d.hour < 16:
            if not (d.hour == 9 and d.minute < 30):
                market_dates.append(d)
    
    dates = pd.DatetimeIndex(market_dates)
    
    # Generate price data with time-based patterns
    np.random.seed(42)
    prices = []
    volumes = []
    
    base_price = 100
    for i, timestamp in enumerate(dates):
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Opening volatility (9:30-10:00)
        if hour == 9 or (hour == 10 and minute < 30):
            volatility = 0.003  # High volatility
            volume_multiplier = 3.0
            drift = 0.0001  # Slight upward bias
            
        # Mid-morning (10:30-11:30)
        elif hour == 10 or hour == 11:
            volatility = 0.0015
            volume_multiplier = 1.5
            drift = -0.00005  # Slight pullback
            
        # Lunch hour (12:00-13:00)
        elif hour == 12:
            volatility = 0.0008  # Low volatility
            volume_multiplier = 0.5  # Low volume
            drift = 0
            
        # Afternoon trend (14:00-15:30)
        elif hour >= 14 and hour < 15:
            volatility = 0.0012
            volume_multiplier = 1.8
            drift = 0.0001  # Afternoon rally
            
        # Closing volatility (15:30-16:00)
        else:
            volatility = 0.002  # Higher volatility
            volume_multiplier = 2.5
            drift = -0.00005
        
        # Add some random anomalies
        if np.random.rand() < 0.05:  # 5% chance of anomaly
            volatility *= 2.5
            volume_multiplier *= 3
        
        # Generate price movement
        if i > 0:
            price_change = drift + volatility * np.random.randn()
            base_price = base_price * (1 + price_change)
        
        prices.append(base_price)
        
        # Generate volume with patterns
        base_volume = 1000000
        volume = base_volume * volume_multiplier * (1 + 0.3 * np.random.randn())
        volumes.append(max(volume, 100000))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + 0.001 * abs(np.random.randn())) for p in prices],
        'Low': [p * (1 - 0.001 * abs(np.random.randn())) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Create indicator
    seasonality_indicator = IntradaySeasonalityIndicator(
        lookback_days=20,
        time_buckets=26,  # 15-minute buckets for market hours
        min_samples=10,
        volume_threshold=2.0,
        volatility_threshold=2.0,
        detect_anomalies=True
    )
    
    # Calculate seasonality
    print("Analyzing intraday patterns...\n")
    result = seasonality_indicator.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("INTRADAY SEASONALITY ANALYSIS:")
    print("=" * 60)
    print(f"Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Pattern Strength: {result.value:.1f}/100")
    print(f"Opportunity Score: {result.metadata.get('opportunity_score', 0):.1f}/100")
    
    # Show current pattern
    if 'current_pattern' in result.metadata:
        pattern = result.metadata['current_pattern']
        print(f"\nCurrent Time Pattern ({pattern['time_range']}):")
        print(f"  Expected Return: {pattern['expected_return']*100:.3f}%")
        print(f"  Expected Volatility: {pattern['expected_volatility']*100:.1f}%")
        print(f"  Historical Win Rate: {pattern['historical_win_rate']*100:.1f}%")
        print(f"  Sample Size: {pattern['sample_size']} periods")
    
    # Show anomalies
    if 'anomalies' in result.metadata:
        print("\nAnomalies Detected:")
        for anomaly_type, details in result.metadata['anomalies'].items():
            print(f"  {anomaly_type}: {details['type']} (Z-score: {details['zscore']:.2f})")
    
    # Show best/worst times
    if 'best_times' in result.metadata:
        print("\nBest Trading Times:")
        for period in result.metadata['best_times']:
            print(f"  {period['time']}: {period['avg_return']*100:.3f}% avg return, "
                  f"{period['win_rate']*100:.1f}% win rate")
    
    if 'worst_times' in result.metadata:
        print("\nWorst Trading Times:")
        for period in result.metadata['worst_times']:
            print(f"  {period['time']}: {period['avg_return']*100:.3f}% avg return, "
                  f"{period['win_rate']*100:.1f}% win rate")
    
    # Show zone analysis
    if 'zone_analysis' in result.metadata:
        print("\nTime Zone Analysis:")
        for zone, analysis in result.metadata['zone_analysis'].items():
            print(f"  {zone.title()}: {analysis['behavior']}, "
                  f"{analysis['avg_volatility']*100:.1f}% vol, "
                  f"{analysis['win_rate']*100:.1f}% win rate")
    
    print("\n💡 Seasonality Trading Tips:")
    print("- Opening (9:30-10:00): High volatility, wait for direction")
    print("- Mid-morning (10:30): Common reversal point")
    print("- Lunch (12:00-13:00): Low volume, avoid trading")
    print("- Power hour (15:00-16:00): Increased activity, positioning")
    print("- Volume/volatility anomalies often precede moves")


if __name__ == "__main__":
    demonstrate_intraday_seasonality()