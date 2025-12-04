"""
📊 Dynamic Z-Score Bands - Adaptive Mean Reversion
Volatility-adjusted standard deviation bands for improved risk/reward
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class DynamicZScoreIndicator(BaseIndicator):
    """
    Dynamic Z-Score indicator with volatility-adjusted bands
    
    Traditional z-score assumes constant volatility, but markets exhibit:
    - Volatility clustering (GARCH effects)
    - Regime changes
    - Non-normal distributions
    - Time-varying correlations
    
    This indicator addresses these issues by:
    - Using rolling volatility estimates
    - Adjusting for volatility regimes
    - Incorporating tail risk measures
    - Dynamic threshold adaptation
    """
    
    def __init__(self,
                 lookback_window: int = 20,
                 volatility_window: int = 10,
                 ewma_span: int = 10,
                 min_zscore_threshold: float = 1.5,
                 max_zscore_threshold: float = 3.0,
                 use_robust_stats: bool = True,
                 adapt_to_regime: bool = True):
        """
        Initialize Dynamic Z-Score indicator
        
        Args:
            lookback_window: Period for mean calculation
            volatility_window: Period for volatility estimation
            ewma_span: EWMA span for dynamic adaptation
            min_zscore_threshold: Minimum z-score for signals
            max_zscore_threshold: Maximum z-score (cap)
            use_robust_stats: Use median/MAD instead of mean/std
            adapt_to_regime: Adapt thresholds to market regime
        """
        super().__init__(
            name="DynamicZScore",
            timeframe=TimeFrame.DAILY,
            lookback_periods=max(lookback_window, volatility_window) * 2,
            params={
                'lookback_window': lookback_window,
                'volatility_window': volatility_window,
                'ewma_span': ewma_span,
                'min_zscore_threshold': min_zscore_threshold,
                'max_zscore_threshold': max_zscore_threshold,
                'use_robust_stats': use_robust_stats,
                'adapt_to_regime': adapt_to_regime
            }
        )
        
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.ewma_span = ewma_span
        self.min_zscore_threshold = min_zscore_threshold
        self.max_zscore_threshold = max_zscore_threshold
        self.use_robust_stats = use_robust_stats
        self.adapt_to_regime = adapt_to_regime
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate dynamic z-score and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with dynamic z-score analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
            highs = data['High'].values if 'High' in data else data['high'].values
            lows = data['Low'].values if 'Low' in data else data['low'].values
            volumes = data['Volume'].values if 'Volume' in data else data.get('volume', np.ones_like(prices)).values
        else:
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = np.array(data)
            highs = prices
            lows = prices
            volumes = np.ones_like(prices)
            
        # Calculate returns for volatility estimation
        returns = np.diff(np.log(prices))
        
        # Calculate dynamic statistics
        if self.use_robust_stats:
            center, scale = self._calculate_robust_stats(prices, returns)
        else:
            center, scale = self._calculate_normal_stats(prices, returns)
            
        # Calculate raw z-score
        current_price = prices[-1]
        raw_zscore = (current_price - center) / scale if scale > 0 else 0
        
        # Calculate volatility regime
        volatility_regime = self._analyze_volatility_regime(returns, highs, lows)
        
        # Adjust z-score for regime
        if self.adapt_to_regime:
            adjusted_zscore, thresholds = self._adapt_zscore_to_regime(
                raw_zscore, volatility_regime
            )
        else:
            adjusted_zscore = raw_zscore
            thresholds = {
                'entry': self.min_zscore_threshold,
                'exit': self.max_zscore_threshold
            }
            
        # Calculate dynamic bands
        bands = self._calculate_dynamic_bands(
            center, scale, volatility_regime, thresholds
        )
        
        # Analyze distribution properties
        distribution_analysis = self._analyze_distribution(returns)
        
        # Calculate volume-weighted adjustments
        volume_adjustment = self._calculate_volume_adjustment(
            prices, volumes, raw_zscore
        )
        
        # Generate comprehensive score
        composite_score = self._calculate_composite_score(
            adjusted_zscore, volatility_regime, distribution_analysis,
            volume_adjustment
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            composite_score, thresholds, volatility_regime,
            distribution_analysis
        )
        
        # Create metadata
        metadata = self._create_metadata(
            raw_zscore, adjusted_zscore, composite_score,
            center, scale, bands, thresholds,
            volatility_regime, distribution_analysis,
            volume_adjustment, len(prices)
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
    
    def _calculate_robust_stats(self, 
                               prices: np.ndarray,
                               returns: np.ndarray) -> Tuple[float, float]:
        """Calculate robust statistics (median/MAD)"""
        
        # Use median for center
        center = np.median(prices[-self.lookback_window:])
        
        # Use Median Absolute Deviation for scale
        deviations = np.abs(prices[-self.lookback_window:] - center)
        mad = np.median(deviations)
        
        # Convert MAD to standard deviation equivalent
        # MAD * 1.4826 ≈ std for normal distribution
        scale = mad * 1.4826
        
        return center, scale
    
    def _calculate_normal_stats(self,
                               prices: np.ndarray,
                               returns: np.ndarray) -> Tuple[float, float]:
        """Calculate traditional mean/std statistics"""
        
        # Simple moving average
        center = np.mean(prices[-self.lookback_window:])
        
        # Standard deviation with volatility adjustment
        # Use returns volatility to scale price std
        price_std = np.std(prices[-self.lookback_window:])
        
        # EWMA volatility for recent conditions
        ewma_weights = np.exp(-np.arange(self.volatility_window) / self.ewma_span)
        ewma_weights /= ewma_weights.sum()
        
        if len(returns) >= self.volatility_window:
            recent_vol = np.sqrt(np.sum(ewma_weights * returns[-self.volatility_window:]**2))
            historical_vol = np.std(returns[-self.lookback_window:])
            
            # Volatility adjustment factor
            vol_adjustment = recent_vol / historical_vol if historical_vol > 0 else 1
            scale = price_std * vol_adjustment
        else:
            scale = price_std
            
        return center, scale
    
    def _analyze_volatility_regime(self,
                                  returns: np.ndarray,
                                  highs: np.ndarray,
                                  lows: np.ndarray) -> Dict:
        """Analyze current volatility regime"""
        
        if len(returns) < self.volatility_window:
            return {
                'regime': 'unknown',
                'volatility_percentile': 50,
                'volatility_trend': 'stable'
            }
            
        # Calculate various volatility measures
        
        # 1. Standard deviation
        recent_vol = np.std(returns[-self.volatility_window:])
        historical_vol = np.std(returns[-self.lookback_window:])
        
        # 2. High-Low volatility (Parkinson)
        hl_vol = np.sqrt(np.mean(np.log(highs[-self.volatility_window:] / lows[-self.volatility_window:])**2) / (4 * np.log(2)))
        
        # 3. GARCH-style volatility clustering
        squared_returns = returns**2
        vol_autocorr = np.corrcoef(
            squared_returns[-self.volatility_window:-1],
            squared_returns[-self.volatility_window+1:]
        )[0, 1] if len(returns) > self.volatility_window else 0
        
        # Volatility percentile
        all_vols = [np.std(returns[i:i+self.volatility_window]) 
                   for i in range(len(returns) - self.volatility_window)]
        
        if all_vols:
            vol_percentile = stats.percentileofscore(all_vols, recent_vol)
        else:
            vol_percentile = 50
            
        # Determine regime
        if vol_percentile > 80:
            regime = 'high_volatility'
        elif vol_percentile < 20:
            regime = 'low_volatility'
        elif vol_autocorr > 0.3:
            regime = 'clustered_volatility'
        else:
            regime = 'normal_volatility'
            
        # Volatility trend
        if recent_vol > historical_vol * 1.2:
            vol_trend = 'increasing'
        elif recent_vol < historical_vol * 0.8:
            vol_trend = 'decreasing'
        else:
            vol_trend = 'stable'
            
        return {
            'regime': regime,
            'current_volatility': recent_vol,
            'historical_volatility': historical_vol,
            'hl_volatility': hl_vol,
            'volatility_percentile': vol_percentile,
            'volatility_trend': vol_trend,
            'clustering_coefficient': vol_autocorr
        }
    
    def _adapt_zscore_to_regime(self,
                                raw_zscore: float,
                                volatility_regime: Dict) -> Tuple[float, Dict]:
        """Adapt z-score and thresholds to volatility regime"""
        
        regime = volatility_regime['regime']
        vol_percentile = volatility_regime['volatility_percentile']
        
        # Adjust z-score based on regime
        if regime == 'high_volatility':
            # In high volatility, extremes are more common
            # Scale down z-score to reduce false signals
            adjustment_factor = 0.7 + 0.3 * (100 - vol_percentile) / 100
            adjusted_zscore = raw_zscore * adjustment_factor
            
            # Widen thresholds
            entry_threshold = self.min_zscore_threshold * 1.5
            exit_threshold = self.max_zscore_threshold * 1.2
            
        elif regime == 'low_volatility':
            # In low volatility, small moves are significant
            # Scale up z-score to capture opportunities
            adjustment_factor = 1.3 - 0.3 * vol_percentile / 100
            adjusted_zscore = raw_zscore * adjustment_factor
            
            # Tighten thresholds
            entry_threshold = self.min_zscore_threshold * 0.8
            exit_threshold = self.max_zscore_threshold * 0.9
            
        elif regime == 'clustered_volatility':
            # GARCH effects - expect persistence
            adjusted_zscore = raw_zscore
            
            # Asymmetric thresholds
            if raw_zscore > 0:
                entry_threshold = self.min_zscore_threshold * 1.2
            else:
                entry_threshold = self.min_zscore_threshold * 0.9
            exit_threshold = self.max_zscore_threshold
            
        else:
            # Normal regime
            adjusted_zscore = raw_zscore
            entry_threshold = self.min_zscore_threshold
            exit_threshold = self.max_zscore_threshold
            
        thresholds = {
            'entry': entry_threshold,
            'exit': exit_threshold,
            'stop': exit_threshold * 1.2
        }
        
        return adjusted_zscore, thresholds
    
    def _calculate_dynamic_bands(self,
                                center: float,
                                scale: float,
                                volatility_regime: Dict,
                                thresholds: Dict) -> Dict:
        """Calculate dynamic bands based on current conditions"""
        
        # Base bands
        bands = {
            'center': center,
            'upper_1': center + scale,
            'upper_2': center + 2 * scale,
            'upper_entry': center + thresholds['entry'] * scale,
            'upper_exit': center + thresholds['exit'] * scale,
            'lower_1': center - scale,
            'lower_2': center - 2 * scale,
            'lower_entry': center - thresholds['entry'] * scale,
            'lower_exit': center - thresholds['exit'] * scale
        }
        
        # Add confidence bands based on regime
        if volatility_regime['regime'] == 'high_volatility':
            # Wider confidence bands
            bands['confidence_upper'] = center + 2.5 * scale
            bands['confidence_lower'] = center - 2.5 * scale
        else:
            # Normal confidence bands
            bands['confidence_upper'] = center + 2 * scale
            bands['confidence_lower'] = center - 2 * scale
            
        return bands
    
    def _analyze_distribution(self, returns: np.ndarray) -> Dict:
        """Analyze return distribution properties"""
        
        if len(returns) < 20:
            return {
                'skewness': 0,
                'kurtosis': 3,
                'normality_pvalue': 0.5,
                'is_normal': True
            }
            
        # Calculate moments
        skewness = stats.skew(returns[-self.lookback_window:])
        kurtosis = stats.kurtosis(returns[-self.lookback_window:])
        
        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns[-self.lookback_window:])
            is_normal = jb_pvalue > 0.05
        except:
            jb_pvalue = 0.5
            is_normal = True
            
        # Tail analysis
        left_tail = np.percentile(returns[-self.lookback_window:], 5)
        right_tail = np.percentile(returns[-self.lookback_window:], 95)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_pvalue': jb_pvalue,
            'is_normal': is_normal,
            'left_tail': left_tail,
            'right_tail': right_tail,
            'tail_ratio': abs(left_tail / right_tail) if right_tail != 0 else 1
        }
    
    def _calculate_volume_adjustment(self,
                                    prices: np.ndarray,
                                    volumes: np.ndarray,
                                    zscore: float) -> float:
        """Calculate volume-based adjustment factor"""
        
        if len(volumes) < self.lookback_window:
            return 1.0
            
        # Volume analysis
        recent_volume = np.mean(volumes[-self.volatility_window:])
        historical_volume = np.mean(volumes[-self.lookback_window:])
        
        # Volume trend
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
        
        # High volume at extremes confirms signal
        if abs(zscore) > self.min_zscore_threshold:
            if volume_ratio > 1.5:
                # High volume confirms extreme
                adjustment = 1.2
            elif volume_ratio < 0.5:
                # Low volume suggests false signal
                adjustment = 0.8
            else:
                adjustment = 1.0
        else:
            adjustment = 1.0
            
        return adjustment
    
    def _calculate_composite_score(self,
                                  adjusted_zscore: float,
                                  volatility_regime: Dict,
                                  distribution_analysis: Dict,
                                  volume_adjustment: float) -> Dict:
        """Calculate composite score combining all factors"""
        
        # Base score is adjusted z-score
        base_score = adjusted_zscore
        
        # Volatility regime adjustment
        if volatility_regime['regime'] == 'high_volatility':
            vol_weight = 0.8
        elif volatility_regime['regime'] == 'low_volatility':
            vol_weight = 1.2
        else:
            vol_weight = 1.0
            
        # Distribution adjustment
        if not distribution_analysis['is_normal']:
            # Non-normal distribution - adjust for skew
            if distribution_analysis['skewness'] > 0.5:
                # Positive skew - downward moves more likely
                dist_weight = 1.1 if adjusted_zscore < 0 else 0.9
            elif distribution_analysis['skewness'] < -0.5:
                # Negative skew - upward moves more likely
                dist_weight = 0.9 if adjusted_zscore < 0 else 1.1
            else:
                dist_weight = 1.0
                
            # Fat tails adjustment
            if distribution_analysis['kurtosis'] > 5:
                # Leptokurtic - extremes more likely
                dist_weight *= 0.9
        else:
            dist_weight = 1.0
            
        # Calculate final score
        composite_score = base_score * vol_weight * dist_weight * volume_adjustment
        
        return {
            'raw_score': adjusted_zscore,
            'composite_score': composite_score,
            'vol_weight': vol_weight,
            'dist_weight': dist_weight,
            'volume_weight': volume_adjustment,
            'percentile': stats.norm.cdf(composite_score) * 100
        }
    
    def _generate_signal(self,
                        composite_score: Dict,
                        thresholds: Dict,
                        volatility_regime: Dict,
                        distribution_analysis: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from dynamic z-score analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        score = composite_score['composite_score']
        
        # Entry signals
        if score < -thresholds['entry']:
            # Oversold
            signal = SignalType.BUY
            confidence = min(abs(score) * 20, 80)
            
        elif score > thresholds['entry']:
            # Overbought
            signal = SignalType.SELL
            confidence = min(abs(score) * 20, 80)
            
        # Adjust confidence based on market conditions
        if signal != SignalType.HOLD:
            # Volatility regime adjustment
            if volatility_regime['regime'] == 'high_volatility':
                confidence *= 0.8  # Less confident in volatile markets
            elif volatility_regime['regime'] == 'low_volatility':
                confidence *= 1.1  # More confident in calm markets
                
            # Distribution adjustment
            if not distribution_analysis['is_normal']:
                if distribution_analysis['kurtosis'] > 5:
                    # Fat tails - be cautious
                    confidence *= 0.9
                    
            # Volume confirmation
            volume_weight = composite_score['volume_weight']
            if volume_weight > 1.1:
                confidence *= 1.1  # Volume confirms
            elif volume_weight < 0.9:
                confidence *= 0.9  # Volume diverges
                
        confidence = min(confidence, 85)
        
        # Value represents extremeness of score
        value = min(abs(score) * 20, 100)
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        raw_zscore: float,
                        adjusted_zscore: float,
                        composite_score: Dict,
                        center: float,
                        scale: float,
                        bands: Dict,
                        thresholds: Dict,
                        volatility_regime: Dict,
                        distribution_analysis: Dict,
                        volume_adjustment: float,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'z_scores': {
                'raw': raw_zscore,
                'adjusted': adjusted_zscore,
                'composite': composite_score['composite_score']
            },
            'statistics': {
                'center': center,
                'scale': scale,
                'method': 'robust' if self.use_robust_stats else 'normal'
            },
            'bands': bands,
            'thresholds': thresholds,
            'volatility_regime': volatility_regime,
            'distribution': distribution_analysis,
            'adjustments': {
                'volatility_weight': composite_score['vol_weight'],
                'distribution_weight': composite_score['dist_weight'],
                'volume_weight': volume_adjustment
            },
            'percentile': composite_score['percentile']
        }
        
        # Add trading insights
        if abs(composite_score['composite_score']) > thresholds['exit']:
            metadata['insight'] = f"Extreme deviation ({composite_score['composite_score']:.1f}σ) - reversal imminent"
        elif abs(composite_score['composite_score']) > thresholds['entry']:
            metadata['insight'] = f"Significant deviation - {volatility_regime['regime'].replace('_', ' ')} regime"
        elif volatility_regime['regime'] == 'high_volatility':
            metadata['insight'] = "High volatility regime - wider bands active"
        elif not distribution_analysis['is_normal']:
            metadata['insight'] = "Non-normal distribution detected - adjusted thresholds"
        else:
            metadata['insight'] = "Within normal range"
            
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
            metadata={'error': 'Insufficient data for dynamic z-score calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            required = ['close'] if 'close' in data.columns else ['Close']
            has_required = any(col in data.columns for col in required)
            return has_required and len(data) >= self.lookback_periods
        else:
            return len(data) >= self.lookback_periods


def demonstrate_dynamic_zscore():
    """Demonstration of Dynamic Z-Score indicator"""
    
    print("📊 Dynamic Z-Score Bands Demonstration\n")
    
    # Generate synthetic data with regime changes
    print("Generating synthetic data with volatility regimes...\n")
    
    np.random.seed(42)
    n_points = 200
    
    prices = []
    volumes = []
    price = 100
    
    for i in range(n_points):
        if i < 50:
            # Low volatility regime
            volatility = 0.005
            volume_mult = 1.0
            print_regime = "Low volatility" if i == 0 else None
        elif i < 100:
            # High volatility regime
            volatility = 0.02
            volume_mult = 1.5
            print_regime = "High volatility" if i == 50 else None
        elif i < 150:
            # Mean reversion regime
            volatility = 0.01
            # Mean revert to 100
            drift = (100 - price) * 0.05
            volume_mult = 1.2
            print_regime = "Mean reversion" if i == 100 else None
        else:
            # Trending regime with vol clustering
            volatility = 0.015 + 0.005 * np.sin(i/10)
            drift = 0.001
            volume_mult = 1.0 + 0.5 * np.sin(i/10)
            print_regime = "Volatility clustering" if i == 150 else None
            
        if print_regime:
            print(f"  Period {i}-{min(i+50, n_points)}: {print_regime}")
            
        # Add drift for some periods
        if i >= 150:
            price *= (1 + drift + volatility * np.random.randn())
        elif 100 <= i < 150:
            price = price + drift + volatility * np.random.randn() * price
        else:
            price *= (1 + volatility * np.random.randn())
            
        # Volume with regime characteristics
        volume = 1000000 * volume_mult * (1 + 0.3 * abs(np.random.randn()))
        
        prices.append(price)
        volumes.append(volume)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + 0.001 * abs(np.random.randn())) for p in prices],
        'Low': [p * (1 - 0.001 * abs(np.random.randn())) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='D'))
    
    # Create indicators with different settings
    print("\nTesting Dynamic Z-Score with different configurations...\n")
    
    # 1. Standard dynamic z-score
    standard_zscore = DynamicZScoreIndicator(
        lookback_window=20,
        volatility_window=10,
        use_robust_stats=False,
        adapt_to_regime=True
    )
    
    result_standard = standard_zscore.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("STANDARD DYNAMIC Z-SCORE:")
    print("=" * 60)
    
    # Z-score values
    print(f"\nZ-Score Values:")
    print(f"  Raw Z-Score: {result_standard.metadata['z_scores']['raw']:.3f}")
    print(f"  Adjusted Z-Score: {result_standard.metadata['z_scores']['adjusted']:.3f}")
    print(f"  Composite Score: {result_standard.metadata['z_scores']['composite']:.3f}")
    print(f"  Percentile: {result_standard.metadata['percentile']:.1f}%")
    
    # Volatility regime
    vol_regime = result_standard.metadata['volatility_regime']
    print(f"\nVolatility Regime: {vol_regime['regime'].upper()}")
    print(f"  Current Volatility: {vol_regime['current_volatility']*100:.2f}%")
    print(f"  Historical Volatility: {vol_regime['historical_volatility']*100:.2f}%")
    print(f"  Volatility Percentile: {vol_regime['volatility_percentile']:.1f}%")
    print(f"  Trend: {vol_regime['volatility_trend'].upper()}")
    
    # Thresholds
    thresholds = result_standard.metadata['thresholds']
    print(f"\nAdaptive Thresholds:")
    print(f"  Entry: ±{thresholds['entry']:.2f}σ")
    print(f"  Exit: ±{thresholds['exit']:.2f}σ")
    
    # Trading signal
    print(f"\nSignal: {result_standard.signal.value}")
    print(f"Confidence: {result_standard.confidence:.1f}%")
    
    # 2. Robust statistics version
    print("\n" + "=" * 60)
    print("ROBUST STATISTICS VERSION:")
    print("=" * 60)
    
    robust_zscore = DynamicZScoreIndicator(
        lookback_window=20,
        volatility_window=10,
        use_robust_stats=True,
        adapt_to_regime=True
    )
    
    result_robust = robust_zscore.calculate(data, "SYNTHETIC")
    
    print(f"\nRobust Z-Score: {result_robust.metadata['z_scores']['composite']:.3f}")
    print(f"Method: Median/MAD")
    
    # Distribution analysis
    dist = result_robust.metadata['distribution']
    print(f"\nDistribution Analysis:")
    print(f"  Skewness: {dist['skewness']:.3f}")
    print(f"  Kurtosis: {dist['kurtosis']:.3f}")
    print(f"  Is Normal: {dist['is_normal']}")
    
    # 3. Test on extreme data
    print("\n" + "=" * 60)
    print("TESTING EXTREME SCENARIOS:")
    print("=" * 60)
    
    # Add extreme spike
    extreme_data = data.copy()
    extreme_data.iloc[-1, extreme_data.columns.get_loc('Close')] = 120  # 20% spike
    extreme_data.iloc[-1, extreme_data.columns.get_loc('Volume')] = 5000000  # High volume
    
    result_extreme = standard_zscore.calculate(extreme_data, "EXTREME")
    
    print(f"\nExtreme Spike Analysis:")
    print(f"  Composite Z-Score: {result_extreme.metadata['z_scores']['composite']:.3f}")
    print(f"  Volume Adjustment: {result_extreme.metadata['adjustments']['volume_weight']:.2f}")
    print(f"  Signal: {result_extreme.signal.value}")
    print(f"  Confidence: {result_extreme.confidence:.1f}%")
    
    # Key bands
    bands = result_extreme.metadata['bands']
    current_price = extreme_data['Close'].iloc[-1]
    print(f"\nPrice Bands:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Center: ${bands['center']:.2f}")
    print(f"  Entry Bands: ${bands['lower_entry']:.2f} - ${bands['upper_entry']:.2f}")
    print(f"  Exit Bands: ${bands['lower_exit']:.2f} - ${bands['upper_exit']:.2f}")
    
    # Insight
    print(f"\nInsight: {result_extreme.metadata.get('insight', 'No specific insight')}")
    
    print("\n💡 Dynamic Z-Score Trading Tips:")
    print("- Adapts to volatility regimes automatically")
    print("- Robust stats (median/MAD) handle outliers better")
    print("- Volume confirmation improves signal quality")
    print("- Non-normal distributions get special treatment")
    print("- Composite score combines multiple factors")
    print("- Use wider bands in high volatility periods")


if __name__ == "__main__":
    demonstrate_dynamic_zscore()