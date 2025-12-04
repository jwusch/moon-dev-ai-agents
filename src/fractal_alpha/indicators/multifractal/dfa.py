"""
🌊 DFA (Detrended Fluctuation Analysis) - Multi-Scale Market Analysis
Detects self-similar patterns and long-range correlations across multiple time scales
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class DFAIndicator(BaseIndicator):
    """
    Detrended Fluctuation Analysis (DFA) for multi-scale market analysis
    
    DFA reveals:
    - Long-range correlations in price movements
    - Multi-fractal characteristics of markets
    - Scale-invariant patterns
    - Persistence vs anti-persistence at different time scales
    
    DFA scaling exponent interpretation:
    - α < 0.5: Anti-persistent (mean-reverting at this scale)
    - α = 0.5: Uncorrelated (random walk)
    - α > 0.5: Persistent (trending at this scale)
    - α = 1.0: 1/f noise (pink noise)
    - α > 1.0: Non-stationary, unbounded
    """
    
    def __init__(self,
                 min_scale: int = 4,
                 max_scale: int = 100,
                 n_scales: int = 20,
                 detrend_order: int = 1,
                 multi_scale: bool = True):
        """
        Initialize DFA indicator
        
        Args:
            min_scale: Minimum box size for analysis
            max_scale: Maximum box size (will be capped by data length)
            n_scales: Number of scales to analyze
            detrend_order: Polynomial order for detrending (1=linear, 2=quadratic)
            multi_scale: Whether to perform multi-fractal DFA
        """
        super().__init__(
            name="DFA",
            timeframe=TimeFrame.DAILY,
            lookback_periods=max_scale * 2,
            params={
                'min_scale': min_scale,
                'max_scale': max_scale,
                'n_scales': n_scales,
                'detrend_order': detrend_order,
                'multi_scale': multi_scale
            }
        )
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.detrend_order = detrend_order
        self.multi_scale = multi_scale
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate DFA scaling exponents
        
        Args:
            data: Price data (DataFrame with OHLCV or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with DFA analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.min_scale * 4:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
        else:
            if len(data) < self.min_scale * 4:
                return self._empty_result(symbol)
            prices = np.array(data)
            
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.min_scale * 4:
            return self._empty_result(symbol)
            
        # Perform DFA analysis
        if self.multi_scale:
            dfa_results = self._multifractal_dfa(returns)
        else:
            dfa_results = self._monofractal_dfa(returns)
            
        # Extract scaling behavior
        scaling_exponent = dfa_results['scaling_exponent']
        
        # Analyze scale-dependent behavior
        scale_analysis = self._analyze_scales(dfa_results)
        
        # Determine market characteristics
        market_state = self._classify_market_state(scaling_exponent, scale_analysis)
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            scaling_exponent, scale_analysis, market_state
        )
        
        # Create metadata
        metadata = self._create_metadata(
            dfa_results, scale_analysis, market_state, len(returns)
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
    
    def _monofractal_dfa(self, returns: np.ndarray) -> Dict:
        """Perform monofractal DFA analysis"""
        
        # Integrate the series (cumulative sum)
        profile = np.cumsum(returns - np.mean(returns))
        
        # Define scales
        max_scale_actual = min(self.max_scale, len(profile) // 4)
        scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale_actual), self.n_scales).astype(int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        for scale in scales:
            # Number of boxes
            n_boxes = len(profile) // scale
            
            if n_boxes < 2:
                continue
                
            # Calculate fluctuation for each box
            box_flucts = []
            
            for i in range(n_boxes):
                start = i * scale
                end = (i + 1) * scale
                
                if end > len(profile):
                    continue
                    
                # Extract segment
                segment = profile[start:end]
                
                # Detrend
                x = np.arange(len(segment))
                
                if self.detrend_order == 1:
                    # Linear detrending
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                elif self.detrend_order == 2:
                    # Quadratic detrending
                    coeffs = np.polyfit(x, segment, 2)
                    trend = np.polyval(coeffs, x)
                else:
                    # Higher order
                    coeffs = np.polyfit(x, segment, self.detrend_order)
                    trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                detrended = segment - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                
                if fluctuation > 0 and np.isfinite(fluctuation):
                    box_flucts.append(fluctuation)
            
            if box_flucts:
                avg_fluctuation = np.mean(box_flucts)
                fluctuations.append((scale, avg_fluctuation))
        
        if len(fluctuations) < 3:
            return {
                'scaling_exponent': 0.5,
                'scales': scales,
                'fluctuations': [],
                'r_squared': 0,
                'intercept': 0
            }
        
        # Calculate scaling exponent
        log_scales = np.log([f[0] for f in fluctuations])
        log_flucts = np.log([f[1] for f in fluctuations])
        
        # Linear regression
        coeffs = np.polyfit(log_scales, log_flucts, 1)
        scaling_exponent = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate R-squared
        predicted = np.polyval(coeffs, log_scales)
        r_squared = 1 - np.sum((log_flucts - predicted)**2) / np.sum((log_flucts - np.mean(log_flucts))**2)
        
        return {
            'scaling_exponent': scaling_exponent,
            'scales': [f[0] for f in fluctuations],
            'fluctuations': [f[1] for f in fluctuations],
            'r_squared': r_squared,
            'intercept': intercept,
            'log_scales': log_scales,
            'log_fluctuations': log_flucts
        }
    
    def _multifractal_dfa(self, returns: np.ndarray) -> Dict:
        """Perform multifractal DFA analysis"""
        
        # Start with monofractal analysis
        mono_results = self._monofractal_dfa(returns)
        
        # Analyze different moments
        q_values = [-5, -3, -1, 0, 1, 3, 5]  # Different moments
        hq_values = []  # Generalized Hurst exponents
        
        profile = np.cumsum(returns - np.mean(returns))
        
        for q in q_values:
            if q == 0:
                # Special case for q=0 (uses geometric mean)
                h_q = self._calculate_hq_zero(profile)
            else:
                h_q = self._calculate_hq(profile, q)
            
            hq_values.append(h_q)
        
        # Calculate singularity spectrum
        singularity_spectrum = self._calculate_singularity_spectrum(q_values, hq_values)
        
        results = mono_results.copy()
        results.update({
            'q_values': q_values,
            'hq_values': hq_values,
            'singularity_spectrum': singularity_spectrum,
            'multifractal_width': singularity_spectrum.get('width', 0)
        })
        
        return results
    
    def _calculate_hq(self, profile: np.ndarray, q: float) -> float:
        """Calculate generalized Hurst exponent for moment q"""
        
        max_scale = min(self.max_scale, len(profile) // 4)
        scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale), min(self.n_scales, 10)).astype(int)
        scales = np.unique(scales)
        
        fluctuations_q = []
        
        for scale in scales:
            n_boxes = len(profile) // scale
            
            if n_boxes < 2:
                continue
            
            box_flucts = []
            
            for i in range(n_boxes):
                start = i * scale
                end = min((i + 1) * scale, len(profile))
                
                segment = profile[start:end]
                
                if len(segment) < 3:
                    continue
                
                # Detrend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                fluctuation = np.sqrt(np.mean(detrended**2))
                
                if fluctuation > 0:
                    box_flucts.append(fluctuation**q)
            
            if box_flucts:
                if q >= 0:
                    fq = np.mean(box_flucts)**(1/q) if q != 0 else np.exp(np.mean(np.log(box_flucts)))
                else:
                    # For negative q, use harmonic mean
                    fq = len(box_flucts) / np.sum(1 / np.array(box_flucts))**(1/abs(q))
                    
                if fq > 0 and np.isfinite(fq):
                    fluctuations_q.append((scale, fq))
        
        if len(fluctuations_q) < 3:
            return 0.5
        
        # Calculate scaling
        log_scales = np.log([f[0] for f in fluctuations_q])
        log_flucts = np.log([f[1] for f in fluctuations_q])
        
        hq = np.polyfit(log_scales, log_flucts, 1)[0]
        
        return hq
    
    def _calculate_hq_zero(self, profile: np.ndarray) -> float:
        """Special case calculation for q=0"""
        # For q=0, we use logarithmic averaging
        return self._calculate_hq(profile, 0.01)  # Approximate with small q
    
    def _calculate_singularity_spectrum(self, q_values: List[float], hq_values: List[float]) -> Dict:
        """Calculate the singularity spectrum f(α)"""
        
        # Calculate α(q) = d(τ(q))/dq where τ(q) = qh(q) - 1
        tau_q = [q * h - 1 for q, h in zip(q_values, hq_values)]
        
        # Numerical derivative
        alpha_values = []
        for i in range(len(q_values)):
            if i == 0:
                alpha = (tau_q[1] - tau_q[0]) / (q_values[1] - q_values[0])
            elif i == len(q_values) - 1:
                alpha = (tau_q[-1] - tau_q[-2]) / (q_values[-1] - q_values[-2])
            else:
                alpha = (tau_q[i+1] - tau_q[i-1]) / (q_values[i+1] - q_values[i-1])
            alpha_values.append(alpha)
        
        # Calculate f(α) = qα - τ(q)
        f_alpha = [q * a - t for q, a, t in zip(q_values, alpha_values, tau_q)]
        
        # Spectrum width (measure of multifractality)
        width = max(alpha_values) - min(alpha_values)
        
        return {
            'alpha': alpha_values,
            'f_alpha': f_alpha,
            'width': width,
            'alpha_max': max(alpha_values),
            'alpha_min': min(alpha_values)
        }
    
    def _analyze_scales(self, dfa_results: Dict) -> Dict:
        """Analyze behavior at different scales"""
        
        if 'scales' not in dfa_results or len(dfa_results['scales']) < 3:
            return {'crossover_scales': [], 'local_exponents': {}}
        
        scales = dfa_results['scales']
        log_scales = dfa_results.get('log_scales', np.log(scales))
        log_flucts = dfa_results.get('log_fluctuations', [])
        
        # Find crossover points (changes in scaling behavior)
        crossover_scales = []
        local_exponents = {}
        
        # Analyze local scaling in windows
        window_size = max(3, len(scales) // 4)
        
        for i in range(len(scales) - window_size + 1):
            window_scales = log_scales[i:i+window_size]
            window_flucts = log_flucts[i:i+window_size]
            
            if len(window_scales) >= 3:
                local_alpha = np.polyfit(window_scales, window_flucts, 1)[0]
                scale_center = scales[i + window_size // 2]
                local_exponents[scale_center] = local_alpha
        
        # Detect crossovers (significant changes in local scaling)
        if local_exponents:
            exponent_values = list(local_exponents.values())
            scale_keys = list(local_exponents.keys())
            
            for i in range(1, len(exponent_values)):
                if abs(exponent_values[i] - exponent_values[i-1]) > 0.1:
                    crossover_scales.append(scale_keys[i])
        
        return {
            'crossover_scales': crossover_scales,
            'local_exponents': local_exponents,
            'has_crossover': len(crossover_scales) > 0
        }
    
    def _classify_market_state(self, scaling_exponent: float, scale_analysis: Dict) -> Dict:
        """Classify market state based on DFA results"""
        
        # Basic classification from scaling exponent
        if scaling_exponent < 0.4:
            persistence_type = "strongly_anti_persistent"
            market_behavior = "mean_reverting"
        elif scaling_exponent < 0.5:
            persistence_type = "anti_persistent"
            market_behavior = "weakly_mean_reverting"
        elif scaling_exponent < 0.6:
            persistence_type = "random_walk"
            market_behavior = "efficient"
        elif scaling_exponent < 0.7:
            persistence_type = "persistent"
            market_behavior = "trending"
        else:
            persistence_type = "strongly_persistent"
            market_behavior = "strong_trending"
        
        # Check for scale-dependent behavior
        scale_dependent = scale_analysis.get('has_crossover', False)
        
        # Multi-scale characteristics
        if scale_dependent:
            if 'local_exponents' in scale_analysis:
                # Find dominant behavior at different scales
                local_exp = scale_analysis['local_exponents']
                short_scale_exp = np.mean([v for k, v in local_exp.items() if k < 20])
                long_scale_exp = np.mean([v for k, v in local_exp.items() if k >= 20])
                
                if short_scale_exp < 0.5 and long_scale_exp > 0.5:
                    market_behavior = "mean_reverting_short_trending_long"
                elif short_scale_exp > 0.5 and long_scale_exp < 0.5:
                    market_behavior = "trending_short_mean_reverting_long"
        
        return {
            'persistence_type': persistence_type,
            'market_behavior': market_behavior,
            'scale_dependent': scale_dependent,
            'scaling_exponent': scaling_exponent
        }
    
    def _generate_signal(self, 
                        scaling_exponent: float,
                        scale_analysis: Dict,
                        market_state: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from DFA analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        market_behavior = market_state['market_behavior']
        
        # Signal generation based on market behavior
        if market_behavior == "mean_reverting" or market_behavior == "strongly_anti_persistent":
            # Strong mean reversion - contrarian signals
            signal = SignalType.BUY  # Assuming we're looking for reversals
            confidence = min((0.5 - scaling_exponent) * 200, 80)  # Higher confidence for stronger anti-persistence
            
        elif market_behavior == "trending" or market_behavior == "strong_trending":
            # Trending market - momentum signals
            signal = SignalType.BUY  # Follow the trend
            confidence = min((scaling_exponent - 0.5) * 150, 75)
            
        elif market_behavior == "mean_reverting_short_trending_long":
            # Complex multi-scale behavior
            signal = SignalType.BUY
            confidence = 60  # Moderate confidence due to mixed signals
            
        elif market_behavior == "efficient":
            # Random walk - no edge
            signal = SignalType.HOLD
            confidence = 0
        
        # Adjust confidence based on scale consistency
        if not market_state['scale_dependent']:
            confidence *= 1.2  # Higher confidence for consistent behavior
        else:
            confidence *= 0.8  # Lower confidence for scale-dependent behavior
        
        confidence = min(confidence, 85)
        
        # Value represents the scaling exponent (normalized to 0-100)
        value = scaling_exponent * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        dfa_results: Dict,
                        scale_analysis: Dict,
                        market_state: Dict,
                        data_length: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'scaling_exponent': dfa_results['scaling_exponent'],
            'r_squared': dfa_results['r_squared'],
            'persistence_type': market_state['persistence_type'],
            'market_behavior': market_state['market_behavior'],
            'scale_dependent': market_state['scale_dependent'],
            'data_points': data_length,
            'n_scales_analyzed': len(dfa_results.get('scales', []))
        }
        
        # Add scale analysis details
        if 'crossover_scales' in scale_analysis:
            metadata['crossover_scales'] = scale_analysis['crossover_scales']
            metadata['n_crossovers'] = len(scale_analysis['crossover_scales'])
        
        # Add local scaling exponents at key scales
        if 'local_exponents' in scale_analysis and scale_analysis['local_exponents']:
            # Sample at specific scales
            for target_scale in [10, 20, 50, 100]:
                closest_scale = min(scale_analysis['local_exponents'].keys(), 
                                  key=lambda x: abs(x - target_scale))
                if abs(closest_scale - target_scale) < 10:
                    metadata[f'alpha_scale_{target_scale}'] = scale_analysis['local_exponents'][closest_scale]
        
        # Add multifractal analysis if available
        if 'multifractal_width' in dfa_results:
            metadata.update({
                'multifractal': True,
                'multifractal_width': dfa_results['multifractal_width'],
                'q_values': dfa_results.get('q_values', []),
                'hq_values': dfa_results.get('hq_values', [])
            })
            
            # Singularity spectrum characteristics
            if 'singularity_spectrum' in dfa_results:
                spectrum = dfa_results['singularity_spectrum']
                metadata.update({
                    'alpha_max': spectrum.get('alpha_max', 0),
                    'alpha_min': spectrum.get('alpha_min', 0),
                    'spectrum_width': spectrum.get('width', 0)
                })
        
        # Trading implications
        if dfa_results['scaling_exponent'] < 0.5:
            metadata['trading_strategy'] = "mean_reversion"
            metadata['optimal_holding_period'] = "short"
        elif dfa_results['scaling_exponent'] > 0.6:
            metadata['trading_strategy'] = "trend_following"
            metadata['optimal_holding_period'] = "medium_to_long"
        else:
            metadata['trading_strategy'] = "mixed_or_neutral"
            metadata['optimal_holding_period'] = "flexible"
        
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,  # Neutral (random walk)
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for DFA analysis'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            required = ['close'] if 'close' in data.columns else ['Close']
            has_required = any(col in data.columns for col in required)
            return has_required and len(data) >= self.min_scale * 4
        else:
            return len(data) >= self.min_scale * 4


def demonstrate_dfa():
    """Demonstration of DFA indicator"""
    
    print("🌊 DFA (Detrended Fluctuation Analysis) Demonstration\n")
    
    # Generate synthetic data with known characteristics
    np.random.seed(42)
    n_points = 1000
    
    # Create different market regimes
    print("Generating synthetic market data with varying regimes...\n")
    
    # Phase 1: Anti-persistent (mean-reverting)
    print("Phase 1: Mean-reverting market (200 points)...")
    mean_reverting = []
    price = 100
    for i in range(200):
        # Anti-persistent increments
        increment = np.random.randn() * 0.5
        if i > 0:
            # Add mean reversion
            increment -= (price - 100) * 0.05
        price += increment
        mean_reverting.append(price)
    
    # Phase 2: Random walk
    print("Phase 2: Random walk (300 points)...")
    random_walk = []
    for i in range(300):
        price += np.random.randn() * 0.5
        random_walk.append(price)
    
    # Phase 3: Persistent (trending)
    print("Phase 3: Trending market (500 points)...")
    trending = []
    trend = 0.02
    momentum = 0
    for i in range(500):
        # Add momentum
        momentum = 0.8 * momentum + np.random.randn() * 0.3
        price += trend + momentum
        trending.append(price)
    
    # Combine all phases
    full_series = mean_reverting + random_walk + trending
    
    # Create DataFrame
    data = pd.DataFrame({
        'Close': full_series
    }, index=pd.date_range('2024-01-01', periods=len(full_series), freq='D'))
    
    # Create DFA indicator
    dfa_indicator = DFAIndicator(
        min_scale=10,
        max_scale=200,
        n_scales=20,
        detrend_order=1,
        multi_scale=True
    )
    
    # Analyze full series
    print("\nAnalyzing complete time series...\n")
    result_full = dfa_indicator.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("COMPLETE SERIES DFA ANALYSIS:")
    print("=" * 60)
    print(f"Scaling Exponent (α): {result_full.metadata['scaling_exponent']:.3f}")
    print(f"R-squared: {result_full.metadata['r_squared']:.3f}")
    print(f"Persistence Type: {result_full.metadata['persistence_type']}")
    print(f"Market Behavior: {result_full.metadata['market_behavior']}")
    print(f"Scale-Dependent: {result_full.metadata['scale_dependent']}")
    
    if 'crossover_scales' in result_full.metadata and result_full.metadata['crossover_scales']:
        print(f"Crossover Scales: {result_full.metadata['crossover_scales']}")
    
    # Analyze individual phases
    print("\n" + "=" * 60)
    print("PHASE-BY-PHASE ANALYSIS:")
    print("=" * 60)
    
    # Phase 1: Mean-reverting
    phase1_data = pd.DataFrame({'Close': mean_reverting})
    result_phase1 = dfa_indicator.calculate(phase1_data, "PHASE1")
    
    print(f"\nPhase 1 (Mean-Reverting):")
    print(f"  α = {result_phase1.metadata['scaling_exponent']:.3f}")
    print(f"  Behavior: {result_phase1.metadata['market_behavior']}")
    print(f"  Signal: {result_phase1.signal.value}")
    print(f"  Confidence: {result_phase1.confidence:.1f}%")
    
    # Phase 2: Random walk
    phase2_data = pd.DataFrame({'Close': random_walk[100:]})  # Use latter part
    result_phase2 = dfa_indicator.calculate(phase2_data, "PHASE2")
    
    print(f"\nPhase 2 (Random Walk):")
    print(f"  α = {result_phase2.metadata['scaling_exponent']:.3f}")
    print(f"  Behavior: {result_phase2.metadata['market_behavior']}")
    print(f"  Signal: {result_phase2.signal.value}")
    print(f"  Confidence: {result_phase2.confidence:.1f}%")
    
    # Phase 3: Trending
    phase3_data = pd.DataFrame({'Close': trending[100:]})  # Use latter part
    result_phase3 = dfa_indicator.calculate(phase3_data, "PHASE3")
    
    print(f"\nPhase 3 (Trending):")
    print(f"  α = {result_phase3.metadata['scaling_exponent']:.3f}")
    print(f"  Behavior: {result_phase3.metadata['market_behavior']}")
    print(f"  Signal: {result_phase3.signal.value}")
    print(f"  Confidence: {result_phase3.confidence:.1f}%")
    
    # Multifractal analysis
    if 'multifractal_width' in result_full.metadata:
        print("\n" + "=" * 60)
        print("MULTIFRACTAL ANALYSIS:")
        print("=" * 60)
        print(f"Multifractal Width: {result_full.metadata['multifractal_width']:.3f}")
        
        if result_full.metadata['multifractal_width'] > 0.2:
            print("Market exhibits significant multifractal behavior")
            print("Different scales show different persistence characteristics")
        else:
            print("Market is approximately monofractal")
            print("Similar behavior across all scales")
    
    # Trading recommendations
    print("\n" + "=" * 60)
    print("TRADING RECOMMENDATIONS:")
    print("=" * 60)
    print(f"Overall Signal: {result_full.signal.value}")
    print(f"Confidence: {result_full.confidence:.1f}%")
    print(f"Strategy: {result_full.metadata.get('trading_strategy', 'Unknown')}")
    print(f"Optimal Holding Period: {result_full.metadata.get('optimal_holding_period', 'Unknown')}")
    
    print("\n💡 DFA Interpretation Guide:")
    print("- α < 0.5: Anti-persistent (mean-reverting)")
    print("- α = 0.5: Random walk (efficient market)")
    print("- α > 0.5: Persistent (trending)")
    print("- α ≈ 1.0: 1/f noise (complex dynamics)")
    print("- Crossovers indicate regime changes at different scales")
    print("- Wide multifractal spectrum indicates rich market dynamics")


if __name__ == "__main__":
    demonstrate_dfa()