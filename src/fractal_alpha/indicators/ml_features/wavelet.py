"""
🌊 Wavelet Decomposition Analysis - Multi-Scale Signal Processing
Separates signal from noise using time-frequency localization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
import pywt
from scipy import signal

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class WaveletIndicator(BaseIndicator):
    """
    Wavelet decomposition indicator for multi-scale price analysis
    
    Wavelets provide time-frequency localization, unlike Fourier transforms:
    - Capture both frequency and location of patterns
    - Separate trends from noise at multiple scales
    - Identify regime changes and breakpoints
    - Extract cycles without assuming stationarity
    
    This indicator implements:
    - Discrete Wavelet Transform (DWT) for decomposition
    - Multi-resolution analysis (MRA)
    - Wavelet denoising for signal extraction
    - Scalogram analysis for pattern detection
    - Wavelet coherence for correlation dynamics
    """
    
    def __init__(self,
                 wavelet_type: str = 'db4',
                 decomposition_levels: int = 4,
                 denoising_threshold: str = 'soft',
                 min_signal_strength: float = 0.7,
                 analyze_coherence: bool = True,
                 use_packet_decomposition: bool = False):
        """
        Initialize Wavelet indicator
        
        Args:
            wavelet_type: Wavelet family ('db4', 'sym5', 'coif3', 'bior3.5')
            decomposition_levels: Number of decomposition levels
            denoising_threshold: 'soft' or 'hard' thresholding
            min_signal_strength: Minimum strength for signal detection
            analyze_coherence: Analyze wavelet coherence
            use_packet_decomposition: Use wavelet packet for finer analysis
        """
        super().__init__(
            name="Wavelet",
            timeframe=TimeFrame.DAILY,
            lookback_periods=2 ** (decomposition_levels + 2),  # Need enough data for decomposition
            params={
                'wavelet_type': wavelet_type,
                'decomposition_levels': decomposition_levels,
                'denoising_threshold': denoising_threshold,
                'min_signal_strength': min_signal_strength,
                'analyze_coherence': analyze_coherence,
                'use_packet_decomposition': use_packet_decomposition
            }
        )
        
        self.wavelet_type = wavelet_type
        self.decomposition_levels = decomposition_levels
        self.denoising_threshold = denoising_threshold
        self.min_signal_strength = min_signal_strength
        self.analyze_coherence = analyze_coherence
        self.use_packet_decomposition = use_packet_decomposition
        
        # Wavelet characteristics
        self.wavelet_info = {
            'db4': {'name': 'Daubechies 4', 'vanishing_moments': 4, 'smoothness': 'medium'},
            'sym5': {'name': 'Symlet 5', 'vanishing_moments': 5, 'smoothness': 'high'},
            'coif3': {'name': 'Coiflet 3', 'vanishing_moments': 6, 'smoothness': 'high'},
            'bior3.5': {'name': 'Biorthogonal 3.5', 'vanishing_moments': 3, 'smoothness': 'low'}
        }
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate wavelet decomposition and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with wavelet analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
            volumes = data['Volume'].values if 'Volume' in data else data.get('volume', np.ones_like(prices)).values
        else:
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = np.array(data)
            volumes = np.ones_like(prices)
            
        # Perform wavelet decomposition
        decomposition = self._wavelet_decompose(prices)
        
        # Extract signal components
        signal_components = self._extract_signal_components(
            decomposition, prices
        )
        
        # Analyze energy distribution
        energy_analysis = self._analyze_energy_distribution(decomposition)
        
        # Detect patterns at different scales
        pattern_analysis = self._detect_scale_patterns(
            decomposition, signal_components
        )
        
        # Perform denoising
        denoised_signal = self._wavelet_denoise(prices, decomposition)
        
        # Analyze coherence if requested
        coherence_analysis = {}
        if self.analyze_coherence and len(prices) == len(volumes):
            coherence_analysis = self._analyze_wavelet_coherence(
                prices, volumes
            )
        
        # Detect regime changes
        regime_analysis = self._detect_regime_changes(
            decomposition, energy_analysis
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            signal_components, pattern_analysis, regime_analysis,
            energy_analysis, denoised_signal, prices
        )
        
        # Create metadata
        metadata = self._create_metadata(
            decomposition, signal_components, energy_analysis,
            pattern_analysis, regime_analysis, coherence_analysis,
            len(prices)
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
    
    def _wavelet_decompose(self, prices: np.ndarray) -> Dict:
        """Perform discrete wavelet transform decomposition"""
        
        # Use log prices for better decomposition
        log_prices = np.log(prices)
        
        if self.use_packet_decomposition:
            # Wavelet packet decomposition (more detailed)
            wp = pywt.WaveletPacket(log_prices, self.wavelet_type, maxlevel=self.decomposition_levels)
            
            # Extract all nodes at final level
            level_nodes = [node for node in wp.get_level(self.decomposition_levels, 'natural')]
            coefficients = [node.data for node in level_nodes]
            
            decomposition = {
                'type': 'packet',
                'coefficients': coefficients,
                'nodes': level_nodes,
                'tree': wp
            }
        else:
            # Standard DWT decomposition
            coeffs = pywt.wavedec(log_prices, self.wavelet_type, level=self.decomposition_levels)
            
            # Separate approximation and details
            approximation = coeffs[0]
            details = coeffs[1:]
            
            decomposition = {
                'type': 'dwt',
                'approximation': approximation,
                'details': details,
                'coefficients': coeffs,
                'levels': self.decomposition_levels
            }
            
        return decomposition
    
    def _extract_signal_components(self, 
                                  decomposition: Dict,
                                  prices: np.ndarray) -> Dict:
        """Extract meaningful signal components from decomposition"""
        
        components = {}
        
        if decomposition['type'] == 'dwt':
            # Reconstruct signals at each scale
            coeffs = decomposition['coefficients']
            
            # Trend (approximation)
            trend_coeffs = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
            components['trend'] = pywt.waverec(trend_coeffs, self.wavelet_type)
            
            # Details at each scale
            for i, detail in enumerate(decomposition['details']):
                detail_coeffs = [np.zeros_like(coeffs[0])]
                detail_coeffs.extend([np.zeros_like(d) for d in coeffs[1:i+1]])
                if i < len(coeffs) - 1:
                    detail_coeffs.append(detail)
                    detail_coeffs.extend([np.zeros_like(d) for d in coeffs[i+2:]])
                else:
                    detail_coeffs.append(detail)
                    
                reconstructed = pywt.waverec(detail_coeffs, self.wavelet_type)
                components[f'detail_{i+1}'] = reconstructed[:len(prices)]
                
            # Ensure all components have correct length
            for key in components:
                if len(components[key]) > len(prices):
                    components[key] = components[key][:len(prices)]
                    
        else:
            # Packet decomposition
            wp = decomposition['tree']
            
            # Get key frequency bands
            components['low_freq'] = wp['a' * self.decomposition_levels].data
            components['high_freq'] = wp['d' * self.decomposition_levels].data
            
            # Mid-frequency components
            if self.decomposition_levels >= 2:
                components['mid_freq'] = wp['ad'].data + wp['da'].data
                
        # Calculate signal strength for each component
        total_energy = np.sum(prices**2)
        strength_values = {}
        for name, component in list(components.items()):
            if len(component) == len(prices):
                energy = np.sum(component**2)
                strength_values[f'{name}_strength'] = energy / total_energy if total_energy > 0 else 0
        
        # Add strength values
        components.update(strength_values)
                
        return components
    
    def _analyze_energy_distribution(self, decomposition: Dict) -> Dict:
        """Analyze energy distribution across scales"""
        
        energy_dist = {}
        
        if decomposition['type'] == 'dwt':
            # Calculate energy at each level
            total_energy = 0
            level_energies = {}
            
            # Approximation energy
            approx_energy = np.sum(decomposition['approximation']**2)
            level_energies['approximation'] = approx_energy
            total_energy += approx_energy
            
            # Detail energies
            for i, detail in enumerate(decomposition['details']):
                detail_energy = np.sum(detail**2)
                level_energies[f'detail_{i+1}'] = detail_energy
                total_energy += detail_energy
                
            # Energy percentages
            energy_percentages = {}
            for level, energy in level_energies.items():
                energy_percentages[level] = (energy / total_energy * 100) if total_energy > 0 else 0
                
            # Dominant scale
            dominant_scale = max(energy_percentages.items(), key=lambda x: x[1])[0]
            
            # Shannon entropy of energy distribution
            probs = np.array(list(energy_percentages.values())) / 100
            probs = probs[probs > 0]
            energy_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
            
            energy_dist = {
                'total_energy': total_energy,
                'level_energies': level_energies,
                'energy_percentages': energy_percentages,
                'dominant_scale': dominant_scale,
                'energy_entropy': energy_entropy,
                'energy_concentration': max(energy_percentages.values()) if energy_percentages else 0
            }
            
        return energy_dist
    
    def _detect_scale_patterns(self,
                              decomposition: Dict,
                              signal_components: Dict) -> Dict:
        """Detect patterns at different scales"""
        
        patterns = {
            'trend_strength': 0,
            'cycle_detected': False,
            'noise_level': 0,
            'breakpoints': [],
            'dominant_period': 0
        }
        
        if 'trend' in signal_components:
            # Trend strength
            trend = signal_components['trend']
            if len(trend) > 1:
                trend_change = abs(trend[-1] - trend[0])
                trend_volatility = np.std(np.diff(trend))
                patterns['trend_strength'] = trend_change / trend_volatility if trend_volatility > 0 else 0
                
        # Analyze details for cycles
        for i in range(1, min(4, self.decomposition_levels + 1)):
            if f'detail_{i}' in signal_components:
                detail = signal_components[f'detail_{i}']
                
                # Check for cyclic behavior
                if len(detail) > 10:
                    # Autocorrelation to detect periodicity
                    # Ensure detail is 1D array
                    detail_1d = np.asarray(detail).flatten()
                    autocorr = np.correlate(detail_1d, detail_1d, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    autocorr = autocorr / autocorr[0]  # Normalize
                    
                    # Find peaks in autocorrelation
                    peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
                    
                    if len(peaks) > 0:
                        # Dominant period at this scale
                        period = peaks[0] + 1
                        scale_period = period * (2**i)  # Adjust for scale
                        
                        if patterns['dominant_period'] == 0 or scale_period < patterns['dominant_period']:
                            patterns['dominant_period'] = scale_period
                            patterns['cycle_detected'] = True
                            
        # Noise level estimation (highest frequency detail)
        if decomposition['type'] == 'dwt' and len(decomposition['details']) > 0:
            highest_detail = decomposition['details'][-1]
            noise_std = np.std(highest_detail)
            
            # Noise as percentage of signal
            signal_std = np.std(decomposition['approximation'])
            patterns['noise_level'] = (noise_std / signal_std * 100) if signal_std > 0 else 0
            
        # Detect breakpoints using detail coefficients
        if decomposition['type'] == 'dwt':
            for i, detail in enumerate(decomposition['details'][:3]):  # Focus on lower frequency details
                if len(detail) > 3:
                    # Look for large coefficients
                    threshold = 3 * np.std(detail)
                    breakpoint_indices = np.where(np.abs(detail) > threshold)[0]
                    
                    # Scale indices back to original time series
                    scale_factor = len(signal_components.get('trend', [])) // len(detail) if 'trend' in signal_components else 1
                    scaled_indices = breakpoint_indices * scale_factor
                    
                    patterns['breakpoints'].extend(scaled_indices.tolist())
                    
        patterns['breakpoints'] = sorted(list(set(patterns['breakpoints'])))[:5]  # Keep top 5
        
        return patterns
    
    def _wavelet_denoise(self, 
                        prices: np.ndarray,
                        decomposition: Dict) -> np.ndarray:
        """Denoise signal using wavelet thresholding"""
        
        if decomposition['type'] != 'dwt':
            return prices  # Only implemented for DWT
            
        coeffs = decomposition['coefficients'].copy()
        
        # Universal threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # MAD estimator
        threshold = sigma * np.sqrt(2 * np.log(len(prices)))
        
        # Apply thresholding to detail coefficients
        if self.denoising_threshold == 'soft':
            # Soft thresholding
            denoised_coeffs = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                denoised = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                denoised_coeffs.append(denoised)
        else:
            # Hard thresholding
            denoised_coeffs = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                denoised = detail * (np.abs(detail) > threshold)
                denoised_coeffs.append(denoised)
                
        # Reconstruct denoised signal
        denoised_log_prices = pywt.waverec(denoised_coeffs, self.wavelet_type)
        
        # Convert back from log space
        denoised_prices = np.exp(denoised_log_prices[:len(prices)])
        
        return denoised_prices
    
    def _analyze_wavelet_coherence(self,
                                  prices: np.ndarray,
                                  volumes: np.ndarray) -> Dict:
        """Analyze wavelet coherence between price and volume"""
        
        # Simplified coherence analysis
        # In practice, would use continuous wavelet transform
        
        coherence_info = {}
        
        # Normalize series
        norm_prices = (prices - prices.mean()) / prices.std()
        norm_volumes = (volumes - volumes.mean()) / volumes.std()
        
        # Cross-correlation at different scales
        correlations = []
        scales = [2**i for i in range(1, min(5, self.decomposition_levels))]
        
        for scale in scales:
            if scale < len(prices) // 4:
                # Smooth at this scale
                smoothed_prices = pd.Series(norm_prices).rolling(scale).mean().dropna()
                smoothed_volumes = pd.Series(norm_volumes).rolling(scale).mean().dropna()
                
                if len(smoothed_prices) > scale:
                    corr = np.corrcoef(smoothed_prices, smoothed_volumes[:len(smoothed_prices)])[0, 1]
                    correlations.append({
                        'scale': scale,
                        'correlation': corr,
                        'period_days': scale
                    })
                    
        if correlations:
            # Find scale with highest correlation
            max_corr = max(correlations, key=lambda x: abs(x['correlation']))
            
            coherence_info = {
                'scale_correlations': correlations,
                'dominant_coherence_scale': max_corr['scale'],
                'max_correlation': max_corr['correlation'],
                'coherence_type': 'positive' if max_corr['correlation'] > 0 else 'negative'
            }
            
        return coherence_info
    
    def _detect_regime_changes(self,
                              decomposition: Dict,
                              energy_analysis: Dict) -> Dict:
        """Detect regime changes from wavelet coefficients"""
        
        regime_info = {
            'regime_change_detected': False,
            'change_location': None,
            'regime_type': 'stable',
            'confidence': 0
        }
        
        if decomposition['type'] == 'dwt' and len(decomposition['details']) > 0:
            # Use second-level details for regime detection
            if len(decomposition['details']) >= 2:
                detail2 = decomposition['details'][1]
                
                # Sliding window variance
                window = max(3, len(detail2) // 10)
                if len(detail2) > window * 2:
                    variances = []
                    for i in range(len(detail2) - window):
                        window_var = np.var(detail2[i:i+window])
                        variances.append(window_var)
                        
                    variances = np.array(variances)
                    
                    # Detect significant variance changes
                    if len(variances) > 2:
                        var_change = np.abs(np.diff(variances))
                        threshold = 3 * np.std(var_change)
                        
                        change_points = np.where(var_change > threshold)[0]
                        
                        if len(change_points) > 0:
                            # Most recent change
                            recent_change = change_points[-1]
                            
                            # Scale to original time series
                            scale_factor = len(decomposition['details']) * 4  # Approximate
                            change_location = recent_change * scale_factor
                            
                            regime_info['regime_change_detected'] = True
                            regime_info['change_location'] = int(change_location)
                            
                            # Determine regime type based on energy shift
                            if 'energy_entropy' in energy_analysis:
                                if energy_analysis['energy_entropy'] > 1.5:
                                    regime_info['regime_type'] = 'high_complexity'
                                elif energy_analysis['energy_entropy'] < 0.5:
                                    regime_info['regime_type'] = 'trending'
                                else:
                                    regime_info['regime_type'] = 'transitional'
                                    
                            regime_info['confidence'] = min(len(change_points) * 20, 80)
                            
        return regime_info
    
    def _generate_signal(self,
                        signal_components: Dict,
                        pattern_analysis: Dict,
                        regime_analysis: Dict,
                        energy_analysis: Dict,
                        denoised_signal: np.ndarray,
                        original_prices: np.ndarray) -> Tuple[SignalType, float, float]:
        """Generate trading signal from wavelet analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Check denoised signal trend
        if len(denoised_signal) >= 5:
            recent_trend = (denoised_signal[-1] / denoised_signal[-5]) - 1
            
            # Strong trends in denoised signal
            if recent_trend > 0.02:  # 2% up move
                signal = SignalType.BUY
                confidence = 60
            elif recent_trend < -0.02:  # 2% down move
                signal = SignalType.SELL
                confidence = 60
                
        # Adjust for pattern analysis
        if pattern_analysis['cycle_detected'] and pattern_analysis['dominant_period'] > 0:
            # Check position in cycle
            period = int(pattern_analysis['dominant_period'])
            if period < len(original_prices):
                cycle_position = len(original_prices) % period
                cycle_phase = cycle_position / period
                
                if cycle_phase < 0.25:  # Early in cycle
                    if signal == SignalType.HOLD:
                        signal = SignalType.BUY
                        confidence = 50
                    else:
                        confidence += 10
                elif cycle_phase > 0.75:  # Late in cycle
                    if signal == SignalType.BUY:
                        confidence *= 0.8
                        
        # Regime change impacts
        if regime_analysis['regime_change_detected']:
            regime_type = regime_analysis['regime_type']
            
            if regime_type == 'trending':
                # New trend starting
                if signal != SignalType.HOLD:
                    confidence += 15
            elif regime_type == 'high_complexity':
                # Market becoming chaotic
                confidence *= 0.7
                
        # Energy distribution impacts
        if 'dominant_scale' in energy_analysis:
            if energy_analysis['dominant_scale'] == 'approximation':
                # Trend dominates
                if signal != SignalType.HOLD:
                    confidence += 10
            elif 'detail_1' in energy_analysis['dominant_scale']:
                # Short-term volatility dominates
                confidence *= 0.9
                
        # Noise level adjustment
        if pattern_analysis['noise_level'] > 30:
            # High noise - reduce confidence
            confidence *= 0.8
        elif pattern_analysis['noise_level'] < 10:
            # Low noise - increase confidence
            confidence *= 1.1
            
        # Breakpoint detection
        if pattern_analysis['breakpoints']:
            # Recent breakpoint suggests new regime
            if len(original_prices) - pattern_analysis['breakpoints'][-1] < 5:
                confidence += 10
                
        confidence = min(confidence, 85)
        
        # Value represents signal-to-noise ratio
        if pattern_analysis['noise_level'] > 0:
            value = min((100 - pattern_analysis['noise_level']), 100)
        else:
            value = 75
            
        return signal, confidence, value
    
    def _create_metadata(self,
                        decomposition: Dict,
                        signal_components: Dict,
                        energy_analysis: Dict,
                        pattern_analysis: Dict,
                        regime_analysis: Dict,
                        coherence_analysis: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'wavelet_type': self.wavelet_type,
            'decomposition_levels': self.decomposition_levels,
            'signal_components': {}
        }
        
        # Add component strengths
        for name, value in signal_components.items():
            if '_strength' in name:
                metadata['signal_components'][name] = value
                
        # Energy distribution
        if energy_analysis:
            metadata['energy_distribution'] = {
                'dominant_scale': energy_analysis.get('dominant_scale'),
                'energy_entropy': energy_analysis.get('energy_entropy', 0),
                'energy_concentration': energy_analysis.get('energy_concentration', 0)
            }
            
            # Add top energy levels
            if 'energy_percentages' in energy_analysis:
                sorted_energies = sorted(
                    energy_analysis['energy_percentages'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                metadata['top_energy_scales'] = sorted_energies
                
        # Pattern analysis
        metadata['patterns'] = pattern_analysis
        
        # Regime information
        metadata['regime_analysis'] = regime_analysis
        
        # Coherence information
        if coherence_analysis:
            metadata['price_volume_coherence'] = coherence_analysis
            
        # Trading insights
        if regime_analysis['regime_change_detected']:
            metadata['insight'] = f"Regime change detected: {regime_analysis['regime_type'].replace('_', ' ')}"
        elif pattern_analysis['cycle_detected']:
            metadata['insight'] = f"Cyclic pattern detected with period {pattern_analysis['dominant_period']} days"
        elif pattern_analysis['noise_level'] > 30:
            metadata['insight'] = "High noise level - caution advised"
        elif energy_analysis.get('dominant_scale') == 'approximation':
            metadata['insight'] = "Strong trending behavior detected"
        else:
            metadata['insight'] = "Multi-scale analysis shows mixed signals"
            
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
            metadata={'error': 'Insufficient data for wavelet analysis'},
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


def demonstrate_wavelet():
    """Demonstration of Wavelet indicator"""
    
    print("🌊 Wavelet Decomposition Analysis Demonstration\n")
    
    # Generate synthetic data with multiple components
    print("Generating synthetic data with trend, cycles, and noise...\n")
    
    np.random.seed(42)
    n_points = 256  # Power of 2 for wavelet transform
    t = np.linspace(0, 1, n_points)
    
    # Components
    trend = 100 + 20 * t  # Linear trend
    cycle1 = 5 * np.sin(2 * np.pi * 4 * t)  # 4 cycles
    cycle2 = 3 * np.sin(2 * np.pi * 16 * t)  # 16 cycles
    noise = np.random.randn(n_points) * 2
    
    # Regime change
    regime_change = np.zeros(n_points)
    regime_change[n_points//2:] = 10  # Step change
    
    # Combine components
    prices = trend + cycle1 + cycle2 + noise + regime_change
    
    # Add some volume correlation
    volumes = 1000000 + 500000 * np.sin(2 * np.pi * 4 * t + np.pi/4) + 100000 * np.random.randn(n_points)
    volumes = np.abs(volumes)
    
    print("Signal components:")
    print(f"  - Linear trend: 100 → 120")
    print(f"  - Low frequency cycle: 4 periods")
    print(f"  - High frequency cycle: 16 periods")
    print(f"  - Gaussian noise: σ=2")
    print(f"  - Regime change at t=0.5\n")
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': volumes
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='D'))
    
    # Create indicator
    wavelet_indicator = WaveletIndicator(
        wavelet_type='db4',
        decomposition_levels=4,
        denoising_threshold='soft',
        analyze_coherence=True
    )
    
    # Calculate
    result = wavelet_indicator.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("WAVELET ANALYSIS:")
    print("=" * 60)
    
    # Signal components
    print("\nSignal Component Strengths:")
    for name, strength in result.metadata.get('signal_components', {}).items():
        print(f"  {name}: {strength*100:.1f}%")
    
    # Energy distribution
    energy = result.metadata.get('energy_distribution', {})
    if energy:
        print(f"\nEnergy Distribution:")
        print(f"  Dominant Scale: {energy.get('dominant_scale', 'Unknown')}")
        print(f"  Energy Entropy: {energy.get('energy_entropy', 0):.2f}")
        print(f"  Energy Concentration: {energy.get('energy_concentration', 0):.1f}%")
        
        # Top scales
        if 'top_energy_scales' in result.metadata:
            print("\n  Top Energy Scales:")
            for scale, pct in result.metadata['top_energy_scales']:
                print(f"    {scale}: {pct:.1f}%")
    
    # Pattern analysis
    patterns = result.metadata.get('patterns', {})
    if patterns:
        print(f"\nPattern Detection:")
        print(f"  Trend Strength: {patterns.get('trend_strength', 0):.2f}")
        print(f"  Cycle Detected: {patterns.get('cycle_detected', False)}")
        if patterns.get('dominant_period', 0) > 0:
            print(f"  Dominant Period: {patterns['dominant_period']} days")
        print(f"  Noise Level: {patterns.get('noise_level', 0):.1f}%")
        
        if patterns.get('breakpoints'):
            print(f"  Breakpoints Detected: {patterns['breakpoints']}")
    
    # Regime analysis
    regime = result.metadata.get('regime_analysis', {})
    if regime:
        print(f"\nRegime Analysis:")
        print(f"  Regime Change: {regime.get('regime_change_detected', False)}")
        if regime.get('regime_change_detected'):
            print(f"  Change Location: {regime.get('change_location', 'Unknown')}")
            print(f"  Regime Type: {regime.get('regime_type', 'Unknown')}")
            print(f"  Confidence: {regime.get('confidence', 0):.0f}%")
    
    # Price-Volume coherence
    coherence = result.metadata.get('price_volume_coherence', {})
    if coherence:
        print(f"\nPrice-Volume Coherence:")
        print(f"  Dominant Scale: {coherence.get('dominant_coherence_scale', 0)} days")
        print(f"  Max Correlation: {coherence.get('max_correlation', 0):.3f}")
        print(f"  Type: {coherence.get('coherence_type', 'Unknown')}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Signal-to-Noise: {result.value:.1f}/100")
    
    # Insight
    print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    # Test with different wavelet types
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT WAVELET FAMILIES:")
    print("=" * 60)
    
    wavelet_types = ['db4', 'sym5', 'coif3']
    
    for wavelet_type in wavelet_types:
        test_indicator = WaveletIndicator(
            wavelet_type=wavelet_type,
            decomposition_levels=3
        )
        
        test_result = test_indicator.calculate(data, "TEST")
        
        print(f"\n{wavelet_type.upper()} Wavelet:")
        print(f"  Signal: {test_result.signal.value}")
        print(f"  Confidence: {test_result.confidence:.1f}%")
        print(f"  Dominant Scale: {test_result.metadata.get('energy_distribution', {}).get('dominant_scale', 'Unknown')}")
    
    print("\n💡 Wavelet Trading Tips:")
    print("- Use db4 for balanced smoothness and localization")
    print("- Higher decomposition levels = more detail but need more data")
    print("- Low noise level (<10%) indicates clear signals")
    print("- Regime changes often precede major moves")
    print("- Combine with other indicators for confirmation")
    print("- Energy concentration shows where information lies")


if __name__ == "__main__":
    demonstrate_wavelet()