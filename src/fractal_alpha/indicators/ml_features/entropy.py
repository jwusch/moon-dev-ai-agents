"""
🔬 Entropy-Based Indicators - Information Flow Detection
Shannon entropy and related measures for regime transition detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class EntropyIndicator(BaseIndicator):
    """
    Entropy-based indicator for information flow and regime detection
    
    Shannon entropy measures the uncertainty/randomness in price movements:
    - High entropy = high uncertainty, random market
    - Low entropy = low uncertainty, trending/predictable market
    - Entropy changes = regime transitions
    
    This indicator implements:
    - Shannon entropy of returns
    - Relative entropy (KL divergence) for regime shifts
    - Approximate entropy for complexity analysis
    - Transfer entropy for directional information flow
    - Entropy rate for predictability measurement
    """
    
    def __init__(self,
                 lookback_periods: int = 50,
                 entropy_window: int = 20,
                 n_bins: int = 10,
                 entropy_types: List[str] = None,
                 min_entropy_change: float = 0.15,
                 use_log_returns: bool = True):
        """
        Initialize Entropy indicator
        
        Args:
            lookback_periods: Periods for baseline entropy
            entropy_window: Rolling window for entropy calculation
            n_bins: Number of bins for discretization
            entropy_types: Types to calculate ['shannon', 'relative', 'approximate', 'transfer']
            min_entropy_change: Minimum change for regime detection
            use_log_returns: Use log returns instead of simple returns
        """
        if entropy_types is None:
            entropy_types = ['shannon', 'relative', 'approximate']
            
        super().__init__(
            name="Entropy",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_periods + entropy_window,
            params={
                'lookback_periods': lookback_periods,
                'entropy_window': entropy_window,
                'n_bins': n_bins,
                'entropy_types': entropy_types,
                'min_entropy_change': min_entropy_change,
                'use_log_returns': use_log_returns
            }
        )
        
        self.lookback_periods = lookback_periods
        self.entropy_window = entropy_window
        self.n_bins = n_bins
        self.entropy_types = entropy_types
        self.min_entropy_change = min_entropy_change
        self.use_log_returns = use_log_returns
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate entropy indicators and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with entropy analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.lookback_periods + self.entropy_window:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
            volumes = data['Volume'].values if 'Volume' in data else data.get('volume', np.ones_like(prices)).values
            highs = data['High'].values if 'High' in data else data['high'].values
            lows = data['Low'].values if 'Low' in data else data['low'].values
        else:
            if len(data) < self.lookback_periods + self.entropy_window:
                return self._empty_result(symbol)
            prices = np.array(data)
            volumes = np.ones_like(prices)
            highs = prices
            lows = prices
            
        # Calculate returns
        if self.use_log_returns:
            returns = np.diff(np.log(prices))
        else:
            returns = np.diff(prices) / prices[:-1]
            
        # Calculate various entropy measures
        entropy_results = {}
        
        if 'shannon' in self.entropy_types:
            shannon_entropy = self._calculate_shannon_entropy(
                returns, self.entropy_window, self.n_bins
            )
            entropy_results['shannon'] = shannon_entropy
            
        if 'relative' in self.entropy_types:
            relative_entropy = self._calculate_relative_entropy(
                returns, self.entropy_window, self.lookback_periods
            )
            entropy_results['relative'] = relative_entropy
            
        if 'approximate' in self.entropy_types:
            approx_entropy = self._calculate_approximate_entropy(
                returns[-self.entropy_window:], m=2, r=0.2
            )
            entropy_results['approximate'] = approx_entropy
            
        if 'transfer' in self.entropy_types:
            transfer_entropy = self._calculate_transfer_entropy(
                returns, volumes, self.entropy_window
            )
            entropy_results['transfer'] = transfer_entropy
            
        # Calculate entropy rate (predictability)
        entropy_rate = self._calculate_entropy_rate(
            returns, self.entropy_window
        )
        
        # Detect regime transitions
        regime_analysis = self._analyze_regime_transitions(
            entropy_results, returns, self.min_entropy_change
        )
        
        # Analyze complexity patterns
        complexity_analysis = self._analyze_complexity_patterns(
            entropy_results, returns, highs, lows
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            entropy_results, regime_analysis, complexity_analysis,
            entropy_rate
        )
        
        # Create metadata
        metadata = self._create_metadata(
            entropy_results, entropy_rate, regime_analysis,
            complexity_analysis, len(prices)
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
    
    def _calculate_shannon_entropy(self, 
                                  returns: np.ndarray,
                                  window: int,
                                  n_bins: int) -> Dict:
        """Calculate Shannon entropy of returns"""
        
        if len(returns) < window:
            return {'current': 0, 'historical': 0, 'percentile': 50}
            
        # Calculate rolling entropy
        entropies = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            
            # Discretize returns into bins
            if window_returns.std() > 0:
                # Create bins based on return distribution
                hist, bins = np.histogram(window_returns, bins=n_bins)
                
                # Calculate probabilities
                probs = hist / len(window_returns)
                probs = probs[probs > 0]  # Remove zero probabilities
                
                # Shannon entropy
                entropy = -np.sum(probs * np.log2(probs))
            else:
                entropy = 0
                
            entropies.append(entropy)
            
        if not entropies:
            return {'current': 0, 'historical': 0, 'percentile': 50}
            
        current_entropy = entropies[-1]
        historical_entropy = np.mean(entropies)
        
        # Normalize entropy (0-1 scale)
        max_entropy = np.log2(n_bins)  # Maximum possible entropy
        normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0
        
        # Percentile ranking
        percentile = stats.percentileofscore(entropies, current_entropy)
        
        # Entropy change rate
        if len(entropies) >= 5:
            recent_change = (entropies[-1] - entropies[-5]) / 5
        else:
            recent_change = 0
            
        return {
            'current': current_entropy,
            'normalized': normalized_entropy,
            'historical': historical_entropy,
            'percentile': percentile,
            'change_rate': recent_change,
            'max_entropy': max_entropy
        }
    
    def _calculate_relative_entropy(self,
                                   returns: np.ndarray,
                                   window: int,
                                   baseline_periods: int) -> Dict:
        """Calculate relative entropy (KL divergence) between recent and baseline"""
        
        if len(returns) < baseline_periods + window:
            return {'kl_divergence': 0, 'is_significant': False}
            
        # Baseline distribution (historical)
        baseline_returns = returns[-baseline_periods-window:-window]
        
        # Recent distribution
        recent_returns = returns[-window:]
        
        # Create probability distributions
        try:
            # Use KDE for continuous distributions
            from scipy.stats import gaussian_kde
            
            baseline_kde = gaussian_kde(baseline_returns)
            recent_kde = gaussian_kde(recent_returns)
            
            # Sample points for KL calculation
            x_range = np.linspace(
                min(baseline_returns.min(), recent_returns.min()),
                max(baseline_returns.max(), recent_returns.max()),
                100
            )
            
            # Calculate PDFs
            p = baseline_kde(x_range)
            q = recent_kde(x_range)
            
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            
            # KL divergence
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            kl_div = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
            
            # Symmetric KL (Jensen-Shannon divergence)
            js_div = 0.5 * kl_div + 0.5 * np.sum(q * np.log((q + epsilon) / (p + epsilon)))
            
        except:
            # Fallback to histogram method
            bins = np.histogram_bin_edges(
                np.concatenate([baseline_returns, recent_returns]), 
                bins=self.n_bins
            )
            
            hist_baseline, _ = np.histogram(baseline_returns, bins=bins)
            hist_recent, _ = np.histogram(recent_returns, bins=bins)
            
            # Convert to probabilities
            p = hist_baseline / hist_baseline.sum()
            q = hist_recent / hist_recent.sum()
            
            # KL divergence
            epsilon = 1e-10
            mask = (p > 0) & (q > 0)
            kl_div = np.sum(p[mask] * np.log((p[mask] + epsilon) / (q[mask] + epsilon)))
            js_div = kl_div  # Simplified
            
        # Test for significance
        is_significant = kl_div > 0.1  # Threshold for regime change
        
        return {
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'is_significant': is_significant,
            'baseline_mean': baseline_returns.mean(),
            'recent_mean': recent_returns.mean(),
            'distribution_shift': recent_returns.mean() - baseline_returns.mean()
        }
    
    def _calculate_approximate_entropy(self, 
                                      returns: np.ndarray,
                                      m: int = 2,
                                      r: float = 0.2) -> float:
        """Calculate approximate entropy (ApEn) for complexity measurement"""
        
        if len(returns) < m + 1:
            return 0
            
        N = len(returns)
        
        # Normalize returns
        if returns.std() > 0:
            normalized_returns = (returns - returns.mean()) / returns.std()
        else:
            return 0
            
        # Scale threshold by standard deviation
        threshold = r * returns.std()
        
        # Count patterns of length m
        def _count_patterns(data, pattern_length, tolerance):
            patterns = np.zeros(N - pattern_length + 1)
            
            for i in range(N - pattern_length + 1):
                template = data[i:i + pattern_length]
                for j in range(N - pattern_length + 1):
                    if np.max(np.abs(template - data[j:j + pattern_length])) < tolerance:
                        patterns[i] += 1
                        
            return patterns
            
        # Calculate phi(m) and phi(m+1)
        patterns_m = _count_patterns(normalized_returns, m, threshold)
        patterns_m1 = _count_patterns(normalized_returns, m + 1, threshold)
        
        # Calculate approximate entropy
        phi_m = 0
        phi_m1 = 0
        
        for i in range(N - m + 1):
            if patterns_m[i] > 0:
                phi_m += np.log(patterns_m[i] / (N - m + 1))
                
        for i in range(N - m):
            if patterns_m1[i] > 0:
                phi_m1 += np.log(patterns_m1[i] / (N - m))
                
        phi_m = phi_m / (N - m + 1)
        phi_m1 = phi_m1 / (N - m)
        
        approx_entropy = phi_m - phi_m1
        
        return max(0, approx_entropy)  # ApEn should be non-negative
    
    def _calculate_transfer_entropy(self,
                                   returns: np.ndarray,
                                   volumes: np.ndarray,
                                   window: int) -> Dict:
        """Calculate transfer entropy from volume to price"""
        
        if len(returns) < window or len(volumes) < window + 1:
            return {'volume_to_price': 0, 'price_to_volume': 0}
            
        # Get volume changes
        volume_changes = np.diff(np.log(volumes + 1))[-window:]
        price_returns = returns[-window:]
        
        if len(volume_changes) < 2:
            return {'volume_to_price': 0, 'price_to_volume': 0}
            
        # Discretize for entropy calculation
        n_states = min(5, len(volume_changes) // 10)  # Adaptive binning
        
        if n_states < 2:
            return {'volume_to_price': 0, 'price_to_volume': 0}
            
        # Create discrete states
        vol_states = pd.qcut(volume_changes, n_states, labels=False, duplicates='drop')
        price_states = pd.qcut(price_returns, n_states, labels=False, duplicates='drop')
        
        # Calculate transfer entropy (simplified)
        # TE(X→Y) = H(Y_future|Y_past) - H(Y_future|Y_past,X_past)
        
        # This is a simplified approximation
        # In practice, would use more sophisticated methods
        
        # Volume → Price information flow
        vol_price_corr = np.corrcoef(volume_changes[:-1], price_returns[1:])[0, 1]
        volume_to_price_te = abs(vol_price_corr) * np.std(price_returns)
        
        # Price → Volume information flow  
        price_vol_corr = np.corrcoef(price_returns[:-1], volume_changes[1:])[0, 1]
        price_to_volume_te = abs(price_vol_corr) * np.std(volume_changes)
        
        # Determine dominant direction
        if volume_to_price_te > price_to_volume_te * 1.2:
            dominant_direction = 'volume_leads'
        elif price_to_volume_te > volume_to_price_te * 1.2:
            dominant_direction = 'price_leads'
        else:
            dominant_direction = 'bidirectional'
            
        return {
            'volume_to_price': volume_to_price_te,
            'price_to_volume': price_to_volume_te,
            'dominant_direction': dominant_direction,
            'information_asymmetry': abs(volume_to_price_te - price_to_volume_te)
        }
    
    def _calculate_entropy_rate(self, returns: np.ndarray, window: int) -> Dict:
        """Calculate entropy rate (predictability measure)"""
        
        if len(returns) < window:
            return {'entropy_rate': 0, 'predictability': 0}
            
        # Get recent returns
        recent_returns = returns[-window:]
        
        # Calculate conditional entropy H(X_n|X_n-1)
        # Using first-order Markov approximation
        
        # Discretize returns
        if recent_returns.std() > 0:
            n_states = min(5, len(recent_returns) // 5)
            states = pd.qcut(recent_returns, n_states, labels=False, duplicates='drop')
        else:
            return {'entropy_rate': 0, 'predictability': 0}
            
        # Build transition matrix
        transitions = {}
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            
            if current not in transitions:
                transitions[current] = {}
            if next_state not in transitions[current]:
                transitions[current][next_state] = 0
            transitions[current][next_state] += 1
            
        # Calculate entropy rate
        entropy_rate = 0
        total_transitions = len(states) - 1
        
        for current_state, next_states in transitions.items():
            state_prob = sum(next_states.values()) / total_transitions
            
            for next_state, count in next_states.items():
                transition_prob = count / sum(next_states.values())
                if transition_prob > 0:
                    entropy_rate += state_prob * transition_prob * np.log2(1/transition_prob)
                    
        # Maximum possible entropy rate
        max_entropy_rate = np.log2(len(set(states)))
        
        # Predictability (inverse of normalized entropy rate)
        predictability = 1 - (entropy_rate / max_entropy_rate) if max_entropy_rate > 0 else 0
        
        return {
            'entropy_rate': entropy_rate,
            'predictability': predictability,
            'max_entropy_rate': max_entropy_rate
        }
    
    def _analyze_regime_transitions(self,
                                   entropy_results: Dict,
                                   returns: np.ndarray,
                                   min_change: float) -> Dict:
        """Analyze regime transitions from entropy changes"""
        
        regime_info = {
            'current_regime': 'normal',
            'transition_detected': False,
            'transition_type': None,
            'confidence': 0
        }
        
        # Shannon entropy analysis
        if 'shannon' in entropy_results:
            shannon = entropy_results['shannon']
            
            # Determine current regime based on entropy level
            if shannon['normalized'] > 0.8:
                regime_info['current_regime'] = 'high_uncertainty'
            elif shannon['normalized'] < 0.3:
                regime_info['current_regime'] = 'low_uncertainty'
            else:
                regime_info['current_regime'] = 'normal'
                
            # Detect transitions based on entropy changes
            if abs(shannon['change_rate']) > min_change:
                regime_info['transition_detected'] = True
                
                if shannon['change_rate'] > 0:
                    regime_info['transition_type'] = 'increasing_uncertainty'
                else:
                    regime_info['transition_type'] = 'decreasing_uncertainty'
                    
        # Relative entropy (KL divergence) analysis
        if 'relative' in entropy_results:
            relative = entropy_results['relative']
            
            if relative['is_significant']:
                regime_info['transition_detected'] = True
                regime_info['transition_type'] = 'distribution_shift'
                regime_info['confidence'] = min(relative['kl_divergence'] * 100, 85)
                
        # Complexity analysis from approximate entropy
        if 'approximate' in entropy_results:
            approx = entropy_results['approximate']
            
            # Low ApEn = more regular/predictable
            # High ApEn = more random/complex
            if approx < 0.5:
                regime_info['complexity'] = 'low'
                regime_info['pattern_type'] = 'regular'
            elif approx > 1.5:
                regime_info['complexity'] = 'high'  
                regime_info['pattern_type'] = 'random'
            else:
                regime_info['complexity'] = 'moderate'
                regime_info['pattern_type'] = 'mixed'
                
        return regime_info
    
    def _analyze_complexity_patterns(self,
                                    entropy_results: Dict,
                                    returns: np.ndarray,
                                    highs: np.ndarray,
                                    lows: np.ndarray) -> Dict:
        """Analyze market complexity patterns"""
        
        patterns = {
            'volatility_clustering': False,
            'information_efficiency': 'normal',
            'market_state': 'normal',
            'trading_opportunity': 'neutral'
        }
        
        # Check for volatility clustering
        if len(returns) >= 20:
            squared_returns = returns**2
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            patterns['volatility_clustering'] = autocorr > 0.3
            
        # Information efficiency based on predictability
        if 'entropy_rate' in locals():
            entropy_rate = self._calculate_entropy_rate(returns, min(20, len(returns)))
            predictability = entropy_rate.get('predictability', 0.5)
            
            if predictability > 0.7:
                patterns['information_efficiency'] = 'low'
                patterns['trading_opportunity'] = 'high'
            elif predictability < 0.3:
                patterns['information_efficiency'] = 'high'
                patterns['trading_opportunity'] = 'low'
                
        # Market state based on multiple entropy measures
        if 'shannon' in entropy_results and 'approximate' in entropy_results:
            shannon_norm = entropy_results['shannon']['normalized']
            approx_entropy = entropy_results['approximate']
            
            # High Shannon + Low ApEn = Trending with noise
            if shannon_norm > 0.7 and approx_entropy < 0.5:
                patterns['market_state'] = 'noisy_trend'
                
            # Low Shannon + Low ApEn = Clear trend
            elif shannon_norm < 0.3 and approx_entropy < 0.5:
                patterns['market_state'] = 'strong_trend'
                
            # High Shannon + High ApEn = Random walk
            elif shannon_norm > 0.7 and approx_entropy > 1.5:
                patterns['market_state'] = 'random_walk'
                
            # Low Shannon + High ApEn = Choppy/whipsaw
            elif shannon_norm < 0.3 and approx_entropy > 1.5:
                patterns['market_state'] = 'choppy'
                
        # Volume-price dynamics
        if 'transfer' in entropy_results:
            transfer = entropy_results['transfer']
            
            if transfer['dominant_direction'] == 'volume_leads':
                patterns['information_flow'] = 'volume_driven'
            elif transfer['dominant_direction'] == 'price_leads':
                patterns['information_flow'] = 'price_driven'
            else:
                patterns['information_flow'] = 'balanced'
                
        return patterns
    
    def _generate_signal(self,
                        entropy_results: Dict,
                        regime_analysis: Dict,
                        complexity_analysis: Dict,
                        entropy_rate: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from entropy analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Base signal on regime transitions
        if regime_analysis['transition_detected']:
            transition_type = regime_analysis['transition_type']
            
            if transition_type == 'decreasing_uncertainty':
                # Market becoming more predictable - trend forming
                signal = SignalType.BUY
                confidence = 60
                
            elif transition_type == 'distribution_shift':
                # Significant regime change detected
                # Could be either direction - need more context
                if 'relative' in entropy_results:
                    shift = entropy_results['relative']['distribution_shift']
                    if shift > 0:
                        signal = SignalType.BUY
                    else:
                        signal = SignalType.SELL
                    confidence = 65
                    
        # Adjust for complexity patterns
        market_state = complexity_analysis['market_state']
        
        if market_state == 'strong_trend':
            # Clear trend with low entropy
            if signal == SignalType.BUY:
                confidence += 15
            elif signal == SignalType.HOLD:
                # Might want to join trend
                signal = SignalType.BUY
                confidence = 55
                
        elif market_state == 'random_walk':
            # High randomness - reduce confidence
            confidence *= 0.7
            
        elif market_state == 'choppy':
            # Whipsaw market - avoid
            if signal != SignalType.HOLD:
                confidence *= 0.5
                
        # Information flow analysis
        if 'transfer' in entropy_results:
            transfer = entropy_results['transfer']
            
            if transfer['dominant_direction'] == 'volume_leads':
                # Volume leading price - stronger signal
                confidence *= 1.1
            elif transfer['information_asymmetry'] > 0.5:
                # High information asymmetry - potential opportunity
                confidence += 5
                
        # Predictability bonus
        predictability = entropy_rate.get('predictability', 0.5)
        
        if predictability > 0.7 and signal != SignalType.HOLD:
            confidence += 10
        elif predictability < 0.3:
            confidence *= 0.8
            
        confidence = min(confidence, 85)
        
        # Value represents information content (0-100)
        if 'shannon' in entropy_results:
            value = (1 - entropy_results['shannon']['normalized']) * 100
        else:
            value = 50
            
        return signal, confidence, value
    
    def _create_metadata(self,
                        entropy_results: Dict,
                        entropy_rate: Dict,
                        regime_analysis: Dict,
                        complexity_analysis: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'entropy_measures': {}
        }
        
        # Add entropy results
        if 'shannon' in entropy_results:
            shannon = entropy_results['shannon']
            metadata['entropy_measures']['shannon'] = {
                'current': shannon['current'],
                'normalized': shannon['normalized'],
                'percentile': shannon['percentile'],
                'change_rate': shannon['change_rate']
            }
            
        if 'relative' in entropy_results:
            relative = entropy_results['relative']
            metadata['entropy_measures']['relative'] = {
                'kl_divergence': relative['kl_divergence'],
                'js_divergence': relative['js_divergence'],
                'distribution_shift': relative['distribution_shift']
            }
            
        if 'approximate' in entropy_results:
            metadata['entropy_measures']['approximate'] = entropy_results['approximate']
            
        if 'transfer' in entropy_results:
            metadata['entropy_measures']['transfer'] = entropy_results['transfer']
            
        # Add other analyses
        metadata['entropy_rate'] = entropy_rate
        metadata['regime_analysis'] = regime_analysis
        metadata['complexity_patterns'] = complexity_analysis
        
        # Add interpretation
        if regime_analysis['transition_detected']:
            metadata['insight'] = f"Regime transition detected: {regime_analysis['transition_type'].replace('_', ' ')}"
        elif complexity_analysis['market_state'] == 'strong_trend':
            metadata['insight'] = "Low entropy environment - strong trending conditions"
        elif complexity_analysis['market_state'] == 'random_walk':
            metadata['insight'] = "High entropy - market in random walk phase"
        elif entropy_rate['predictability'] > 0.7:
            metadata['insight'] = "High predictability detected - favorable for systematic strategies"
        else:
            metadata['insight'] = f"Market in {complexity_analysis['market_state'].replace('_', ' ')} state"
            
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
            metadata={'error': 'Insufficient data for entropy calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            required = ['close'] if 'close' in data.columns else ['Close']
            has_required = any(col in data.columns for col in required)
            return has_required and len(data) >= self.lookback_periods + self.entropy_window
        else:
            return len(data) >= self.lookback_periods + self.entropy_window


def demonstrate_entropy():
    """Demonstration of Entropy indicator"""
    
    print("🔬 Entropy-Based Indicators Demonstration\n")
    
    # Generate synthetic data with regime changes
    print("Generating synthetic data with information regime changes...\n")
    
    np.random.seed(42)
    n_points = 200
    
    prices = []
    price = 100
    
    for i in range(n_points):
        if i < 50:
            # Low entropy regime (trending)
            trend = 0.001
            volatility = 0.005
            regime = "Low entropy (trending)" if i == 0 else None
        elif i < 100:
            # Transition regime
            trend = 0
            volatility = 0.01 + 0.01 * ((i - 50) / 50)  # Increasing volatility
            regime = "Entropy transition" if i == 50 else None
        elif i < 150:
            # High entropy regime (random)
            trend = 0
            volatility = 0.02
            regime = "High entropy (random)" if i == 100 else None
        else:
            # Return to low entropy
            trend = -0.0005
            volatility = 0.005 + 0.005 * np.sin(i/10)
            regime = "Decreasing entropy" if i == 150 else None
            
        if regime:
            print(f"  Period {i}-{min(i+50, n_points)}: {regime}")
            
        # Generate price movement
        return_val = trend + volatility * np.random.randn()
        price *= (1 + return_val)
        prices.append(price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + 0.001 * abs(np.random.randn())) for p in prices],
        'Low': [p * (1 - 0.001 * abs(np.random.randn())) for p in prices],
        'Close': prices,
        'Volume': [1000000 * (1 + 0.5 * abs(np.random.randn())) for _ in prices]
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='D'))
    
    # Create indicator
    entropy_indicator = EntropyIndicator(
        lookback_periods=50,
        entropy_window=20,
        n_bins=10,
        entropy_types=['shannon', 'relative', 'approximate', 'transfer']
    )
    
    # Calculate
    result = entropy_indicator.calculate(data, "SYNTHETIC")
    
    print("\n" + "=" * 60)
    print("ENTROPY ANALYSIS:")
    print("=" * 60)
    
    # Shannon entropy
    if 'shannon' in result.metadata['entropy_measures']:
        shannon = result.metadata['entropy_measures']['shannon']
        print(f"\nShannon Entropy:")
        print(f"  Current: {shannon['current']:.3f}")
        print(f"  Normalized (0-1): {shannon['normalized']:.3f}")
        print(f"  Percentile: {shannon['percentile']:.1f}%")
        print(f"  Change Rate: {shannon['change_rate']:.3f}")
    
    # Relative entropy
    if 'relative' in result.metadata['entropy_measures']:
        relative = result.metadata['entropy_measures']['relative']
        print(f"\nRelative Entropy (KL Divergence):")
        print(f"  KL Divergence: {relative['kl_divergence']:.3f}")
        print(f"  JS Divergence: {relative['js_divergence']:.3f}")
        print(f"  Distribution Shift: {relative['distribution_shift']:.4f}")
    
    # Approximate entropy
    if 'approximate' in result.metadata['entropy_measures']:
        approx = result.metadata['entropy_measures']['approximate']
        print(f"\nApproximate Entropy: {approx:.3f}")
    
    # Transfer entropy
    if 'transfer' in result.metadata['entropy_measures']:
        transfer = result.metadata['entropy_measures']['transfer']
        print(f"\nTransfer Entropy:")
        print(f"  Volume → Price: {transfer['volume_to_price']:.3f}")
        print(f"  Price → Volume: {transfer['price_to_volume']:.3f}")
        print(f"  Dominant Direction: {transfer['dominant_direction'].upper()}")
    
    # Entropy rate
    entropy_rate = result.metadata['entropy_rate']
    print(f"\nEntropy Rate:")
    print(f"  Rate: {entropy_rate['entropy_rate']:.3f}")
    print(f"  Predictability: {entropy_rate['predictability']:.1%}")
    
    # Regime analysis
    regime = result.metadata['regime_analysis']
    print(f"\nRegime Analysis:")
    print(f"  Current Regime: {regime['current_regime'].replace('_', ' ').upper()}")
    print(f"  Transition Detected: {regime['transition_detected']}")
    if regime['transition_type']:
        print(f"  Transition Type: {regime['transition_type'].replace('_', ' ')}")
    
    # Complexity patterns
    complexity = result.metadata['complexity_patterns']
    print(f"\nComplexity Patterns:")
    print(f"  Market State: {complexity['market_state'].replace('_', ' ').upper()}")
    print(f"  Information Efficiency: {complexity['information_efficiency'].upper()}")
    print(f"  Trading Opportunity: {complexity['trading_opportunity'].upper()}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Information Content: {result.value:.1f}/100")
    
    # Insight
    print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    print("\n💡 Entropy Trading Tips:")
    print("- Low entropy = trending markets (momentum strategies)")
    print("- High entropy = random markets (mean reversion)")
    print("- Entropy transitions = regime changes (adjust strategy)")
    print("- Transfer entropy shows information flow direction")
    print("- Approximate entropy measures market complexity")
    print("- Combine with other indicators for confirmation")


if __name__ == "__main__":
    demonstrate_entropy()