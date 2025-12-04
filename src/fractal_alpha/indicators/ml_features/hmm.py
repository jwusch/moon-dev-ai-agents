"""
🔮 Hidden Markov Model - Probabilistic Regime Detection
Identifies market states and predicts regime transitions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not installed. Using simplified HMM implementation.")

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class HMMIndicator(BaseIndicator):
    """
    Hidden Markov Model indicator for regime detection
    
    HMM models the market as having hidden states (regimes) that:
    - Are not directly observable
    - Influence the observable returns/volatility
    - Transition probabilistically between states
    
    This indicator implements:
    - Gaussian HMM for return distributions
    - State identification (trending, ranging, volatile)
    - Transition probability matrix
    - Forward-looking state predictions
    - Dynamic strategy allocation based on regime
    """
    
    def __init__(self,
                 n_states: int = 3,
                 lookback_periods: int = 100,
                 n_features: int = 3,
                 use_volatility: bool = True,
                 use_volume: bool = True,
                 min_state_confidence: float = 0.7,
                 predict_transitions: bool = True):
        """
        Initialize HMM indicator
        
        Args:
            n_states: Number of hidden states/regimes
            lookback_periods: Training window for HMM
            n_features: Number of features (returns, volatility, volume)
            use_volatility: Include volatility as observable
            use_volume: Include volume as observable
            min_state_confidence: Minimum confidence for state classification
            predict_transitions: Predict next state transitions
        """
        super().__init__(
            name="HMM",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_periods + 20,  # Extra for feature calculation
            params={
                'n_states': n_states,
                'lookback_periods': lookback_periods,
                'n_features': n_features,
                'use_volatility': use_volatility,
                'use_volume': use_volume,
                'min_state_confidence': min_state_confidence,
                'predict_transitions': predict_transitions
            }
        )
        
        self.n_states = n_states
        self.lookback_periods = lookback_periods
        self.n_features = n_features
        self.use_volatility = use_volatility
        self.use_volume = use_volume
        self.min_state_confidence = min_state_confidence
        self.predict_transitions = predict_transitions
        
        # State interpretations (for 3-state model)
        self.state_names = {
            0: 'low_volatility_trending',
            1: 'high_volatility_ranging',
            2: 'moderate_volatility_transitional'
        }
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate HMM states and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with HMM analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.lookback_periods + 20:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
            volumes = data['Volume'].values if 'Volume' in data else data.get('volume', np.ones_like(prices)).values
            highs = data['High'].values if 'High' in data else data['high'].values
            lows = data['Low'].values if 'Low' in data else data['low'].values
        else:
            if len(data) < self.lookback_periods + 20:
                return self._empty_result(symbol)
            prices = np.array(data)
            volumes = np.ones_like(prices)
            highs = prices
            lows = prices
            
        # Prepare features for HMM
        features = self._prepare_features(prices, highs, lows, volumes)
        
        if features is None or len(features) < self.lookback_periods:
            return self._empty_result(symbol)
            
        # Fit HMM model
        if HMM_AVAILABLE:
            model, states, log_likelihood = self._fit_hmm(features)
        else:
            model, states, log_likelihood = self._fit_simple_hmm(features)
            
        if model is None:
            return self._empty_result(symbol)
            
        # Get current state and probabilities
        current_state = states[-1]
        state_probabilities = self._get_state_probabilities(model, features[-1:])
        
        # Analyze state characteristics
        state_analysis = self._analyze_states(model, features, states)
        
        # Get transition matrix
        transition_matrix = self._get_transition_matrix(model, states)
        
        # Predict next state if requested
        next_state_prediction = {}
        if self.predict_transitions:
            next_state_prediction = self._predict_next_state(
                model, current_state, state_probabilities, transition_matrix
            )
        
        # Analyze regime stability
        regime_stability = self._analyze_regime_stability(states, transition_matrix)
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            current_state, state_probabilities, state_analysis,
            next_state_prediction, regime_stability, prices
        )
        
        # Create metadata
        metadata = self._create_metadata(
            current_state, state_probabilities, state_analysis,
            transition_matrix, next_state_prediction, regime_stability,
            log_likelihood, len(prices)
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
    
    def _prepare_features(self, 
                         prices: np.ndarray,
                         highs: np.ndarray,
                         lows: np.ndarray,
                         volumes: np.ndarray) -> Optional[np.ndarray]:
        """Prepare feature matrix for HMM"""
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Start with returns as first feature
        features = [returns]
        
        # Add volatility if requested
        if self.use_volatility:
            # Rolling volatility (5-day)
            volatilities = []
            for i in range(5, len(returns) + 1):
                vol = np.std(returns[i-5:i])
                volatilities.append(vol)
            
            # Pad beginning
            volatilities = [volatilities[0]] * 4 + volatilities
            features.append(np.array(volatilities))
            
            # Add high-low volatility
            hl_vol = np.log(highs[1:] / lows[1:])
            features.append(hl_vol)
        
        # Add volume if requested
        if self.use_volume:
            # Volume ratio
            volume_ma = pd.Series(volumes).rolling(10).mean().bfill().values
            volume_ratio = volumes / (volume_ma + 1e-10)
            features.append(volume_ratio[1:])  # Match returns length
            
        # Stack features
        if len(features) > 1:
            # Ensure all features have same length
            min_len = min(len(f) for f in features)
            features = [f[-min_len:] for f in features]
            feature_matrix = np.column_stack(features)
        else:
            feature_matrix = features[0].reshape(-1, 1)
            
        # Standardize features
        if len(feature_matrix) > 0:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            
        return feature_matrix
    
    def _fit_hmm(self, features: np.ndarray) -> Tuple[Optional[object], np.ndarray, float]:
        """Fit HMM using hmmlearn library"""
        
        if not HMM_AVAILABLE:
            return self._fit_simple_hmm(features)
            
        try:
            # Create Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            model.fit(features[-self.lookback_periods:])
            
            # Get states
            states = model.predict(features)
            
            # Get log likelihood
            log_likelihood = model.score(features)
            
            return model, states, log_likelihood
            
        except Exception as e:
            warnings.warn(f"HMM fitting failed: {e}")
            return None, np.array([]), 0
    
    def _fit_simple_hmm(self, features: np.ndarray) -> Tuple[Optional[Dict], np.ndarray, float]:
        """Simple HMM implementation when hmmlearn not available"""
        
        # K-means clustering as simple state assignment
        from sklearn.cluster import KMeans
        
        try:
            # Use k-means to identify states
            kmeans = KMeans(n_clusters=self.n_states, random_state=42)
            states = kmeans.fit_predict(features)
            
            # Calculate simple transition matrix
            transition_counts = np.zeros((self.n_states, self.n_states))
            for i in range(len(states) - 1):
                transition_counts[states[i], states[i+1]] += 1
                
            # Normalize to get probabilities
            transition_matrix = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-10)
            
            # Calculate state means and variances
            state_stats = {}
            for state in range(self.n_states):
                state_mask = states == state
                if state_mask.any():
                    state_features = features[state_mask]
                    state_stats[state] = {
                        'mean': state_features.mean(axis=0),
                        'std': state_features.std(axis=0),
                        'count': state_mask.sum()
                    }
                    
            # Simple model object
            model = {
                'type': 'simple',
                'kmeans': kmeans,
                'transition_matrix': transition_matrix,
                'state_stats': state_stats,
                'n_states': self.n_states
            }
            
            # Pseudo log-likelihood
            log_likelihood = -kmeans.inertia_
            
            return model, states, log_likelihood
            
        except Exception as e:
            warnings.warn(f"Simple HMM fitting failed: {e}")
            return None, np.array([]), 0
    
    def _get_state_probabilities(self, model: Union[object, Dict], features: np.ndarray) -> np.ndarray:
        """Get probability of each state for current observation"""
        
        if HMM_AVAILABLE and hasattr(model, 'predict_proba'):
            # Use hmmlearn's predict_proba
            return model.predict_proba(features)[-1]
        elif isinstance(model, dict) and model['type'] == 'simple':
            # Calculate distances to cluster centers
            kmeans = model['kmeans']
            distances = kmeans.transform(features)
            
            # Convert distances to probabilities (softmax)
            neg_distances = -distances[0]
            exp_distances = np.exp(neg_distances - neg_distances.max())
            probabilities = exp_distances / exp_distances.sum()
            
            return probabilities
        else:
            # Default uniform probabilities
            return np.ones(self.n_states) / self.n_states
    
    def _analyze_states(self, model: Union[object, Dict], features: np.ndarray, states: np.ndarray) -> Dict:
        """Analyze characteristics of each state"""
        
        state_info = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            if not state_mask.any():
                continue
                
            state_features = features[state_mask]
            
            # Basic statistics
            returns = state_features[:, 0]  # First feature is returns
            
            state_info[state] = {
                'mean_return': returns.mean() * 252,  # Annualized
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'frequency': state_mask.sum() / len(states),
                'avg_duration': self._calculate_avg_duration(states, state),
                'interpretation': self._interpret_state(returns)
            }
            
            # Add volatility info if available
            if features.shape[1] > 1 and self.use_volatility:
                volatilities = state_features[:, 1]
                state_info[state]['avg_volatility'] = volatilities.mean()
                
        return state_info
    
    def _calculate_avg_duration(self, states: np.ndarray, target_state: int) -> float:
        """Calculate average duration in a given state"""
        
        durations = []
        current_duration = 0
        
        for state in states:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
                
        if current_duration > 0:
            durations.append(current_duration)
            
        return np.mean(durations) if durations else 0
    
    def _interpret_state(self, returns: np.ndarray) -> str:
        """Interpret state based on return characteristics"""
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Simple interpretation rules
        if volatility < np.percentile(returns.std(), 33):
            if mean_return > 0:
                return "low_vol_uptrend"
            else:
                return "low_vol_downtrend"
        elif volatility > np.percentile(returns.std(), 67):
            if abs(mean_return) < volatility * 0.5:
                return "high_vol_ranging"
            else:
                return "high_vol_trending"
        else:
            return "moderate_volatility"
    
    def _get_transition_matrix(self, model: Union[object, Dict], states: np.ndarray) -> np.ndarray:
        """Get state transition probability matrix"""
        
        if HMM_AVAILABLE and hasattr(model, 'transmat_'):
            return model.transmat_
        elif isinstance(model, dict) and 'transition_matrix' in model:
            return model['transition_matrix']
        else:
            # Calculate from states
            transition_matrix = np.zeros((self.n_states, self.n_states))
            
            for i in range(len(states) - 1):
                transition_matrix[states[i], states[i+1]] += 1
                
            # Normalize
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = transition_matrix / (row_sums + 1e-10)
            
            return transition_matrix
    
    def _predict_next_state(self,
                           model: Union[object, Dict],
                           current_state: int,
                           state_probabilities: np.ndarray,
                           transition_matrix: np.ndarray) -> Dict:
        """Predict most likely next state"""
        
        # Get transition probabilities from current state
        transition_probs = transition_matrix[current_state]
        
        # Most likely next state
        next_state = np.argmax(transition_probs)
        next_state_prob = transition_probs[next_state]
        
        # Calculate expected state (weighted by probabilities)
        expected_state = np.sum(np.arange(self.n_states) * transition_probs)
        
        # State change probability
        state_change_prob = 1 - transition_probs[current_state]
        
        return {
            'next_state': int(next_state),
            'next_state_probability': float(next_state_prob),
            'expected_state': float(expected_state),
            'state_change_probability': float(state_change_prob),
            'transition_probabilities': transition_probs.tolist()
        }
    
    def _analyze_regime_stability(self, states: np.ndarray, transition_matrix: np.ndarray) -> Dict:
        """Analyze regime stability and persistence"""
        
        # Diagonal values indicate state persistence
        persistence = np.diag(transition_matrix)
        
        # Average regime duration (geometric distribution)
        avg_durations = 1 / (1 - persistence + 1e-10)
        
        # Regime switching frequency
        switches = np.sum(states[1:] != states[:-1])
        switch_frequency = switches / len(states)
        
        # Find most stable state
        most_stable_state = np.argmax(persistence)
        
        # Calculate entropy of transition matrix (regime uncertainty)
        transition_entropy = 0
        for row in transition_matrix:
            row_entropy = -np.sum(row * np.log(row + 1e-10))
            transition_entropy += row_entropy
        transition_entropy /= self.n_states
        
        return {
            'state_persistence': persistence.tolist(),
            'avg_state_durations': avg_durations.tolist(),
            'switch_frequency': float(switch_frequency),
            'most_stable_state': int(most_stable_state),
            'regime_entropy': float(transition_entropy),
            'current_regime_duration': self._get_current_regime_duration(states)
        }
    
    def _get_current_regime_duration(self, states: np.ndarray) -> int:
        """Get duration of current regime"""
        
        current_state = states[-1]
        duration = 1
        
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_state:
                duration += 1
            else:
                break
                
        return duration
    
    def _generate_signal(self,
                        current_state: int,
                        state_probabilities: np.ndarray,
                        state_analysis: Dict,
                        next_state_prediction: Dict,
                        regime_stability: Dict,
                        prices: np.ndarray) -> Tuple[SignalType, float, float]:
        """Generate trading signal from HMM analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Get state characteristics
        if current_state in state_analysis:
            state_info = state_analysis[current_state]
            interpretation = state_info['interpretation']
            
            # State-based signals
            if 'uptrend' in interpretation:
                signal = SignalType.BUY
                confidence = 60
            elif 'downtrend' in interpretation:
                signal = SignalType.SELL
                confidence = 60
            elif 'ranging' in interpretation:
                # Mean reversion in ranging market
                price_percentile = stats.percentileofscore(prices[-20:], prices[-1])
                if price_percentile < 30:
                    signal = SignalType.BUY
                    confidence = 50
                elif price_percentile > 70:
                    signal = SignalType.SELL
                    confidence = 50
                    
        # Adjust confidence based on state probability
        max_prob = state_probabilities.max()
        if max_prob < self.min_state_confidence:
            # Low confidence in state classification
            confidence *= 0.7
        else:
            # High confidence in state
            confidence *= (0.8 + 0.2 * max_prob)
            
        # Transition-based adjustments
        if next_state_prediction:
            next_state = next_state_prediction['next_state']
            change_prob = next_state_prediction['state_change_probability']
            
            # If regime change likely
            if change_prob > 0.5 and next_state != current_state:
                if next_state in state_analysis:
                    next_interpretation = state_analysis[next_state]['interpretation']
                    
                    # Anticipate regime change
                    if 'uptrend' in next_interpretation and signal != SignalType.BUY:
                        signal = SignalType.BUY
                        confidence = 55
                    elif 'downtrend' in next_interpretation and signal != SignalType.SELL:
                        signal = SignalType.SELL
                        confidence = 55
                        
        # Stability adjustments
        current_duration = regime_stability['current_regime_duration']
        avg_duration = regime_stability['avg_state_durations'][current_state] if current_state < len(regime_stability['avg_state_durations']) else 10
        
        if current_duration > avg_duration * 1.5:
            # Regime overstaying - potential reversal
            confidence *= 0.85
            
        confidence = min(confidence, 85)
        
        # Value represents regime clarity (inverse of entropy)
        regime_entropy = regime_stability['regime_entropy']
        value = (1 - regime_entropy / np.log(self.n_states)) * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        current_state: int,
                        state_probabilities: np.ndarray,
                        state_analysis: Dict,
                        transition_matrix: np.ndarray,
                        next_state_prediction: Dict,
                        regime_stability: Dict,
                        log_likelihood: float,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'n_states': self.n_states,
            'current_state': {
                'id': int(current_state),
                'name': self.state_names.get(current_state, f'state_{current_state}'),
                'probability': float(state_probabilities[current_state]),
                'all_probabilities': state_probabilities.tolist()
            },
            'log_likelihood': float(log_likelihood),
            'state_analysis': {}
        }
        
        # Add state details
        for state, info in state_analysis.items():
            metadata['state_analysis'][f'state_{state}'] = {
                'interpretation': info['interpretation'],
                'mean_return_annual': f"{info['mean_return']:.1f}%",
                'volatility_annual': f"{info['volatility']:.1f}%",
                'frequency': f"{info['frequency']:.1%}",
                'avg_duration': f"{info['avg_duration']:.1f} days"
            }
            
        # Transition matrix
        metadata['transition_matrix'] = transition_matrix.tolist()
        
        # Next state prediction
        if next_state_prediction:
            metadata['next_state_prediction'] = next_state_prediction
            
        # Regime stability
        metadata['regime_stability'] = regime_stability
        
        # Trading insight
        if current_state in state_analysis:
            current_interpretation = state_analysis[current_state]['interpretation']
            duration = regime_stability['current_regime_duration']
            
            if 'trending' in current_interpretation:
                metadata['insight'] = f"Market in {current_interpretation.replace('_', ' ')} for {duration} days"
            elif duration > 20:
                metadata['insight'] = f"Extended {current_interpretation.replace('_', ' ')} regime - watch for reversal"
            elif regime_stability['switch_frequency'] > 0.2:
                metadata['insight'] = "High regime switching - unstable market conditions"
            else:
                metadata['insight'] = f"Stable {current_interpretation.replace('_', ' ')} regime"
        else:
            metadata['insight'] = "Analyzing market regime probabilities"
            
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
            metadata={'error': 'Insufficient data for HMM analysis'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, List[float]]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            required = ['close'] if 'close' in data.columns else ['Close']
            has_required = any(col in data.columns for col in required)
            return has_required and len(data) >= self.lookback_periods + 20
        else:
            return len(data) >= self.lookback_periods + 20


def demonstrate_hmm():
    """Demonstration of HMM indicator"""
    
    print("🔮 Hidden Markov Model Regime Detection Demonstration\n")
    
    # Generate synthetic data with regime changes
    print("Generating synthetic data with distinct market regimes...\n")
    
    np.random.seed(42)
    n_points = 300
    
    prices = []
    price = 100
    
    # Define regimes
    for i in range(n_points):
        if i < 100:
            # Regime 1: Low volatility uptrend
            returns = np.random.normal(0.0005, 0.005)
            regime = "Low volatility uptrend" if i == 0 else None
        elif i < 200:
            # Regime 2: High volatility ranging
            returns = np.random.normal(0, 0.02)
            regime = "High volatility ranging" if i == 100 else None
        else:
            # Regime 3: Moderate volatility downtrend
            returns = np.random.normal(-0.0003, 0.01)
            regime = "Moderate volatility downtrend" if i == 200 else None
            
        if regime:
            print(f"  Period {i}-{min(i+100, n_points)}: {regime}")
            
        price *= (1 + returns)
        prices.append(price)
        
    # Add volume with regime characteristics
    volumes = []
    for i in range(n_points):
        if i < 100:
            vol = 1000000 * (1 + 0.1 * np.random.randn())
        elif i < 200:
            vol = 2000000 * (1 + 0.3 * np.random.randn())
        else:
            vol = 1500000 * (1 + 0.2 * np.random.randn())
        volumes.append(vol)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + 0.001 * abs(np.random.randn())) for p in prices],
        'Low': [p * (1 - 0.001 * abs(np.random.randn())) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='D'))
    
    # Create indicator
    hmm_indicator = HMMIndicator(
        n_states=3,
        lookback_periods=100,
        use_volatility=True,
        use_volume=True,
        predict_transitions=True
    )
    
    # Calculate
    result = hmm_indicator.calculate(data, "SYNTHETIC")
    
    print("\n" + "=" * 60)
    print("HMM REGIME ANALYSIS:")
    print("=" * 60)
    
    # Current state
    current = result.metadata['current_state']
    print(f"\nCurrent Market State:")
    print(f"  State ID: {current['id']}")
    print(f"  State Name: {current['name']}")
    print(f"  Confidence: {current['probability']:.1%}")
    print(f"\n  All State Probabilities:")
    for i, prob in enumerate(current['all_probabilities']):
        print(f"    State {i}: {prob:.1%}")
    
    # State characteristics
    print("\nState Characteristics:")
    for state_name, info in result.metadata['state_analysis'].items():
        print(f"\n  {state_name.upper()}:")
        print(f"    Interpretation: {info['interpretation']}")
        print(f"    Annual Return: {info['mean_return_annual']}")
        print(f"    Annual Volatility: {info['volatility_annual']}")
        print(f"    Frequency: {info['frequency']}")
        print(f"    Avg Duration: {info['avg_duration']}")
    
    # Transition matrix
    print("\nTransition Probability Matrix:")
    trans_mat = result.metadata['transition_matrix']
    print("       To: State 0   State 1   State 2")
    for i, row in enumerate(trans_mat):
        print(f"From State {i}: {row[0]:.3f}    {row[1]:.3f}    {row[2]:.3f}")
    
    # Next state prediction
    if 'next_state_prediction' in result.metadata:
        pred = result.metadata['next_state_prediction']
        print(f"\nNext State Prediction:")
        print(f"  Most Likely Next State: {pred['next_state']}")
        print(f"  Probability: {pred['next_state_probability']:.1%}")
        print(f"  State Change Probability: {pred['state_change_probability']:.1%}")
    
    # Regime stability
    stability = result.metadata['regime_stability']
    print(f"\nRegime Stability Analysis:")
    print(f"  Current Regime Duration: {stability['current_regime_duration']} days")
    print(f"  Regime Switch Frequency: {stability['switch_frequency']:.1%} per day")
    print(f"  Most Stable State: {stability['most_stable_state']}")
    print(f"  Regime Entropy: {stability['regime_entropy']:.2f}")
    
    print("\n  State Persistence (diagonal of transition matrix):")
    for i, persistence in enumerate(stability['state_persistence']):
        avg_duration = stability['avg_state_durations'][i]
        print(f"    State {i}: {persistence:.1%} (avg duration: {avg_duration:.1f} days)")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Regime Clarity: {result.value:.1f}/100")
    
    # Model quality
    print(f"\nModel Quality:")
    print(f"  Log-Likelihood: {result.metadata['log_likelihood']:.2f}")
    
    # Insight
    print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    print("\n💡 HMM Trading Tips:")
    print("- High state persistence = stable regimes")
    print("- Low regime entropy = clear market structure")
    print("- Watch for regime transitions at extremes")
    print("- Combine state + transition probabilities")
    print("- 3-5 states typically optimal for daily data")
    print("- Use multiple features for better state detection")


if __name__ == "__main__":
    demonstrate_hmm()