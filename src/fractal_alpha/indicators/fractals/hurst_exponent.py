"""
📈 Hurst Exponent Calculator
Measures the long-term memory of time series for regime detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType, MarketRegime


class HurstExponentIndicator(BaseIndicator):
    """
    Calculates the Hurst Exponent to identify market regimes
    
    H > 0.5: Trending (persistence)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting (anti-persistence)
    
    Methods:
    1. Rescaled Range (R/S) Analysis - Classic method
    2. Detrended Fluctuation Analysis (DFA) - More robust
    3. Variance ratio test - Quick approximation
    """
    
    def __init__(self,
                 method: str = 'rs',
                 min_window: int = 10,
                 max_window: Optional[int] = None,
                 lookback_periods: int = 100,
                 rolling_window: int = 50):
        """
        Initialize Hurst Exponent indicator
        
        Args:
            method: Calculation method ('rs', 'dfa', 'variance')
            min_window: Minimum window for R/S analysis
            max_window: Maximum window (default: lookback/4)
            lookback_periods: Total periods to analyze
            rolling_window: Window for rolling Hurst calculation
        """
        super().__init__(
            name="HurstExponent",
            timeframe=TimeFrame.FIFTEEN_MIN,
            lookback_periods=lookback_periods,
            params={
                'method': method,
                'min_window': min_window,
                'max_window': max_window,
                'rolling_window': rolling_window
            }
        )
        
        self.method = method
        self.min_window = min_window
        self.max_window = max_window or lookback_periods // 4
        self.rolling_window = rolling_window
        
    def calculate(self, 
                  data: Union[pd.DataFrame, np.ndarray], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Hurst exponent and generate regime signals
        
        Args:
            data: Price data (OHLCV)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with regime analysis
        """
        if isinstance(data, np.ndarray):
            # Assume close prices
            prices = data
        else:
            prices = data['close'].values
            
        if len(prices) < self.lookback_periods:
            return self._empty_result(symbol)
            
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate Hurst exponent
        if self.method == 'rs':
            hurst = self._calculate_rs_hurst(returns)
        elif self.method == 'dfa':
            hurst = self._calculate_dfa_hurst(returns)
        elif self.method == 'variance':
            hurst = self._calculate_variance_hurst(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Calculate rolling Hurst for trend
        rolling_hurst = self._calculate_rolling_hurst(returns)
        
        # Determine regime
        regime = self._determine_regime(hurst)
        
        # Generate trading signal
        signal, confidence, value = self._generate_signal(
            hurst, rolling_hurst, regime, prices
        )
        
        # Create metadata
        metadata = {
            'hurst_exponent': hurst,
            'regime': regime.value,
            'rolling_hurst': rolling_hurst[-1] if len(rolling_hurst) > 0 else hurst,
            'hurst_trend': self._calculate_hurst_trend(rolling_hurst),
            'confidence_interval': self._estimate_confidence_interval(len(returns)),
            'regime_strength': abs(hurst - 0.5) * 200  # 0-100 scale
        }
        
        timestamp = int(datetime.now().timestamp() * 1000)
        if isinstance(data, pd.DataFrame) and not data.empty:
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
    
    def _calculate_rs_hurst(self, returns: np.ndarray) -> float:
        """Calculate Hurst using Rescaled Range analysis"""
        
        # Remove any NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < self.min_window * 2:
            return 0.5
            
        # Range of scales
        scales = self._get_scales(len(returns))
        
        # Calculate R/S for each scale
        rs_values = []
        
        for scale in scales:
            rs_scale = []
            
            # Divide series into chunks
            for start in range(0, len(returns) - scale + 1, scale):
                chunk = returns[start:start + scale]
                
                if len(chunk) < scale:
                    continue
                    
                # Calculate mean
                mean = np.mean(chunk)
                
                # Mean-adjusted series
                y = chunk - mean
                
                # Cumulative sum
                z = np.cumsum(y)
                
                # Range
                R = np.max(z) - np.min(z)
                
                # Standard deviation
                S = np.std(chunk, ddof=1)
                
                if S > 0:
                    rs_scale.append(R / S)
                    
            if rs_scale:
                rs_values.append(np.mean(rs_scale))
                
        if len(rs_values) < 2:
            return 0.5
            
        # Log-log regression
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove any invalid values
        mask = np.isfinite(log_scales) & np.isfinite(log_rs)
        log_scales = log_scales[mask]
        log_rs = log_rs[mask]
        
        if len(log_scales) < 2:
            return 0.5
            
        # Linear regression
        hurst = np.polyfit(log_scales, log_rs, 1)[0]
        
        # Bound between 0 and 1
        return np.clip(hurst, 0.01, 0.99)
    
    def _calculate_dfa_hurst(self, returns: np.ndarray) -> float:
        """Calculate Hurst using Detrended Fluctuation Analysis"""
        
        # Cumulative sum
        y = np.cumsum(returns - np.mean(returns))
        
        scales = self._get_scales(len(y))
        fluctuations = []
        
        for scale in scales:
            # Number of segments
            n_segments = len(y) // scale
            
            if n_segments < 1:
                continue
                
            # Reshape into segments
            segments = y[:n_segments * scale].reshape(n_segments, scale)
            
            # Fit polynomial trend to each segment
            x = np.arange(scale)
            detrended = []
            
            for segment in segments:
                # Fit linear trend
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                detrended.append(np.sqrt(np.mean((segment - fit) ** 2)))
                
            if detrended:
                fluctuations.append(np.mean(detrended))
                
        if len(fluctuations) < 2:
            return 0.5
            
        # Log-log regression
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        
        # Linear regression
        hurst = np.polyfit(log_scales, log_fluct, 1)[0]
        
        return np.clip(hurst, 0.01, 0.99)
    
    def _calculate_variance_hurst(self, returns: np.ndarray) -> float:
        """Quick Hurst approximation using variance ratios"""
        
        # Calculate variance ratios for different lags
        lags = [2, 4, 8, 16]
        var_ratios = []
        
        var_1 = np.var(returns)
        
        for lag in lags:
            if len(returns) < lag * 10:
                continue
                
            # Returns over lag periods
            returns_lag = np.array([np.sum(returns[i:i+lag]) 
                                   for i in range(0, len(returns)-lag+1, lag)])
            
            var_lag = np.var(returns_lag) / lag
            
            if var_1 > 0:
                var_ratios.append(var_lag / var_1)
                
        if not var_ratios:
            return 0.5
            
        # Estimate Hurst from variance ratios
        # For fBm: Var(lag) / Var(1) = lag^(2H)
        log_lags = np.log(lags[:len(var_ratios)])
        log_ratios = np.log(var_ratios)
        
        # Remove invalid values
        mask = np.isfinite(log_lags) & np.isfinite(log_ratios)
        
        if np.sum(mask) < 2:
            return 0.5
            
        # Regression
        h_est = np.polyfit(log_lags[mask], log_ratios[mask], 1)[0] / 2
        
        return np.clip(h_est, 0.01, 0.99)
    
    def _calculate_rolling_hurst(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling Hurst exponent"""
        
        if len(returns) < self.rolling_window:
            return np.array([])
            
        rolling_hurst = []
        
        for i in range(self.rolling_window, len(returns) + 1):
            window_returns = returns[i - self.rolling_window:i]
            
            # Use faster variance method for rolling calculation
            h = self._calculate_variance_hurst(window_returns)
            rolling_hurst.append(h)
            
        return np.array(rolling_hurst)
    
    def _get_scales(self, n: int) -> np.ndarray:
        """Get scales for R/S or DFA analysis"""
        
        # Logarithmic scales
        min_scale = max(self.min_window, 4)
        max_scale = min(self.max_window, n // 4)
        
        if max_scale <= min_scale:
            return np.array([min_scale])
            
        # Generate logarithmically spaced scales
        n_scales = int(np.log2(max_scale / min_scale)) + 1
        scales = np.unique(np.logspace(
            np.log2(min_scale), 
            np.log2(max_scale), 
            n_scales, 
            base=2
        ).astype(int))
        
        return scales[scales <= n // 2]
    
    def _determine_regime(self, hurst: float) -> MarketRegime:
        """Determine market regime from Hurst exponent"""
        
        if hurst > 0.65:
            return MarketRegime.TRENDING_UP  # Strong trend
        elif hurst > 0.55:
            return MarketRegime.TRENDING_UP if np.random.random() > 0.5 else MarketRegime.TRENDING_DOWN
        elif hurst < 0.35:
            return MarketRegime.MEAN_REVERTING  # Strong mean reversion
        elif hurst < 0.45:
            return MarketRegime.MEAN_REVERTING  # Weak mean reversion
        else:
            return MarketRegime.UNKNOWN  # Random walk
    
    def _generate_signal(self, 
                        hurst: float,
                        rolling_hurst: np.ndarray,
                        regime: MarketRegime,
                        prices: np.ndarray) -> Tuple[SignalType, float, float]:
        """Generate trading signal based on regime"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Regime strength
        regime_strength = abs(hurst - 0.5)
        
        # For trending regime
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Follow the trend
            if len(prices) >= 20:
                short_ma = np.mean(prices[-10:])
                long_ma = np.mean(prices[-20:])
                
                if short_ma > long_ma:
                    signal = SignalType.BUY
                else:
                    signal = SignalType.SELL
                    
                confidence = min(regime_strength * 200, 85)
                
        # For mean-reverting regime
        elif regime == MarketRegime.MEAN_REVERTING:
            # Trade reversions
            if len(prices) >= 20:
                mean = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                z_score = (prices[-1] - mean) / std if std > 0 else 0
                
                if z_score < -1.5:  # Oversold
                    signal = SignalType.BUY
                    confidence = min(abs(z_score) * 20 + regime_strength * 100, 85)
                elif z_score > 1.5:  # Overbought
                    signal = SignalType.SELL
                    confidence = min(abs(z_score) * 20 + regime_strength * 100, 85)
        
        # Check regime transitions
        if len(rolling_hurst) >= 10:
            # Regime changing from mean-reverting to trending
            if rolling_hurst[-10] < 0.45 and rolling_hurst[-1] > 0.55:
                # New trend beginning
                signal = SignalType.BUY if prices[-1] > prices[-2] else SignalType.SELL
                confidence = 70
                
            # Regime changing from trending to mean-reverting
            elif rolling_hurst[-10] > 0.55 and rolling_hurst[-1] < 0.45:
                # Trend ending, prepare for reversions
                signal = SignalType.HOLD
                confidence = 0
        
        # Value is normalized Hurst (0-100 scale)
        value = hurst * 100
        
        return signal, confidence, value
    
    def _calculate_hurst_trend(self, rolling_hurst: np.ndarray) -> str:
        """Determine if Hurst is increasing or decreasing"""
        
        if len(rolling_hurst) < 5:
            return "stable"
            
        recent = rolling_hurst[-5:]
        trend = np.polyfit(range(5), recent, 1)[0]
        
        if trend > 0.01:
            return "increasing"  # Moving toward trending
        elif trend < -0.01:
            return "decreasing"  # Moving toward mean reversion
        else:
            return "stable"
    
    def _estimate_confidence_interval(self, n: int) -> float:
        """Estimate confidence interval for Hurst estimate"""
        
        # Approximate standard error
        # This is a simplified estimate
        se = 0.5 / np.sqrt(n)
        
        # 95% confidence interval width
        return 1.96 * se
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,  # Neutral
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate input data"""
        
        if isinstance(data, pd.DataFrame):
            return 'close' in data.columns and len(data) >= self.lookback_periods
        else:
            return len(data) >= self.lookback_periods


def demonstrate_hurst_exponent():
    """Demonstration of Hurst Exponent calculation"""
    
    print("📈 Hurst Exponent Demonstration\n")
    
    # Generate different regime data
    np.random.seed(42)
    n = 500
    
    # 1. Trending data (H > 0.5)
    trend = np.cumsum(np.random.randn(n) + 0.1)
    trend_prices = 100 * np.exp(trend * 0.01)
    
    # 2. Mean-reverting data (H < 0.5)
    mr_returns = []
    price = 0
    for _ in range(n):
        price = 0.9 * price + np.random.randn()
        mr_returns.append(price)
    mr_prices = 100 * np.exp(np.cumsum(np.array(mr_returns) * 0.01))
    
    # 3. Random walk (H ≈ 0.5)
    rw_prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    # Test each regime
    scenarios = [
        ("Trending Market", trend_prices),
        ("Mean-Reverting Market", mr_prices),
        ("Random Walk", rw_prices)
    ]
    
    for name, prices in scenarios:
        print(f"\n{name}:")
        print("-" * 50)
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices
        }, index=pd.date_range('2024-01-01', periods=len(prices), freq='15T'))
        
        # Test different methods
        for method in ['rs', 'dfa', 'variance']:
            indicator = HurstExponentIndicator(method=method)
            result = indicator.calculate(df, name)
            
            h = result.metadata['hurst_exponent']
            regime = result.metadata['regime']
            
            print(f"  {method.upper()} Method:")
            print(f"    Hurst Exponent: {h:.3f}")
            print(f"    Regime: {regime}")
            print(f"    Signal: {result.signal.value}")
            print(f"    Confidence: {result.confidence:.1f}%")
    
    # Show rolling Hurst evolution
    print("\n\nRolling Hurst Evolution:")
    print("="*50)
    
    # Create transitioning market
    transition_returns = np.concatenate([
        np.random.randn(200) * 0.01,  # Random
        np.cumsum(np.random.randn(200) * 0.01 + 0.001),  # Trending
        mr_returns[:100]  # Mean-reverting
    ])
    
    transition_prices = 100 * np.exp(np.cumsum(transition_returns))
    
    df = pd.DataFrame({
        'close': transition_prices
    }, index=pd.date_range('2024-01-01', periods=len(transition_prices), freq='15T'))
    
    indicator = HurstExponentIndicator(method='variance', rolling_window=50)
    result = indicator.calculate(df, "Transitioning Market")
    
    print(f"Final Hurst: {result.metadata['hurst_exponent']:.3f}")
    print(f"Hurst Trend: {result.metadata['hurst_trend']}")
    print(f"Current Regime: {result.metadata['regime']}")
    print(f"Regime Strength: {result.metadata['regime_strength']:.1f}/100")


if __name__ == "__main__":
    demonstrate_hurst_exponent()