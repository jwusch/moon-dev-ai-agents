"""
📐 Ornstein-Uhlenbeck Process - Statistical Mean Reversion
Models mean-reverting price dynamics with drift and volatility
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from scipy.optimize import minimize

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class OUProcessIndicator(BaseIndicator):
    """
    Ornstein-Uhlenbeck (OU) Process indicator for mean reversion
    
    The OU process models price as:
    dX(t) = θ(μ - X(t))dt + σdW(t)
    
    Where:
    - θ (theta): Mean reversion speed
    - μ (mu): Long-term mean level
    - σ (sigma): Volatility
    - dW(t): Brownian motion
    
    Key insights:
    - Half-life = ln(2)/θ indicates reversion time
    - Higher θ = faster mean reversion
    - Statistical arbitrage when price deviates from μ
    - Combines with cointegration for pairs trading
    """
    
    def __init__(self,
                 lookback_periods: int = 100,
                 estimation_method: str = "mle",
                 min_half_life: float = 1.0,
                 max_half_life: float = 50.0,
                 z_score_threshold: float = 2.0,
                 confidence_level: float = 0.95):
        """
        Initialize OU Process indicator
        
        Args:
            lookback_periods: Periods for parameter estimation
            estimation_method: "mle" or "ols" for parameter estimation
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
            z_score_threshold: Z-score for entry signals
            confidence_level: Confidence level for bands
        """
        super().__init__(
            name="OUProcess",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_periods,
            params={
                'lookback_periods': lookback_periods,
                'estimation_method': estimation_method,
                'min_half_life': min_half_life,
                'max_half_life': max_half_life,
                'z_score_threshold': z_score_threshold,
                'confidence_level': confidence_level
            }
        )
        
        self.lookback_periods = lookback_periods
        self.estimation_method = estimation_method
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.z_score_threshold = z_score_threshold
        self.confidence_level = confidence_level
        
    def calculate(self, 
                  data: Union[pd.DataFrame, List[float]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate OU process parameters and generate signals
        
        Args:
            data: Price data (DataFrame with OHLC or list of prices)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with OU process analysis
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = data['Close'].values if 'Close' in data else data['close'].values
        else:
            if len(data) < self.lookback_periods:
                return self._empty_result(symbol)
            prices = np.array(data)
            
        # Take log prices for OU process
        log_prices = np.log(prices)
        
        # Estimate OU parameters
        if self.estimation_method == "mle":
            params = self._estimate_mle(log_prices)
        else:
            params = self._estimate_ols(log_prices)
            
        if params is None:
            return self._empty_result(symbol)
            
        theta, mu, sigma = params
        
        # Calculate half-life
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        # Check if process is mean-reverting within acceptable range
        is_mean_reverting = self.min_half_life <= half_life <= self.max_half_life
        
        # Calculate current deviation from mean
        current_log_price = log_prices[-1]
        deviation = current_log_price - mu
        z_score = deviation / (sigma / np.sqrt(2 * theta)) if theta > 0 else 0
        
        # Calculate expected value and confidence bands
        expected_value, confidence_bands = self._calculate_expected_path(
            current_log_price, theta, mu, sigma
        )
        
        # Analyze mean reversion opportunity
        reversion_analysis = self._analyze_reversion_opportunity(
            z_score, half_life, theta, sigma
        )
        
        # Test stationarity
        stationarity_test = self._test_stationarity(log_prices)
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            z_score, is_mean_reverting, reversion_analysis,
            stationarity_test, half_life
        )
        
        # Create metadata
        metadata = self._create_metadata(
            theta, mu, sigma, half_life, z_score,
            reversion_analysis, stationarity_test,
            expected_value, confidence_bands,
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
    
    def _estimate_mle(self, log_prices: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Estimate OU parameters using Maximum Likelihood Estimation"""
        
        n = len(log_prices)
        if n < 3:
            return None
            
        dt = 1  # Daily data
        
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            theta, mu, sigma = params
            
            if theta <= 0 or sigma <= 0:
                return np.inf
                
            # Calculate log-likelihood
            ll = 0
            for i in range(1, n):
                expected = log_prices[i-1] + theta * (mu - log_prices[i-1]) * dt
                variance = sigma**2 * dt
                
                # Add to log-likelihood
                ll += -0.5 * np.log(2 * np.pi * variance)
                ll += -0.5 * ((log_prices[i] - expected)**2) / variance
                
            return -ll
        
        # Initial parameter guesses
        mu_init = np.mean(log_prices)
        sigma_init = np.std(np.diff(log_prices))
        theta_init = -np.log(np.corrcoef(log_prices[:-1], log_prices[1:])[0, 1]) / dt
        
        if np.isnan(theta_init) or theta_init <= 0:
            theta_init = 0.1
            
        initial_params = [theta_init, mu_init, sigma_init]
        
        # Optimize
        try:
            result = minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=[(0.001, 10), (None, None), (0.001, 10)]
            )
            
            if result.success:
                return tuple(result.x)
            else:
                # Fall back to OLS
                return self._estimate_ols(log_prices)
                
        except:
            return self._estimate_ols(log_prices)
    
    def _estimate_ols(self, log_prices: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Estimate OU parameters using Ordinary Least Squares"""
        
        if len(log_prices) < 3:
            return None
            
        # Regression: X(t) = a + b*X(t-1) + error
        y = log_prices[1:]
        x = log_prices[:-1]
        
        # Add constant term
        X = np.column_stack([np.ones(len(x)), x])
        
        # OLS estimation
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            a, b = beta
            
            # Convert to OU parameters
            theta = -np.log(b) if b > 0 and b < 1 else 0.1
            mu = a / (1 - b) if b != 1 else np.mean(log_prices)
            
            # Estimate sigma from residuals
            residuals = y - (a + b * x)
            sigma = np.std(residuals) * np.sqrt(2 * theta)
            
            return theta, mu, sigma
            
        except:
            return None
    
    def _calculate_expected_path(self, 
                                current_value: float,
                                theta: float,
                                mu: float,
                                sigma: float) -> Tuple[float, Dict]:
        """Calculate expected value and confidence bands"""
        
        # Expected value after 1 period
        expected_1d = current_value * np.exp(-theta) + mu * (1 - np.exp(-theta))
        
        # Variance of the process
        if theta > 0:
            variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta))
        else:
            variance = sigma**2
            
        std_dev = np.sqrt(variance)
        
        # Confidence bands
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        confidence_bands = {
            'upper': expected_1d + z_score * std_dev,
            'lower': expected_1d - z_score * std_dev,
            'expected': expected_1d,
            'std_dev': std_dev
        }
        
        return expected_1d, confidence_bands
    
    def _analyze_reversion_opportunity(self,
                                      z_score: float,
                                      half_life: float,
                                      theta: float,
                                      sigma: float) -> Dict:
        """Analyze mean reversion trading opportunity"""
        
        # Expected return to mean
        if theta > 0:
            expected_return_time = half_life
            probability_of_reversion = 1 - np.exp(-theta * half_life)
        else:
            expected_return_time = np.inf
            probability_of_reversion = 0
            
        # Risk-reward ratio
        if abs(z_score) > 0.5:
            # Expected profit if reversion occurs
            expected_profit = abs(z_score) * (sigma / np.sqrt(2 * theta)) if theta > 0 else 0
            # Risk is one standard deviation move against us
            risk = sigma / np.sqrt(2 * theta) if theta > 0 else sigma
            risk_reward = expected_profit / risk if risk > 0 else 0
        else:
            risk_reward = 0
            
        # Reversion strength
        if abs(z_score) > 3:
            strength = 'extreme'
        elif abs(z_score) > 2:
            strength = 'strong'
        elif abs(z_score) > 1:
            strength = 'moderate'
        else:
            strength = 'weak'
            
        return {
            'z_score': z_score,
            'strength': strength,
            'expected_return_time': expected_return_time,
            'probability_of_reversion': probability_of_reversion,
            'risk_reward_ratio': risk_reward,
            'entry_quality': 'good' if abs(z_score) > self.z_score_threshold and risk_reward > 1.5 else 'poor'
        }
    
    def _test_stationarity(self, log_prices: np.ndarray) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Run ADF test
            result = adfuller(log_prices, autolag='AIC')
            
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Determine if stationary
            is_stationary = p_value < 0.05
            
            return {
                'is_stationary': is_stationary,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_1pct': critical_values.get('1%', np.nan),
                'critical_5pct': critical_values.get('5%', np.nan),
                'critical_10pct': critical_values.get('10%', np.nan)
            }
            
        except ImportError:
            # Simplified stationarity check
            returns = np.diff(log_prices)
            
            # Check if returns have stable mean and variance
            first_half_mean = np.mean(returns[:len(returns)//2])
            second_half_mean = np.mean(returns[len(returns)//2:])
            
            first_half_std = np.std(returns[:len(returns)//2])
            second_half_std = np.std(returns[len(returns)//2:])
            
            mean_stable = abs(first_half_mean - second_half_mean) < 0.001
            variance_stable = abs(first_half_std - second_half_std) / first_half_std < 0.3
            
            return {
                'is_stationary': mean_stable and variance_stable,
                'mean_stable': mean_stable,
                'variance_stable': variance_stable,
                'test_type': 'simplified'
            }
    
    def _generate_signal(self,
                        z_score: float,
                        is_mean_reverting: bool,
                        reversion_analysis: Dict,
                        stationarity_test: Dict,
                        half_life: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from OU process analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Check basic conditions
        if not is_mean_reverting:
            # Process not mean-reverting in acceptable timeframe
            return signal, confidence, 50.0
            
        if not stationarity_test.get('is_stationary', False):
            # Not stationary - reduce confidence
            confidence_multiplier = 0.7
        else:
            confidence_multiplier = 1.0
            
        # Generate signal based on Z-score
        if z_score < -self.z_score_threshold:
            # Oversold - Buy signal
            signal = SignalType.BUY
            confidence = min(abs(z_score) * 25, 80) * confidence_multiplier
            
        elif z_score > self.z_score_threshold:
            # Overbought - Sell signal
            signal = SignalType.SELL
            confidence = min(abs(z_score) * 25, 80) * confidence_multiplier
            
        else:
            # Within normal range
            signal = SignalType.HOLD
            confidence = 0
            
        # Adjust for reversion quality
        if reversion_analysis['entry_quality'] == 'good':
            confidence *= 1.2
        elif reversion_analysis['entry_quality'] == 'poor':
            confidence *= 0.8
            
        # Adjust for half-life
        if half_life < 5:
            # Very fast mean reversion
            confidence *= 1.1
        elif half_life > 30:
            # Slow mean reversion
            confidence *= 0.9
            
        confidence = min(confidence, 85)
        
        # Value represents mean reversion strength (0-100)
        value = min(abs(z_score) * 20, 100)
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        theta: float,
                        mu: float,
                        sigma: float,
                        half_life: float,
                        z_score: float,
                        reversion_analysis: Dict,
                        stationarity_test: Dict,
                        expected_value: float,
                        confidence_bands: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'ou_parameters': {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life': half_life
            },
            'current_state': {
                'z_score': z_score,
                'deviation_from_mean': z_score * (sigma / np.sqrt(2 * theta)) if theta > 0 else 0,
                'percentile': stats.norm.cdf(z_score) * 100
            },
            'mean_reversion': {
                'is_mean_reverting': self.min_half_life <= half_life <= self.max_half_life,
                'strength': reversion_analysis['strength'],
                'expected_return_time': reversion_analysis['expected_return_time'],
                'probability': reversion_analysis['probability_of_reversion'],
                'risk_reward': reversion_analysis['risk_reward_ratio']
            },
            'forecast': {
                'expected_1d': np.exp(expected_value),  # Convert back from log
                'upper_band': np.exp(confidence_bands['upper']),
                'lower_band': np.exp(confidence_bands['lower']),
                'confidence_level': self.confidence_level
            },
            'stationarity': stationarity_test
        }
        
        # Add trading insights
        if abs(z_score) > 3:
            metadata['insight'] = f"Extreme deviation ({z_score:.1f}σ) - high probability reversal"
        elif abs(z_score) > 2:
            metadata['insight'] = f"Significant deviation - {reversion_analysis['entry_quality']} entry"
        elif half_life > self.max_half_life:
            metadata['insight'] = "Process too slow - not suitable for mean reversion"
        elif not stationarity_test.get('is_stationary', False):
            metadata['insight'] = "Non-stationary process - use with caution"
        else:
            metadata['insight'] = "Normal range - wait for better opportunity"
            
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
            metadata={'error': 'Insufficient data for OU process estimation'},
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


def demonstrate_ou_process():
    """Demonstration of OU Process indicator"""
    
    print("📐 Ornstein-Uhlenbeck Process Demonstration\n")
    
    # Generate synthetic mean-reverting data
    print("Generating synthetic mean-reverting price data...\n")
    
    np.random.seed(42)
    n_points = 200
    dt = 1  # Daily
    
    # OU process parameters
    true_theta = 0.15  # Mean reversion speed
    true_mu = np.log(100)  # Long-term mean (log price)
    true_sigma = 0.02  # Volatility
    
    print(f"True parameters:")
    print(f"  θ (theta): {true_theta:.3f}")
    print(f"  μ (mu): {true_mu:.3f}")
    print(f"  σ (sigma): {true_sigma:.3f}")
    print(f"  Half-life: {np.log(2)/true_theta:.1f} days\n")
    
    # Simulate OU process
    log_prices = [true_mu]
    
    for i in range(1, n_points):
        # OU process update
        drift = true_theta * (true_mu - log_prices[-1]) * dt
        diffusion = true_sigma * np.sqrt(dt) * np.random.randn()
        
        new_log_price = log_prices[-1] + drift + diffusion
        log_prices.append(new_log_price)
    
    # Convert to prices
    prices = np.exp(log_prices)
    
    # Add some trends to test robustness
    if n_points > 150:
        # Add uptrend in middle
        prices[50:100] *= np.linspace(1, 1.2, 50)
        # Add downtrend at end
        prices[150:] *= np.linspace(1, 0.9, len(prices[150:]))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices
    }, index=pd.date_range('2024-01-01', periods=n_points, freq='D'))
    
    # Create indicator
    ou_indicator = OUProcessIndicator(
        lookback_periods=100,
        estimation_method='mle',
        min_half_life=1.0,
        max_half_life=50.0,
        z_score_threshold=2.0
    )
    
    # Calculate
    result = ou_indicator.calculate(data, "SYNTHETIC")
    
    print("=" * 60)
    print("OU PROCESS ANALYSIS:")
    print("=" * 60)
    
    # Estimated parameters
    ou_params = result.metadata['ou_parameters']
    print(f"\nEstimated Parameters:")
    print(f"  θ (theta): {ou_params['theta']:.3f}")
    print(f"  μ (mu): {ou_params['mu']:.3f}")
    print(f"  σ (sigma): {ou_params['sigma']:.3f}")
    print(f"  Half-life: {ou_params['half_life']:.1f} days")
    
    # Current state
    current = result.metadata['current_state']
    print(f"\nCurrent State:")
    print(f"  Z-score: {current['z_score']:.2f}")
    print(f"  Deviation: {current['deviation_from_mean']:.3f}")
    print(f"  Percentile: {current['percentile']:.1f}%")
    
    # Mean reversion analysis
    mr = result.metadata['mean_reversion']
    print(f"\nMean Reversion Analysis:")
    print(f"  Is Mean-Reverting: {mr['is_mean_reverting']}")
    print(f"  Strength: {mr['strength'].upper()}")
    print(f"  Expected Return Time: {mr['expected_return_time']:.1f} days")
    print(f"  Probability of Reversion: {mr['probability']:.1%}")
    print(f"  Risk/Reward Ratio: {mr['risk_reward']:.2f}")
    
    # Forecast
    forecast = result.metadata['forecast']
    current_price = prices[-1]
    print(f"\nPrice Forecast:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Expected 1-Day: ${forecast['expected_1d']:.2f}")
    print(f"  Upper Band ({ou_indicator.confidence_level:.0%}): ${forecast['upper_band']:.2f}")
    print(f"  Lower Band ({ou_indicator.confidence_level:.0%}): ${forecast['lower_band']:.2f}")
    
    # Stationarity test
    stat_test = result.metadata['stationarity']
    print(f"\nStationarity Test:")
    print(f"  Is Stationary: {stat_test.get('is_stationary', False)}")
    if 'p_value' in stat_test:
        print(f"  ADF p-value: {stat_test['p_value']:.4f}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Mean Reversion Strength: {result.value:.1f}/100")
    
    # Insight
    print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    # Test with different scenarios
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT SCENARIOS:")
    print("=" * 60)
    
    # Scenario 1: Strong trend (non-stationary)
    trend_prices = prices * np.linspace(1, 2, len(prices))
    trend_data = pd.DataFrame({'Close': trend_prices}, index=data.index)
    
    trend_result = ou_indicator.calculate(trend_data, "TREND")
    print(f"\n1. Strong Trend Data:")
    print(f"   Is Mean-Reverting: {trend_result.metadata['mean_reversion']['is_mean_reverting']}")
    print(f"   Half-life: {trend_result.metadata['ou_parameters']['half_life']:.1f} days")
    print(f"   Signal: {trend_result.signal.value}")
    
    # Scenario 2: Pure noise (no mean reversion)
    noise_prices = 100 + np.random.randn(len(prices)) * 5
    noise_data = pd.DataFrame({'Close': noise_prices}, index=data.index)
    
    noise_result = ou_indicator.calculate(noise_data, "NOISE")
    print(f"\n2. Pure Noise Data:")
    print(f"   Is Mean-Reverting: {noise_result.metadata['mean_reversion']['is_mean_reverting']}")
    print(f"   Half-life: {noise_result.metadata['ou_parameters']['half_life']:.1f} days")
    print(f"   Signal: {noise_result.signal.value}")
    
    print("\n💡 OU Process Trading Tips:")
    print("- Half-life 5-20 days ideal for daily trading")
    print("- Z-score > 2 suggests oversold (buy)")
    print("- Z-score < -2 suggests overbought (sell)")
    print("- Check stationarity before trusting signals")
    print("- Combine with pairs trading for best results")
    print("- Higher theta = faster mean reversion = more trades")


if __name__ == "__main__":
    demonstrate_ou_process()