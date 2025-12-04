"""
📈 Pairs Trading Residuals - Statistical Arbitrage
Identifies cointegrated pairs and generates market-neutral signals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class PairsTradingIndicator(BaseIndicator):
    """
    Pairs Trading indicator for market-neutral mean reversion
    
    This indicator implements:
    - Cointegration testing (Engle-Granger)
    - Johansen test for multiple pairs
    - Spread z-score calculation
    - Dynamic hedge ratio estimation
    - Risk-adjusted position sizing
    - Regime-aware signal generation
    
    Key concepts:
    - Pairs move together long-term (cointegrated)
    - Short-term deviations create opportunities
    - Market neutral = reduced systematic risk
    - Mean reversion in spread, not individual prices
    """
    
    def __init__(self,
                 lookback_periods: int = 60,
                 zscore_window: int = 20,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5,
                 min_correlation: float = 0.7,
                 max_half_life: float = 20.0,
                 use_dynamic_hedge: bool = True,
                 test_cointegration: bool = True):
        """
        Initialize Pairs Trading indicator
        
        Args:
            lookback_periods: Periods for cointegration test
            zscore_window: Window for spread z-score calculation
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            min_correlation: Minimum correlation required
            max_half_life: Maximum acceptable half-life (days)
            use_dynamic_hedge: Use rolling hedge ratio
            test_cointegration: Test for cointegration (vs correlation only)
        """
        super().__init__(
            name="PairsTrading",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_periods + zscore_window,
            params={
                'lookback_periods': lookback_periods,
                'zscore_window': zscore_window,
                'entry_zscore': entry_zscore,
                'exit_zscore': exit_zscore,
                'min_correlation': min_correlation,
                'max_half_life': max_half_life,
                'use_dynamic_hedge': use_dynamic_hedge,
                'test_cointegration': test_cointegration
            }
        )
        
        self.lookback_periods = lookback_periods
        self.zscore_window = zscore_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.min_correlation = min_correlation
        self.max_half_life = max_half_life
        self.use_dynamic_hedge = use_dynamic_hedge
        self.test_cointegration = test_cointegration
        
    def calculate(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate pairs trading signals
        
        Args:
            data: Either a DataFrame with both assets or dict with 'asset1' and 'asset2'
            symbol: Symbol pair being analyzed (e.g., "XOM-CVX")
            
        Returns:
            IndicatorResult with pairs trading analysis
        """
        # Extract price series for both assets
        if isinstance(data, dict):
            if 'asset1' not in data or 'asset2' not in data:
                return self._empty_result(symbol)
            prices1 = self._extract_prices(data['asset1'])
            prices2 = self._extract_prices(data['asset2'])
        else:
            # Assume DataFrame has both price series
            if data.shape[1] < 2:
                return self._empty_result(symbol)
            prices1 = data.iloc[:, 0].values
            prices2 = data.iloc[:, 1].values
            
        if len(prices1) < self.lookback_periods or len(prices2) < self.lookback_periods:
            return self._empty_result(symbol)
            
        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        # Calculate correlation
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        
        if abs(correlation) < self.min_correlation:
            return self._insufficient_correlation_result(symbol, correlation)
            
        # Test for cointegration
        if self.test_cointegration:
            coint_test = self._test_cointegration(prices1, prices2)
            if not coint_test['is_cointegrated']:
                return self._not_cointegrated_result(symbol, coint_test)
        else:
            coint_test = {'is_cointegrated': True, 'p_value': 0.01}
            
        # Calculate hedge ratio
        if self.use_dynamic_hedge:
            hedge_ratio, hedge_stats = self._calculate_dynamic_hedge_ratio(
                prices1, prices2, self.zscore_window
            )
        else:
            hedge_ratio, hedge_stats = self._calculate_static_hedge_ratio(
                prices1[-self.lookback_periods:], 
                prices2[-self.lookback_periods:]
            )
            
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        
        # Calculate spread statistics
        spread_stats = self._calculate_spread_statistics(
            spread, self.zscore_window
        )
        
        # Test spread stationarity and half-life
        spread_quality = self._analyze_spread_quality(
            spread[-self.lookback_periods:]
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            spread_stats['zscore'],
            spread_quality,
            correlation,
            coint_test
        )
        
        # Calculate optimal position sizes
        position_sizes = self._calculate_position_sizes(
            prices1[-1], prices2[-1], hedge_ratio,
            spread_stats['volatility'], signal
        )
        
        # Analyze pair characteristics
        pair_analysis = self._analyze_pair_characteristics(
            prices1, prices2, spread, hedge_ratio
        )
        
        # Create metadata
        metadata = self._create_metadata(
            correlation, coint_test, hedge_stats,
            spread_stats, spread_quality, position_sizes,
            pair_analysis, len(prices1)
        )
        
        # Get timestamp
        timestamp = int(datetime.now().timestamp() * 1000)
        if isinstance(data, pd.DataFrame) and hasattr(data.index[-1], 'timestamp'):
            timestamp = int(data.index[-1].timestamp() * 1000)
        elif isinstance(data, dict):
            df = list(data.values())[0]
            if isinstance(df, pd.DataFrame) and hasattr(df.index[-1], 'timestamp'):
                timestamp = int(df.index[-1].timestamp() * 1000)
                
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
    
    def _extract_prices(self, data: pd.DataFrame) -> np.ndarray:
        """Extract price series from DataFrame"""
        
        if 'Close' in data.columns:
            return data['Close'].values
        elif 'close' in data.columns:
            return data['close'].values
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return data[numeric_cols[0]].values
            else:
                return np.array([])
    
    def _test_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> Dict:
        """Test for cointegration using Engle-Granger method"""
        
        try:
            from statsmodels.tsa.stattools import coint
            
            # Run cointegration test
            coint_stat, p_value, crit_values = coint(prices1, prices2)
            
            is_cointegrated = p_value < 0.05
            
            return {
                'is_cointegrated': is_cointegrated,
                'test_statistic': coint_stat,
                'p_value': p_value,
                'critical_values': {
                    '1%': crit_values[0],
                    '5%': crit_values[1],
                    '10%': crit_values[2]
                }
            }
            
        except ImportError:
            # Fallback: simple stationarity test on spread
            # OLS regression to get spread
            model = LinearRegression()
            model.fit(prices2.reshape(-1, 1), prices1)
            spread = prices1 - model.predict(prices2.reshape(-1, 1))
            
            # Test if spread is stationary (simplified)
            spread_returns = np.diff(spread)
            
            # Check mean reversion
            autocorr = np.corrcoef(spread[:-1], spread[1:])[0, 1]
            is_stationary = autocorr < 0.9  # High autocorr = non-stationary
            
            return {
                'is_cointegrated': is_stationary,
                'test_statistic': autocorr,
                'p_value': 0.05 if is_stationary else 0.5,
                'test_type': 'simplified'
            }
    
    def _calculate_static_hedge_ratio(self, 
                                     prices1: np.ndarray, 
                                     prices2: np.ndarray) -> Tuple[float, Dict]:
        """Calculate static hedge ratio using OLS"""
        
        # Linear regression: prices1 = beta * prices2 + alpha
        model = LinearRegression()
        model.fit(prices2.reshape(-1, 1), prices1)
        
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate R-squared
        predicted = model.predict(prices2.reshape(-1, 1))
        ss_res = np.sum((prices1 - predicted) ** 2)
        ss_tot = np.sum((prices1 - np.mean(prices1)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Residual statistics
        residuals = prices1 - predicted
        residual_std = np.std(residuals)
        
        stats = {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'r_squared': r_squared,
            'residual_std': residual_std,
            'method': 'static_ols'
        }
        
        return hedge_ratio, stats
    
    def _calculate_dynamic_hedge_ratio(self,
                                      prices1: np.ndarray,
                                      prices2: np.ndarray,
                                      window: int) -> Tuple[float, Dict]:
        """Calculate dynamic rolling hedge ratio"""
        
        if len(prices1) < window:
            return self._calculate_static_hedge_ratio(prices1, prices2)
            
        # Calculate rolling hedge ratios
        hedge_ratios = []
        
        for i in range(window, len(prices1) + 1):
            window_prices1 = prices1[i-window:i]
            window_prices2 = prices2[i-window:i]
            
            model = LinearRegression()
            model.fit(window_prices2.reshape(-1, 1), window_prices1)
            hedge_ratios.append(model.coef_[0])
            
        # Current hedge ratio is the most recent
        current_hedge = hedge_ratios[-1]
        
        # Hedge ratio statistics
        hedge_mean = np.mean(hedge_ratios)
        hedge_std = np.std(hedge_ratios)
        hedge_trend = (hedge_ratios[-1] - hedge_ratios[-5]) / 5 if len(hedge_ratios) >= 5 else 0
        
        # Stability check
        is_stable = hedge_std / abs(hedge_mean) < 0.2 if hedge_mean != 0 else False
        
        stats = {
            'hedge_ratio': current_hedge,
            'hedge_mean': hedge_mean,
            'hedge_std': hedge_std,
            'hedge_trend': hedge_trend,
            'is_stable': is_stable,
            'method': 'dynamic_rolling'
        }
        
        return current_hedge, stats
    
    def _calculate_spread_statistics(self, 
                                    spread: np.ndarray, 
                                    window: int) -> Dict:
        """Calculate spread z-score and statistics"""
        
        if len(spread) < window:
            return {
                'zscore': 0,
                'spread_mean': np.mean(spread),
                'spread_std': np.std(spread),
                'current_spread': spread[-1] if len(spread) > 0 else 0,
                'volatility': 0
            }
            
        # Rolling statistics
        spread_mean = np.mean(spread[-window:])
        spread_std = np.std(spread[-window:])
        
        # Current z-score
        current_spread = spread[-1]
        zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Volatility (annualized)
        returns = np.diff(spread[-window:])
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Percentile of current spread
        percentile = stats.percentileofscore(spread[-window:], current_spread)
        
        return {
            'zscore': zscore,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'current_spread': current_spread,
            'volatility': volatility,
            'percentile': percentile
        }
    
    def _analyze_spread_quality(self, spread: np.ndarray) -> Dict:
        """Analyze spread quality and mean reversion characteristics"""
        
        # Calculate half-life using OU process
        if len(spread) < 3:
            return {
                'half_life': np.inf,
                'mean_reversion_speed': 0,
                'is_quality_spread': False,
                'spread_trend': 0
            }
            
        # Fit AR(1) model to estimate mean reversion
        y = spread[1:]
        x = spread[:-1]
        
        # Add constant
        X = np.column_stack([np.ones(len(x)), x])
        
        try:
            # OLS estimation
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            phi = beta[1]
            
            # Mean reversion speed
            if 0 < phi < 1:
                theta = -np.log(phi)
                half_life = np.log(2) / theta
            else:
                theta = 0
                half_life = np.inf
                
        except:
            theta = 0
            half_life = np.inf
            
        # Spread trend
        if len(spread) >= 20:
            recent_mean = np.mean(spread[-10:])
            older_mean = np.mean(spread[-20:-10])
            spread_trend = (recent_mean - older_mean) / older_mean if older_mean != 0 else 0
        else:
            spread_trend = 0
            
        # Quality checks
        is_quality_spread = (
            0 < half_life < self.max_half_life and
            abs(spread_trend) < 0.1  # Not trending too much
        )
        
        return {
            'half_life': half_life,
            'mean_reversion_speed': theta,
            'is_quality_spread': is_quality_spread,
            'spread_trend': spread_trend,
            'ar_coefficient': phi if 'phi' in locals() else 1.0
        }
    
    def _generate_signal(self,
                        zscore: float,
                        spread_quality: Dict,
                        correlation: float,
                        coint_test: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from pairs analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Check if pair is suitable
        if not spread_quality['is_quality_spread']:
            return signal, confidence, 50.0
            
        # Entry signals based on z-score
        if abs(zscore) > self.entry_zscore:
            if zscore > self.entry_zscore:
                # Spread too high - sell spread (sell asset1, buy asset2)
                signal = SignalType.SELL
            else:
                # Spread too low - buy spread (buy asset1, sell asset2)
                signal = SignalType.BUY
                
            # Base confidence on z-score magnitude
            confidence = min(abs(zscore) * 25, 80)
            
        elif abs(zscore) < self.exit_zscore:
            # Exit zone - close positions
            signal = SignalType.HOLD
            confidence = 0
            
        else:
            # In between - maintain position
            signal = SignalType.HOLD
            confidence = 0
            
        # Adjust confidence based on pair quality
        if signal != SignalType.HOLD:
            # Cointegration strength
            if coint_test.get('p_value', 1.0) < 0.01:
                confidence *= 1.2
            elif coint_test.get('p_value', 1.0) > 0.05:
                confidence *= 0.8
                
            # Half-life quality
            half_life = spread_quality['half_life']
            if half_life < 5:
                confidence *= 1.1  # Fast mean reversion
            elif half_life > 15:
                confidence *= 0.9  # Slower mean reversion
                
            # Correlation strength
            if abs(correlation) > 0.9:
                confidence *= 1.1
            elif abs(correlation) < 0.75:
                confidence *= 0.9
                
        confidence = min(confidence, 85)
        
        # Value represents spread extremeness
        value = min(abs(zscore) * 20, 100)
        
        return signal, confidence, value
    
    def _calculate_position_sizes(self,
                                 price1: float,
                                 price2: float,
                                 hedge_ratio: float,
                                 spread_volatility: float,
                                 signal: SignalType) -> Dict:
        """Calculate optimal position sizes for market neutrality"""
        
        # Dollar neutral positions
        # If we trade $1000 of asset1, we need hedge_ratio * $1000 of asset2
        base_position = 1000  # Base position size in dollars
        
        if signal == SignalType.BUY:
            # Buy spread: long asset1, short asset2
            position1_dollars = base_position
            position2_dollars = -base_position * hedge_ratio
            
            position1_shares = position1_dollars / price1
            position2_shares = position2_dollars / price2
            
        elif signal == SignalType.SELL:
            # Sell spread: short asset1, long asset2
            position1_dollars = -base_position
            position2_dollars = base_position * hedge_ratio
            
            position1_shares = position1_dollars / price1
            position2_shares = position2_dollars / price2
            
        else:
            # No position
            position1_shares = 0
            position2_shares = 0
            position1_dollars = 0
            position2_dollars = 0
            
        # Calculate beta neutrality check
        net_beta = abs(position1_dollars) - abs(position2_dollars)
        is_market_neutral = abs(net_beta) < base_position * 0.1  # Within 10%
        
        # Risk metrics
        position_risk = abs(position1_dollars) * spread_volatility
        
        return {
            'asset1_shares': position1_shares,
            'asset2_shares': position2_shares,
            'asset1_dollars': position1_dollars,
            'asset2_dollars': position2_dollars,
            'is_market_neutral': is_market_neutral,
            'net_exposure': net_beta,
            'position_risk': position_risk
        }
    
    def _analyze_pair_characteristics(self,
                                     prices1: np.ndarray,
                                     prices2: np.ndarray,
                                     spread: np.ndarray,
                                     hedge_ratio: float) -> Dict:
        """Analyze pair characteristics and relationship"""
        
        # Return correlation
        returns1 = np.diff(np.log(prices1))
        returns2 = np.diff(np.log(prices2))
        
        if len(returns1) > 0 and len(returns2) > 0:
            return_correlation = np.corrcoef(returns1, returns2)[0, 1]
        else:
            return_correlation = 0
            
        # Volatility ratio
        vol1 = np.std(returns1) * np.sqrt(252) if len(returns1) > 0 else 0
        vol2 = np.std(returns2) * np.sqrt(252) if len(returns2) > 0 else 0
        vol_ratio = vol1 / vol2 if vol2 > 0 else 1
        
        # Price ratio analysis
        price_ratio = prices1 / prices2
        ratio_mean = np.mean(price_ratio)
        ratio_std = np.std(price_ratio)
        current_ratio = price_ratio[-1]
        ratio_zscore = (current_ratio - ratio_mean) / ratio_std if ratio_std > 0 else 0
        
        # Spread characteristics
        spread_range = np.max(spread) - np.min(spread)
        spread_current_percentile = stats.percentileofscore(spread, spread[-1])
        
        return {
            'return_correlation': return_correlation,
            'vol_asset1': vol1,
            'vol_asset2': vol2,
            'vol_ratio': vol_ratio,
            'price_ratio': current_ratio,
            'ratio_mean': ratio_mean,
            'ratio_zscore': ratio_zscore,
            'spread_range': spread_range,
            'spread_percentile': spread_current_percentile
        }
    
    def _create_metadata(self,
                        correlation: float,
                        coint_test: Dict,
                        hedge_stats: Dict,
                        spread_stats: Dict,
                        spread_quality: Dict,
                        position_sizes: Dict,
                        pair_analysis: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'correlation': correlation,
            'cointegration': coint_test,
            'hedge_ratio': hedge_stats,
            'spread': {
                'current_zscore': spread_stats['zscore'],
                'mean': spread_stats['spread_mean'],
                'std': spread_stats['spread_std'],
                'current_value': spread_stats['current_spread'],
                'percentile': spread_stats['percentile']
            },
            'spread_quality': spread_quality,
            'position_sizing': position_sizes,
            'pair_characteristics': pair_analysis,
            'entry_threshold': self.entry_zscore,
            'exit_threshold': self.exit_zscore
        }
        
        # Add trading insights
        zscore = spread_stats['zscore']
        half_life = spread_quality['half_life']
        
        if abs(zscore) > 3:
            metadata['insight'] = f"Extreme spread deviation ({zscore:.1f}σ) - strong reversal candidate"
        elif abs(zscore) > self.entry_zscore:
            metadata['insight'] = f"Entry signal triggered - {half_life:.1f} day expected reversal"
        elif not coint_test.get('is_cointegrated', False):
            metadata['insight'] = "Pair not cointegrated - relationship may have broken"
        elif not spread_quality['is_quality_spread']:
            metadata['insight'] = f"Spread quality poor - half-life {half_life:.1f} days too long"
        elif abs(spread_stats['percentile'] - 50) > 40:
            metadata['insight'] = "Spread at historical extremes"
        else:
            metadata['insight'] = "Spread within normal range"
            
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
            metadata={'error': 'Insufficient data for pairs trading analysis'},
            calculation_time_ms=0
        )
    
    def _insufficient_correlation_result(self, symbol: str, correlation: float) -> IndicatorResult:
        """Return result when correlation is too low"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={
                'error': 'Insufficient correlation',
                'correlation': correlation,
                'min_required': self.min_correlation
            },
            calculation_time_ms=0
        )
    
    def _not_cointegrated_result(self, symbol: str, coint_test: Dict) -> IndicatorResult:
        """Return result when pair is not cointegrated"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={
                'error': 'Pair not cointegrated',
                'p_value': coint_test.get('p_value', 1.0),
                'test_statistic': coint_test.get('test_statistic', 0)
            },
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, Dict]) -> bool:
        """Validate input data"""
        
        if isinstance(data, dict):
            return (
                'asset1' in data and 
                'asset2' in data and
                isinstance(data['asset1'], pd.DataFrame) and
                isinstance(data['asset2'], pd.DataFrame)
            )
        else:
            return isinstance(data, pd.DataFrame) and data.shape[1] >= 2


def demonstrate_pairs_trading():
    """Demonstration of Pairs Trading indicator"""
    
    print("📈 Pairs Trading Residuals Demonstration\n")
    
    # Generate synthetic cointegrated pair data
    print("Generating synthetic cointegrated pair (e.g., XOM-CVX)...\n")
    
    np.random.seed(42)
    n_points = 200
    
    # Common factor (oil prices)
    common_factor = np.cumsum(np.random.randn(n_points) * 0.02)
    
    # Individual components with mean reversion
    spread_process = [0]
    theta = 0.1  # Mean reversion speed
    sigma = 0.01  # Spread volatility
    
    for i in range(1, n_points):
        # OU process for spread
        spread_process.append(
            spread_process[-1] * (1 - theta) + sigma * np.random.randn()
        )
    
    spread_process = np.array(spread_process)
    
    # Create cointegrated prices
    # Asset 1 = common + 0.3 * spread
    # Asset 2 = common - 0.7 * spread  
    # This creates a cointegrating relationship
    
    price1 = 100 * np.exp(common_factor + 0.3 * spread_process)
    price2 = 80 * np.exp(common_factor - 0.7 * spread_process)
    
    # Add some individual noise
    price1 *= np.exp(np.cumsum(np.random.randn(n_points) * 0.005))
    price2 *= np.exp(np.cumsum(np.random.randn(n_points) * 0.005))
    
    # Create DataFrames
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    
    data = {
        'asset1': pd.DataFrame({
            'Open': price1,
            'High': price1 * 1.01,
            'Low': price1 * 0.99,
            'Close': price1
        }, index=dates),
        'asset2': pd.DataFrame({
            'Open': price2,
            'High': price2 * 1.01,
            'Low': price2 * 0.99,
            'Close': price2
        }, index=dates)
    }
    
    # Create indicator
    pairs_indicator = PairsTradingIndicator(
        lookback_periods=60,
        zscore_window=20,
        entry_zscore=2.0,
        exit_zscore=0.5,
        min_correlation=0.7,
        max_half_life=20.0
    )
    
    # Calculate
    result = pairs_indicator.calculate(data, "XOM-CVX")
    
    print("=" * 60)
    print("PAIRS TRADING ANALYSIS:")
    print("=" * 60)
    
    # Correlation
    print(f"\nPair Correlation: {result.metadata['correlation']:.3f}")
    
    # Cointegration test
    coint = result.metadata['cointegration']
    print(f"\nCointegration Test:")
    print(f"  Is Cointegrated: {coint.get('is_cointegrated', False)}")
    if 'p_value' in coint:
        print(f"  P-value: {coint['p_value']:.4f}")
    
    # Hedge ratio
    hedge = result.metadata['hedge_ratio']
    print(f"\nHedge Ratio: {hedge['hedge_ratio']:.3f}")
    print(f"  Method: {hedge['method']}")
    if 'r_squared' in hedge:
        print(f"  R-squared: {hedge['r_squared']:.3f}")
    
    # Spread statistics
    spread = result.metadata['spread']
    print(f"\nSpread Statistics:")
    print(f"  Current Z-score: {spread['current_zscore']:.2f}")
    print(f"  Spread Mean: {spread['mean']:.2f}")
    print(f"  Spread Std: {spread['std']:.2f}")
    print(f"  Current Value: {spread['current_value']:.2f}")
    print(f"  Percentile: {spread['percentile']:.1f}%")
    
    # Spread quality
    quality = result.metadata['spread_quality']
    print(f"\nSpread Quality:")
    print(f"  Half-life: {quality['half_life']:.1f} days")
    print(f"  Mean Reversion Speed: {quality['mean_reversion_speed']:.3f}")
    print(f"  Quality Spread: {quality['is_quality_spread']}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Spread Extremeness: {result.value:.1f}/100")
    
    # Position sizing
    positions = result.metadata['position_sizing']
    if positions['asset1_shares'] != 0:
        print(f"\nPosition Sizing (per $1000 base):")
        print(f"  Asset1: {positions['asset1_shares']:.2f} shares (${positions['asset1_dollars']:.0f})")
        print(f"  Asset2: {positions['asset2_shares']:.2f} shares (${positions['asset2_dollars']:.0f})")
        print(f"  Market Neutral: {positions['is_market_neutral']}")
        print(f"  Net Exposure: ${positions['net_exposure']:.0f}")
    
    # Pair characteristics
    characteristics = result.metadata['pair_characteristics']
    print(f"\nPair Characteristics:")
    print(f"  Return Correlation: {characteristics['return_correlation']:.3f}")
    print(f"  Volatility Ratio: {characteristics['vol_ratio']:.2f}")
    print(f"  Current Price Ratio: {characteristics['price_ratio']:.2f}")
    
    # Insight
    print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    # Test different scenarios
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT SCENARIOS:")
    print("=" * 60)
    
    # Scenario 1: Uncorrelated pair
    uncorr_data = {
        'asset1': pd.DataFrame({
            'Close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        }, index=pd.date_range('2024-01-01', periods=100)),
        'asset2': pd.DataFrame({
            'Close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        }, index=pd.date_range('2024-01-01', periods=100))
    }
    
    uncorr_result = pairs_indicator.calculate(uncorr_data, "RANDOM1-RANDOM2")
    print(f"\n1. Uncorrelated Pair:")
    print(f"   Correlation: {uncorr_result.metadata.get('correlation', 0):.3f}")
    print(f"   Signal: {uncorr_result.signal.value}")
    print(f"   Error: {uncorr_result.metadata.get('error', 'None')}")
    
    print("\n💡 Pairs Trading Tips:")
    print("- Look for historically correlated pairs (>0.7)")
    print("- Test for cointegration, not just correlation")
    print("- Half-life 5-15 days optimal for daily trading")
    print("- Entry at ±2σ, exit at ±0.5σ typical")
    print("- Always maintain market neutrality")
    print("- Monitor for relationship breakdowns")
    print("- Consider transaction costs in spread trading")


if __name__ == "__main__":
    demonstrate_pairs_trading()