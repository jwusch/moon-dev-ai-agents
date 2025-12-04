"""
🔍 PPIN (Probability of Informed Trading) - Information Asymmetry Detection
Estimates the probability that trades are informed using structural microstructure models
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
import warnings
from scipy.optimize import minimize

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class PPINIndicator(BaseIndicator):
    """
    Probability of Informed Trading (PPIN) using the EHO model
    
    PPIN estimates the probability that any given trade is informed
    based on order flow and trade direction clustering.
    
    Model components:
    - α (alpha): Probability of information event occurring
    - δ (delta): Probability that information is good news (vs bad news)  
    - μ (mu): Rate of informed trading
    - εb (epsilon_buy): Rate of uninformed buy orders
    - εs (epsilon_sell): Rate of uninformed sell orders
    
    PPIN = (α * μ) / (α * μ + εb + εs)
    
    Higher PPIN indicates:
    - More informed trading
    - Greater information asymmetry
    - Potential price movements ahead
    - Market inefficiency opportunities
    
    Lower PPIN indicates:
    - Mostly noise trading
    - Efficient price discovery
    - Lower information content
    """
    
    def __init__(self,
                 estimation_window: int = 60,
                 min_trades_per_day: int = 100,
                 max_iterations: int = 100,
                 convergence_tolerance: float = 1e-6):
        """
        Initialize PPIN indicator
        
        Args:
            estimation_window: Number of periods for estimation
            min_trades_per_day: Minimum trades required per period
            max_iterations: Maximum iterations for optimization
            convergence_tolerance: Convergence criteria for optimization
        """
        super().__init__(
            name="PPIN",
            timeframe=TimeFrame.DAILY,  # Typically estimated on daily data
            lookback_periods=estimation_window,
            params={
                'estimation_window': estimation_window,
                'min_trades_per_day': min_trades_per_day,
                'max_iterations': max_iterations,
                'convergence_tolerance': convergence_tolerance
            }
        )
        
        self.estimation_window = estimation_window
        self.min_trades_per_day = min_trades_per_day
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        # Storage for PPIN estimation
        self.daily_data: Deque[Dict] = deque(maxlen=estimation_window)
        self.ppin_history: Deque[float] = deque(maxlen=60)  # 60-day PPIN history
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate PPIN from trade data
        
        Args:
            data: Trade data (tick data or aggregated daily data)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with PPIN analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Convert OHLCV to synthetic trade data
            daily_periods = self._convert_ohlcv_to_daily_periods(data)
        else:
            # Aggregate tick data into daily periods
            daily_periods = self._aggregate_ticks_to_daily(data)
            
        if len(daily_periods) < 20:
            return self._empty_result(symbol)
            
        # Store daily data
        for period in daily_periods[-min(len(daily_periods), 10):]:  # Last 10 days
            self.daily_data.append(period)
            
        if len(self.daily_data) < 20:
            return self._empty_result(symbol)
            
        # Estimate PPIN using Maximum Likelihood
        ppin_estimate, model_params = self._estimate_ppin()
        
        if ppin_estimate is None:
            return self._empty_result(symbol)
            
        # Store in history
        self.ppin_history.append(ppin_estimate)
        
        # Calculate additional metrics
        information_asymmetry = self._calculate_information_asymmetry(model_params)
        trading_intensity = self._calculate_trading_intensity(model_params)
        market_efficiency = self._calculate_market_efficiency()
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            ppin_estimate, information_asymmetry, trading_intensity
        )
        
        # Create metadata
        metadata = self._create_metadata(
            ppin_estimate, model_params, information_asymmetry,
            trading_intensity, market_efficiency
        )
        
        # Get latest timestamp
        latest_period = daily_periods[-1] if daily_periods else {}
        timestamp = latest_period.get('timestamp', int(datetime.now().timestamp() * 1000))
        
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
    
    def _convert_ohlcv_to_daily_periods(self, data: pd.DataFrame) -> List[Dict]:
        """Convert OHLCV data to synthetic daily trade periods"""
        
        daily_periods = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            # Generate synthetic buy/sell counts based on volume and price movement
            daily_volume = row['volume']
            price_change = (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0
            
            # Estimate buy/sell split based on price direction
            if price_change > 0:
                # Positive day - more buying
                buy_ratio = 0.6 + min(0.3, abs(price_change) * 5)
            elif price_change < 0:
                # Negative day - more selling  
                buy_ratio = 0.4 - min(0.3, abs(price_change) * 5)
            else:
                # Neutral - balanced
                buy_ratio = 0.5
                
            buy_ratio = max(0.1, min(0.9, buy_ratio))  # Keep reasonable bounds
            
            # Estimate number of trades (synthetic)
            estimated_trades = int(daily_volume / 100)  # Assume avg 100 shares per trade
            estimated_trades = max(self.min_trades_per_day, estimated_trades)
            
            buy_count = int(estimated_trades * buy_ratio)
            sell_count = estimated_trades - buy_count
            
            period = {
                'date': i,
                'timestamp': int(datetime.now().timestamp() * 1000) + i * 86400000,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_trades': estimated_trades,
                'volume': daily_volume,
                'price_change': price_change
            }
            
            daily_periods.append(period)
            
        return daily_periods
    
    def _aggregate_ticks_to_daily(self, ticks: List[TickData]) -> List[Dict]:
        """Aggregate tick data into daily periods"""
        
        daily_periods = {}
        
        for tick in ticks:
            # Get day key (timestamp in days)
            day_key = tick.timestamp // (24 * 60 * 60 * 1000)
            
            if day_key not in daily_periods:
                daily_periods[day_key] = {
                    'date': day_key,
                    'timestamp': tick.timestamp,
                    'buy_count': 0,
                    'sell_count': 0,
                    'total_volume': 0,
                    'prices': []
                }
                
            period = daily_periods[day_key]
            period['total_volume'] += tick.volume
            period['prices'].append(tick.price)
            
            if tick.side == 1:  # Buy
                period['buy_count'] += 1
            elif tick.side == -1:  # Sell
                period['sell_count'] += 1
        
        # Convert to list and calculate additional metrics
        result = []
        for day_key in sorted(daily_periods.keys()):
            period = daily_periods[day_key]
            
            # Calculate price change for the day
            if len(period['prices']) >= 2:
                price_change = (period['prices'][-1] - period['prices'][0]) / period['prices'][0]
            else:
                price_change = 0
                
            period['total_trades'] = period['buy_count'] + period['sell_count']
            period['volume'] = period['total_volume']
            period['price_change'] = price_change
            
            # Only include periods with sufficient trading
            if period['total_trades'] >= self.min_trades_per_day:
                result.append(period)
                
        return result
    
    def _estimate_ppin(self) -> Tuple[Optional[float], Optional[Dict]]:
        """Estimate PPIN using Maximum Likelihood Estimation"""
        
        if len(self.daily_data) < 20:
            return None, None
            
        # Prepare data for estimation
        buy_counts = []
        sell_counts = []
        
        for period in self.daily_data:
            buy_counts.append(period['buy_count'])
            sell_counts.append(period['sell_count'])
            
        buy_counts = np.array(buy_counts)
        sell_counts = np.array(sell_counts)
        
        # Initial parameter guesses
        # α (alpha): probability of info event ~ 0.2
        # δ (delta): probability of good news ~ 0.5  
        # μ (mu): informed trading rate
        # εb, εs: uninformed trading rates
        
        initial_params = np.array([
            0.2,  # alpha
            0.5,  # delta
            np.mean(buy_counts + sell_counts) * 0.3,  # mu
            np.mean(buy_counts) * 0.7,  # epsilon_buy
            np.mean(sell_counts) * 0.7   # epsilon_sell
        ])
        
        # Parameter bounds
        bounds = [
            (0.01, 0.99),  # alpha
            (0.01, 0.99),  # delta  
            (0.1, np.max(buy_counts + sell_counts) * 2),  # mu
            (0.1, np.max(buy_counts) * 2),  # epsilon_buy
            (0.1, np.max(sell_counts) * 2)  # epsilon_sell
        ]
        
        try:
            # Optimize likelihood function
            result = minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(buy_counts, sell_counts),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                alpha, delta, mu, epsilon_buy, epsilon_sell = result.x
                
                # Calculate PPIN
                ppin = (alpha * mu) / (alpha * mu + epsilon_buy + epsilon_sell)
                
                model_params = {
                    'alpha': alpha,
                    'delta': delta,
                    'mu': mu,
                    'epsilon_buy': epsilon_buy,
                    'epsilon_sell': epsilon_sell,
                    'likelihood': -result.fun,
                    'success': True
                }
                
                return ppin, model_params
                
            else:
                return None, None
                
        except Exception:
            return None, None
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                               buy_counts: np.ndarray, 
                               sell_counts: np.ndarray) -> float:
        """Calculate negative log-likelihood for PPIN model"""
        
        alpha, delta, mu, epsilon_buy, epsilon_sell = params
        
        # Ensure parameters are positive
        if any(p <= 0 for p in params) or alpha >= 1 or delta >= 1:
            return 1e10  # Large penalty for invalid params
            
        try:
            log_likelihood = 0
            
            for B, S in zip(buy_counts, sell_counts):
                # Three scenarios:
                # 1. No information event: Poisson(εb), Poisson(εs)
                # 2. Good news: Poisson(εb + μ), Poisson(εs)
                # 3. Bad news: Poisson(εb), Poisson(εs + μ)
                
                # Calculate Poisson probabilities (using log for numerical stability)
                from scipy.special import gammaln
                
                # Log Poisson PMF: log(λ^k * exp(-λ) / k!) = k*log(λ) - λ - log(k!)
                def log_poisson(k, lam):
                    if lam <= 0:
                        return -1e10
                    return k * np.log(lam) - lam - gammaln(k + 1)
                
                # Scenario probabilities
                prob_no_info = log_poisson(B, epsilon_buy) + log_poisson(S, epsilon_sell)
                prob_good_news = log_poisson(B, epsilon_buy + mu) + log_poisson(S, epsilon_sell)
                prob_bad_news = log_poisson(B, epsilon_buy) + log_poisson(S, epsilon_sell + mu)
                
                # Weighted likelihood
                likelihood_day = np.log(
                    (1 - alpha) * np.exp(prob_no_info) +
                    alpha * delta * np.exp(prob_good_news) +
                    alpha * (1 - delta) * np.exp(prob_bad_news)
                )
                
                log_likelihood += likelihood_day
                
            return -log_likelihood  # Return negative for minimization
            
        except (OverflowError, ValueError, RuntimeWarning):
            return 1e10  # Large penalty for numerical issues
    
    def _calculate_information_asymmetry(self, model_params: Optional[Dict]) -> float:
        """Calculate information asymmetry metric"""
        
        if not model_params:
            return 0.0
            
        # Information asymmetry increases with alpha and mu
        alpha = model_params.get('alpha', 0)
        mu = model_params.get('mu', 0)
        epsilon_buy = model_params.get('epsilon_buy', 1)
        epsilon_sell = model_params.get('epsilon_sell', 1)
        
        # Ratio of informed to uninformed trading
        uninformed_rate = epsilon_buy + epsilon_sell
        if uninformed_rate > 0:
            asymmetry_ratio = (alpha * mu) / uninformed_rate
        else:
            asymmetry_ratio = 0
            
        return min(asymmetry_ratio, 2.0)  # Cap at reasonable level
    
    def _calculate_trading_intensity(self, model_params: Optional[Dict]) -> float:
        """Calculate overall trading intensity"""
        
        if not model_params:
            return 0.0
            
        mu = model_params.get('mu', 0)
        epsilon_buy = model_params.get('epsilon_buy', 0)
        epsilon_sell = model_params.get('epsilon_sell', 0)
        
        return mu + epsilon_buy + epsilon_sell
    
    def _calculate_market_efficiency(self) -> str:
        """Classify market efficiency based on PPIN trend"""
        
        if len(self.ppin_history) < 10:
            return "unknown"
            
        recent_ppin = list(self.ppin_history)[-10:]
        avg_ppin = np.mean(recent_ppin)
        
        if avg_ppin < 0.1:
            return "highly_efficient"
        elif avg_ppin < 0.2:
            return "efficient"
        elif avg_ppin < 0.4:
            return "moderately_efficient"
        elif avg_ppin < 0.6:
            return "inefficient"
        else:
            return "highly_inefficient"
    
    def _generate_signal(self,
                        ppin: float,
                        information_asymmetry: float,
                        trading_intensity: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from PPIN analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # High PPIN suggests informed trading - follow the informed traders
        if ppin > 0.4:
            # Check recent order flow direction
            if len(self.daily_data) >= 3:
                recent_periods = list(self.daily_data)[-3:]
                net_buying = sum(p['buy_count'] - p['sell_count'] for p in recent_periods)
                
                if net_buying > 0:
                    signal = SignalType.BUY  # Follow informed buying
                    confidence = min(ppin * 100, 75)
                elif net_buying < 0:
                    signal = SignalType.SELL  # Follow informed selling
                    confidence = min(ppin * 100, 75)
                    
        elif ppin > 0.6:
            # Very high PPIN - strong informed trading
            confidence *= 1.3
            
        # Adjust confidence based on information asymmetry
        if information_asymmetry > 1.0:
            confidence *= 1.2
        elif information_asymmetry < 0.3:
            confidence *= 0.8
            
        # PPIN trend adjustment
        if len(self.ppin_history) >= 5:
            recent_ppins = list(self.ppin_history)[-5:]
            if len(recent_ppins) >= 3:
                trend = np.polyfit(range(len(recent_ppins)), recent_ppins, 1)[0]
                if trend > 0:  # Rising PPIN
                    confidence *= 1.1
                    
        confidence = min(confidence, 80)
        
        # Value represents information content (PPIN * 100)
        value = ppin * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        ppin: float,
                        model_params: Optional[Dict],
                        information_asymmetry: float,
                        trading_intensity: float,
                        market_efficiency: str) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'ppin': ppin,
            'information_asymmetry': information_asymmetry,
            'trading_intensity': trading_intensity,
            'market_efficiency': market_efficiency,
            'data_periods': len(self.daily_data),
            'ppin_history_length': len(self.ppin_history)
        }
        
        # Add model parameters
        if model_params:
            metadata.update({
                'alpha': model_params.get('alpha', 0),
                'delta': model_params.get('delta', 0),
                'mu': model_params.get('mu', 0),
                'epsilon_buy': model_params.get('epsilon_buy', 0),
                'epsilon_sell': model_params.get('epsilon_sell', 0),
                'likelihood': model_params.get('likelihood', 0),
                'model_success': model_params.get('success', False)
            })
        
        # PPIN statistics
        if self.ppin_history:
            ppin_values = list(self.ppin_history)
            metadata.update({
                'avg_ppin': np.mean(ppin_values),
                'min_ppin': np.min(ppin_values),
                'max_ppin': np.max(ppin_values),
                'ppin_volatility': np.std(ppin_values),
                'ppin_percentile_90': np.percentile(ppin_values, 90),
                'ppin_percentile_10': np.percentile(ppin_values, 10)
            })
            
            # PPIN trend
            if len(ppin_values) >= 5:
                trend = np.polyfit(range(len(ppin_values)), ppin_values, 1)[0]
                metadata['ppin_trend'] = 'increasing' if trend > 0 else 'decreasing'
                metadata['ppin_trend_slope'] = trend
        
        # Trading pattern analysis
        if len(self.daily_data) >= 5:
            recent_data = list(self.daily_data)[-5:]
            
            avg_buy_ratio = np.mean([p['buy_count'] / (p['buy_count'] + p['sell_count']) 
                                   for p in recent_data if p['buy_count'] + p['sell_count'] > 0])
            
            metadata.update({
                'recent_buy_ratio': avg_buy_ratio,
                'recent_avg_trades': np.mean([p['total_trades'] for p in recent_data]),
                'trading_balance': 'buy_heavy' if avg_buy_ratio > 0.55 else 'sell_heavy' if avg_buy_ratio < 0.45 else 'balanced'
            })
        
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=20.0,  # Neutral PPIN
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for PPIN calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= 1000  # Need many ticks for daily aggregation
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 20
        return False


def demonstrate_ppin():
    """Demonstration of PPIN indicator"""
    
    print("🔍 PPIN (Probability of Informed Trading) Demonstration\n")
    
    # Generate synthetic market data with informed trading patterns
    np.random.seed(42)
    
    print("Generating synthetic market data with information events...\n")
    
    # Create synthetic daily data with varying informed trading
    n_days = 80
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Base parameters
    base_volume = 2_000_000
    base_price = 100.0
    
    # Generate data with different informed trading regimes
    data_records = []
    current_price = base_price
    
    for i in range(n_days):
        # Information events occur randomly
        info_event = np.random.random() < 0.3  # 30% chance of info event
        
        if info_event:
            # Information event - create informed trading
            good_news = np.random.random() < 0.5
            
            if good_news:
                # Good news - price rises, more informed buying
                price_change = np.random.uniform(0.02, 0.05)
                volume_multiplier = np.random.uniform(1.5, 2.5)
            else:
                # Bad news - price falls, more informed selling
                price_change = np.random.uniform(-0.05, -0.02)
                volume_multiplier = np.random.uniform(1.5, 2.5)
        else:
            # No information - random walk
            price_change = np.random.normal(0, 0.015)
            volume_multiplier = np.random.uniform(0.8, 1.2)
            
        # Update price
        current_price *= (1 + price_change)
        
        # Generate OHLC
        daily_vol = base_volume * volume_multiplier
        high = current_price * (1 + abs(price_change) * 0.5)
        low = current_price * (1 - abs(price_change) * 0.5)
        
        data_records.append({
            'open': current_price / (1 + price_change),
            'high': high,
            'low': low,
            'close': current_price,
            'volume': daily_vol
        })
    
    data = pd.DataFrame(data_records, index=dates)
    
    # Create PPIN indicator
    ppin_indicator = PPINIndicator(
        estimation_window=50,
        min_trades_per_day=50
    )
    
    # Process data incrementally
    print("Analyzing PPIN evolution with information events...\n")
    
    results = []
    for i in range(25, len(data), 5):
        batch = data.iloc[:i]
        result = ppin_indicator.calculate(batch, "TEST")
        results.append(result)
        
        if result.metadata.get('model_success', False):
            print(f"Day {i}:")
            print(f"  PPIN: {result.metadata['ppin']:.3f}")
            print(f"  Information Asymmetry: {result.metadata['information_asymmetry']:.2f}")
            print(f"  Market Efficiency: {result.metadata['market_efficiency']}")
            print(f"  Alpha (Info Events): {result.metadata['alpha']:.3f}")
            print(f"  Signal: {result.signal.value}")
            print(f"  Confidence: {result.confidence:.1f}%")
            print()
    
    # Final comprehensive analysis
    if results and results[-1].metadata.get('model_success'):
        final_result = results[-1]
        print("=" * 60)
        print("FINAL PPIN ANALYSIS:")
        print("=" * 60)
        
        print(f"Current PPIN: {final_result.metadata['ppin']:.3f}")
        print(f"Information Asymmetry: {final_result.metadata['information_asymmetry']:.2f}")
        print(f"Market Efficiency: {final_result.metadata['market_efficiency']}")
        
        print(f"\nModel Parameters:")
        print(f"  α (Info Event Prob): {final_result.metadata['alpha']:.3f}")
        print(f"  δ (Good News Prob): {final_result.metadata['delta']:.3f}")
        print(f"  μ (Informed Rate): {final_result.metadata['mu']:.1f}")
        print(f"  εb (Uninformed Buy): {final_result.metadata['epsilon_buy']:.1f}")
        print(f"  εs (Uninformed Sell): {final_result.metadata['epsilon_sell']:.1f}")
        
        if 'avg_ppin' in final_result.metadata:
            print(f"\nPPIN Statistics:")
            print(f"  Average PPIN: {final_result.metadata['avg_ppin']:.3f}")
            print(f"  PPIN Range: {final_result.metadata['min_ppin']:.3f} - {final_result.metadata['max_ppin']:.3f}")
            print(f"  PPIN Volatility: {final_result.metadata['ppin_volatility']:.3f}")
            
        if 'ppin_trend' in final_result.metadata:
            print(f"  PPIN Trend: {final_result.metadata['ppin_trend']}")
            
        print(f"\nTrading Signal: {final_result.signal.value}")
        print(f"Confidence: {final_result.confidence:.1f}%")
    
    print("\n💡 PPIN Interpretation:")
    print("- PPIN > 0.3: Significant informed trading present")
    print("- PPIN > 0.5: High information asymmetry")  
    print("- Rising PPIN: Information events becoming more frequent")
    print("- α shows probability of information events")
    print("- μ/(εb+εs) ratio shows informed vs uninformed trading")
    print("- Used by institutions to detect information leakage")


if __name__ == "__main__":
    demonstrate_ppin()