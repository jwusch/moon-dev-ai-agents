"""
💵 Dollar Correlation Analyzer - Currency Impact on Assets
Analyzes correlations with US Dollar Index (DXY) for macro insights
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class DollarCorrelationIndicator(BaseIndicator):
    """
    Dollar Correlation analyzer for macro regime detection
    
    The US Dollar Index (DXY) correlations reveal:
    - Risk sentiment (dollar as safe haven)
    - Commodity relationships (inverse correlation)
    - Emerging market stress (dollar strength = EM weakness)
    - Global liquidity conditions
    - Fed policy expectations
    
    Key relationships:
    - Commodities: Usually -0.6 to -0.8 correlation
    - Gold: Complex (safe haven vs dollar strength)
    - EM Assets: Strong negative correlation
    - US Equities: Time-varying (sometimes positive in risk-off)
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 correlation_window: int = 20,
                 dollar_strength_threshold: float = 100,
                 detect_divergence: bool = True,
                 asset_type: str = "equity"):
        """
        Initialize Dollar Correlation indicator
        
        Args:
            lookback_days: Days of history for analysis
            correlation_window: Rolling correlation window
            dollar_strength_threshold: DXY level for "strong dollar"
            detect_divergence: Detect correlation divergences
            asset_type: Type of asset ("equity", "commodity", "currency", "crypto")
        """
        super().__init__(
            name="DollarCorrelation",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_days + correlation_window,
            params={
                'lookback_days': lookback_days,
                'correlation_window': correlation_window,
                'dollar_strength_threshold': dollar_strength_threshold,
                'detect_divergence': detect_divergence,
                'asset_type': asset_type
            }
        )
        
        self.lookback_days = lookback_days
        self.correlation_window = correlation_window
        self.dollar_strength_threshold = dollar_strength_threshold
        self.detect_divergence = detect_divergence
        self.asset_type = asset_type
        
        # Expected correlations by asset type
        self.expected_correlations = {
            'equity': 0.0,      # Time-varying
            'commodity': -0.7,   # Strong inverse
            'currency': -0.8,    # Strong inverse (non-USD)
            'crypto': -0.3       # Moderate inverse
        }
        
    def calculate(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Dollar correlations and generate signals
        
        Args:
            data: Either a dict with 'asset' and 'dollar' DataFrames,
                  or a single DataFrame (will simulate dollar)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with Dollar correlation analysis
        """
        # Handle different data formats
        if isinstance(data, dict):
            if 'asset' not in data or 'dollar' not in data:
                return self._empty_result(symbol)
            asset_data = data['asset']
            dollar_data = data['dollar']
        else:
            # Single DataFrame - simulate DXY
            asset_data = data
            dollar_data = self._simulate_dollar_index(asset_data)
            
        if len(asset_data) < self.correlation_window or len(dollar_data) < self.correlation_window:
            return self._empty_result(symbol)
            
        # Calculate returns
        asset_returns = self._calculate_returns(asset_data)
        dollar_returns = self._calculate_returns(dollar_data)
        
        # Calculate rolling correlations
        correlations = self._calculate_rolling_correlation(
            asset_returns, dollar_returns
        )
        
        # Analyze Dollar strength
        dollar_analysis = self._analyze_dollar_strength(dollar_data)
        
        # Detect divergences
        divergence_analysis = {}
        if self.detect_divergence:
            divergence_analysis = self._detect_divergences(
                asset_data, dollar_data, correlations
            )
        
        # Analyze correlation stability
        stability_analysis = self._analyze_correlation_stability(correlations)
        
        # Detect macro regime
        macro_regime = self._detect_macro_regime(
            dollar_analysis, correlations, self.asset_type
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            correlations, dollar_analysis, divergence_analysis,
            stability_analysis, macro_regime
        )
        
        # Create metadata
        metadata = self._create_metadata(
            correlations, dollar_analysis, divergence_analysis,
            stability_analysis, macro_regime, len(asset_data)
        )
        
        # Get timestamp
        if isinstance(asset_data, pd.DataFrame):
            timestamp = int(asset_data.index[-1].timestamp() * 1000) if hasattr(asset_data.index[-1], 'timestamp') else int(datetime.now().timestamp() * 1000)
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
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data"""
        
        if 'Close' in data.columns:
            prices = data['Close']
        elif 'close' in data.columns:
            prices = data['close']
        else:
            prices = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
            
        return prices.pct_change().dropna()
    
    def _calculate_rolling_correlation(self,
                                     asset_returns: pd.Series,
                                     dollar_returns: pd.Series) -> pd.Series:
        """Calculate rolling correlation between asset and Dollar"""
        
        # Align the series
        aligned_asset = asset_returns.iloc[-len(dollar_returns):]
        aligned_dollar = dollar_returns.iloc[-len(asset_returns):]
        
        # Ensure same index
        common_index = aligned_asset.index.intersection(aligned_dollar.index)
        aligned_asset = aligned_asset.loc[common_index]
        aligned_dollar = aligned_dollar.loc[common_index]
        
        # Calculate rolling correlation
        correlations = aligned_asset.rolling(
            window=self.correlation_window
        ).corr(aligned_dollar)
        
        return correlations
    
    def _analyze_dollar_strength(self, dollar_data: pd.DataFrame) -> Dict:
        """Analyze Dollar Index levels and trends"""
        
        # Get Dollar values
        if 'Close' in dollar_data.columns:
            dollar_values = dollar_data['Close']
        elif 'close' in dollar_data.columns:
            dollar_values = dollar_data['close']
        else:
            dollar_values = dollar_data.iloc[:, 0]
            
        current_dollar = dollar_values.iloc[-1]
        
        # Moving averages
        dxy_ma20 = dollar_values.rolling(20).mean().iloc[-1]
        dxy_ma50 = dollar_values.rolling(50).mean().iloc[-1] if len(dollar_values) >= 50 else dxy_ma20
        dxy_ma200 = dollar_values.rolling(200).mean().iloc[-1] if len(dollar_values) >= 200 else dxy_ma50
        
        # Trend analysis
        trend = "bullish" if current_dollar > dxy_ma20 > dxy_ma50 else "bearish"
        
        # Strength classification
        if current_dollar > self.dollar_strength_threshold:
            strength = "strong"
        elif current_dollar > 95:
            strength = "moderate"
        else:
            strength = "weak"
            
        # Rate of change
        dxy_1d_change = (current_dollar / dollar_values.iloc[-2] - 1) * 100 if len(dollar_values) > 1 else 0
        dxy_5d_change = (current_dollar / dollar_values.iloc[-5] - 1) * 100 if len(dollar_values) > 5 else 0
        dxy_20d_change = (current_dollar / dollar_values.iloc[-20] - 1) * 100 if len(dollar_values) > 20 else 0
        
        # Historical percentile
        dxy_percentile = (dollar_values < current_dollar).mean() * 100
        
        return {
            'current_level': current_dollar,
            'strength': strength,
            'trend': trend,
            'ma20': dxy_ma20,
            'ma50': dxy_ma50,
            'ma200': dxy_ma200,
            'above_ma20': current_dollar > dxy_ma20,
            'above_ma50': current_dollar > dxy_ma50,
            'above_ma200': current_dollar > dxy_ma200 if len(dollar_values) >= 200 else None,
            '1d_change': dxy_1d_change,
            '5d_change': dxy_5d_change,
            '20d_change': dxy_20d_change,
            'percentile': dxy_percentile,
            'recent_high': dollar_values.iloc[-20:].max(),
            'recent_low': dollar_values.iloc[-20:].min()
        }
    
    def _detect_divergences(self,
                           asset_data: pd.DataFrame,
                           dollar_data: pd.DataFrame,
                           correlations: pd.Series) -> Dict:
        """Detect price divergences between asset and dollar"""
        
        # Get prices
        asset_prices = asset_data['Close'] if 'Close' in asset_data else asset_data['close']
        dollar_prices = dollar_data['Close'] if 'Close' in dollar_data else dollar_data['close']
        
        # Calculate recent performance
        lookback = min(20, len(asset_prices) - 1)
        
        asset_perf = (asset_prices.iloc[-1] / asset_prices.iloc[-lookback] - 1) * 100
        dollar_perf = (dollar_prices.iloc[-1] / dollar_prices.iloc[-lookback] - 1) * 100
        
        # Expected relationship based on correlation
        current_corr = correlations.iloc[-1] if len(correlations) > 0 else 0
        expected_corr = self.expected_correlations.get(self.asset_type, -0.5)
        
        # Detect divergence types
        divergences = {
            'performance_divergence': asset_perf - dollar_perf,
            'asset_performance': asset_perf,
            'dollar_performance': dollar_perf
        }
        
        # Classic divergences
        if self.asset_type in ['commodity', 'currency']:
            # Should move opposite
            if asset_perf > 0 and dollar_perf > 0:
                divergences['type'] = 'bearish_divergence'
                divergences['strength'] = 'strong' if asset_perf > 5 and dollar_perf > 2 else 'moderate'
            elif asset_perf < 0 and dollar_perf < 0:
                divergences['type'] = 'bullish_divergence'
                divergences['strength'] = 'strong' if asset_perf < -5 and dollar_perf < -2 else 'moderate'
            else:
                divergences['type'] = 'normal'
                divergences['strength'] = 'none'
        else:
            # Equity/Crypto - more complex
            if abs(current_corr - expected_corr) > 0.3:
                divergences['type'] = 'correlation_divergence'
                divergences['strength'] = 'strong' if abs(current_corr - expected_corr) > 0.5 else 'moderate'
            else:
                divergences['type'] = 'normal'
                divergences['strength'] = 'none'
                
        return divergences
    
    def _analyze_correlation_stability(self, correlations: pd.Series) -> Dict:
        """Analyze correlation stability over time"""
        
        if len(correlations.dropna()) < 10:
            return {'stable': True, 'volatility': 0}
            
        # Rolling standard deviation of correlation
        corr_volatility = correlations.rolling(10).std().iloc[-1]
        
        # Correlation range
        recent_corr = correlations.iloc[-20:]
        corr_range = recent_corr.max() - recent_corr.min()
        
        # Stability classification
        if corr_volatility < 0.1 and corr_range < 0.3:
            stability = 'stable'
        elif corr_volatility < 0.2 and corr_range < 0.5:
            stability = 'moderately_stable'
        else:
            stability = 'unstable'
            
        # Trend in correlation
        if len(correlations) >= 20:
            corr_ma5 = correlations.rolling(5).mean()
            corr_ma20 = correlations.rolling(20).mean()
            
            if corr_ma5.iloc[-1] > corr_ma20.iloc[-1] + 0.1:
                corr_trend = 'strengthening'
            elif corr_ma5.iloc[-1] < corr_ma20.iloc[-1] - 0.1:
                corr_trend = 'weakening'
            else:
                corr_trend = 'stable'
        else:
            corr_trend = 'unknown'
            
        return {
            'stability': stability,
            'volatility': corr_volatility,
            'range': corr_range,
            'trend': corr_trend
        }
    
    def _detect_macro_regime(self,
                            dollar_analysis: Dict,
                            correlations: pd.Series,
                            asset_type: str) -> Dict:
        """Detect macro regime based on dollar dynamics"""
        
        dollar_strength = dollar_analysis['strength']
        dollar_trend = dollar_analysis['trend']
        dollar_roc = dollar_analysis['5d_change']
        
        current_corr = correlations.iloc[-1] if len(correlations) > 0 else 0
        
        # Macro regime detection
        if dollar_strength == 'strong' and dollar_trend == 'bullish':
            if dollar_roc > 2:
                regime = 'dollar_surge'
                implication = 'risk_off'
            else:
                regime = 'strong_dollar'
                implication = 'em_weakness'
        elif dollar_strength == 'weak' and dollar_trend == 'bearish':
            regime = 'weak_dollar'
            implication = 'risk_on'
        else:
            regime = 'neutral_dollar'
            implication = 'balanced'
            
        # Asset-specific implications
        asset_implications = {
            'commodity': {
                'dollar_surge': 'very_bearish',
                'strong_dollar': 'bearish',
                'weak_dollar': 'bullish',
                'neutral_dollar': 'neutral'
            },
            'currency': {
                'dollar_surge': 'very_bearish',
                'strong_dollar': 'bearish',
                'weak_dollar': 'bullish',
                'neutral_dollar': 'neutral'
            },
            'equity': {
                'dollar_surge': 'bearish' if current_corr < -0.3 else 'mixed',
                'strong_dollar': 'mixed',
                'weak_dollar': 'bullish',
                'neutral_dollar': 'neutral'
            },
            'crypto': {
                'dollar_surge': 'bearish',
                'strong_dollar': 'slightly_bearish',
                'weak_dollar': 'bullish',
                'neutral_dollar': 'neutral'
            }
        }
        
        asset_impact = asset_implications.get(asset_type, {}).get(regime, 'neutral')
        
        return {
            'regime': regime,
            'market_implication': implication,
            'asset_impact': asset_impact,
            'dollar_momentum': 'accelerating' if abs(dollar_roc) > 3 else 'steady'
        }
    
    def _simulate_dollar_index(self, asset_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate DXY if not provided"""
        
        # Start at typical DXY level
        dxy_values = []
        dxy = 95.0
        
        # Create inverse correlation with asset
        asset_returns = self._calculate_returns(asset_data)
        
        for i, ret in enumerate(asset_returns):
            if i == 0:
                dxy_values.append(dxy)
                continue
                
            # Dollar tends to strengthen in risk-off
            # Inverse correlation for most assets
            if self.asset_type in ['commodity', 'currency']:
                dxy_change = -ret * 0.7  # Strong inverse
            else:
                dxy_change = -ret * 0.3  # Moderate inverse
                
            # Add some independent movement
            dxy_change += np.random.normal(0, 0.002)
            
            # Mean reversion
            if dxy > 100:
                dxy_change -= 0.001
            elif dxy < 90:
                dxy_change += 0.001
                
            dxy *= (1 + dxy_change)
            dxy_values.append(dxy)
            
        # Create DataFrame
        dxy_df = pd.DataFrame(
            {'close': dxy_values},
            index=asset_data.index[1:]  # Skip first due to returns calculation
        )
        
        # Add first row
        dxy_df = pd.concat([
            pd.DataFrame({'close': [95.0]}, index=[asset_data.index[0]]),
            dxy_df
        ])
        
        return dxy_df
    
    def _generate_signal(self,
                        correlations: pd.Series,
                        dollar_analysis: Dict,
                        divergence_analysis: Dict,
                        stability_analysis: Dict,
                        macro_regime: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from Dollar correlation analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Base signal on asset impact
        asset_impact = macro_regime['asset_impact']
        
        if 'very_bullish' in asset_impact:
            signal = SignalType.BUY
            confidence = 70
        elif 'bullish' in asset_impact:
            signal = SignalType.BUY
            confidence = 60
        elif 'very_bearish' in asset_impact:
            signal = SignalType.SELL
            confidence = 70
        elif 'bearish' in asset_impact:
            signal = SignalType.SELL
            confidence = 60
        else:
            signal = SignalType.HOLD
            confidence = 30
            
        # Adjust for divergences
        if divergence_analysis:
            div_type = divergence_analysis.get('type', 'normal')
            div_strength = divergence_analysis.get('strength', 'none')
            
            if div_type == 'bullish_divergence' and div_strength == 'strong':
                signal = SignalType.BUY
                confidence += 15
            elif div_type == 'bearish_divergence' and div_strength == 'strong':
                signal = SignalType.SELL
                confidence += 15
            elif div_type == 'correlation_divergence':
                # Unusual correlation = caution
                confidence *= 0.8
                
        # Adjust for correlation stability
        if stability_analysis.get('stability') == 'unstable':
            # Unstable correlations = lower confidence
            confidence *= 0.8
        elif stability_analysis.get('stability') == 'stable':
            # Stable correlations = higher confidence
            confidence *= 1.1
            
        # Dollar momentum adjustments
        dollar_roc = dollar_analysis['5d_change']
        
        if abs(dollar_roc) > 5:  # Big dollar move
            if self.asset_type in ['commodity', 'currency']:
                # Strong reaction expected
                confidence += 10
            else:
                # May see delayed reaction
                confidence += 5
                
        # Extreme dollar levels
        if dollar_analysis['percentile'] > 90:
            # Extreme high dollar
            if self.asset_type in ['commodity', 'currency'] and signal != SignalType.BUY:
                signal = SignalType.BUY  # Contrarian
                confidence = 65
        elif dollar_analysis['percentile'] < 10:
            # Extreme low dollar
            if self.asset_type in ['commodity', 'currency'] and signal != SignalType.SELL:
                signal = SignalType.SELL  # Contrarian
                confidence = 65
                
        confidence = min(confidence, 85)
        
        # Value represents correlation stability (0-100)
        stability_score = (1 - stability_analysis.get('volatility', 0.5)) * 100
        value = min(stability_score, 100)
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        correlations: pd.Series,
                        dollar_analysis: Dict,
                        divergence_analysis: Dict,
                        stability_analysis: Dict,
                        macro_regime: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'asset_type': self.asset_type,
            'data_points': data_points,
            'dollar_level': dollar_analysis['current_level'],
            'dollar_strength': dollar_analysis['strength'],
            'dollar_trend': dollar_analysis['trend'],
            'dollar_percentile': dollar_analysis['percentile'],
            'dollar_5d_change': dollar_analysis['5d_change']
        }
        
        # Add correlation stats
        if len(correlations.dropna()) > 0:
            metadata['current_correlation'] = correlations.iloc[-1]
            metadata['correlation_mean'] = correlations.dropna().mean()
            metadata['expected_correlation'] = self.expected_correlations.get(self.asset_type, -0.5)
            metadata['correlation_deviation'] = abs(correlations.iloc[-1] - metadata['expected_correlation'])
        
        # Add divergence info
        if divergence_analysis:
            metadata['divergence_type'] = divergence_analysis.get('type', 'normal')
            metadata['divergence_strength'] = divergence_analysis.get('strength', 'none')
            metadata['asset_vs_dollar_perf'] = divergence_analysis.get('performance_divergence', 0)
        
        # Add stability info
        if stability_analysis:
            metadata['correlation_stability'] = stability_analysis.get('stability', 'unknown')
            metadata['correlation_trend'] = stability_analysis.get('trend', 'unknown')
            
        # Add regime info
        metadata['macro_regime'] = macro_regime['regime']
        metadata['asset_impact'] = macro_regime['asset_impact']
        metadata['market_implication'] = macro_regime['market_implication']
        
        # Trading insights
        if macro_regime['regime'] == 'dollar_surge':
            metadata['insight'] = "Dollar surge - risk-off environment"
        elif divergence_analysis.get('type') == 'bullish_divergence':
            metadata['insight'] = "Bullish divergence - potential reversal"
        elif divergence_analysis.get('type') == 'bearish_divergence':
            metadata['insight'] = "Bearish divergence - caution advised"
        else:
            metadata['insight'] = f"Normal {self.asset_type}-dollar relationship"
            
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
            metadata={'error': 'Insufficient data for Dollar correlation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, Dict]) -> bool:
        """Validate input data"""
        
        if isinstance(data, dict):
            return 'asset' in data and 'dollar' in data
        else:
            return isinstance(data, pd.DataFrame) and len(data) >= self.correlation_window


def demonstrate_dollar_correlation():
    """Demonstration of Dollar Correlation indicator"""
    
    print("💵 Dollar Correlation Analysis Demonstration\n")
    
    # Test different asset types
    asset_types = ['commodity', 'equity', 'crypto']
    
    for asset_type in asset_types:
        print(f"\n{'=' * 60}")
        print(f"Testing {asset_type.upper()} vs Dollar")
        print('=' * 60)
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Dollar index data
        dxy_values = []
        dxy = 95.0
        
        # Asset data
        asset_values = []
        asset_price = 100.0 if asset_type != 'commodity' else 50.0
        
        for i in range(len(dates)):
            # Dollar movements
            if i < 30:
                # Dollar strengthening
                dxy_change = np.random.normal(0.001, 0.003)
            elif i < 60:
                # Dollar weakening
                dxy_change = np.random.normal(-0.001, 0.003)
            else:
                # Volatile period
                dxy_change = np.random.normal(0, 0.005)
                
            dxy *= (1 + dxy_change)
            dxy_values.append(dxy)
            
            # Asset movements based on type
            if asset_type == 'commodity':
                # Strong inverse correlation
                asset_change = -dxy_change * 1.5 + np.random.normal(0, 0.002)
            elif asset_type == 'equity':
                # Variable correlation
                if dxy_change > 0.005:  # Big dollar move
                    asset_change = -dxy_change * 0.5  # Risk-off
                else:
                    asset_change = np.random.normal(0.0005, 0.01)
            else:  # crypto
                # Moderate inverse correlation
                asset_change = -dxy_change * 0.8 + np.random.normal(0, 0.015)
                
            asset_price *= (1 + asset_change)
            asset_values.append(asset_price)
        
        # Create DataFrames
        asset_df = pd.DataFrame({
            'Open': asset_values,
            'High': [p * 1.01 for p in asset_values],
            'Low': [p * 0.99 for p in asset_values],
            'Close': asset_values
        }, index=dates)
        
        dxy_df = pd.DataFrame({
            'Close': dxy_values
        }, index=dates)
        
        # Create indicator
        dollar_corr = DollarCorrelationIndicator(
            lookback_days=20,
            correlation_window=20,
            detect_divergence=True,
            asset_type=asset_type
        )
        
        # Calculate
        data_dict = {
            'asset': asset_df,
            'dollar': dxy_df
        }
        
        result = dollar_corr.calculate(data_dict, f"SYNTHETIC_{asset_type.upper()}")
        
        # Display results
        print(f"\nDollar Analysis:")
        print(f"Current DXY Level: {result.metadata['dollar_level']:.2f}")
        print(f"Dollar Strength: {result.metadata['dollar_strength'].upper()}")
        print(f"Dollar Trend: {result.metadata['dollar_trend'].upper()}")
        print(f"5-Day Change: {result.metadata['dollar_5d_change']:+.1f}%")
        
        print(f"\nCorrelation Analysis:")
        print(f"Current Correlation: {result.metadata.get('current_correlation', 0):.3f}")
        print(f"Expected Correlation: {result.metadata.get('expected_correlation', 0):.3f}")
        print(f"Correlation Stability: {result.metadata.get('correlation_stability', 'Unknown').upper()}")
        
        if 'divergence_type' in result.metadata:
            print(f"\nDivergence Detected: {result.metadata['divergence_type'].upper()}")
            print(f"Divergence Strength: {result.metadata['divergence_strength'].upper()}")
        
        print(f"\nMacro Regime: {result.metadata['macro_regime'].upper()}")
        print(f"Asset Impact: {result.metadata['asset_impact'].upper()}")
        
        print(f"\nTrading Signal: {result.signal.value}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Stability Score: {result.value:.1f}/100")
        
        print(f"\nInsight: {result.metadata.get('insight', 'No specific insight')}")
    
    print("\n💡 Dollar Correlation Trading Tips:")
    print("- Commodities: Strong inverse correlation (weaker dollar = higher prices)")
    print("- EM Assets: Suffer when dollar strengthens")
    print("- US Equities: Complex relationship (sometimes positive in risk-off)")
    print("- Divergences often signal reversals")
    print("- Extreme dollar levels (>100 or <90) often mean revert")
    print("- Monitor correlation stability for regime changes")


if __name__ == "__main__":
    demonstrate_dollar_correlation()