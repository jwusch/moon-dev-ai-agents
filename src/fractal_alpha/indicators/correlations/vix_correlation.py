"""
😱 VIX Correlation Analyzer - Fear Gauge Relationships
Analyzes correlations with volatility index for regime detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class VIXCorrelationIndicator(BaseIndicator):
    """
    VIX Correlation analyzer for market regime detection
    
    The VIX (Volatility Index) is known as the "fear gauge" and shows:
    - Inverse correlation with equities (usually -0.7 to -0.9)
    - Regime changes when correlation breaks down
    - Risk-on/risk-off sentiment shifts
    - Potential market turning points
    
    Key insights:
    - Decorrelation often precedes major moves
    - Extreme VIX levels mark sentiment extremes
    - VIX term structure reveals market expectations
    - Cross-asset correlations shift with VIX regimes
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 correlation_window: int = 20,
                 vix_threshold_low: float = 12,
                 vix_threshold_high: float = 20,
                 vix_extreme_high: float = 30,
                 detect_regime_shifts: bool = True):
        """
        Initialize VIX Correlation indicator
        
        Args:
            lookback_days: Days of history for analysis
            correlation_window: Rolling correlation window
            vix_threshold_low: Low VIX threshold (complacency)
            vix_threshold_high: High VIX threshold (fear)
            vix_extreme_high: Extreme VIX threshold (panic)
            detect_regime_shifts: Detect correlation regime changes
        """
        super().__init__(
            name="VIXCorrelation",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_days + correlation_window,
            params={
                'lookback_days': lookback_days,
                'correlation_window': correlation_window,
                'vix_threshold_low': vix_threshold_low,
                'vix_threshold_high': vix_threshold_high,
                'vix_extreme_high': vix_extreme_high,
                'detect_regime_shifts': detect_regime_shifts
            }
        )
        
        self.lookback_days = lookback_days
        self.correlation_window = correlation_window
        self.vix_threshold_low = vix_threshold_low
        self.vix_threshold_high = vix_threshold_high
        self.vix_extreme_high = vix_extreme_high
        self.detect_regime_shifts = detect_regime_shifts
        
    def calculate(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate VIX correlations and generate signals
        
        Args:
            data: Either a dict with 'asset' and 'vix' DataFrames, 
                  or a single DataFrame (will need VIX data)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with VIX correlation analysis
        """
        # Handle different data formats
        if isinstance(data, dict):
            if 'asset' not in data or 'vix' not in data:
                return self._empty_result(symbol)
            asset_data = data['asset']
            vix_data = data['vix']
        else:
            # Single DataFrame - need to fetch or simulate VIX
            asset_data = data
            vix_data = self._get_or_simulate_vix(asset_data)
            
        if len(asset_data) < self.correlation_window or len(vix_data) < self.correlation_window:
            return self._empty_result(symbol)
            
        # Calculate returns
        asset_returns = self._calculate_returns(asset_data)
        vix_returns = self._calculate_returns(vix_data)
        
        # Calculate rolling correlations
        correlations = self._calculate_rolling_correlation(
            asset_returns, vix_returns
        )
        
        # Analyze VIX levels
        vix_analysis = self._analyze_vix_levels(vix_data)
        
        # Detect regime shifts
        regime_analysis = {}
        if self.detect_regime_shifts:
            regime_analysis = self._detect_regime_shifts(
                correlations, vix_analysis
            )
        
        # Calculate term structure if available
        term_structure = self._analyze_term_structure(data)
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            correlations, vix_analysis, regime_analysis, term_structure
        )
        
        # Create metadata
        metadata = self._create_metadata(
            correlations, vix_analysis, regime_analysis, 
            term_structure, len(asset_data)
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
            # Assume it's VIX data with direct values
            prices = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
            
        return prices.pct_change().dropna()
    
    def _calculate_rolling_correlation(self, 
                                     asset_returns: pd.Series,
                                     vix_returns: pd.Series) -> pd.Series:
        """Calculate rolling correlation between asset and VIX"""
        
        # Align the series
        aligned_asset = asset_returns.iloc[-len(vix_returns):]
        aligned_vix = vix_returns.iloc[-len(asset_returns):]
        
        # Ensure same index
        common_index = aligned_asset.index.intersection(aligned_vix.index)
        aligned_asset = aligned_asset.loc[common_index]
        aligned_vix = aligned_vix.loc[common_index]
        
        # Calculate rolling correlation
        correlations = aligned_asset.rolling(
            window=self.correlation_window
        ).corr(aligned_vix)
        
        return correlations
    
    def _analyze_vix_levels(self, vix_data: pd.DataFrame) -> Dict:
        """Analyze VIX absolute levels and percentiles"""
        
        # Get VIX values
        if 'Close' in vix_data.columns:
            vix_values = vix_data['Close']
        elif 'close' in vix_data.columns:
            vix_values = vix_data['close']
        else:
            vix_values = vix_data.iloc[:, 0]
            
        current_vix = vix_values.iloc[-1]
        
        # Calculate percentiles
        vix_percentile = (vix_values < current_vix).mean() * 100
        
        # Moving averages
        vix_ma20 = vix_values.rolling(20).mean().iloc[-1]
        vix_ma50 = vix_values.rolling(50).mean().iloc[-1] if len(vix_values) >= 50 else vix_ma20
        
        # Determine regime
        if current_vix < self.vix_threshold_low:
            vix_regime = "complacency"
        elif current_vix < self.vix_threshold_high:
            vix_regime = "normal"
        elif current_vix < self.vix_extreme_high:
            vix_regime = "elevated"
        else:
            vix_regime = "extreme_fear"
            
        # Rate of change
        vix_1d_change = (current_vix / vix_values.iloc[-2] - 1) * 100 if len(vix_values) > 1 else 0
        vix_5d_change = (current_vix / vix_values.iloc[-5] - 1) * 100 if len(vix_values) > 5 else 0
        
        return {
            'current_level': current_vix,
            'percentile': vix_percentile,
            'regime': vix_regime,
            'ma20': vix_ma20,
            'ma50': vix_ma50,
            'above_ma20': current_vix > vix_ma20,
            'above_ma50': current_vix > vix_ma50,
            '1d_change': vix_1d_change,
            '5d_change': vix_5d_change,
            'recent_high': vix_values.iloc[-20:].max(),
            'recent_low': vix_values.iloc[-20:].min()
        }
    
    def _detect_regime_shifts(self, 
                             correlations: pd.Series,
                             vix_analysis: Dict) -> Dict:
        """Detect correlation regime shifts"""
        
        if len(correlations.dropna()) < 10:
            return {}
            
        # Current correlation
        current_corr = correlations.iloc[-1]
        
        # Historical correlation statistics
        corr_mean = correlations.dropna().mean()
        corr_std = correlations.dropna().std()
        
        # Typical correlation ranges by VIX regime
        expected_corr = {
            'complacency': -0.5,  # Lower inverse correlation
            'normal': -0.7,       # Normal inverse correlation  
            'elevated': -0.8,     # Strong inverse correlation
            'extreme_fear': -0.9  # Very strong inverse correlation
        }
        
        vix_regime = vix_analysis['regime']
        expected = expected_corr.get(vix_regime, -0.7)
        
        # Detect anomalies
        correlation_zscore = (current_corr - corr_mean) / (corr_std + 1e-8)
        is_anomaly = abs(correlation_zscore) > 2
        
        # Correlation breakdown
        correlation_breakdown = current_corr > -0.3  # Lost inverse relationship
        
        # Trend analysis
        recent_corr = correlations.iloc[-5:].mean()
        corr_trend = "strengthening" if recent_corr < corr_mean else "weakening"
        
        return {
            'current_correlation': current_corr,
            'expected_correlation': expected,
            'correlation_mean': corr_mean,
            'correlation_std': corr_std,
            'correlation_zscore': correlation_zscore,
            'is_anomaly': is_anomaly,
            'correlation_breakdown': correlation_breakdown,
            'correlation_trend': corr_trend,
            'regime_match': abs(current_corr - expected) < 0.2
        }
    
    def _analyze_term_structure(self, data: Union[Dict, pd.DataFrame]) -> Dict:
        """Analyze VIX term structure if available"""
        
        if not isinstance(data, dict):
            return {}
            
        # Check for VIX9D (9-day) and VIX data
        if 'vix9d' in data and 'vix' in data:
            vix9d = data['vix9d'].iloc[-1, 0] if isinstance(data['vix9d'], pd.DataFrame) else data['vix9d'].iloc[-1]
            vix = data['vix'].iloc[-1, 0] if isinstance(data['vix'], pd.DataFrame) else data['vix'].iloc[-1]
            
            # Term structure
            term_spread = vix - vix9d
            contango = term_spread > 0  # Normal market
            
            return {
                'has_term_structure': True,
                'vix9d': vix9d,
                'vix': vix,
                'term_spread': term_spread,
                'structure': 'contango' if contango else 'backwardation',
                'stress_signal': not contango  # Backwardation = stress
            }
            
        return {'has_term_structure': False}
    
    def _get_or_simulate_vix(self, asset_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate VIX if not provided (using realized volatility)"""
        
        # Calculate 20-day realized volatility as VIX proxy
        returns = self._calculate_returns(asset_data)
        
        # Annualized volatility
        realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100
        
        # Add some mean reversion to make it more VIX-like
        vix_simulated = realized_vol.ewm(span=5).mean()
        
        # Create DataFrame
        vix_df = pd.DataFrame(
            {'close': vix_simulated},
            index=asset_data.index
        )
        
        return vix_df
    
    def _generate_signal(self,
                        correlations: pd.Series,
                        vix_analysis: Dict,
                        regime_analysis: Dict,
                        term_structure: Dict) -> Tuple[SignalType, float, float]:
        """Generate trading signal from VIX correlation analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Base signal on VIX regime
        vix_level = vix_analysis['current_level']
        vix_regime = vix_analysis['regime']
        
        if vix_regime == "extreme_fear":
            # Contrarian buy at extreme fear
            signal = SignalType.BUY
            confidence = 70
            
        elif vix_regime == "complacency":
            # Caution at extreme low VIX
            signal = SignalType.SELL
            confidence = 50
            
        # Adjust for correlation regime
        if regime_analysis:
            if regime_analysis.get('correlation_breakdown', False):
                # Correlation breakdown = regime change
                confidence += 20
                if vix_analysis['5d_change'] > 20:
                    signal = SignalType.SELL  # Risk-off
                elif vix_analysis['5d_change'] < -20:
                    signal = SignalType.BUY   # Risk-on
                    
            if regime_analysis.get('is_anomaly', False):
                # Anomalous correlation
                confidence += 10
        
        # Term structure signals
        if term_structure.get('has_term_structure', False):
            if term_structure.get('stress_signal', False):
                # Backwardation = market stress
                if signal != SignalType.SELL:
                    confidence *= 0.8
            else:
                # Contango = normal market
                if signal == SignalType.BUY:
                    confidence *= 1.1
        
        # VIX spike signals
        if vix_analysis['1d_change'] > 15:  # 15% VIX spike
            signal = SignalType.SELL
            confidence = 65
        elif vix_analysis['1d_change'] < -15:  # 15% VIX crush
            signal = SignalType.BUY
            confidence = 60
            
        # Mean reversion at extremes
        if vix_analysis['percentile'] > 90:
            signal = SignalType.BUY  # Contrarian
            confidence = min(confidence + 15, 80)
        elif vix_analysis['percentile'] < 10:
            signal = SignalType.HOLD  # Too complacent
            confidence = 30
            
        confidence = min(confidence, 85)
        
        # Value represents correlation strength (0-100)
        current_corr = correlations.iloc[-1] if len(correlations) > 0 else -0.7
        value = abs(current_corr) * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        correlations: pd.Series,
                        vix_analysis: Dict,
                        regime_analysis: Dict,
                        term_structure: Dict,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'vix_level': vix_analysis['current_level'],
            'vix_regime': vix_analysis['regime'],
            'vix_percentile': vix_analysis['percentile'],
            'vix_1d_change': vix_analysis['1d_change'],
            'vix_5d_change': vix_analysis['5d_change']
        }
        
        # Add correlation stats
        if len(correlations.dropna()) > 0:
            metadata['current_correlation'] = correlations.iloc[-1]
            metadata['correlation_mean'] = correlations.dropna().mean()
            metadata['correlation_std'] = correlations.dropna().std()
            metadata['correlation_min'] = correlations.dropna().min()
            metadata['correlation_max'] = correlations.dropna().max()
        
        # Add regime analysis
        if regime_analysis:
            metadata.update({
                'expected_correlation': regime_analysis.get('expected_correlation', -0.7),
                'correlation_zscore': regime_analysis.get('correlation_zscore', 0),
                'correlation_anomaly': regime_analysis.get('is_anomaly', False),
                'correlation_breakdown': regime_analysis.get('correlation_breakdown', False),
                'correlation_trend': regime_analysis.get('correlation_trend', 'stable')
            })
        
        # Add term structure
        if term_structure.get('has_term_structure', False):
            metadata.update({
                'vix_term_structure': term_structure['structure'],
                'term_spread': term_structure['term_spread'],
                'backwardation_signal': term_structure.get('stress_signal', False)
            })
        
        # Trading insights
        if vix_analysis['regime'] == 'extreme_fear':
            metadata['insight'] = "Extreme fear - contrarian buy opportunity"
        elif vix_analysis['regime'] == 'complacency':
            metadata['insight'] = "Market complacency - risk of surprise"
        elif regime_analysis.get('correlation_breakdown', False):
            metadata['insight'] = "Correlation breakdown - regime change likely"
        else:
            metadata['insight'] = "Normal market conditions"
            
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=70.0,  # Default correlation strength
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for VIX correlation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, Dict]) -> bool:
        """Validate input data"""
        
        if isinstance(data, dict):
            return 'asset' in data and 'vix' in data
        else:
            # Single DataFrame is ok - we'll simulate VIX
            return isinstance(data, pd.DataFrame) and len(data) >= self.correlation_window


def demonstrate_vix_correlation():
    """Demonstration of VIX Correlation indicator"""
    
    print("😱 VIX Correlation Analysis Demonstration\n")
    
    # Generate synthetic data
    print("Generating synthetic asset and VIX data...\n")
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Asset prices with regime changes
    asset_prices = []
    vix_values = []
    
    price = 100
    vix = 15
    
    for i in range(len(dates)):
        # Create different market regimes
        if i < 30:
            # Bull market - low VIX
            price *= 1 + np.random.normal(0.001, 0.01)
            vix = 12 + np.random.normal(0, 1)
        elif i < 50:
            # Increased volatility
            price *= 1 + np.random.normal(0, 0.015)
            vix = 18 + np.random.normal(0, 2)
        elif i < 70:
            # Market stress
            price *= 1 + np.random.normal(-0.002, 0.02)
            vix = 25 + np.random.normal(0, 3)
        else:
            # Recovery
            price *= 1 + np.random.normal(0.001, 0.012)
            vix = 16 + np.random.normal(0, 1.5)
            
        # Add inverse correlation
        if i > 0:
            price_change = (price / asset_prices[-1] - 1) if asset_prices else 0
            vix += -price_change * 100 * np.random.uniform(0.5, 1.5)
            
        asset_prices.append(price)
        vix_values.append(max(vix, 10))  # VIX floor at 10
    
    # Create DataFrames
    asset_df = pd.DataFrame({
        'Open': asset_prices,
        'High': [p * 1.01 for p in asset_prices],
        'Low': [p * 0.99 for p in asset_prices],
        'Close': asset_prices
    }, index=dates)
    
    vix_df = pd.DataFrame({
        'Close': vix_values
    }, index=dates)
    
    # Create indicator
    vix_corr = VIXCorrelationIndicator(
        lookback_days=20,
        correlation_window=20,
        detect_regime_shifts=True
    )
    
    # Calculate with both asset and VIX data
    data_dict = {
        'asset': asset_df,
        'vix': vix_df
    }
    
    result = vix_corr.calculate(data_dict, "SYNTHETIC")
    
    print("=" * 60)
    print("VIX CORRELATION ANALYSIS:")
    print("=" * 60)
    
    # Current state
    print(f"Current VIX Level: {result.metadata['vix_level']:.2f}")
    print(f"VIX Regime: {result.metadata['vix_regime'].upper()}")
    print(f"VIX Percentile: {result.metadata['vix_percentile']:.1f}%")
    print(f"VIX 1-Day Change: {result.metadata['vix_1d_change']:+.1f}%")
    print(f"VIX 5-Day Change: {result.metadata['vix_5d_change']:+.1f}%")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    print(f"Current Correlation: {result.metadata.get('current_correlation', 0):.3f}")
    print(f"Historical Mean: {result.metadata.get('correlation_mean', 0):.3f}")
    print(f"Expected for Regime: {result.metadata.get('expected_correlation', 0):.3f}")
    
    if result.metadata.get('correlation_anomaly', False):
        print("⚠️ ANOMALOUS CORRELATION DETECTED")
    
    if result.metadata.get('correlation_breakdown', False):
        print("🚨 CORRELATION BREAKDOWN - REGIME CHANGE LIKELY")
    
    print(f"Correlation Trend: {result.metadata.get('correlation_trend', 'Unknown').upper()}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Correlation Strength: {result.value:.1f}/100")
    
    # Insight
    print(f"\nMarket Insight: {result.metadata.get('insight', 'No specific insight')}")
    
    # Now test with automatic VIX simulation
    print("\n" + "=" * 60)
    print("TESTING WITH SIMULATED VIX:")
    print("=" * 60)
    
    result_sim = vix_corr.calculate(asset_df, "SYNTHETIC")
    
    print(f"Simulated VIX Level: {result_sim.metadata['vix_level']:.2f}")
    print(f"Signal: {result_sim.signal.value}")
    print(f"Confidence: {result_sim.confidence:.1f}%")
    
    print("\n💡 VIX Correlation Trading Tips:")
    print("- Normal correlation: -0.7 to -0.9 (inverse)")
    print("- Breakdown > -0.3 signals regime change")
    print("- VIX > 30 often marks bottoms (contrarian buy)")
    print("- VIX < 12 shows complacency (caution)")
    print("- Decorrelation precedes volatility events")
    print("- Term structure backwardation = immediate stress")


if __name__ == "__main__":
    demonstrate_vix_correlation()