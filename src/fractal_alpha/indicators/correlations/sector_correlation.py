"""
🏢 Sector Correlation Analyzer - Market Leadership Detection
Analyzes correlations across sectors to identify rotations and themes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


class SectorCorrelationIndicator(BaseIndicator):
    """
    Sector Correlation analyzer for market regime and rotation detection
    
    This indicator tracks:
    - Sector rotation patterns
    - Risk-on/risk-off behavior
    - Market breadth and divergences
    - Leadership changes
    - Correlation breakdowns signaling regime shifts
    
    Key sectors analyzed:
    - Technology (XLK) - Growth/Risk-on
    - Financials (XLF) - Economic health
    - Energy (XLE) - Inflation/Commodities
    - Utilities (XLU) - Defensive/Risk-off
    - Consumer Discretionary (XLY) - Consumer strength
    - Healthcare (XLV) - Defensive growth
    - Real Estate (XLRE) - Rate sensitive
    - Materials (XLB) - Economic cycle
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 correlation_window: int = 20,
                 min_correlation_strength: float = 0.7,
                 detect_rotation: bool = True,
                 primary_sector: Optional[str] = None):
        """
        Initialize Sector Correlation indicator
        
        Args:
            lookback_days: Days of history for analysis
            correlation_window: Rolling correlation window
            min_correlation_strength: Minimum correlation for "high correlation"
            detect_rotation: Detect sector rotation patterns
            primary_sector: Primary sector to focus on (if analyzing individual stock)
        """
        super().__init__(
            name="SectorCorrelation",
            timeframe=TimeFrame.DAILY,
            lookback_periods=lookback_days + correlation_window,
            params={
                'lookback_days': lookback_days,
                'correlation_window': correlation_window,
                'min_correlation_strength': min_correlation_strength,
                'detect_rotation': detect_rotation,
                'primary_sector': primary_sector
            }
        )
        
        self.lookback_days = lookback_days
        self.correlation_window = correlation_window
        self.min_correlation_strength = min_correlation_strength
        self.detect_rotation = detect_rotation
        self.primary_sector = primary_sector
        
        # Sector characteristics
        self.sector_profiles = {
            'XLK': {'type': 'growth', 'sensitivity': 'risk_on', 'beta': 1.2},
            'XLF': {'type': 'cyclical', 'sensitivity': 'rates', 'beta': 1.1},
            'XLE': {'type': 'commodity', 'sensitivity': 'inflation', 'beta': 1.3},
            'XLU': {'type': 'defensive', 'sensitivity': 'risk_off', 'beta': 0.6},
            'XLY': {'type': 'consumer', 'sensitivity': 'growth', 'beta': 1.1},
            'XLV': {'type': 'defensive_growth', 'sensitivity': 'neutral', 'beta': 0.8},
            'XLRE': {'type': 'rate_sensitive', 'sensitivity': 'rates', 'beta': 0.9},
            'XLB': {'type': 'cyclical', 'sensitivity': 'growth', 'beta': 1.0}
        }
        
    def calculate(self, 
                  data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Sector correlations and generate signals
        
        Args:
            data: Either a dict with sector DataFrames or single asset DataFrame
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with Sector correlation analysis
        """
        # Handle different data formats
        if isinstance(data, dict):
            sector_data = data
            asset_data = data.get('asset', list(data.values())[0])
        else:
            # Single DataFrame - simulate sector behavior
            asset_data = data
            sector_data = self._simulate_sector_data(asset_data)
            
        if not sector_data or len(asset_data) < self.correlation_window:
            return self._empty_result(symbol)
            
        # Calculate sector returns
        sector_returns = self._calculate_sector_returns(sector_data)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(sector_returns)
        
        # Analyze sector performance
        sector_performance = self._analyze_sector_performance(sector_returns)
        
        # Detect market regime
        market_regime = self._detect_market_regime(
            sector_performance, correlation_matrix
        )
        
        # Detect sector rotation
        rotation_analysis = {}
        if self.detect_rotation:
            rotation_analysis = self._detect_sector_rotation(
                sector_performance, sector_returns
            )
        
        # Analyze breadth and divergences
        breadth_analysis = self._analyze_market_breadth(
            sector_performance, correlation_matrix
        )
        
        # Generate trading signals
        signal, confidence, value = self._generate_signal(
            sector_performance, market_regime, rotation_analysis,
            breadth_analysis, correlation_matrix
        )
        
        # Create metadata
        metadata = self._create_metadata(
            sector_performance, market_regime, rotation_analysis,
            breadth_analysis, correlation_matrix, len(asset_data)
        )
        
        # Get timestamp
        timestamp = int(datetime.now().timestamp() * 1000)
        if isinstance(asset_data, pd.DataFrame) and hasattr(asset_data.index[-1], 'timestamp'):
            timestamp = int(asset_data.index[-1].timestamp() * 1000)
            
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
    
    def _calculate_sector_returns(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Calculate returns for each sector"""
        
        sector_returns = {}
        
        for sector, data in sector_data.items():
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    prices = data['Close']
                elif 'close' in data.columns:
                    prices = data['close']
                else:
                    prices = data.iloc[:, 0]
                    
                returns = prices.pct_change().dropna()
                sector_returns[sector] = returns
                
        return sector_returns
    
    def _calculate_correlation_matrix(self, sector_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix between sectors"""
        
        # Create DataFrame from returns
        returns_df = pd.DataFrame(sector_returns)
        
        # Calculate rolling correlations
        correlation_matrix = returns_df.iloc[-self.correlation_window:].corr()
        
        return correlation_matrix
    
    def _analyze_sector_performance(self, sector_returns: Dict[str, pd.Series]) -> Dict:
        """Analyze individual sector performance"""
        
        performance = {}
        
        for sector, returns in sector_returns.items():
            if len(returns) < 20:
                continue
                
            # Performance metrics
            perf_1d = returns.iloc[-1] * 100 if len(returns) > 0 else 0
            perf_5d = (returns.iloc[-5:].sum() * 100) if len(returns) >= 5 else 0
            perf_20d = (returns.iloc[-20:].sum() * 100) if len(returns) >= 20 else 0
            
            # Volatility
            volatility = returns.iloc[-20:].std() * np.sqrt(252) * 100
            
            # Momentum
            if len(returns) >= 20:
                ma5 = returns.iloc[-5:].mean()
                ma20 = returns.iloc[-20:].mean()
                momentum = 'positive' if ma5 > ma20 else 'negative'
            else:
                momentum = 'neutral'
                
            # Relative strength (vs average of all sectors)
            avg_return = np.mean([r.iloc[-20:].sum() for r in sector_returns.values() if len(r) >= 20])
            relative_strength = perf_20d - (avg_return * 100)
            
            performance[sector] = {
                '1d_return': perf_1d,
                '5d_return': perf_5d,
                '20d_return': perf_20d,
                'volatility': volatility,
                'momentum': momentum,
                'relative_strength': relative_strength,
                'rank': 0  # Will be filled later
            }
        
        # Rank sectors by 20d performance
        sorted_sectors = sorted(performance.items(), 
                               key=lambda x: x[1]['20d_return'], 
                               reverse=True)
        
        for rank, (sector, _) in enumerate(sorted_sectors):
            performance[sector]['rank'] = rank + 1
            
        return performance
    
    def _detect_market_regime(self, 
                             sector_performance: Dict,
                             correlation_matrix: pd.DataFrame) -> Dict:
        """Detect overall market regime from sector behavior"""
        
        # Count outperforming sectors
        risk_on_sectors = ['XLK', 'XLY', 'XLF']
        risk_off_sectors = ['XLU', 'XLV', 'XLRE']
        
        risk_on_strength = 0
        risk_off_strength = 0
        
        for sector, perf in sector_performance.items():
            if sector in risk_on_sectors and perf['relative_strength'] > 0:
                risk_on_strength += 1
            elif sector in risk_off_sectors and perf['relative_strength'] > 0:
                risk_off_strength += 1
                
        # Average correlation level
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean()
        
        # Determine regime
        if risk_on_strength > risk_off_strength:
            sentiment = 'risk_on'
        elif risk_off_strength > risk_on_strength:
            sentiment = 'risk_off'
        else:
            sentiment = 'neutral'
            
        # Market condition based on correlations
        if avg_correlation > 0.8:
            condition = 'highly_correlated'
            implication = 'macro_driven'
        elif avg_correlation > 0.6:
            condition = 'correlated'
            implication = 'trending'
        elif avg_correlation < 0.3:
            condition = 'decorrelated'
            implication = 'stock_picking'
        else:
            condition = 'normal'
            implication = 'mixed'
            
        # Growth vs Value
        growth_sectors = ['XLK', 'XLY']
        value_sectors = ['XLF', 'XLE', 'XLU']
        
        growth_avg = np.mean([sector_performance.get(s, {}).get('20d_return', 0) 
                             for s in growth_sectors if s in sector_performance])
        value_avg = np.mean([sector_performance.get(s, {}).get('20d_return', 0) 
                            for s in value_sectors if s in sector_performance])
        
        style = 'growth' if growth_avg > value_avg else 'value'
        
        return {
            'sentiment': sentiment,
            'condition': condition,
            'market_implication': implication,
            'style_leadership': style,
            'avg_correlation': avg_correlation,
            'risk_on_strength': risk_on_strength,
            'risk_off_strength': risk_off_strength
        }
    
    def _detect_sector_rotation(self,
                               sector_performance: Dict,
                               sector_returns: Dict[str, pd.Series]) -> Dict:
        """Detect sector rotation patterns"""
        
        # Leadership changes
        current_leader = min(sector_performance.items(), 
                            key=lambda x: x[1]['rank'])[0]
        
        # Historical leaders (5 days ago)
        hist_performance = {}
        for sector, returns in sector_returns.items():
            if len(returns) >= 25:
                hist_perf = returns.iloc[-25:-5].sum() * 100
                hist_performance[sector] = hist_perf
                
        if hist_performance:
            past_leader = max(hist_performance.items(), key=lambda x: x[1])[0]
            leadership_change = current_leader != past_leader
        else:
            past_leader = None
            leadership_change = False
            
        # Rotation patterns
        rotation_pattern = 'none'
        
        # Classic rotations
        if leadership_change:
            if past_leader == 'XLK' and current_leader in ['XLF', 'XLE']:
                rotation_pattern = 'growth_to_value'
            elif past_leader in ['XLF', 'XLE'] and current_leader == 'XLK':
                rotation_pattern = 'value_to_growth'
            elif past_leader in ['XLK', 'XLY'] and current_leader in ['XLU', 'XLV']:
                rotation_pattern = 'risk_on_to_risk_off'
            elif past_leader in ['XLU', 'XLV'] and current_leader in ['XLK', 'XLY']:
                rotation_pattern = 'risk_off_to_risk_on'
            else:
                rotation_pattern = 'sector_specific'
                
        # Rotation strength (how decisive)
        if sector_performance:
            leader_margin = (sector_performance[current_leader]['20d_return'] - 
                           np.mean([p['20d_return'] for p in sector_performance.values()]))
            rotation_strength = 'strong' if abs(leader_margin) > 5 else 'moderate'
        else:
            rotation_strength = 'weak'
            
        return {
            'current_leader': current_leader,
            'past_leader': past_leader,
            'leadership_change': leadership_change,
            'rotation_pattern': rotation_pattern,
            'rotation_strength': rotation_strength,
            'leader_outperformance': leader_margin if sector_performance else 0
        }
    
    def _analyze_market_breadth(self,
                               sector_performance: Dict,
                               correlation_matrix: pd.DataFrame) -> Dict:
        """Analyze market breadth and divergences"""
        
        if not sector_performance:
            return {}
            
        # Breadth metrics
        positive_sectors = sum(1 for p in sector_performance.values() 
                              if p['20d_return'] > 0)
        total_sectors = len(sector_performance)
        
        breadth_ratio = positive_sectors / total_sectors if total_sectors > 0 else 0.5
        
        # Divergence detection
        divergences = []
        
        # Check for major divergences
        if 'XLK' in sector_performance and 'XLF' in sector_performance:
            tech_perf = sector_performance['XLK']['20d_return']
            fin_perf = sector_performance['XLF']['20d_return']
            
            if tech_perf > 5 and fin_perf < -5:
                divergences.append('tech_financial_divergence')
            elif tech_perf < -5 and fin_perf > 5:
                divergences.append('financial_tech_divergence')
                
        # Defensive vs Cyclical divergence
        defensive_avg = np.mean([sector_performance.get(s, {}).get('20d_return', 0) 
                                for s in ['XLU', 'XLV'] if s in sector_performance])
        cyclical_avg = np.mean([sector_performance.get(s, {}).get('20d_return', 0) 
                               for s in ['XLF', 'XLE', 'XLB'] if s in sector_performance])
        
        if defensive_avg > 5 and cyclical_avg < -5:
            divergences.append('defensive_outperformance')
        elif defensive_avg < -5 and cyclical_avg > 5:
            divergences.append('cyclical_outperformance')
            
        # Breadth quality
        if breadth_ratio > 0.7:
            breadth_quality = 'strong'
        elif breadth_ratio < 0.3:
            breadth_quality = 'weak'
        else:
            breadth_quality = 'mixed'
            
        return {
            'breadth_ratio': breadth_ratio,
            'positive_sectors': positive_sectors,
            'total_sectors': total_sectors,
            'breadth_quality': breadth_quality,
            'divergences': divergences,
            'has_divergence': len(divergences) > 0
        }
    
    def _simulate_sector_data(self, asset_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Simulate sector data if not provided"""
        
        sector_data = {}
        base_returns = asset_data.pct_change().dropna()
        
        # Simulate each sector with characteristic behavior
        for sector, profile in self.sector_profiles.items():
            sector_prices = []
            price = 100
            
            for i, (idx, base_ret) in enumerate(base_returns.items()):
                # Base movement correlated with asset
                movement = base_ret * profile['beta']
                
                # Add sector-specific behavior
                if profile['sensitivity'] == 'risk_on':
                    movement += np.random.normal(0.0002, 0.005)
                elif profile['sensitivity'] == 'risk_off':
                    movement -= base_ret * 0.3  # Inverse during risk-off
                elif profile['sensitivity'] == 'rates':
                    movement += np.random.normal(0, 0.006)
                elif profile['sensitivity'] == 'inflation':
                    movement += np.random.normal(0.0001, 0.008)
                    
                price *= (1 + movement)
                sector_prices.append(price)
                
            sector_df = pd.DataFrame({
                'close': sector_prices
            }, index=base_returns.index)
            
            sector_data[sector] = sector_df
            
        return sector_data
    
    def _generate_signal(self,
                        sector_performance: Dict,
                        market_regime: Dict,
                        rotation_analysis: Dict,
                        breadth_analysis: Dict,
                        correlation_matrix: pd.DataFrame) -> Tuple[SignalType, float, float]:
        """Generate trading signal from Sector correlation analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Base signal on market regime
        sentiment = market_regime['sentiment']
        
        if sentiment == 'risk_on':
            signal = SignalType.BUY
            confidence = 60
        elif sentiment == 'risk_off':
            signal = SignalType.SELL
            confidence = 60
        else:
            signal = SignalType.HOLD
            confidence = 40
            
        # Adjust for breadth
        breadth_quality = breadth_analysis.get('breadth_quality', 'mixed')
        
        if breadth_quality == 'strong':
            # Broad participation
            if signal == SignalType.BUY:
                confidence += 15
            elif signal == SignalType.SELL:
                confidence -= 10  # Strong breadth conflicts with sell
        elif breadth_quality == 'weak':
            # Narrow market
            if signal == SignalType.SELL:
                confidence += 15
            elif signal == SignalType.BUY:
                confidence -= 10  # Weak breadth conflicts with buy
                
        # Rotation signals
        if rotation_analysis.get('rotation_pattern'):
            pattern = rotation_analysis['rotation_pattern']
            
            if pattern == 'risk_off_to_risk_on':
                signal = SignalType.BUY
                confidence += 10
            elif pattern == 'risk_on_to_risk_off':
                signal = SignalType.SELL
                confidence += 10
            elif pattern == 'growth_to_value' and self.primary_sector in ['XLK', 'tech']:
                # Negative for growth stocks
                signal = SignalType.SELL
                confidence += 5
                
        # Divergence warnings
        if breadth_analysis.get('has_divergence', False):
            # Divergences = caution
            confidence *= 0.8
            
            divergences = breadth_analysis.get('divergences', [])
            if 'defensive_outperformance' in divergences:
                signal = SignalType.SELL
                confidence = 65
                
        # Correlation regime
        avg_corr = market_regime.get('avg_correlation', 0.5)
        
        if avg_corr > 0.85:
            # Extremely high correlation = macro event
            confidence += 10  # Strong directional move
        elif avg_corr < 0.3:
            # Low correlation = stock picking market
            confidence *= 0.7  # Index-level signals less reliable
            
        confidence = min(confidence, 85)
        
        # Value represents market breadth strength
        value = breadth_analysis.get('breadth_ratio', 0.5) * 100
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        sector_performance: Dict,
                        market_regime: Dict,
                        rotation_analysis: Dict,
                        breadth_analysis: Dict,
                        correlation_matrix: pd.DataFrame,
                        data_points: int) -> Dict:
        """Create comprehensive metadata"""
        
        metadata = {
            'data_points': data_points,
            'market_sentiment': market_regime['sentiment'],
            'market_condition': market_regime['condition'],
            'style_leadership': market_regime['style_leadership'],
            'avg_sector_correlation': market_regime['avg_correlation']
        }
        
        # Add top/bottom performers
        if sector_performance:
            sorted_sectors = sorted(sector_performance.items(),
                                   key=lambda x: x[1]['20d_return'],
                                   reverse=True)
            
            metadata['top_sectors'] = [
                {
                    'sector': s[0],
                    'return': s[1]['20d_return'],
                    'momentum': s[1]['momentum']
                }
                for s in sorted_sectors[:3]
            ]
            
            metadata['bottom_sectors'] = [
                {
                    'sector': s[0],
                    'return': s[1]['20d_return'],
                    'momentum': s[1]['momentum']
                }
                for s in sorted_sectors[-3:]
            ]
            
        # Add rotation info
        if rotation_analysis:
            metadata.update({
                'current_leader': rotation_analysis.get('current_leader'),
                'rotation_pattern': rotation_analysis.get('rotation_pattern', 'none'),
                'leadership_change': rotation_analysis.get('leadership_change', False)
            })
            
        # Add breadth info
        if breadth_analysis:
            metadata.update({
                'market_breadth': breadth_analysis.get('breadth_ratio', 0.5),
                'breadth_quality': breadth_analysis.get('breadth_quality', 'unknown'),
                'divergences': breadth_analysis.get('divergences', [])
            })
            
        # Trading insights
        if market_regime['sentiment'] == 'risk_on' and breadth_analysis.get('breadth_quality') == 'strong':
            metadata['insight'] = "Strong risk-on with broad participation"
        elif rotation_analysis.get('rotation_pattern') == 'risk_on_to_risk_off':
            metadata['insight'] = "Rotation to defensive sectors detected"
        elif breadth_analysis.get('has_divergence', False):
            metadata['insight'] = "Sector divergences suggest caution"
        elif market_regime['avg_correlation'] > 0.85:
            metadata['insight'] = "High correlation - macro-driven market"
        else:
            metadata['insight'] = "Normal sector behavior"
            
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
            metadata={'error': 'Insufficient data for Sector correlation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[pd.DataFrame, Dict]) -> bool:
        """Validate input data"""
        
        if isinstance(data, dict):
            # Need at least some sectors
            return len(data) >= 3
        else:
            return isinstance(data, pd.DataFrame) and len(data) >= self.correlation_window


def demonstrate_sector_correlation():
    """Demonstration of Sector Correlation indicator"""
    
    print("🏢 Sector Correlation Analysis Demonstration\n")
    
    # Generate synthetic sector data
    print("Generating synthetic sector data with rotation...\n")
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create sector data with different regimes
    sector_data = {}
    
    # Base market movement
    market_trend = []
    trend = 0
    for i in range(len(dates)):
        if i < 30:
            # Risk-on phase
            trend += np.random.normal(0.001, 0.01)
        elif i < 60:
            # Rotation phase
            trend += np.random.normal(0, 0.012)
        else:
            # Risk-off phase
            trend += np.random.normal(-0.0005, 0.015)
        market_trend.append(trend)
    
    # Generate each sector
    for sector, profile in {'XLK': {'beta': 1.3, 'type': 'growth'},
                           'XLF': {'beta': 1.1, 'type': 'cyclical'},
                           'XLU': {'beta': 0.6, 'type': 'defensive'},
                           'XLE': {'beta': 1.4, 'type': 'commodity'},
                           'XLY': {'beta': 1.2, 'type': 'consumer'},
                           'XLV': {'beta': 0.8, 'type': 'defensive_growth'}}.items():
        
        prices = []
        price = 100
        
        for i, base_move in enumerate(market_trend):
            # Sector-specific behavior
            if i < 30:  # Risk-on
                if profile['type'] in ['growth', 'consumer']:
                    move = base_move * profile['beta'] * 1.2
                else:
                    move = base_move * profile['beta'] * 0.8
            elif i < 60:  # Rotation
                if profile['type'] in ['cyclical', 'commodity']:
                    move = base_move * profile['beta'] * 1.1
                else:
                    move = base_move * profile['beta'] * 0.9
            else:  # Risk-off
                if profile['type'] in ['defensive', 'defensive_growth']:
                    move = -base_move * 0.5  # Defensive outperformance
                else:
                    move = base_move * profile['beta'] * 1.3
                    
            # Add sector noise
            move += np.random.normal(0, 0.005)
            
            if i > 0:
                price *= (1 + move)
            prices.append(price)
            
        sector_data[sector] = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices
        }, index=dates)
    
    # Create indicator
    sector_corr = SectorCorrelationIndicator(
        lookback_days=20,
        correlation_window=20,
        detect_rotation=True
    )
    
    # Calculate
    result = sector_corr.calculate(sector_data, "MARKET")
    
    print("=" * 60)
    print("SECTOR CORRELATION ANALYSIS:")
    print("=" * 60)
    
    # Market regime
    print(f"Market Sentiment: {result.metadata['market_sentiment'].upper()}")
    print(f"Market Condition: {result.metadata['market_condition'].upper()}")
    print(f"Style Leadership: {result.metadata['style_leadership'].upper()}")
    print(f"Average Correlation: {result.metadata['avg_sector_correlation']:.3f}")
    
    # Top/Bottom sectors
    print("\nTop Performing Sectors:")
    for sector in result.metadata.get('top_sectors', []):
        print(f"  {sector['sector']}: {sector['return']:+.1f}% ({sector['momentum']})")
        
    print("\nBottom Performing Sectors:")
    for sector in result.metadata.get('bottom_sectors', []):
        print(f"  {sector['sector']}: {sector['return']:+.1f}% ({sector['momentum']})")
    
    # Rotation
    print(f"\nCurrent Leader: {result.metadata.get('current_leader', 'Unknown')}")
    if result.metadata.get('leadership_change', False):
        print("⚠️ LEADERSHIP CHANGE DETECTED")
    print(f"Rotation Pattern: {result.metadata.get('rotation_pattern', 'None').upper()}")
    
    # Breadth
    print(f"\nMarket Breadth: {result.metadata.get('market_breadth', 0.5):.1%}")
    print(f"Breadth Quality: {result.metadata.get('breadth_quality', 'Unknown').upper()}")
    
    # Divergences
    divergences = result.metadata.get('divergences', [])
    if divergences:
        print("\nDivergences Detected:")
        for div in divergences:
            print(f"  - {div.replace('_', ' ').title()}")
    
    # Trading signal
    print(f"\nTrading Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Breadth Strength: {result.value:.1f}/100")
    
    # Insight
    print(f"\nMarket Insight: {result.metadata.get('insight', 'No specific insight')}")
    
    print("\n💡 Sector Correlation Trading Tips:")
    print("- High correlation (>0.8) = Macro-driven market")
    print("- Low correlation (<0.3) = Stock picking environment")
    print("- Tech leadership = Risk-on sentiment")
    print("- Utilities/Healthcare leadership = Risk-off")
    print("- Rotation patterns often precede trend changes")
    print("- Breadth divergences warn of market weakness")


if __name__ == "__main__":
    demonstrate_sector_correlation()