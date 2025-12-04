"""
🌀 FRACTAL ALPHA INDICATORS UNIFIED MODULE
Complete collection of advanced fractal market indicators for regime detection,
microstructure analysis, and alpha generation across multiple timeframes.

This module provides a unified interface to all fractal indicators with:
- Easy-to-use factory methods
- Pre-configured indicator ensembles  
- Regime-aware signal generation
- Performance monitoring and caching
- Backtesting integration
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all fractal indicators
from .base.indicator import BaseIndicator, IndicatorResult
from .base.types import TimeFrame, SignalType, MarketRegime

# Fractal Indicators
from .indicators.fractals.hurst_exponent import HurstExponentIndicator
from .indicators.fractals.williams_fractals import WilliamsFractalsIndicator

# Microstructure Indicators
from .indicators.microstructure.vpin import VPINIndicator
from .indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from .indicators.microstructure.amihud_illiquidity import AmihudIlliquidityIndicator
from .indicators.microstructure.tick_volume_imbalance import TickVolumeImbalanceIndicator
from .indicators.microstructure.bid_ask_dynamics import BidAskDynamicsIndicator
from .indicators.microstructure.order_flow_divergence import OrderFlowDivergenceIndicator

# Mean Reversion Indicators
from .indicators.mean_reversion.ou_process import OrnsteinUhlenbeckIndicator
from .indicators.mean_reversion.dynamic_zscore import DynamicZScoreIndicator
from .indicators.mean_reversion.pairs_trading import PairsTradingIndicator

# Time Pattern Indicators
from .indicators.time_patterns.volume_bars import VolumeClockIndicator
from .indicators.time_patterns.renko_bars import RenkoBarIndicator
from .indicators.time_patterns.intraday_seasonality import IntradaySeasonalityIndicator

# Correlation Indicators
from .indicators.correlations.vix_correlation import VIXCorrelationIndicator
from .indicators.correlations.dollar_correlation import DollarCorrelationIndicator
from .indicators.correlations.sector_correlation import SectorCorrelationIndicator

# ML Feature Indicators
from .indicators.ml_features.entropy import EntropyIndicator
from .indicators.ml_features.wavelet import WaveletIndicator
from .indicators.ml_features.hmm import HiddenMarkovModelIndicator

# Multifractal Indicators  
from .indicators.multifractal.dfa import DFAIndicator

# Ensemble and utilities
from .ensemble.signal_combiner import SignalCombiner
from .utils.cache import IndicatorCache
from .utils.performance import PerformanceProfiler


class IndicatorCategory(Enum):
    """Categories of fractal indicators"""
    FRACTALS = "fractals"
    MICROSTRUCTURE = "microstructure"
    MEAN_REVERSION = "mean_reversion"
    TIME_PATTERNS = "time_patterns"
    CORRELATIONS = "correlations"
    ML_FEATURES = "ml_features"
    MULTIFRACTAL = "multifractal"


@dataclass
class IndicatorEnsembleResult:
    """Results from running an ensemble of indicators"""
    symbol: str
    timestamp: datetime
    individual_results: Dict[str, IndicatorResult]
    combined_signal: SignalType
    combined_confidence: float
    regime: MarketRegime
    ensemble_score: float
    execution_time_ms: float
    
    def get_signals_by_category(self, category: IndicatorCategory) -> List[IndicatorResult]:
        """Get all signals from a specific category"""
        return [result for name, result in self.individual_results.items() 
                if category.value in name.lower()]
    
    def get_top_signals(self, n: int = 5) -> List[IndicatorResult]:
        """Get top N signals by confidence"""
        return sorted(self.individual_results.values(), 
                     key=lambda x: x.confidence, reverse=True)[:n]


class FractalIndicatorFactory:
    """Factory for creating and configuring fractal indicators"""
    
    @staticmethod
    def create_hurst_exponent(timeframe: TimeFrame = TimeFrame.HOUR_1,
                             lookback: int = 100) -> HurstExponentIndicator:
        """Create Hurst Exponent indicator for regime detection"""
        return HurstExponentIndicator(
            timeframe=timeframe,
            lookback_periods=lookback,
            params={'method': 'rs', 'min_periods': 20}
        )
    
    @staticmethod
    def create_vpin(timeframe: TimeFrame = TimeFrame.MIN_5,
                   bucket_size: int = 50) -> VPINIndicator:
        """Create VPIN indicator for informed trading detection"""
        return VPINIndicator(
            timeframe=timeframe,
            lookback_periods=bucket_size,
            params={'bucket_size': bucket_size, 'alpha': 0.95}
        )
    
    @staticmethod
    def create_ou_process(timeframe: TimeFrame = TimeFrame.MIN_15,
                         lookback: int = 50) -> OrnsteinUhlenbeckIndicator:
        """Create Ornstein-Uhlenbeck mean reversion indicator"""
        return OrnsteinUhlenbeckIndicator(
            timeframe=timeframe,
            lookback_periods=lookback,
            params={'confidence_threshold': 0.8}
        )
    
    @staticmethod
    def create_dynamic_zscore(timeframe: TimeFrame = TimeFrame.MIN_5,
                            lookback: int = 20) -> DynamicZScoreIndicator:
        """Create adaptive Z-Score bands"""
        return DynamicZScoreIndicator(
            timeframe=timeframe,
            lookback_periods=lookback,
            params={'alpha': 0.1, 'buy_threshold': -2, 'sell_threshold': 2}
        )
    
    @staticmethod
    def create_entropy_indicator(timeframe: TimeFrame = TimeFrame.MIN_15,
                               lookback: int = 30) -> EntropyIndicator:
        """Create Shannon entropy indicator for regime change detection"""
        return EntropyIndicator(
            timeframe=timeframe,
            lookback_periods=lookback,
            params={'bins': 10, 'threshold': 1.5}
        )
    
    @staticmethod
    def create_williams_fractals(timeframe: TimeFrame = TimeFrame.MIN_15,
                               periods: int = 5) -> WilliamsFractalsIndicator:
        """Create Williams Fractals indicator"""
        return WilliamsFractalsIndicator(
            timeframe=timeframe,
            lookback_periods=periods,
            params={'periods': periods, 'require_confirmation': True}
        )
    
    @staticmethod
    def create_volume_clock(volume_threshold: int = 1000000) -> VolumeClockIndicator:
        """Create volume-based bar aggregation"""
        return VolumeClockIndicator(
            timeframe=TimeFrame.TICK,
            lookback_periods=50,
            params={'volume_threshold': volume_threshold}
        )


class FractalIndicatorEnsemble:
    """Ensemble of fractal indicators with regime-aware signal combination"""
    
    def __init__(self, 
                 timeframe: TimeFrame = TimeFrame.MIN_15,
                 enable_caching: bool = True,
                 performance_monitoring: bool = True):
        self.timeframe = timeframe
        self.indicators: Dict[str, BaseIndicator] = {}
        self.signal_combiner = SignalCombiner()
        self.cache = IndicatorCache() if enable_caching else None
        self.profiler = PerformanceProfiler() if performance_monitoring else None
        
        # Initialize default indicator set
        self._initialize_default_indicators()
    
    def _initialize_default_indicators(self):
        """Initialize a balanced set of indicators for alpha generation"""
        factory = FractalIndicatorFactory()
        
        # Core regime detection
        self.indicators['hurst_exponent'] = factory.create_hurst_exponent(self.timeframe)
        self.indicators['entropy'] = factory.create_entropy_indicator(self.timeframe)
        
        # Mean reversion signals
        self.indicators['ou_process'] = factory.create_ou_process(self.timeframe)
        self.indicators['dynamic_zscore'] = factory.create_dynamic_zscore(self.timeframe)
        
        # Microstructure insights
        if self.timeframe in [TimeFrame.MIN_1, TimeFrame.MIN_5, TimeFrame.MIN_15]:
            self.indicators['vpin'] = factory.create_vpin(self.timeframe)
            self.indicators['tick_imbalance'] = TickVolumeImbalanceIndicator(
                timeframe=self.timeframe,
                lookback_periods=30,
                params={'imbalance_threshold': 0.6}
            )
        
        # Pattern detection
        self.indicators['williams_fractals'] = factory.create_williams_fractals(self.timeframe)
        
        # Advanced features for longer timeframes
        if self.timeframe in [TimeFrame.HOUR_1, TimeFrame.HOUR_4, TimeFrame.DAY_1]:
            self.indicators['hmm_regime'] = HiddenMarkovModelIndicator(
                timeframe=self.timeframe,
                lookback_periods=100,
                params={'n_states': 3, 'covariance_type': 'full'}
            )
    
    def add_indicator(self, name: str, indicator: BaseIndicator):
        """Add custom indicator to ensemble"""
        self.indicators[name] = indicator
    
    def remove_indicator(self, name: str):
        """Remove indicator from ensemble"""
        if name in self.indicators:
            del self.indicators[name]
    
    def get_indicator_names(self) -> List[str]:
        """Get list of all indicator names"""
        return list(self.indicators.keys())
    
    def analyze_symbol(self, symbol: str, 
                      data: Optional[pd.DataFrame] = None) -> IndicatorEnsembleResult:
        """Run complete fractal analysis on a symbol"""
        start_time = datetime.now()
        
        # Get data if not provided
        if data is None:
            data = self._fetch_data(symbol)
        
        if data is None or len(data) < 50:  # Minimum data requirement
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Run all indicators
        individual_results = {}
        execution_times = []
        
        for name, indicator in self.indicators.items():
            try:
                # Check cache first
                cache_key = f"{symbol}_{name}_{self.timeframe.value}_{len(data)}"
                
                if self.cache and self.cache.has_key(cache_key):
                    result = self.cache.get(cache_key)
                else:
                    # Calculate indicator
                    result = indicator.calculate_with_timing(data, symbol)
                    
                    # Cache result
                    if self.cache:
                        self.cache.set(cache_key, result, ttl_minutes=15)
                
                individual_results[name] = result
                execution_times.append(result.calculation_time_ms)
                
                # Profile performance
                if self.profiler:
                    self.profiler.record_indicator_performance(
                        indicator_name=name,
                        symbol=symbol,
                        execution_time_ms=result.calculation_time_ms,
                        confidence=result.confidence
                    )
                    
            except Exception as e:
                print(f"Error calculating {name} for {symbol}: {str(e)}")
                continue
        
        if not individual_results:
            raise ValueError(f"No indicators calculated successfully for {symbol}")
        
        # Combine signals
        signals = list(individual_results.values())
        combined_signal, combined_confidence, regime = self.signal_combiner.combine_signals(signals)
        
        # Calculate ensemble score
        ensemble_score = self._calculate_ensemble_score(individual_results, combined_confidence)
        
        total_execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IndicatorEnsembleResult(
            symbol=symbol,
            timestamp=datetime.now(),
            individual_results=individual_results,
            combined_signal=combined_signal,
            combined_confidence=combined_confidence,
            regime=regime,
            ensemble_score=ensemble_score,
            execution_time_ms=total_execution_time
        )
    
    def batch_analyze(self, symbols: List[str]) -> Dict[str, IndicatorEnsembleResult]:
        """Analyze multiple symbols in batch"""
        results = {}
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol)
                results[symbol] = result
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_regime_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed regime analysis for a symbol"""
        result = self.analyze_symbol(symbol)
        
        regime_indicators = {
            'hurst_exponent': result.individual_results.get('hurst_exponent'),
            'entropy': result.individual_results.get('entropy'),
            'hmm_regime': result.individual_results.get('hmm_regime')
        }
        
        # Filter out None results
        regime_indicators = {k: v for k, v in regime_indicators.items() if v is not None}
        
        return {
            'current_regime': result.regime,
            'confidence': result.combined_confidence,
            'regime_indicators': regime_indicators,
            'regime_stability': self._assess_regime_stability(regime_indicators)
        }
    
    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for analysis"""
        try:
            # Determine period based on timeframe
            if self.timeframe in [TimeFrame.MIN_1, TimeFrame.MIN_5]:
                period = "5d"
                interval = "1m"
            elif self.timeframe == TimeFrame.MIN_15:
                period = "30d"
                interval = "15m"
            elif self.timeframe == TimeFrame.HOUR_1:
                period = "90d"
                interval = "1h"
            else:
                period = "1y"
                interval = "1d"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            data.reset_index(inplace=True)
            
            # Add timestamp column
            if 'datetime' in data.columns:
                data['timestamp'] = pd.to_datetime(data['datetime']).astype(np.int64) // 10**9
            elif 'date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['date']).astype(np.int64) // 10**9
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _calculate_ensemble_score(self, 
                                individual_results: Dict[str, IndicatorResult],
                                combined_confidence: float) -> float:
        """Calculate overall ensemble score 0-100"""
        if not individual_results:
            return 0.0
        
        # Weight different indicator categories
        category_weights = {
            IndicatorCategory.FRACTALS: 0.25,
            IndicatorCategory.MICROSTRUCTURE: 0.20,
            IndicatorCategory.MEAN_REVERSION: 0.25,
            IndicatorCategory.ML_FEATURES: 0.15,
            IndicatorCategory.TIME_PATTERNS: 0.10,
            IndicatorCategory.CORRELATIONS: 0.05
        }
        
        category_scores = {}
        
        for category in IndicatorCategory:
            category_results = []
            for name, result in individual_results.items():
                if category.value in name.lower():
                    category_results.append(result.confidence)
            
            if category_results:
                category_scores[category] = np.mean(category_results)
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, weight in category_weights.items():
            if category in category_scores:
                weighted_score += category_scores[category] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        
        # Combine with signal confidence
        ensemble_score = (weighted_score * 0.7) + (combined_confidence * 0.3)
        
        return min(100.0, max(0.0, ensemble_score))
    
    def _assess_regime_stability(self, regime_indicators: Dict[str, IndicatorResult]) -> float:
        """Assess how stable the current regime is (0-100)"""
        if not regime_indicators:
            return 50.0
        
        # Calculate variance in regime-related signals
        confidences = [result.confidence for result in regime_indicators.values()]
        
        if len(confidences) < 2:
            return confidences[0] if confidences else 50.0
        
        # High stability = low variance in high-confidence signals
        mean_confidence = np.mean(confidences)
        variance = np.var(confidences)
        
        # Normalize variance (lower is more stable)
        stability = mean_confidence * (1 - min(variance / 1000, 0.5))
        
        return min(100.0, max(0.0, stability))


class FractalIndicatorConfig:
    """Configuration presets for different trading strategies"""
    
    @staticmethod
    def scalping_config(timeframe: TimeFrame = TimeFrame.MIN_1) -> FractalIndicatorEnsemble:
        """Configuration optimized for scalping (1-5 min trades)"""
        ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
        
        # Add scalping-specific indicators
        factory = FractalIndicatorFactory()
        
        ensemble.add_indicator('tick_imbalance', 
                             factory.create_volume_clock(volume_threshold=100000))
        ensemble.add_indicator('bid_ask_dynamics',
                             BidAskDynamicsIndicator(
                                 timeframe=timeframe,
                                 lookback_periods=20,
                                 params={'spread_threshold': 0.001}
                             ))
        
        return ensemble
    
    @staticmethod
    def swing_trading_config(timeframe: TimeFrame = TimeFrame.HOUR_1) -> FractalIndicatorEnsemble:
        """Configuration optimized for swing trading (1-5 day holds)"""
        ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
        
        # Add swing trading indicators
        ensemble.add_indicator('sector_correlation',
                             SectorCorrelationIndicator(
                                 timeframe=timeframe,
                                 lookback_periods=50,
                                 params={'correlation_threshold': 0.7}
                             ))
        ensemble.add_indicator('vix_correlation',
                             VIXCorrelationIndicator(
                                 timeframe=timeframe,
                                 lookback_periods=30,
                                 params={'fear_threshold': 20}
                             ))
        
        return ensemble
    
    @staticmethod
    def mean_reversion_config(timeframe: TimeFrame = TimeFrame.MIN_15) -> FractalIndicatorEnsemble:
        """Configuration optimized for mean reversion strategies"""
        ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
        
        # Focus on mean reversion indicators
        factory = FractalIndicatorFactory()
        
        # Remove momentum-based indicators
        ensemble.remove_indicator('williams_fractals')
        
        # Add specialized mean reversion indicators
        ensemble.add_indicator('pairs_trading',
                             PairsTradingIndicator(
                                 timeframe=timeframe,
                                 lookback_periods=60,
                                 params={'cointegration_threshold': 0.05}
                             ))
        ensemble.add_indicator('amihud_illiquidity',
                             AmihudIlliquidityIndicator(
                                 timeframe=timeframe,
                                 lookback_periods=30,
                                 params={'illiquidity_threshold': 0.1}
                             ))
        
        return ensemble


# Convenience functions for quick access
def analyze_symbol_quick(symbol: str, 
                        timeframe: TimeFrame = TimeFrame.MIN_15) -> IndicatorEnsembleResult:
    """Quick analysis of a single symbol with default settings"""
    ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
    return ensemble.analyze_symbol(symbol)


def get_regime_quick(symbol: str, 
                    timeframe: TimeFrame = TimeFrame.HOUR_1) -> MarketRegime:
    """Quick regime detection for a symbol"""
    ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
    result = ensemble.analyze_symbol(symbol)
    return result.regime


def screen_symbols_fractal(symbols: List[str], 
                          timeframe: TimeFrame = TimeFrame.MIN_15,
                          min_ensemble_score: float = 70.0) -> List[Tuple[str, float]]:
    """Screen symbols using fractal indicators and return top candidates"""
    ensemble = FractalIndicatorEnsemble(timeframe=timeframe)
    results = ensemble.batch_analyze(symbols)
    
    # Filter and sort by ensemble score
    qualified = [(symbol, result.ensemble_score) 
                for symbol, result in results.items() 
                if result.ensemble_score >= min_ensemble_score]
    
    return sorted(qualified, key=lambda x: x[1], reverse=True)


# Export main classes and functions
__all__ = [
    'FractalIndicatorFactory',
    'FractalIndicatorEnsemble', 
    'FractalIndicatorConfig',
    'IndicatorEnsembleResult',
    'IndicatorCategory',
    'analyze_symbol_quick',
    'get_regime_quick',
    'screen_symbols_fractal'
]