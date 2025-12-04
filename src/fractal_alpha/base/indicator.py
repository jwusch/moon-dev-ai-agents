"""Base indicator class for all fractal alpha indicators"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from datetime import datetime
import time

from .types import TimeFrame, SignalType, MarketRegime

@dataclass
class IndicatorResult:
    """Standard result format for all indicators"""
    timestamp: int
    symbol: str
    indicator_name: str
    value: float
    signal: SignalType
    confidence: float  # 0-100
    timeframe: TimeFrame
    metadata: Dict[str, Any]
    calculation_time_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'indicator_name': self.indicator_name,
            'value': self.value,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'timeframe': self.timeframe.value,
            'metadata': self.metadata,
            'calculation_time_ms': self.calculation_time_ms
        }

class BaseIndicator(ABC):
    """Base class for all fractal indicators"""
    
    def __init__(self, 
                 name: str,
                 timeframe: TimeFrame,
                 lookback_periods: int,
                 params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.params = params or {}
        self._cache = {}
        self._last_calculation_time = 0
        
    @abstractmethod
    def calculate(self, 
                  data: Union[pd.DataFrame, np.ndarray], 
                  symbol: str) -> IndicatorResult:
        """Calculate indicator value and generate signal"""
        pass
        
    @abstractmethod
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate input data meets requirements"""
        pass
        
    def warmup_periods(self) -> int:
        """Number of periods needed before first valid signal"""
        return self.lookback_periods
        
    def get_required_columns(self) -> List[str]:
        """List of required data columns"""
        return ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
    def calculate_with_timing(self, 
                            data: Union[pd.DataFrame, np.ndarray], 
                            symbol: str) -> IndicatorResult:
        """Calculate indicator with performance timing"""
        start_time = time.time()
        
        # Validate data
        if not self.validate_data(data):
            raise ValueError(f"Invalid data for {self.name} indicator")
            
        # Calculate
        result = self.calculate(data, symbol)
        
        # Add timing
        result.calculation_time_ms = (time.time() - start_time) * 1000
        
        return result
        
    def _generate_signal(self, 
                        value: float, 
                        threshold_buy: float,
                        threshold_sell: float) -> SignalType:
        """Generate signal based on thresholds"""
        if value >= threshold_buy:
            return SignalType.BUY
        elif value <= threshold_sell:
            return SignalType.SELL
        else:
            return SignalType.HOLD
            
    def _calculate_confidence(self, 
                            value: float,
                            min_val: float,
                            max_val: float) -> float:
        """Calculate confidence score 0-100"""
        if max_val == min_val:
            return 50.0
            
        normalized = (value - min_val) / (max_val - min_val)
        
        # Convert to 0-100, where extremes have high confidence
        if normalized >= 0.5:
            confidence = 50 + (normalized - 0.5) * 100
        else:
            confidence = 50 + (0.5 - normalized) * 100
            
        return min(100.0, max(0.0, confidence))
        
    def _detect_regime(self, 
                      returns: np.ndarray,
                      window: int = 20) -> MarketRegime:
        """Detect current market regime"""
        if len(returns) < window:
            return MarketRegime.UNKNOWN
            
        recent_returns = returns[-window:]
        volatility = np.std(recent_returns)
        trend = np.mean(recent_returns)
        
        # Volatility thresholds
        high_vol_threshold = np.percentile(np.abs(returns), 80)
        low_vol_threshold = np.percentile(np.abs(returns), 20)
        
        # Classify regime
        if volatility > high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < low_vol_threshold:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.001:  # 0.1% average return
            return MarketRegime.TRENDING_UP
        elif trend < -0.001:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.MEAN_REVERTING