"""Signal combination and ensemble voting logic"""

from typing import List, Dict, Optional
import numpy as np
from ..base.indicator import IndicatorResult
from ..base.types import SignalType


class FractalEnsemble:
    """Combines signals from multiple indicators"""
    
    def __init__(self):
        self.weights = {}
        self.min_signals = 2
        
    def combine_signals(self, signals: List[IndicatorResult]) -> Optional[IndicatorResult]:
        """Combine multiple indicator signals into ensemble signal"""
        
        if len(signals) < self.min_signals:
            return None
            
        # For now, simple weighted voting
        # Will enhance with ML later
        buy_score = 0
        sell_score = 0
        total_confidence = 0
        
        for signal in signals:
            weight = self.weights.get(signal.indicator_name, 1.0)
            confidence = signal.confidence / 100.0
            
            if signal.signal == SignalType.BUY:
                buy_score += weight * confidence
            elif signal.signal == SignalType.SELL:
                sell_score += weight * confidence
                
            total_confidence += confidence
            
        # Determine ensemble signal
        if buy_score > sell_score * 1.5:  # 50% stronger
            ensemble_signal = SignalType.BUY
        elif sell_score > buy_score * 1.5:
            ensemble_signal = SignalType.SELL
        else:
            ensemble_signal = SignalType.HOLD
            
        # Calculate ensemble confidence
        avg_confidence = (total_confidence / len(signals)) * 100
        
        # Create ensemble result
        return IndicatorResult(
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            indicator_name="FractalEnsemble",
            value=buy_score - sell_score,
            signal=ensemble_signal,
            confidence=avg_confidence,
            timeframe=signals[0].timeframe,
            metadata={
                'num_signals': len(signals),
                'buy_score': buy_score,
                'sell_score': sell_score
            },
            calculation_time_ms=0
        )