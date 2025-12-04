"""
🌀 AEGS Fractal Alpha Enhancement System
Multi-temporal alpha capture through fractal market analysis
"""

__version__ = "0.1.0"
__author__ = "Moon Dev"

from .base.indicator import BaseIndicator, IndicatorResult
from .base.data_manager import FractalDataManager
from .ensemble.signal_combiner import FractalEnsemble

__all__ = [
    "BaseIndicator",
    "IndicatorResult", 
    "FractalDataManager",
    "FractalEnsemble"
]