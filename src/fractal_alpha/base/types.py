"""Common types and enums for fractal alpha system"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

class TimeFrame(Enum):
    """Supported timeframes for fractal analysis"""
    TICK = "tick"
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"
    
    @property
    def seconds(self) -> int:
        """Convert timeframe to seconds"""
        mapping = {
            "tick": 0,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        return mapping.get(self.value, 0)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"

@dataclass
class TickData:
    """Single tick data point"""
    timestamp: int  # Unix timestamp in milliseconds
    price: float
    volume: int
    side: int  # 1 for buy, -1 for sell
    
@dataclass
class BarData:
    """OHLCV bar data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_volume: Optional[int] = None
    sell_volume: Optional[int] = None
    tick_count: Optional[int] = None
    vwap: Optional[float] = None
    
@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: int
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]  # [[price, size], ...]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
        
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0
        
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
        
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
        
@dataclass
class MicrostructureMetrics:
    """Market microstructure measurements"""
    timestamp: int
    tick_imbalance: float
    order_flow_ratio: float
    bid_ask_spread: float
    effective_spread: float
    kyle_lambda: Optional[float] = None
    vpin: Optional[float] = None
    amihud_ratio: Optional[float] = None