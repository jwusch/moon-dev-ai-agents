"""
💹 Bid-Ask Dynamics Analyzer
Analyzes bid-ask spread patterns and microstructure for liquidity insights
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
from dataclasses import dataclass

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TimeFrame, SignalType


@dataclass
class QuoteData:
    """Level 1 quote data"""
    timestamp: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    

@dataclass
class TradeData:
    """Trade execution data"""
    timestamp: int
    price: float
    volume: int
    side: int  # 1 for buy, -1 for sell
    

class BidAskDynamicsAnalyzer(BaseIndicator):
    """
    Analyzes bid-ask spread dynamics to detect:
    1. Liquidity conditions
    2. Informed trading
    3. Market maker behavior
    4. Hidden liquidity
    
    Key metrics:
    - Spread: Absolute and relative spreads
    - Depth imbalance: Bid vs ask size
    - Quote intensity: Rate of quote changes
    - Trade location: Where trades occur in spread
    """
    
    def __init__(self,
                 window_size: int = 100,
                 spread_threshold: float = 0.001,  # 0.1% threshold
                 imbalance_threshold: float = 0.3,
                 quote_intensity_window: int = 20):
        """
        Initialize Bid-Ask Dynamics Analyzer
        
        Args:
            window_size: Number of quotes to analyze
            spread_threshold: Threshold for wide spread detection
            imbalance_threshold: Threshold for depth imbalance signals
            quote_intensity_window: Window for quote change rate
        """
        super().__init__(
            name="BidAskDynamics",
            timeframe=TimeFrame.TICK,
            lookback_periods=window_size,
            params={
                'window_size': window_size,
                'spread_threshold': spread_threshold,
                'imbalance_threshold': imbalance_threshold,
                'quote_intensity_window': quote_intensity_window
            }
        )
        
        self.window_size = window_size
        self.spread_threshold = spread_threshold
        self.imbalance_threshold = imbalance_threshold
        self.quote_intensity_window = quote_intensity_window
        
        # Quote storage
        self.quote_window: Deque[QuoteData] = deque(maxlen=window_size)
        self.trade_window: Deque[TradeData] = deque(maxlen=window_size)
        
        # Tracking metrics
        self.spread_history = deque(maxlen=window_size)
        self.imbalance_history = deque(maxlen=window_size)
        self.quote_changes = deque(maxlen=quote_intensity_window)
        
    def calculate(self, 
                  data: Union[Dict, pd.DataFrame, List], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate bid-ask dynamics
        
        Args:
            data: Dictionary with 'quotes' and 'trades', or DataFrame
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with spread analysis
        """
        # Handle different data formats
        quotes, trades = self._parse_input_data(data)
        
        if not quotes:
            return self._empty_result(symbol)
            
        # Process quotes and trades
        for quote in quotes:
            self._process_quote(quote)
            
        for trade in trades:
            self._process_trade(trade)
            
        if len(self.quote_window) < 10:
            return self._empty_result(symbol)
            
        # Calculate metrics
        spread_metrics = self._calculate_spread_metrics()
        depth_metrics = self._calculate_depth_metrics()
        flow_metrics = self._calculate_flow_toxicity()
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            spread_metrics, depth_metrics, flow_metrics
        )
        
        # Create metadata
        metadata = {
            **spread_metrics,
            **depth_metrics,
            **flow_metrics,
            'quote_count': len(self.quote_window),
            'trade_count': len(self.trade_window)
        }
        
        # Get latest timestamp
        timestamp = quotes[-1].timestamp if quotes else int(datetime.now().timestamp() * 1000)
        
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
    
    def _parse_input_data(self, data) -> Tuple[List[QuoteData], List[TradeData]]:
        """Parse input data into quotes and trades"""
        
        quotes = []
        trades = []
        
        if isinstance(data, dict):
            # Direct quote/trade data
            if 'quotes' in data:
                for q in data['quotes']:
                    quotes.append(QuoteData(
                        timestamp=q['timestamp'],
                        bid=q['bid'],
                        ask=q['ask'],
                        bid_size=q.get('bid_size', 100),
                        ask_size=q.get('ask_size', 100)
                    ))
                    
            if 'trades' in data:
                for t in data['trades']:
                    trades.append(TradeData(
                        timestamp=t['timestamp'],
                        price=t['price'],
                        volume=t.get('volume', 100),
                        side=t.get('side', 0)
                    ))
                    
        elif isinstance(data, pd.DataFrame):
            # Generate synthetic quotes from OHLCV
            for _, row in data.iterrows():
                mid = (row['high'] + row['low']) / 2
                spread = (row['high'] - row['low']) * 0.1  # Approximate spread
                
                quotes.append(QuoteData(
                    timestamp=int(row.name.timestamp() * 1000),
                    bid=mid - spread/2,
                    ask=mid + spread/2,
                    bid_size=int(row['volume'] * 0.4),
                    ask_size=int(row['volume'] * 0.6)
                ))
                
                # Generate synthetic trades
                # Buy at ask, sell at bid
                if row['close'] > row['open']:
                    trades.append(TradeData(
                        timestamp=int(row.name.timestamp() * 1000),
                        price=mid + spread/2,
                        volume=int(row['volume'] * 0.6),
                        side=1
                    ))
                else:
                    trades.append(TradeData(
                        timestamp=int(row.name.timestamp() * 1000),
                        price=mid - spread/2,
                        volume=int(row['volume'] * 0.6),
                        side=-1
                    ))
                    
        return quotes, trades
    
    def _process_quote(self, quote: QuoteData):
        """Process a single quote"""
        
        # Check if quote changed
        if self.quote_window and (
            quote.bid != self.quote_window[-1].bid or 
            quote.ask != self.quote_window[-1].ask
        ):
            self.quote_changes.append(1)
        else:
            self.quote_changes.append(0)
            
        self.quote_window.append(quote)
        
        # Calculate spread
        spread = quote.ask - quote.bid
        relative_spread = spread / ((quote.ask + quote.bid) / 2)
        self.spread_history.append(relative_spread)
        
        # Calculate depth imbalance
        total_depth = quote.bid_size + quote.ask_size
        if total_depth > 0:
            imbalance = (quote.bid_size - quote.ask_size) / total_depth
        else:
            imbalance = 0
        self.imbalance_history.append(imbalance)
    
    def _process_trade(self, trade: TradeData):
        """Process a single trade"""
        self.trade_window.append(trade)
    
    def _calculate_spread_metrics(self) -> Dict[str, float]:
        """Calculate spread-based metrics"""
        
        if not self.spread_history:
            return {
                'avg_spread': 0,
                'spread_volatility': 0,
                'wide_spread_ratio': 0,
                'spread_trend': 0
            }
            
        spreads = list(self.spread_history)
        
        # Average spread
        avg_spread = np.mean(spreads)
        
        # Spread volatility
        spread_volatility = np.std(spreads) if len(spreads) > 1 else 0
        
        # Wide spread occurrences
        wide_spreads = sum(1 for s in spreads if s > self.spread_threshold)
        wide_spread_ratio = wide_spreads / len(spreads)
        
        # Spread trend
        if len(spreads) >= 5:
            recent_spreads = spreads[-10:]
            spread_trend = np.polyfit(range(len(recent_spreads)), recent_spreads, 1)[0]
        else:
            spread_trend = 0
            
        return {
            'avg_spread': avg_spread,
            'spread_volatility': spread_volatility,
            'wide_spread_ratio': wide_spread_ratio,
            'spread_trend': spread_trend
        }
    
    def _calculate_depth_metrics(self) -> Dict[str, float]:
        """Calculate order book depth metrics"""
        
        if not self.quote_window:
            return {
                'avg_depth_imbalance': 0,
                'imbalance_persistence': 0,
                'quote_intensity': 0,
                'depth_asymmetry': 0
            }
            
        # Average depth imbalance
        imbalances = list(self.imbalance_history)
        avg_imbalance = np.mean(imbalances) if imbalances else 0
        
        # Imbalance persistence (how long it stays on one side)
        if len(imbalances) >= 5:
            persistence_score = self._calculate_persistence(imbalances)
        else:
            persistence_score = 0
            
        # Quote intensity (changes per period)
        quote_intensity = np.mean(list(self.quote_changes)) if self.quote_changes else 0
        
        # Depth asymmetry (variation in imbalance)
        depth_asymmetry = np.std(imbalances) if len(imbalances) > 1 else 0
        
        return {
            'avg_depth_imbalance': avg_imbalance,
            'imbalance_persistence': persistence_score,
            'quote_intensity': quote_intensity,
            'depth_asymmetry': depth_asymmetry
        }
    
    def _calculate_flow_toxicity(self) -> Dict[str, float]:
        """Calculate order flow toxicity metrics"""
        
        if not self.trade_window or not self.quote_window:
            return {
                'price_impact': 0,
                'adverse_selection': 0,
                'trade_location_score': 0,
                'informed_trading_probability': 0
            }
            
        # Price impact: How much trades move the market
        price_impacts = []
        quote_list = list(self.quote_window)
        
        for i, trade in enumerate(self.trade_window):
            # Find corresponding quote
            quote_idx = self._find_nearest_quote(trade.timestamp, quote_list)
            if quote_idx >= 0 and quote_idx < len(quote_list) - 1:
                pre_quote = quote_list[quote_idx]
                post_quote = quote_list[quote_idx + 1]
                
                mid_pre = (pre_quote.bid + pre_quote.ask) / 2
                mid_post = (post_quote.bid + post_quote.ask) / 2
                
                impact = abs(mid_post - mid_pre) / mid_pre
                price_impacts.append(impact)
                
        avg_price_impact = np.mean(price_impacts) if price_impacts else 0
        
        # Trade location in spread
        trade_locations = []
        for trade in list(self.trade_window)[-20:]:  # Recent trades
            # Find nearest quote
            quote_idx = self._find_nearest_quote(trade.timestamp, quote_list)
            if 0 <= quote_idx < len(quote_list):
                quote = quote_list[quote_idx]
                
                # Where in spread did trade occur?
                if quote.ask > quote.bid:
                    location = (trade.price - quote.bid) / (quote.ask - quote.bid)
                    trade_locations.append(location)
                    
        # Average trade location (0=bid, 1=ask, >1=above ask, <0=below bid)
        trade_location_score = np.mean(trade_locations) if trade_locations else 0.5
        
        # Adverse selection (trades at bad prices)
        adverse_trades = sum(1 for loc in trade_locations if loc > 0.8 or loc < 0.2)
        adverse_selection = adverse_trades / len(trade_locations) if trade_locations else 0
        
        # Informed trading probability (PIN approximation)
        buy_trades = sum(1 for t in self.trade_window if t.side == 1)
        sell_trades = sum(1 for t in self.trade_window if t.side == -1)
        total_trades = buy_trades + sell_trades
        
        if total_trades > 0:
            trade_imbalance = abs(buy_trades - sell_trades) / total_trades
            informed_trading_probability = min(trade_imbalance * avg_price_impact * 100, 1.0)
        else:
            informed_trading_probability = 0
            
        return {
            'price_impact': avg_price_impact,
            'adverse_selection': adverse_selection,
            'trade_location_score': trade_location_score,
            'informed_trading_probability': informed_trading_probability
        }
    
    def _find_nearest_quote(self, timestamp: int, quotes: List[QuoteData]) -> int:
        """Find quote index nearest to timestamp"""
        
        if not quotes:
            return -1
            
        # Binary search would be more efficient for large lists
        min_diff = float('inf')
        nearest_idx = -1
        
        for i, quote in enumerate(quotes):
            diff = abs(quote.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i
                
        return nearest_idx
    
    def _calculate_persistence(self, values: List[float]) -> float:
        """Calculate how persistent a signal is"""
        
        if len(values) < 2:
            return 0
            
        # Count consecutive same-sign periods
        streaks = []
        current_streak = 1
        current_sign = np.sign(values[0])
        
        for v in values[1:]:
            if np.sign(v) == current_sign:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
                current_sign = np.sign(v)
                
        streaks.append(current_streak)
        
        # Persistence score based on average streak length
        avg_streak = np.mean(streaks)
        max_possible = len(values)
        
        return avg_streak / max_possible
    
    def _generate_signal(self,
                        spread_metrics: Dict[str, float],
                        depth_metrics: Dict[str, float],
                        flow_metrics: Dict[str, float]) -> Tuple[SignalType, float, float]:
        """Generate trading signal from bid-ask dynamics"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Combine metrics for signal generation
        
        # 1. Liquidity-based signals
        if spread_metrics['avg_spread'] < self.spread_threshold * 0.5:
            # Very tight spread - good liquidity
            liquidity_score = 1.0
        elif spread_metrics['avg_spread'] > self.spread_threshold * 2:
            # Very wide spread - poor liquidity
            liquidity_score = -1.0
        else:
            liquidity_score = 0.0
            
        # 2. Order book imbalance signals
        imbalance = depth_metrics['avg_depth_imbalance']
        if abs(imbalance) > self.imbalance_threshold:
            if imbalance > 0:  # More bid size
                imbalance_signal = SignalType.BUY
                imbalance_confidence = min(abs(imbalance) * 100, 80)
            else:  # More ask size
                imbalance_signal = SignalType.SELL
                imbalance_confidence = min(abs(imbalance) * 100, 80)
                
            # Adjust for persistence
            imbalance_confidence *= depth_metrics['imbalance_persistence']
        else:
            imbalance_signal = SignalType.HOLD
            imbalance_confidence = 0
            
        # 3. Flow toxicity signals
        if flow_metrics['informed_trading_probability'] > 0.6:
            # High probability of informed trading
            if flow_metrics['trade_location_score'] > 0.7:
                # Trades hitting asks - bullish
                flow_signal = SignalType.BUY
                flow_confidence = flow_metrics['informed_trading_probability'] * 80
            elif flow_metrics['trade_location_score'] < 0.3:
                # Trades hitting bids - bearish
                flow_signal = SignalType.SELL
                flow_confidence = flow_metrics['informed_trading_probability'] * 80
            else:
                flow_signal = SignalType.HOLD
                flow_confidence = 0
        else:
            flow_signal = SignalType.HOLD
            flow_confidence = 0
            
        # 4. Combine signals
        if imbalance_confidence > flow_confidence:
            signal = imbalance_signal
            confidence = imbalance_confidence
        else:
            signal = flow_signal
            confidence = flow_confidence
            
        # Adjust for liquidity
        if liquidity_score < 0:
            confidence *= 0.5  # Reduce confidence in poor liquidity
            
        # Adjust for quote intensity
        if depth_metrics['quote_intensity'] > 0.5:
            # High quote changes - unstable market
            confidence *= 0.8
            
        # Value represents market quality (0-100)
        market_quality = (
            (1 - min(spread_metrics['avg_spread'] / self.spread_threshold, 1)) * 40 +
            (1 - depth_metrics['depth_asymmetry']) * 30 +
            (1 - flow_metrics['adverse_selection']) * 30
        )
        
        return signal, confidence, market_quality
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=0.0,
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data) -> bool:
        """Validate input data"""
        
        if isinstance(data, dict):
            return 'quotes' in data or 'trades' in data
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required)
        return False


def demonstrate_bid_ask_dynamics():
    """Demonstration of Bid-Ask Dynamics Analyzer"""
    
    print("💹 Bid-Ask Dynamics Demonstration\n")
    
    # Generate synthetic market data
    np.random.seed(42)
    
    quotes = []
    trades = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Scenario 1: Normal market with balanced order book
    print("Phase 1: Normal market conditions...")
    base_price = 100.0
    
    for i in range(50):
        spread = 0.05 + np.random.normal(0, 0.01)  # ~5 cent spread
        mid = base_price + np.random.normal(0, 0.02)
        
        bid = mid - spread/2
        ask = mid + spread/2
        
        # Balanced book
        bid_size = np.random.randint(800, 1200)
        ask_size = np.random.randint(800, 1200)
        
        quotes.append({
            'timestamp': timestamp + i * 500,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size
        })
        
        # Random trades
        if np.random.random() < 0.5:
            # Buy
            trades.append({
                'timestamp': timestamp + i * 500 + 250,
                'price': ask,
                'volume': np.random.randint(50, 200),
                'side': 1
            })
        else:
            # Sell
            trades.append({
                'timestamp': timestamp + i * 500 + 250,
                'price': bid,
                'volume': np.random.randint(50, 200),
                'side': -1
            })
    
    # Scenario 2: Informed buying - bid/ask imbalance
    print("Phase 2: Informed buying pressure...")
    for i in range(50, 100):
        spread = 0.08 + np.random.normal(0, 0.02)  # Wider spread
        mid = base_price + 0.5 + i * 0.01  # Price rising
        
        bid = mid - spread/2
        ask = mid + spread/2
        
        # Bid size increasing
        bid_size = np.random.randint(1500, 2500)
        ask_size = np.random.randint(500, 800)
        
        quotes.append({
            'timestamp': timestamp + i * 500,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size
        })
        
        # More buys hitting ask
        if np.random.random() < 0.75:
            trades.append({
                'timestamp': timestamp + i * 500 + 250,
                'price': ask + np.random.uniform(0, 0.02),  # Above ask
                'volume': np.random.randint(100, 500),
                'side': 1
            })
        else:
            trades.append({
                'timestamp': timestamp + i * 500 + 250,
                'price': bid,
                'volume': np.random.randint(20, 100),
                'side': -1
            })
    
    # Scenario 3: Liquidity crisis - wide spreads
    print("Phase 3: Liquidity withdrawal...")
    for i in range(100, 150):
        spread = 0.20 + np.random.normal(0, 0.05)  # Very wide spread
        mid = base_price + 1.0 - (i-100) * 0.02  # Price falling
        
        bid = mid - spread/2
        ask = mid + spread/2
        
        # Thin book
        bid_size = np.random.randint(100, 300)
        ask_size = np.random.randint(100, 300)
        
        quotes.append({
            'timestamp': timestamp + i * 500,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size
        })
        
        # Panic selling
        if np.random.random() < 0.8:
            trades.append({
                'timestamp': timestamp + i * 500 + 250,
                'price': bid - np.random.uniform(0, 0.05),  # Below bid
                'volume': np.random.randint(200, 1000),
                'side': -1
            })
    
    # Create analyzer
    analyzer = BidAskDynamicsAnalyzer(
        window_size=40,
        spread_threshold=0.001,
        imbalance_threshold=0.3
    )
    
    # Analyze incrementally
    print("\nAnalyzing bid-ask dynamics...\n")
    
    data_points = [
        (45, "End of normal market"),
        (95, "After informed buying"),
        (145, "After liquidity crisis")
    ]
    
    for idx, description in data_points:
        data = {
            'quotes': quotes[:idx+1],
            'trades': trades[:idx+1]
        }
        
        result = analyzer.calculate(data, "TEST")
        
        print(f"{description}:")
        print(f"  Market Quality: {result.value:.1f}/100")
        print(f"  Signal: {result.signal.value}")
        print(f"  Confidence: {result.confidence:.1f}%")
        print(f"  Avg Spread: {result.metadata['avg_spread']:.4f}")
        print(f"  Depth Imbalance: {result.metadata['avg_depth_imbalance']:+.3f}")
        print(f"  Quote Intensity: {result.metadata['quote_intensity']:.2%}")
        print(f"  Informed Trading Prob: {result.metadata['informed_trading_probability']:.2%}")
        print()
    
    print("\n💡 Key Insights:")
    print("- Tight spreads and balanced books indicate healthy markets")
    print("- Persistent depth imbalance suggests directional pressure")
    print("- Wide spreads signal uncertainty or low liquidity")
    print("- Trade location reveals aggressor behavior")
    print("- High quote intensity indicates unstable conditions")


if __name__ == "__main__":
    demonstrate_bid_ask_dynamics()