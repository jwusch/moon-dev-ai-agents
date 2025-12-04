"""
📏 Effective Spread Calculator - Trading Cost Analysis
Measures the true cost of trading by analyzing price impact and bid-ask spreads
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class EffectiveSpreadIndicator(BaseIndicator):
    """
    Effective Spread measures the true cost of trading
    
    Effective Spread = 2 * |Trade Price - Midpoint| 
    Where midpoint is the theoretical fair value
    
    Key metrics:
    - Quoted Spread: Bid-Ask spread from order book
    - Effective Spread: Actual trading cost including price impact
    - Realized Spread: Spread captured by market makers
    - Price Impact: Permanent price movement from trade
    
    Lower effective spread indicates:
    - Better liquidity
    - Lower trading costs
    - More competitive market making
    - Easier execution
    
    Higher effective spread indicates:
    - Poor liquidity
    - Higher trading costs
    - Wide bid-ask spreads
    - Difficult execution environment
    """
    
    def __init__(self,
                 estimation_window: int = 500,
                 midpoint_method: str = 'volume_weighted',  # 'simple', 'volume_weighted', 'trade_weighted'
                 min_trades: int = 10,
                 outlier_threshold: float = 0.05):  # 5% outlier filter
        """
        Initialize Effective Spread indicator
        
        Args:
            estimation_window: Number of trades for spread estimation
            midpoint_method: Method for calculating midpoint reference price
            min_trades: Minimum trades required for calculation
            outlier_threshold: Filter trades beyond this % from midpoint
        """
        super().__init__(
            name="EffectiveSpread",
            timeframe=TimeFrame.TICK,
            lookback_periods=estimation_window,
            params={
                'estimation_window': estimation_window,
                'midpoint_method': midpoint_method,
                'min_trades': min_trades,
                'outlier_threshold': outlier_threshold
            }
        )
        
        self.estimation_window = estimation_window
        self.midpoint_method = midpoint_method
        self.min_trades = min_trades
        self.outlier_threshold = outlier_threshold
        
        # Storage for spread analysis
        self.trade_window: Deque[Dict] = deque(maxlen=estimation_window)
        self.spread_history: Deque[float] = deque(maxlen=100)
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Effective Spread from trade data
        
        Args:
            data: Trade data (tick data preferred, OHLCV as fallback)
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with spread analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Convert OHLCV to synthetic trade data
            trades = self._convert_ohlcv_to_trades(data)
        else:
            # Process tick data directly
            trades = self._convert_ticks_to_trades(data)
            
        if len(trades) < self.min_trades:
            return self._empty_result(symbol)
            
        # Store recent trades
        for trade in trades[-min(len(trades), 50):]:  # Last 50 trades
            self.trade_window.append(trade)
            
        if len(self.trade_window) < self.min_trades:
            return self._empty_result(symbol)
            
        # Calculate effective spread metrics
        effective_spread = self._calculate_effective_spread()
        
        if effective_spread is None:
            return self._empty_result(symbol)
            
        # Store in history
        self.spread_history.append(effective_spread)
        
        # Calculate additional metrics
        quoted_spread = self._calculate_quoted_spread()
        price_impact = self._calculate_price_impact()
        realized_spread = self._calculate_realized_spread()
        spread_volatility = self._calculate_spread_volatility()
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            effective_spread, quoted_spread, price_impact, spread_volatility
        )
        
        # Create metadata
        metadata = self._create_metadata(
            effective_spread, quoted_spread, price_impact, 
            realized_spread, spread_volatility
        )
        
        # Get latest timestamp
        timestamp = trades[-1]['timestamp'] if trades else int(datetime.now().timestamp() * 1000)
        
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
    
    def _convert_ohlcv_to_trades(self, data: pd.DataFrame) -> List[Dict]:
        """Convert OHLCV bars to synthetic trade data"""
        
        trades = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            # Generate synthetic trades within each bar
            # Use OHLC prices as trade points
            bar_volume = row['volume']
            
            # Distribute volume across OHLC trades
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            
            # Simple model: 4 trades per bar at OHLC prices
            bar_trades = [
                {'price': o, 'volume': bar_volume * 0.3, 'side': 1 if c > o else -1},
                {'price': h, 'volume': bar_volume * 0.2, 'side': 1},
                {'price': l, 'volume': bar_volume * 0.2, 'side': -1},
                {'price': c, 'volume': bar_volume * 0.3, 'side': 1 if c > o else -1}
            ]
            
            for j, trade in enumerate(bar_trades):
                trades.append({
                    'timestamp': int(datetime.now().timestamp() * 1000) + i * 60000 + j * 15000,
                    'price': trade['price'],
                    'volume': trade['volume'],
                    'side': trade['side'],
                    'trade_id': i * 4 + j
                })
                
        return trades
    
    def _convert_ticks_to_trades(self, ticks: List[TickData]) -> List[Dict]:
        """Convert tick data to trade format"""
        
        trades = []
        
        for i, tick in enumerate(ticks):
            trades.append({
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'side': tick.side,
                'trade_id': i
            })
            
        return trades
    
    def _calculate_effective_spread(self) -> Optional[float]:
        """Calculate effective spread using midpoint reference"""
        
        if len(self.trade_window) < self.min_trades:
            return None
            
        trades = list(self.trade_window)
        
        # Calculate midpoint reference price
        midpoint = self._calculate_midpoint(trades)
        
        if midpoint is None:
            return None
            
        # Calculate effective spreads for each trade
        effective_spreads = []
        
        for trade in trades:
            # Effective spread = 2 * |Trade Price - Midpoint|
            spread = 2 * abs(trade['price'] - midpoint)
            
            # Filter outliers
            relative_spread = spread / midpoint if midpoint > 0 else 0
            if relative_spread <= self.outlier_threshold:
                effective_spreads.append(spread)
                
        if not effective_spreads:
            return None
            
        # Return volume-weighted average effective spread
        return np.mean(effective_spreads)
    
    def _calculate_midpoint(self, trades: List[Dict]) -> Optional[float]:
        """Calculate midpoint reference price"""
        
        if not trades:
            return None
            
        if self.midpoint_method == 'simple':
            # Simple average of trade prices
            return np.mean([t['price'] for t in trades])
            
        elif self.midpoint_method == 'volume_weighted':
            # Volume-weighted average price (VWAP)
            total_volume = sum(t['volume'] for t in trades)
            if total_volume == 0:
                return np.mean([t['price'] for t in trades])
                
            vwap = sum(t['price'] * t['volume'] for t in trades) / total_volume
            return vwap
            
        elif self.midpoint_method == 'trade_weighted':
            # Weight recent trades more heavily
            weights = np.exp(np.linspace(-2, 0, len(trades)))  # Exponential decay
            weighted_prices = [t['price'] * w for t, w in zip(trades, weights)]
            return sum(weighted_prices) / sum(weights)
            
        else:
            return np.mean([t['price'] for t in trades])
    
    def _calculate_quoted_spread(self) -> float:
        """Estimate quoted spread from trade data"""
        
        if len(self.trade_window) < 5:
            return 0.0
            
        trades = list(self.trade_window)
        
        # Estimate bid-ask spread from buy/sell trade clustering
        buy_trades = [t for t in trades if t['side'] == 1]
        sell_trades = [t for t in trades if t['side'] == -1]
        
        if not buy_trades or not sell_trades:
            return 0.0
            
        # Estimate bid as average sell price, ask as average buy price
        estimated_bid = np.mean([t['price'] for t in sell_trades])
        estimated_ask = np.mean([t['price'] for t in buy_trades])
        
        quoted_spread = estimated_ask - estimated_bid
        return max(0, quoted_spread)  # Ensure non-negative
    
    def _calculate_price_impact(self) -> float:
        """Calculate average price impact of trades"""
        
        if len(self.trade_window) < 10:
            return 0.0
            
        trades = list(self.trade_window)
        impacts = []
        
        # Measure price change following each trade
        for i in range(len(trades) - 5):
            current_trade = trades[i]
            
            # Look at price 5 trades later
            future_price = trades[i + 5]['price']
            price_change = future_price - current_trade['price']
            
            # Price impact considers trade direction
            if current_trade['side'] == 1:  # Buy trade
                impact = price_change  # Positive impact = price moved up
            else:  # Sell trade
                impact = -price_change  # Positive impact = price moved down
                
            # Normalize by trade price
            if current_trade['price'] > 0:
                relative_impact = impact / current_trade['price']
                impacts.append(relative_impact)
                
        return np.mean(impacts) if impacts else 0.0
    
    def _calculate_realized_spread(self) -> float:
        """Calculate realized spread (spread captured by market makers)"""
        
        effective_spread = self._calculate_effective_spread()
        price_impact = self._calculate_price_impact()
        
        if effective_spread is None:
            return 0.0
            
        # Realized spread = Effective spread - Price impact
        return max(0, effective_spread - abs(price_impact))
    
    def _calculate_spread_volatility(self) -> float:
        """Calculate volatility of spread measurements"""
        
        if len(self.spread_history) < 5:
            return 0.0
            
        spreads = list(self.spread_history)
        return np.std(spreads)
    
    def _generate_signal(self,
                        effective_spread: float,
                        quoted_spread: float,
                        price_impact: float,
                        spread_volatility: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from spread analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        # Convert spreads to basis points for comparison
        trades = list(self.trade_window) if self.trade_window else []
        if not trades:
            return signal, confidence, 50.0
            
        avg_price = np.mean([t['price'] for t in trades])
        if avg_price <= 0:
            return signal, confidence, 50.0
            
        effective_spread_bps = (effective_spread / avg_price) * 10000
        
        # Low spreads = good for trading
        if effective_spread_bps < 10:  # Less than 1 basis point
            # Very low spreads - good environment
            # Check for directional flow
            recent_trades = trades[-10:]
            net_flow = sum(t['side'] * t['volume'] for t in recent_trades)
            
            if net_flow > 0:
                signal = SignalType.BUY
                confidence = 60
            elif net_flow < 0:
                signal = SignalType.SELL
                confidence = 60
            else:
                signal = SignalType.HOLD
                confidence = 30
                
        elif effective_spread_bps > 50:  # Greater than 5 basis points
            # High spreads - poor trading environment
            signal = SignalType.HOLD
            confidence = 0
            
        # Adjust confidence based on spread stability
        if spread_volatility < effective_spread * 0.2:  # Low volatility
            confidence *= 1.2
        elif spread_volatility > effective_spread * 0.5:  # High volatility
            confidence *= 0.7
            
        # Price impact adjustment
        avg_impact_bps = abs(price_impact) * 10000
        if avg_impact_bps < 5:  # Low impact
            confidence *= 1.1
        elif avg_impact_bps > 20:  # High impact
            confidence *= 0.8
            
        confidence = min(confidence, 75)
        
        # Value represents liquidity quality (inverse of spread)
        value = max(0, min(100, 100 - effective_spread_bps))
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        effective_spread: float,
                        quoted_spread: float,
                        price_impact: float,
                        realized_spread: float,
                        spread_volatility: float) -> Dict:
        """Create detailed metadata"""
        
        # Get average price for basis point calculations
        trades = list(self.trade_window) if self.trade_window else []
        avg_price = np.mean([t['price'] for t in trades]) if trades else 100.0
        
        metadata = {
            'effective_spread': effective_spread,
            'quoted_spread': quoted_spread,
            'price_impact': price_impact,
            'realized_spread': realized_spread,
            'spread_volatility': spread_volatility,
            'trade_count': len(self.trade_window),
            'spread_history_length': len(self.spread_history)
        }
        
        # Convert to basis points for easier interpretation
        metadata.update({
            'effective_spread_bps': (effective_spread / avg_price) * 10000,
            'quoted_spread_bps': (quoted_spread / avg_price) * 10000,
            'price_impact_bps': abs(price_impact) * 10000
        })
        
        # Spread statistics
        if self.spread_history:
            spreads = list(self.spread_history)
            metadata.update({
                'avg_effective_spread': np.mean(spreads),
                'min_effective_spread': np.min(spreads),
                'max_effective_spread': np.max(spreads),
                'spread_percentile_90': np.percentile(spreads, 90),
                'spread_percentile_10': np.percentile(spreads, 10)
            })
            
            # Spread trend
            if len(spreads) >= 10:
                recent_avg = np.mean(spreads[-10:])
                earlier_avg = np.mean(spreads[:-10]) if len(spreads) > 10 else recent_avg
                
                if recent_avg > earlier_avg * 1.1:
                    trend = 'widening'
                elif recent_avg < earlier_avg * 0.9:
                    trend = 'tightening'
                else:
                    trend = 'stable'
                    
                metadata['spread_trend'] = trend
        
        # Trading cost assessment
        effective_spread_bps = metadata['effective_spread_bps']
        
        if effective_spread_bps < 5:
            cost_tier = "very_low"
        elif effective_spread_bps < 15:
            cost_tier = "low"
        elif effective_spread_bps < 30:
            cost_tier = "moderate"
        elif effective_spread_bps < 60:
            cost_tier = "high"
        else:
            cost_tier = "very_high"
            
        metadata['trading_cost_tier'] = cost_tier
        
        # Market making profitability (for market makers)
        if realized_spread > 0:
            mm_profit_bps = (realized_spread / avg_price) * 10000
            metadata['mm_profit_bps'] = mm_profit_bps
            
            if mm_profit_bps > 5:
                mm_attractiveness = "attractive"
            elif mm_profit_bps > 2:
                mm_attractiveness = "moderate"
            else:
                mm_attractiveness = "low"
                
            metadata['mm_attractiveness'] = mm_attractiveness
        
        # Trade direction analysis
        if trades:
            buy_count = sum(1 for t in trades if t['side'] == 1)
            sell_count = len(trades) - buy_count
            
            metadata.update({
                'buy_trade_ratio': buy_count / len(trades),
                'trade_balance': 'buy_heavy' if buy_count > sell_count * 1.2 
                               else 'sell_heavy' if sell_count > buy_count * 1.2 
                               else 'balanced'
            })
            
            # Volume analysis
            total_volume = sum(t['volume'] for t in trades)
            avg_trade_size = total_volume / len(trades)
            
            metadata.update({
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size
            })
        
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,  # Neutral spread
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for effective spread calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= self.min_trades
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 5
        return False


def demonstrate_effective_spread():
    """Demonstration of Effective Spread indicator"""
    
    print("📏 Effective Spread Calculator Demonstration\n")
    
    # Generate synthetic market data with varying liquidity conditions
    np.random.seed(42)
    
    print("Generating synthetic trading data with spread regimes...\n")
    
    # Create synthetic tick data with different spread environments
    base_price = 100.0
    ticks = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Phase 1: Tight spreads (high liquidity)
    print("Phase 1: Tight spreads (institutional market)...")
    for i in range(200):
        # Tight bid-ask spread
        if i % 2 == 0:
            # Buy trade slightly above midpoint
            price = base_price * (1 + np.random.normal(0.0001, 0.00005))
            side = 1
        else:
            # Sell trade slightly below midpoint
            price = base_price * (1 - np.random.normal(0.0001, 0.00005))
            side = -1
            
        volume = np.random.randint(100, 500)
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
        
        # Gradual price drift
        base_price *= (1 + np.random.normal(0, 0.0001))
    
    # Phase 2: Moderate spreads
    print("Phase 2: Moderate spreads (retail market)...")
    for i in range(200, 400):
        # Moderate bid-ask spread
        if i % 2 == 0:
            price = base_price * (1 + np.random.normal(0.0005, 0.0002))
            side = 1
        else:
            price = base_price * (1 - np.random.normal(0.0005, 0.0002))
            side = -1
            
        volume = np.random.randint(50, 300)
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
        
        base_price *= (1 + np.random.normal(0, 0.0002))
    
    # Phase 3: Wide spreads (stress conditions)
    print("Phase 3: Wide spreads (market stress)...")
    for i in range(400, 600):
        # Wide bid-ask spread
        if i % 2 == 0:
            price = base_price * (1 + np.random.normal(0.002, 0.001))
            side = 1
        else:
            price = base_price * (1 - np.random.normal(0.002, 0.001))
            side = -1
            
        volume = np.random.randint(20, 150)
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=price,
            volume=volume,
            side=side
        ))
        
        base_price *= (1 + np.random.normal(0, 0.0005))
    
    # Create Effective Spread indicator
    spread_indicator = EffectiveSpreadIndicator(
        estimation_window=100,
        midpoint_method='volume_weighted'
    )
    
    # Process data incrementally
    print("Analyzing spread evolution...\n")
    
    results = []
    for i in range(50, len(ticks), 50):
        batch = ticks[:i]
        result = spread_indicator.calculate(batch, "TEST")
        results.append(result)
        
        phase = "Tight Spreads" if i < 250 else "Moderate Spreads" if i < 450 else "Wide Spreads"
        
        print(f"After {i} trades ({phase}):")
        print(f"  Effective Spread: {result.metadata['effective_spread']:.4f} ({result.metadata['effective_spread_bps']:.1f} bps)")
        print(f"  Quoted Spread: {result.metadata['quoted_spread']:.4f} ({result.metadata['quoted_spread_bps']:.1f} bps)")
        print(f"  Price Impact: {result.metadata['price_impact_bps']:.2f} bps")
        print(f"  Trading Cost Tier: {result.metadata['trading_cost_tier']}")
        print(f"  Signal: {result.signal.value}")
        print(f"  Confidence: {result.confidence:.1f}%")
        print()
    
    # Final comprehensive analysis
    if results:
        final_result = results[-1]
        print("=" * 60)
        print("FINAL EFFECTIVE SPREAD ANALYSIS:")
        print("=" * 60)
        
        print(f"Effective Spread: {final_result.metadata['effective_spread']:.4f}")
        print(f"Effective Spread (bps): {final_result.metadata['effective_spread_bps']:.1f}")
        print(f"Quoted Spread (bps): {final_result.metadata['quoted_spread_bps']:.1f}")
        print(f"Price Impact (bps): {final_result.metadata['price_impact_bps']:.2f}")
        print(f"Trading Cost Tier: {final_result.metadata['trading_cost_tier']}")
        
        if 'spread_trend' in final_result.metadata:
            print(f"Spread Trend: {final_result.metadata['spread_trend']}")
            
        if 'avg_effective_spread' in final_result.metadata:
            print(f"\nSpread Statistics:")
            print(f"  Average Spread: {final_result.metadata['avg_effective_spread']:.4f}")
            print(f"  Spread Range: {final_result.metadata['min_effective_spread']:.4f} - {final_result.metadata['max_effective_spread']:.4f}")
            print(f"  Spread Volatility: {final_result.metadata['spread_volatility']:.4f}")
            
        print(f"\nTrading Analysis:")
        print(f"  Trade Balance: {final_result.metadata.get('trade_balance', 'unknown')}")
        print(f"  Buy Trade Ratio: {final_result.metadata.get('buy_trade_ratio', 0):.1%}")
        
        if 'mm_attractiveness' in final_result.metadata:
            print(f"  Market Making Profit: {final_result.metadata.get('mm_profit_bps', 0):.1f} bps")
            print(f"  MM Attractiveness: {final_result.metadata['mm_attractiveness']}")
        
        print(f"\nTrading Recommendation: {final_result.signal.value}")
        print(f"Confidence: {final_result.confidence:.1f}%")
    
    print("\n💡 Effective Spread Interpretation:")
    print("- Lower spreads = Better liquidity, lower trading costs")
    print("- Spreads < 10 bps = Excellent liquidity")
    print("- Spreads > 50 bps = Poor liquidity, high costs")
    print("- Effective spread > Quoted spread indicates price impact")
    print("- Used by traders for execution cost analysis")
    print("- Critical for algorithmic trading and market making")


if __name__ == "__main__":
    demonstrate_effective_spread()