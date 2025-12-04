"""
⚖️ Kyle's Lambda - Price Impact Measurement
Measures the price impact of order flow - how much price moves per unit of trading
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Deque
from collections import deque
from datetime import datetime
import warnings

from ...base.indicator import BaseIndicator, IndicatorResult
from ...base.types import TickData, TimeFrame, SignalType


class KylesLambdaIndicator(BaseIndicator):
    """
    Kyle's Lambda measures the price impact of trading
    
    Lambda = Change in Price / Signed Order Flow
    
    Higher lambda indicates:
    - Lower liquidity
    - Higher price impact of trades
    - More difficulty trading large sizes
    - Potential informed trading
    
    Lower lambda indicates:
    - Higher liquidity  
    - Lower price impact
    - Easier to trade large sizes
    - Market is more efficient
    """
    
    def __init__(self,
                 estimation_window: int = 100,
                 price_lag: int = 1,
                 rolling_window: int = 50,
                 volume_normalization: bool = True):
        """
        Initialize Kyle's Lambda indicator
        
        Args:
            estimation_window: Window for lambda estimation
            price_lag: Price change measurement lag
            rolling_window: Rolling window for lambda calculation
            volume_normalization: Normalize by volume
        """
        super().__init__(
            name="KylesLambda",
            timeframe=TimeFrame.TICK,
            lookback_periods=estimation_window,
            params={
                'estimation_window': estimation_window,
                'price_lag': price_lag,
                'rolling_window': rolling_window,
                'volume_normalization': volume_normalization
            }
        )
        
        self.estimation_window = estimation_window
        self.price_lag = price_lag
        self.rolling_window = rolling_window
        self.volume_normalization = volume_normalization
        
        # Storage for price impact analysis
        self.tick_window: Deque[TickData] = deque(maxlen=estimation_window)
        self.lambda_history: Deque[float] = deque(maxlen=rolling_window)
        
    def calculate(self, 
                  data: Union[List[TickData], pd.DataFrame], 
                  symbol: str) -> IndicatorResult:
        """
        Calculate Kyle's Lambda
        
        Args:
            data: Tick data or OHLCV bars
            symbol: Symbol being analyzed
            
        Returns:
            IndicatorResult with lambda analysis
        """
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Generate synthetic ticks from bars
            from ...utils.synthetic_ticks import SyntheticTickGenerator
            generator = SyntheticTickGenerator(method='adaptive')
            
            ticks = []
            for _, bar in data.iterrows():
                bar_ticks = generator.generate_ticks(bar, n_ticks=15)
                ticks.extend(bar_ticks)
        else:
            ticks = data
            
        if not ticks:
            return self._empty_result(symbol)
            
        # Process all ticks
        for tick in ticks:
            self.tick_window.append(tick)
            
        if len(self.tick_window) < max(20, self.price_lag + 5):
            return self._empty_result(symbol)
            
        # Calculate Kyle's Lambda
        lambda_estimate = self._calculate_lambda()
        
        if lambda_estimate is not None:
            self.lambda_history.append(lambda_estimate)
            
        # Calculate additional metrics
        price_impact_volatility = self._calculate_price_impact_volatility()
        liquidity_score = self._calculate_liquidity_score()
        trade_difficulty = self._calculate_trade_difficulty()
        
        # Generate signals
        signal, confidence, value = self._generate_signal(
            lambda_estimate, price_impact_volatility, liquidity_score
        )
        
        # Create metadata
        metadata = self._create_metadata(
            lambda_estimate, price_impact_volatility, 
            liquidity_score, trade_difficulty
        )
        
        # Get latest timestamp
        timestamp = ticks[-1].timestamp if ticks else int(datetime.now().timestamp() * 1000)
        
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
    
    def _calculate_lambda(self) -> Optional[float]:
        """Calculate Kyle's Lambda using regression"""
        
        if len(self.tick_window) < 20:
            return None
            
        # Convert to lists for analysis
        ticks = list(self.tick_window)
        
        # Calculate price changes and signed order flows
        price_changes = []
        signed_flows = []
        
        for i in range(self.price_lag, len(ticks)):
            # Price change
            price_change = ticks[i].price - ticks[i - self.price_lag].price
            price_changes.append(price_change)
            
            # Signed order flow (aggregate over lag period)
            flow = 0
            for j in range(i - self.price_lag + 1, i + 1):
                if self.volume_normalization:
                    flow += ticks[j].side * ticks[j].volume
                else:
                    flow += ticks[j].side
                    
            signed_flows.append(flow)
            
        if len(price_changes) < 10:
            return None
            
        # Convert to numpy arrays
        price_changes = np.array(price_changes)
        signed_flows = np.array(signed_flows)
        
        # Remove outliers (beyond 3 standard deviations)
        price_std = np.std(price_changes)
        flow_std = np.std(signed_flows)
        
        if price_std > 0 and flow_std > 0:
            price_mask = np.abs(price_changes) < 3 * price_std
            flow_mask = np.abs(signed_flows) < 3 * flow_std
            mask = price_mask & flow_mask
            
            price_changes = price_changes[mask]
            signed_flows = signed_flows[mask]
        
        if len(price_changes) < 5:
            return None
            
        # Handle zero flows
        nonzero_mask = signed_flows != 0
        if np.sum(nonzero_mask) < 3:
            return 0.0
            
        price_changes = price_changes[nonzero_mask]
        signed_flows = signed_flows[nonzero_mask]
        
        try:
            # Linear regression: price_change = lambda * signed_flow + error
            lambda_estimate = np.cov(price_changes, signed_flows)[0, 1] / np.var(signed_flows)
            
            # Sanity check
            if np.isfinite(lambda_estimate):
                return lambda_estimate
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_price_impact_volatility(self) -> float:
        """Calculate volatility of price impact"""
        
        if len(self.lambda_history) < 5:
            return 0.0
            
        lambdas = list(self.lambda_history)
        return np.std(lambdas)
    
    def _calculate_liquidity_score(self) -> float:
        """Calculate overall liquidity score (0-100)"""
        
        if len(self.lambda_history) < 3:
            return 50.0  # Neutral
            
        # Lower lambda = higher liquidity
        avg_lambda = np.mean(list(self.lambda_history))
        lambda_volatility = self._calculate_price_impact_volatility()
        
        # Normalize to 0-100 scale
        # This is approximate - would need market-specific calibration
        base_score = 100 / (1 + abs(avg_lambda) * 1000)  # Scale factor
        volatility_penalty = lambda_volatility * 100
        
        liquidity_score = max(0, min(100, base_score - volatility_penalty))
        
        return liquidity_score
    
    def _calculate_trade_difficulty(self) -> str:
        """Classify trading difficulty based on lambda"""
        
        if len(self.lambda_history) < 3:
            return "unknown"
            
        avg_lambda = np.mean(list(self.lambda_history))
        
        if abs(avg_lambda) < 0.0001:
            return "very_easy"
        elif abs(avg_lambda) < 0.001:
            return "easy"  
        elif abs(avg_lambda) < 0.01:
            return "moderate"
        elif abs(avg_lambda) < 0.1:
            return "difficult"
        else:
            return "very_difficult"
    
    def _generate_signal(self,
                        lambda_estimate: Optional[float],
                        price_impact_volatility: float,
                        liquidity_score: float) -> Tuple[SignalType, float, float]:
        """Generate trading signal from lambda analysis"""
        
        signal = SignalType.HOLD
        confidence = 0.0
        
        if lambda_estimate is None or len(self.lambda_history) < 3:
            return signal, confidence, 50.0
            
        # High liquidity (low lambda) = good for trading
        if liquidity_score > 70 and price_impact_volatility < 0.01:
            # Good liquidity environment
            # Check for directional flow
            if len(self.tick_window) >= 10:
                recent_ticks = list(self.tick_window)[-10:]
                net_flow = sum(t.side * t.volume for t in recent_ticks)
                
                if net_flow > 0:
                    signal = SignalType.BUY
                    confidence = min(liquidity_score * 0.7, 60)
                elif net_flow < 0:
                    signal = SignalType.SELL
                    confidence = min(liquidity_score * 0.7, 60)
                    
        elif liquidity_score < 30:
            # Poor liquidity - avoid trading
            signal = SignalType.HOLD
            confidence = 0
            
        # Adjust for lambda stability
        if len(self.lambda_history) >= 5:
            recent_lambdas = list(self.lambda_history)[-5:]
            lambda_trend = np.polyfit(range(5), recent_lambdas, 1)[0]
            
            if lambda_trend > 0:  # Increasing lambda = decreasing liquidity
                confidence *= 0.8
            elif lambda_trend < 0:  # Decreasing lambda = improving liquidity
                confidence *= 1.2
                
        confidence = min(confidence, 85)
        
        # Value represents liquidity quality
        value = liquidity_score
        
        return signal, confidence, value
    
    def _create_metadata(self,
                        lambda_estimate: Optional[float],
                        price_impact_volatility: float,
                        liquidity_score: float,
                        trade_difficulty: str) -> Dict:
        """Create detailed metadata"""
        
        metadata = {
            'lambda': lambda_estimate,
            'price_impact_volatility': price_impact_volatility,
            'liquidity_score': liquidity_score,
            'trade_difficulty': trade_difficulty,
            'lambda_history_length': len(self.lambda_history),
            'tick_window_size': len(self.tick_window)
        }
        
        if self.lambda_history:
            lambdas = list(self.lambda_history)
            metadata.update({
                'avg_lambda': np.mean(lambdas),
                'min_lambda': np.min(lambdas),
                'max_lambda': np.max(lambdas),
                'lambda_percentile_90': np.percentile(lambdas, 90),
                'lambda_percentile_10': np.percentile(lambdas, 10)
            })
            
            # Lambda trend
            if len(lambdas) >= 5:
                trend = np.polyfit(range(len(lambdas)), lambdas, 1)[0]
                metadata['lambda_trend'] = 'increasing' if trend > 0 else 'decreasing'
                metadata['lambda_trend_slope'] = trend
        
        # Recent flow analysis
        if len(self.tick_window) >= 20:
            recent_ticks = list(self.tick_window)[-20:]
            total_volume = sum(t.volume for t in recent_ticks)
            net_flow = sum(t.side * t.volume for t in recent_ticks)
            
            metadata.update({
                'recent_total_volume': total_volume,
                'recent_net_flow': net_flow,
                'recent_flow_imbalance': net_flow / total_volume if total_volume > 0 else 0,
                'recent_avg_trade_size': total_volume / len(recent_ticks)
            })
        
        return metadata
    
    def _empty_result(self, symbol: str) -> IndicatorResult:
        """Return empty result when insufficient data"""
        
        return IndicatorResult(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            indicator_name=self.name,
            value=50.0,  # Neutral liquidity
            signal=SignalType.HOLD,
            confidence=0.0,
            timeframe=self.timeframe,
            metadata={'error': 'Insufficient data for lambda calculation'},
            calculation_time_ms=0
        )
    
    def validate_data(self, data: Union[List[TickData], pd.DataFrame]) -> bool:
        """Validate input data"""
        
        if isinstance(data, list):
            return len(data) >= 20
        elif isinstance(data, pd.DataFrame):
            required = ['open', 'high', 'low', 'close', 'volume']
            return all(col in data.columns for col in required) and len(data) >= 5
        return False


def demonstrate_kyles_lambda():
    """Demonstration of Kyle's Lambda indicator"""
    
    print("⚖️ Kyle's Lambda (Price Impact) Demonstration\n")
    
    # Generate synthetic tick data with varying liquidity
    np.random.seed(42)
    ticks = []
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Phase 1: High liquidity (low price impact)
    print("Phase 1: High liquidity market...")
    base_price = 100.0
    for i in range(80):
        # Small price movements despite trading
        price_change = np.random.normal(0, 0.001)  # Very small impact
        base_price += price_change
        
        volume = np.random.randint(100, 500)
        side = np.random.choice([1, -1])
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=base_price,
            volume=volume,
            side=side
        ))
    
    # Phase 2: Deteriorating liquidity (higher price impact)
    print("Phase 2: Liquidity deterioration...")
    for i in range(80, 160):
        # Larger price movements for same trading
        if i > 80:
            # Add price impact based on previous trade
            prev_trade = ticks[-1]
            impact = prev_trade.side * prev_trade.volume * 0.00002  # Higher impact
            base_price += impact
        
        base_price += np.random.normal(0, 0.002)
        
        volume = np.random.randint(200, 800)
        side = np.random.choice([1, -1])
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=base_price,
            volume=volume,
            side=side
        ))
    
    # Phase 3: Very poor liquidity (high price impact)
    print("Phase 3: Poor liquidity (high impact)...")
    for i in range(160, 200):
        # Large price impact
        if i > 160:
            prev_trade = ticks[-1]
            impact = prev_trade.side * prev_trade.volume * 0.0001  # Very high impact
            base_price += impact
            
        base_price += np.random.normal(0, 0.005)
        
        volume = np.random.randint(500, 1500)  # Larger sizes
        side = np.random.choice([1, -1])
        
        ticks.append(TickData(
            timestamp=timestamp + i * 1000,
            price=base_price,
            volume=volume,
            side=side
        ))
    
    # Create Kyle's Lambda indicator
    lambda_indicator = KylesLambdaIndicator(
        estimation_window=60,
        price_lag=1,
        rolling_window=20
    )
    
    # Process ticks incrementally
    print("\nAnalyzing price impact evolution...\n")
    
    results = []
    for i in range(30, len(ticks), 20):
        batch = ticks[:i]
        result = lambda_indicator.calculate(batch, "TEST")
        results.append(result)
        
        if result.metadata.get('lambda') is not None:
            phase = "High Liquidity" if i < 100 else "Medium Liquidity" if i < 180 else "Poor Liquidity"
            
            print(f"After {i} ticks ({phase}):")
            print(f"  Kyle's Lambda: {result.metadata['lambda']:.6f}")
            print(f"  Liquidity Score: {result.metadata['liquidity_score']:.1f}/100")
            print(f"  Trade Difficulty: {result.metadata['trade_difficulty']}")
            print(f"  Price Impact Volatility: {result.metadata['price_impact_volatility']:.6f}")
            print(f"  Signal: {result.signal.value}")
            print(f"  Confidence: {result.confidence:.1f}%")
            print()
    
    # Final analysis
    if results:
        final_result = results[-1]
        print("\n" + "="*60)
        print("FINAL KYLE'S LAMBDA ANALYSIS:")
        print("="*60)
        
        print(f"Current Lambda: {final_result.metadata['lambda']:.6f}")
        print(f"Liquidity Score: {final_result.metadata['liquidity_score']:.1f}/100")
        print(f"Trade Difficulty: {final_result.metadata['trade_difficulty']}")
        print(f"Lambda Trend: {final_result.metadata.get('lambda_trend', 'unknown')}")
        
        if 'avg_lambda' in final_result.metadata:
            print(f"Average Lambda: {final_result.metadata['avg_lambda']:.6f}")
            print(f"Lambda Range: {final_result.metadata['min_lambda']:.6f} to {final_result.metadata['max_lambda']:.6f}")
        
        print(f"\nRecent Market Flow:")
        print(f"  Net Flow: {final_result.metadata.get('recent_net_flow', 0):+,.0f}")
        print(f"  Flow Imbalance: {final_result.metadata.get('recent_flow_imbalance', 0):+.2%}")
        print(f"  Avg Trade Size: {final_result.metadata.get('recent_avg_trade_size', 0):,.0f}")
        
        print(f"\nTrading Recommendation: {final_result.signal.value}")
        print(f"Confidence: {final_result.confidence:.1f}%")
    
    print("\n💡 Kyle's Lambda Interpretation:")
    print("- Lower lambda = Higher liquidity, easier to trade")
    print("- Higher lambda = Lower liquidity, higher price impact")
    print("- Lambda > 0.01 indicates difficult trading conditions")
    print("- Rising lambda trend suggests deteriorating liquidity")
    print("- Used by institutions to optimize execution strategies")


if __name__ == "__main__":
    demonstrate_kyles_lambda()