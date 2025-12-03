"""
🌙 Camarilla Live Trading Example
Shows how to use the Camarilla strategy for live trading signals

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from camarilla_strategy import CamarillaLevels
import time


class CamarillaLiveTrader:
    """Live trading implementation of Camarilla strategy"""
    
    def __init__(self, symbol='BTCUSDT', exchange='BINANCE'):
        self.symbol = symbol
        self.exchange = exchange
        self.current_position = None
        self.levels = None
        self.last_calculation_date = None
        
        # Strategy parameters
        self.range_threshold = 0.001  # 0.1% proximity to levels
        self.stop_loss_buffer = 0.001  # 0.1% beyond S4/R4
        self.position_size_pct = 0.01  # Risk 1% per trade
        
    def update_levels(self, high, low, close):
        """Update Camarilla levels using previous day's data"""
        self.levels = CamarillaLevels.calculate(high, low, close)
        self.last_calculation_date = datetime.now().date()
        
        print(f"\n📊 Camarilla Levels Updated for {self.symbol}:")
        print(f"R4 (Breakout): ${self.levels['r4']:.2f}")
        print(f"R3 (Resistance): ${self.levels['r3']:.2f}")
        print(f"R2: ${self.levels['r2']:.2f}")
        print(f"R1: ${self.levels['r1']:.2f}")
        print(f"Pivot: ${self.levels['pivot']:.2f}")
        print(f"S1: ${self.levels['s1']:.2f}")
        print(f"S2: ${self.levels['s2']:.2f}")
        print(f"S3 (Support): ${self.levels['s3']:.2f}")
        print(f"S4 (Breakout): ${self.levels['s4']:.2f}")
        
    def check_signals(self, current_price):
        """Check for trading signals based on current price"""
        if not self.levels:
            return None
            
        signal = {
            'action': 'HOLD',
            'reason': None,
            'entry': current_price,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward': None
        }
        
        # Skip if we have a position
        if self.current_position:
            return self._check_exit_signals(current_price)
        
        # Check range-bound opportunities
        range_signal = self._check_range_signals(current_price)
        if range_signal['action'] != 'HOLD':
            return range_signal
            
        # Check breakout opportunities
        breakout_signal = self._check_breakout_signals(current_price)
        return breakout_signal
        
    def _check_range_signals(self, price):
        """Check for S3/R3 fade opportunities"""
        signal = {'action': 'HOLD', 'reason': None}
        
        # Long setup near S3
        if self._near_level(price, self.levels['s3']):
            if price > self.levels['s4']:  # Must be above S4
                signal['action'] = 'BUY'
                signal['reason'] = 'Price near S3 support - fade trade'
                signal['entry'] = price
                signal['stop_loss'] = self.levels['s4'] * (1 - self.stop_loss_buffer)
                signal['take_profit'] = self.levels['r1']  # Target R1
                signal['risk_reward'] = self._calculate_risk_reward(
                    price, signal['stop_loss'], signal['take_profit']
                )
                
        # Short setup near R3
        elif self._near_level(price, self.levels['r3']):
            if price < self.levels['r4']:  # Must be below R4
                signal['action'] = 'SELL'
                signal['reason'] = 'Price near R3 resistance - fade trade'
                signal['entry'] = price
                signal['stop_loss'] = self.levels['r4'] * (1 + self.stop_loss_buffer)
                signal['take_profit'] = self.levels['s1']  # Target S1
                signal['risk_reward'] = self._calculate_risk_reward(
                    price, signal['stop_loss'], signal['take_profit']
                )
                
        return signal
        
    def _check_breakout_signals(self, price):
        """Check for S4/R4 breakout opportunities"""
        signal = {'action': 'HOLD', 'reason': None}
        
        # Bullish breakout above R4
        if price > self.levels['r4'] * (1 + self.range_threshold):
            signal['action'] = 'BUY'
            signal['reason'] = 'Breakout above R4 - trend day potential'
            signal['entry'] = price
            signal['stop_loss'] = self.levels['r3']
            risk = price - signal['stop_loss']
            signal['take_profit'] = price + (risk * 2)  # 2:1 RR
            signal['risk_reward'] = 2.0
            
        # Bearish breakout below S4
        elif price < self.levels['s4'] * (1 - self.range_threshold):
            signal['action'] = 'SELL'
            signal['reason'] = 'Breakout below S4 - trend day potential'
            signal['entry'] = price
            signal['stop_loss'] = self.levels['s3']
            risk = signal['stop_loss'] - price
            signal['take_profit'] = price - (risk * 2)  # 2:1 RR
            signal['risk_reward'] = 2.0
            
        return signal
        
    def _check_exit_signals(self, price):
        """Check if current position should be closed"""
        if not self.current_position:
            return {'action': 'HOLD', 'reason': None}
            
        signal = {'action': 'HOLD', 'reason': None}
        pos = self.current_position
        
        # Check stop loss
        if pos['direction'] == 'LONG' and price <= pos['stop_loss']:
            signal['action'] = 'CLOSE'
            signal['reason'] = 'Stop loss hit'
            
        elif pos['direction'] == 'SHORT' and price >= pos['stop_loss']:
            signal['action'] = 'CLOSE'
            signal['reason'] = 'Stop loss hit'
            
        # Check take profit
        elif pos['direction'] == 'LONG' and price >= pos['take_profit']:
            signal['action'] = 'CLOSE'
            signal['reason'] = 'Take profit reached'
            
        elif pos['direction'] == 'SHORT' and price <= pos['take_profit']:
            signal['action'] = 'CLOSE'
            signal['reason'] = 'Take profit reached'
            
        # Range trades: Close at opposite level
        elif pos['type'] == 'range':
            if pos['direction'] == 'LONG' and price >= self.levels['r3']:
                signal['action'] = 'CLOSE'
                signal['reason'] = 'Reached R3 resistance'
                
            elif pos['direction'] == 'SHORT' and price <= self.levels['s3']:
                signal['action'] = 'CLOSE'
                signal['reason'] = 'Reached S3 support'
                
        return signal
        
    def _near_level(self, price, level):
        """Check if price is within threshold of level"""
        return abs(price - level) / level <= self.range_threshold
        
    def _calculate_risk_reward(self, entry, stop, target):
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0
        
    def execute_signal(self, signal, capital=10000):
        """Execute trading signal (paper trading)"""
        if signal['action'] == 'BUY':
            risk_amount = capital * self.position_size_pct
            risk_per_share = signal['entry'] - signal['stop_loss']
            shares = risk_amount / risk_per_share
            
            self.current_position = {
                'direction': 'LONG',
                'type': 'range' if 's3' in signal['reason'].lower() else 'breakout',
                'entry': signal['entry'],
                'shares': shares,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_reward': signal['risk_reward']
            }
            
            print(f"\n🟢 LONG Position Opened:")
            print(f"   Entry: ${signal['entry']:.2f}")
            print(f"   Stop: ${signal['stop_loss']:.2f}")
            print(f"   Target: ${signal['take_profit']:.2f}")
            print(f"   R:R Ratio: {signal['risk_reward']:.1f}")
            
        elif signal['action'] == 'SELL':
            risk_amount = capital * self.position_size_pct
            risk_per_share = signal['stop_loss'] - signal['entry']
            shares = risk_amount / risk_per_share
            
            self.current_position = {
                'direction': 'SHORT',
                'type': 'range' if 'r3' in signal['reason'].lower() else 'breakout',
                'entry': signal['entry'],
                'shares': shares,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_reward': signal['risk_reward']
            }
            
            print(f"\n🔴 SHORT Position Opened:")
            print(f"   Entry: ${signal['entry']:.2f}")
            print(f"   Stop: ${signal['stop_loss']:.2f}")
            print(f"   Target: ${signal['take_profit']:.2f}")
            print(f"   R:R Ratio: {signal['risk_reward']:.1f}")
            
        elif signal['action'] == 'CLOSE' and self.current_position:
            print(f"\n⚪ Position Closed: {signal['reason']}")
            self.current_position = None


def main():
    """Example of live trading with Camarilla levels"""
    trader = CamarillaLiveTrader('BTCUSDT')
    
    # Example: Update levels with yesterday's data
    # In production, fetch from API
    trader.update_levels(
        high=96800,
        low=94200,
        close=95500
    )
    
    # Simulate live price feed
    print("\n🚀 Starting Camarilla Live Trading Simulation...")
    print("=" * 60)
    
    # Generate some price movements
    current_price = 95500
    
    for i in range(50):
        # Simulate price movement
        change = np.random.normal(0, 0.002)  # 0.2% volatility
        current_price *= (1 + change)
        
        # Check for signals
        signal = trader.check_signals(current_price)
        
        if signal and signal['action'] != 'HOLD':
            print(f"\n⚡ Signal at ${current_price:.2f}: {signal['action']}")
            print(f"   Reason: {signal['reason']}")
            
            # Execute signal (paper trading)
            trader.execute_signal(signal)
        
        # Small delay for simulation
        time.sleep(0.1)
    
    print("\n✅ Simulation complete!")


if __name__ == "__main__":
    main()