#!/usr/bin/env python3
"""
AEGS 5-Minute Live Monitor
Real-time signal generation for intraday trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from termcolor import colored
import json
from aegs_5min_strategy import AEGS5MinStrategy

class AEGS5MinLiveMonitor:
    def __init__(self, symbols):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.strategy = AEGS5MinStrategy()
        self.last_signals = {}
        self.positions = {}
        
    def get_latest_data(self, symbol):
        """Get latest 5-minute data"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)  # Get 2 days for indicators
            
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='5m'
            )
            
            if df.empty:
                return None
                
            # Clean columns
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Add indicators
            df = self.strategy.add_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def check_current_signal(self, symbol):
        """Check current signal for a symbol"""
        df = self.get_latest_data(symbol)
        if df is None or len(df) < 50:
            return None
        
        # Generate signals
        df = self.strategy.generate_signals(df)
        
        # Get latest bar info
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal_info = {
            'symbol': symbol,
            'time': df.index[-1],
            'price': latest['close'],
            'signal': int(latest['signal']),
            'signal_strength': latest['signal_strength'],
            'signal_reason': latest['signal_reason'],
            'rsi': latest['rsi'],
            'bb_position': latest['bb_position'],
            'volume_ratio': latest['volume_ratio'],
            'distance_from_mean': latest['distance_from_mean']
        }
        
        # Check if in position
        if symbol in self.positions and self.positions[symbol]['in_position']:
            entry_price = self.positions[symbol]['entry_price']
            current_return = (latest['close'] - entry_price) / entry_price
            bars_held = self.positions[symbol]['bars_held'] + 1
            
            signal_info['in_position'] = True
            signal_info['entry_price'] = entry_price
            signal_info['current_return'] = current_return * 100
            signal_info['bars_held'] = bars_held
            signal_info['minutes_held'] = bars_held * 5
            
            # Check exit conditions
            exit_signal = False
            exit_reason = ''
            
            if current_return <= -self.strategy.params['stop_loss']:
                exit_signal = True
                exit_reason = 'STOP_LOSS'
            elif current_return >= self.strategy.params['take_profit']:
                exit_signal = True
                exit_reason = 'TAKE_PROFIT'
            elif bars_held >= self.strategy.params['max_holding_bars']:
                exit_signal = True
                exit_reason = 'TIME_EXIT'
            elif abs(latest['distance_from_mean']) < 0.005 and bars_held >= 3:
                exit_signal = True
                exit_reason = 'MEAN_REVERSION'
            
            signal_info['exit_signal'] = exit_signal
            signal_info['exit_reason'] = exit_reason
            
        else:
            signal_info['in_position'] = False
        
        return signal_info
    
    def display_signal(self, signal_info):
        """Display signal information"""
        symbol = signal_info['symbol']
        
        # Header
        print(f"\n{'='*60}")
        print(colored(f"📊 {symbol} - {signal_info['time'].strftime('%H:%M:%S')}", 'cyan', attrs=['bold']))
        print(f"Price: ${signal_info['price']:.2f}")
        
        # Position status
        if signal_info['in_position']:
            color = 'green' if signal_info['current_return'] > 0 else 'red'
            print(colored(f"\n📍 IN POSITION", 'yellow'))
            print(f"   Entry: ${signal_info['entry_price']:.2f}")
            print(colored(f"   Return: {signal_info['current_return']:+.2f}%", color))
            print(f"   Held: {signal_info['minutes_held']:.0f} minutes")
            
            if signal_info['exit_signal']:
                print(colored(f"\n⚠️ EXIT SIGNAL: {signal_info['exit_reason']}", 'red', attrs=['bold']))
                print("   Action: SELL ALL SHARES")
        
        # Entry signals
        elif signal_info['signal'] == 1:
            print(colored(f"\n🚀 BUY SIGNAL (Strength: {signal_info['signal_strength']:.0f})", 'green', attrs=['bold']))
            print(f"   Reason: {signal_info['signal_reason']}")
            print("   Action: BUY with 10-15% of capital")
        
        # Market conditions
        print(f"\n📈 Indicators:")
        print(f"   RSI: {signal_info['rsi']:.1f}")
        print(f"   BB Position: {signal_info['bb_position']:.2f}")
        print(f"   Volume Ratio: {signal_info['volume_ratio']:.2f}x")
        print(f"   Distance from Mean: {signal_info['distance_from_mean']*100:+.1f}%")
    
    def update_position(self, symbol, signal_info):
        """Update position tracking"""
        if signal_info['signal'] == 1 and symbol not in self.positions:
            # New position
            self.positions[symbol] = {
                'in_position': True,
                'entry_price': signal_info['price'],
                'entry_time': signal_info['time'],
                'bars_held': 0
            }
        elif signal_info.get('exit_signal') and symbol in self.positions:
            # Exit position
            del self.positions[symbol]
        elif symbol in self.positions:
            # Update bars held
            self.positions[symbol]['bars_held'] += 1
    
    def run_continuous(self, interval_seconds=60):
        """Run continuous monitoring"""
        print(colored("🚀 AEGS 5-MINUTE LIVE MONITOR", 'cyan', attrs=['bold']))
        print(f"Monitoring: {', '.join(self.symbols)}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check market hours (9:30 AM - 4:00 PM EST)
                now = datetime.now()
                market_open = now.replace(hour=9, minute=30, second=0)
                market_close = now.replace(hour=16, minute=0, second=0)
                
                if now < market_open or now > market_close:
                    print(f"\n⏸️ Market closed. Current time: {now.strftime('%H:%M:%S')}")
                    print(f"   Market opens at {market_open.strftime('%H:%M:%S')}")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                # Check each symbol
                for symbol in self.symbols:
                    signal_info = self.check_current_signal(symbol)
                    
                    if signal_info:
                        # Check if signal changed
                        last_signal = self.last_signals.get(symbol, {}).get('signal', 0)
                        current_signal = signal_info['signal']
                        
                        # Display if new signal or position update
                        if (current_signal != last_signal or 
                            signal_info.get('exit_signal') or
                            signal_info.get('in_position')):
                            
                            self.display_signal(signal_info)
                            
                            # Send alert if new buy/sell signal
                            if current_signal == 1 and last_signal == 0:
                                self.send_alert(symbol, "BUY", signal_info)
                            elif signal_info.get('exit_signal'):
                                self.send_alert(symbol, "SELL", signal_info)
                        
                        # Update position
                        self.update_position(symbol, signal_info)
                        
                        # Store last signal
                        self.last_signals[symbol] = signal_info
                
                # Show summary
                self.display_summary()
                
                # Wait for next check
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(colored("\n\n👋 Monitor stopped", 'yellow'))
    
    def display_summary(self):
        """Display position summary"""
        if self.positions:
            print(colored(f"\n📊 ACTIVE POSITIONS", 'yellow'))
            for symbol, pos in self.positions.items():
                print(f"   {symbol}: Held {pos['bars_held']*5} minutes")
    
    def send_alert(self, symbol, action, signal_info):
        """Send alert (placeholder for notification system)"""
        print(colored(f"\n🔔 ALERT: {action} {symbol} @ ${signal_info['price']:.2f}", 'yellow', attrs=['bold', 'blink']))
        
        # Here you could add:
        # - Email notifications
        # - SMS alerts
        # - Discord/Slack webhooks
        # - Sound alerts

def main():
    """Run live monitor"""
    import sys
    
    if len(sys.argv) < 2:
        # Default to monitoring current positions
        symbols = ['SLAI', 'TLRY']
        print(f"Monitoring default symbols: {', '.join(symbols)}")
    else:
        symbols = [s.upper() for s in sys.argv[1:]]
    
    monitor = AEGS5MinLiveMonitor(symbols)
    monitor.run_continuous(interval_seconds=60)  # Check every minute

if __name__ == "__main__":
    main()