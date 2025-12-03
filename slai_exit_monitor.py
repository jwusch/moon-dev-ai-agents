#!/usr/bin/env python3
"""
SLAI Position Exit Monitor
Tracks price and provides real-time exit signals
"""

import yfinance as yf
import json
from datetime import datetime, timedelta
from termcolor import colored
import time

class SLAIExitMonitor:
    def __init__(self):
        # Load position config
        with open('slai_position_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.symbol = self.config['symbol']
        self.entry_price = self.config['entry_price']
        self.shares = self.config['shares']
        self.exit_criteria = self.config['exit_criteria']
        
        # Track highest price for trailing stop
        self.highest_price = self.entry_price
        
    def get_current_price(self):
        """Get current SLAI price"""
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period='1d', interval='1m')
        
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    
    def calculate_metrics(self, current_price):
        """Calculate position metrics"""
        if not current_price:
            return None
            
        position_value = current_price * self.shares
        entry_value = self.entry_price * self.shares
        pnl = position_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        
        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
            
        # Calculate trailing stop
        trailing_stop_price = self.highest_price * (1 - self.exit_criteria['trailing_stop'])
        
        return {
            'current_price': current_price,
            'position_value': position_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'highest_price': self.highest_price,
            'trailing_stop_price': trailing_stop_price
        }
    
    def check_exit_signals(self, metrics):
        """Check if any exit criteria are met"""
        if not metrics:
            return []
            
        signals = []
        current_price = metrics['current_price']
        
        # Check stop loss
        if current_price <= self.exit_criteria['stop_loss']:
            signals.append({
                'type': 'STOP_LOSS',
                'action': 'SELL_ALL',
                'reason': f'Price ${current_price:.2f} hit stop loss ${self.exit_criteria["stop_loss"]:.2f}',
                'priority': 'HIGH'
            })
        
        # Check trailing stop
        if current_price <= metrics['trailing_stop_price']:
            signals.append({
                'type': 'TRAILING_STOP',
                'action': 'SELL_ALL',
                'reason': f'Price ${current_price:.2f} hit trailing stop ${metrics["trailing_stop_price"]:.2f}',
                'priority': 'HIGH'
            })
        
        # Check profit targets
        if current_price >= self.exit_criteria['take_profit_3']:
            signals.append({
                'type': 'PROFIT_TARGET_3',
                'action': 'SELL_ALL',
                'reason': f'Price ${current_price:.2f} hit target 3 (+30%)',
                'priority': 'MEDIUM'
            })
        elif current_price >= self.exit_criteria['take_profit_2']:
            signals.append({
                'type': 'PROFIT_TARGET_2',
                'action': 'SELL_HALF',
                'reason': f'Price ${current_price:.2f} hit target 2 (+20%)',
                'priority': 'MEDIUM'
            })
        elif current_price >= self.exit_criteria['take_profit_1']:
            signals.append({
                'type': 'PROFIT_TARGET_1',
                'action': 'SELL_THIRD',
                'reason': f'Price ${current_price:.2f} hit target 1 (+10%)',
                'priority': 'LOW'
            })
        
        # Check time stop
        entry_date = datetime.strptime(self.config['entry_date'], '%Y-%m-%d')
        days_held = (datetime.now() - entry_date).days
        if days_held >= self.exit_criteria['time_stop']:
            signals.append({
                'type': 'TIME_STOP',
                'action': 'REVIEW',
                'reason': f'Position held for {days_held} days (limit: {self.exit_criteria["time_stop"]})',
                'priority': 'MEDIUM'
            })
        
        return signals
    
    def display_status(self, metrics, signals):
        """Display current status"""
        print("\n" + "="*60)
        print(colored(f"SLAI EXIT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'cyan', attrs=['bold']))
        print("="*60)
        
        if metrics:
            # Position info
            print(f"\n📊 Position Status:")
            print(f"   Entry: ${self.entry_price:.2f} x {self.shares} shares")
            print(f"   Current: ${metrics['current_price']:.2f}")
            print(f"   Value: ${metrics['position_value']:.2f}")
            
            # P&L
            pnl_color = 'green' if metrics['pnl'] >= 0 else 'red'
            print(colored(f"\n💰 P&L: ${metrics['pnl']:.2f} ({metrics['pnl_pct']:+.1f}%)", pnl_color))
            
            # Exit levels
            print(f"\n🎯 Exit Levels:")
            print(f"   Stop Loss: ${self.exit_criteria['stop_loss']:.2f} (from ${metrics['current_price']:.2f}: {((self.exit_criteria['stop_loss']/metrics['current_price'])-1)*100:+.1f}%)")
            print(f"   Trailing Stop: ${metrics['trailing_stop_price']:.2f} (from ${metrics['current_price']:.2f}: {((metrics['trailing_stop_price']/metrics['current_price'])-1)*100:+.1f}%)")
            print(f"   Target 1 (+10%): ${self.exit_criteria['take_profit_1']:.2f}")
            print(f"   Target 2 (+20%): ${self.exit_criteria['take_profit_2']:.2f}")
            print(f"   Target 3 (+30%): ${self.exit_criteria['take_profit_3']:.2f}")
            
            # Signals
            if signals:
                print(colored(f"\n⚠️ EXIT SIGNALS:", 'yellow', attrs=['bold']))
                for signal in signals:
                    color = 'red' if signal['priority'] == 'HIGH' else 'yellow'
                    print(colored(f"   {signal['type']}: {signal['reason']}", color))
                    print(colored(f"   Action: {signal['action']}", color, attrs=['bold']))
            else:
                print(colored("\n✅ No exit signals - Hold position", 'green'))
        else:
            print(colored("❌ Unable to get current price", 'red'))
    
    def run_once(self):
        """Run monitor once"""
        metrics = self.calculate_metrics(self.get_current_price())
        signals = self.check_exit_signals(metrics)
        self.display_status(metrics, signals)
        
        # Return summary
        if metrics:
            return {
                'price': metrics['current_price'],
                'pnl_pct': metrics['pnl_pct'],
                'signals': [s['type'] for s in signals]
            }
        return None
    
    def run_continuous(self, interval_seconds=60):
        """Run monitor continuously"""
        print(colored("Starting SLAI Exit Monitor (Press Ctrl+C to stop)", 'cyan'))
        
        try:
            while True:
                self.run_once()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print(colored("\n\nExit monitor stopped", 'yellow'))

def main():
    """Run exit monitor"""
    monitor = SLAIExitMonitor()
    
    # Run once for immediate status
    result = monitor.run_once()
    
    # Ask if user wants continuous monitoring
    print(f"\nWould you like to run continuous monitoring?")
    response = input("Enter 'y' for continuous monitoring or any other key to exit: ")
    
    if response.lower() == 'y':
        monitor.run_continuous(interval_seconds=60)

if __name__ == "__main__":
    main()