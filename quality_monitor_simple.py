"""
🎯 Simple Quality Score Monitor
Minimalist real-time display for VXX Mean Reversion 15

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import sys
from termcolor import colored
from improved_entry_timing import ImprovedEntryTiming

# Suppress all output during data fetching
class QuietImprover(ImprovedEntryTiming):
    def prepare_data(self, symbol, period="5d"):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            result = super().prepare_data(symbol, period)
            return result
        finally:
            sys.stdout = original_stdout

class SimpleQualityMonitor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['VXX', 'SQQQ', 'AMD', 'NVDA', 'TSLA', 'VIXY']
        self.improver = QuietImprover()
        
    def get_signal_data(self, symbol):
        """Get current signal data for symbol"""
        try:
            df = self.improver.prepare_data(symbol, period="5d")
            if df is None or len(df) < 50:
                return None
                
            # Get last valid market hours bar
            for i in range(len(df)-1, max(0, len(df)-10), -1):
                if 9 <= df.index[i].hour < 16:
                    idx = i
                    break
            else:
                idx = len(df) - 1
            
            # Check for signal
            distance = df['Distance%'].iloc[idx]
            rsi = df['RSI'].iloc[idx]
            
            if pd.isna(distance) or pd.isna(rsi):
                return None
                
            # Entry conditions
            signal = None
            if distance < -1.0 and rsi < 40:
                signal = 'LONG'
            elif distance > 1.0 and rsi > 60:
                signal = 'SHORT'
                
            if signal:
                score, details = self.improver.evaluate_entry_quality(df, idx)
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'score': score,
                    'price': df['Close'].iloc[idx],
                    'distance': distance,
                    'rsi': rsi,
                    'time': df.index[idx]
                }
            
            return None
            
        except:
            return None
    
    def display(self):
        """Display current signals"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(colored("VXX MEAN REVERSION 15 - LIVE SIGNALS", 'cyan', attrs=['bold']))
        print(f"{datetime.now().strftime('%H:%M:%S')}\n")
        
        # Collect signals
        signals = []
        for symbol in self.symbols:
            data = self.get_signal_data(symbol)
            if data:
                signals.append(data)
        
        # Sort by score
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        if signals:
            # Display signals
            for sig in signals:
                # Score bar
                score_pct = sig['score'] / 100
                bar_width = 30
                filled = int(score_pct * bar_width)
                
                if sig['score'] >= 70:
                    bar = colored('█' * filled + '░' * (bar_width - filled), 'green')
                    quality = colored("STRONG", 'green', attrs=['bold'])
                elif sig['score'] >= 50:
                    bar = colored('█' * filled + '░' * (bar_width - filled), 'yellow')
                    quality = colored("MODERATE", 'yellow')
                else:
                    bar = colored('█' * filled + '░' * (bar_width - filled), 'red')
                    quality = colored("WEAK", 'red')
                
                # Signal direction
                if sig['signal'] == 'LONG':
                    direction = colored("↑ LONG", 'green', attrs=['bold'])
                else:
                    direction = colored("↓ SHORT", 'red', attrs=['bold'])
                
                print(f"{sig['symbol']:<6} {direction} @ ${sig['price']:.2f}")
                print(f"Score: {sig['score']}/100 [{bar}] {quality}")
                print(f"Distance: {sig['distance']:.1f}% | RSI: {sig['rsi']:.1f}")
                print("-" * 60)
        else:
            print("No signals currently active")
            print("\nWaiting for entry conditions:")
            print("• LONG: Distance < -1% AND RSI < 40")
            print("• SHORT: Distance > 1% AND RSI > 60")
        
        print(f"\nRefreshing every 5 seconds. Press Ctrl+C to exit.")
    
    def run(self):
        """Run continuous monitoring"""
        try:
            while True:
                self.display()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    monitor = SimpleQualityMonitor()
    monitor.run()