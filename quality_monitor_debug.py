"""
🎯 Quality Monitor with Debug Info
Shows all symbols and their current state

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

class DebugQualityMonitor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['VXX', 'SQQQ', 'AMD', 'NVDA', 'TSLA', 'VIXY', 'SPY', 'QQQ']
        self.improver = QuietImprover()
        
    def get_symbol_data(self, symbol):
        """Get current data for symbol"""
        try:
            df = self.improver.prepare_data(symbol, period="5d")
            if df is None or len(df) < 50:
                return {'symbol': symbol, 'error': 'Insufficient data'}
                
            # Get last bar (even if market closed)
            idx = len(df) - 1
            
            # Get values
            price = df['Close'].iloc[idx]
            sma = df['SMA20'].iloc[idx] if not pd.isna(df['SMA20'].iloc[idx]) else 0
            distance = df['Distance%'].iloc[idx] if not pd.isna(df['Distance%'].iloc[idx]) else 0
            rsi = df['RSI'].iloc[idx] if not pd.isna(df['RSI'].iloc[idx]) else 50
            volume_ratio = df['Volume_Ratio'].iloc[idx] if 'Volume_Ratio' in df and not pd.isna(df['Volume_Ratio'].iloc[idx]) else 1.0
            timestamp = df.index[idx]
            
            # Check entry conditions
            signal = None
            score = 0
            reason = "No signal"
            
            if pd.isna(distance) or pd.isna(rsi):
                reason = "Missing indicators"
            elif distance < -1.0 and rsi < 40:
                signal = 'LONG'
                score, details = self.improver.evaluate_entry_quality(df, idx)
                reason = f"LONG signal (score: {score})"
            elif distance > 1.0 and rsi > 60:
                signal = 'SHORT'
                score, details = self.improver.evaluate_entry_quality(df, idx)
                reason = f"SHORT signal (score: {score})"
            elif distance < -1.0:
                reason = f"Distance OK ({distance:.1f}%) but RSI too high ({rsi:.0f})"
            elif rsi < 40:
                reason = f"RSI OK ({rsi:.0f}) but distance not enough ({distance:.1f}%)"
            else:
                reason = f"No conditions met (Dist: {distance:.1f}%, RSI: {rsi:.0f})"
            
            return {
                'symbol': symbol,
                'price': price,
                'sma': sma,
                'distance': distance,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'signal': signal,
                'score': score,
                'reason': reason,
                'timestamp': timestamp,
                'error': None
            }
            
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
    
    def display(self):
        """Display current state of all symbols"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(colored("╔" + "═" * 80 + "╗", 'cyan'))
        print(colored("║", 'cyan') + " " * 25 + colored("VXX MEAN REVERSION 15", 'white', attrs=['bold']) + " " * 26 + colored("║", 'cyan'))
        print(colored("║", 'cyan') + " " * 23 + "Debug Quality Monitor" + " " * 28 + colored("║", 'cyan'))
        print(colored("╚" + "═" * 80 + "╝", 'cyan'))
        
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Collect data
        all_data = []
        for symbol in self.symbols:
            data = self.get_symbol_data(symbol)
            all_data.append(data)
        
        # Separate signals from non-signals
        signals = [d for d in all_data if not d.get('error') and d.get('signal')]
        no_signals = [d for d in all_data if not d.get('error') and not d.get('signal')]
        errors = [d for d in all_data if d.get('error')]
        
        # Display active signals first
        if signals:
            print(f"\n{colored('📊 ACTIVE SIGNALS', 'green', attrs=['bold'])}:")
            print("─" * 80)
            
            signals.sort(key=lambda x: x['score'], reverse=True)
            
            for sig in signals:
                if sig['signal'] == 'LONG':
                    signal_str = colored("LONG ↑", 'green', attrs=['bold'])
                else:
                    signal_str = colored("SHORT ↓", 'red', attrs=['bold'])
                
                # Score visualization
                score_pct = sig['score'] / 100
                bar_width = 20
                filled = int(score_pct * bar_width)
                
                if sig['score'] >= 70:
                    bar = colored('█' * filled, 'green') + '░' * (bar_width - filled)
                    quality = colored("STRONG", 'green')
                elif sig['score'] >= 50:
                    bar = colored('█' * filled, 'yellow') + '░' * (bar_width - filled)
                    quality = colored("MODERATE", 'yellow')
                else:
                    bar = colored('█' * filled, 'red') + '░' * (bar_width - filled)
                    quality = colored("WEAK", 'red')
                
                print(f"{sig['symbol']:<6} {signal_str} @ ${sig['price']:>8.2f} | "
                      f"Score: {sig['score']:>3}/100 [{bar}] {quality}")
                print(f"       Distance: {sig['distance']:>6.1f}% | RSI: {sig['rsi']:>5.1f} | "
                      f"Volume: {sig['volume_ratio']:>4.1f}x | {sig['timestamp'].strftime('%H:%M')}")
                print()
        
        # Display symbols without signals
        print(f"\n{colored('📈 MARKET STATUS', 'yellow', attrs=['bold'])} (No Signals):")
        print("─" * 80)
        print(f"{'Symbol':<8} {'Price':>10} {'Distance':>10} {'RSI':>8} {'Status'}")
        print("─" * 80)
        
        for data in no_signals:
            # Color code distance and RSI
            dist_str = f"{data['distance']:>9.1f}%"
            if data['distance'] < -2:
                dist_str = colored(dist_str, 'green')
            elif data['distance'] > 2:
                dist_str = colored(dist_str, 'red')
                
            rsi_str = f"{data['rsi']:>7.1f}"
            if data['rsi'] < 30:
                rsi_str = colored(rsi_str, 'green')
            elif data['rsi'] > 70:
                rsi_str = colored(rsi_str, 'red')
            
            print(f"{data['symbol']:<8} ${data['price']:>9.2f} {dist_str} {rsi_str}  {data['reason']}")
        
        # Show any errors
        if errors:
            print(f"\n{colored('❌ ERRORS', 'red')}:")
            for err in errors:
                print(f"  {err['symbol']}: {err['error']}")
        
        # Entry conditions reminder
        print("\n" + "─" * 80)
        print("Entry Conditions:")
        print(f"  • {colored('LONG', 'green')}: Distance < -1.0% AND RSI < 40")
        print(f"  • {colored('SHORT', 'red')}: Distance > 1.0% AND RSI > 60")
        print("\nQuality Score = Distance(20) + RSI(20) + Volume(15) + Momentum(15) + HTF(10) + Micro(10) + Time(10)")
        
        print(f"\nRefreshing in 10 seconds... Press Ctrl+C to exit")
    
    def run(self):
        """Run continuous monitoring"""
        try:
            while True:
                self.display()
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    monitor = DebugQualityMonitor()
    monitor.run()