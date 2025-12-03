"""
🎯 Live Quality Score Display
Compact real-time entry quality scores

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import time
from termcolor import colored
from yfinance_cache_demo import YFinanceCache
from improved_entry_timing import ImprovedEntryTiming

class LiveQualityScore:
    def __init__(self, symbol='VXX'):
        self.symbol = symbol
        self.cache = YFinanceCache()
        self.improver = ImprovedEntryTiming()
        self.history = []
        
    def get_live_score(self):
        """Get current quality score"""
        try:
            # Get fresh data
            df = self.cache.get_data(self.symbol, period="5d", interval="15m", force_refresh=True)
            
            if len(df) < 50:
                return None, None, {}
                
            # Calculate all indicators
            df = self.improver.prepare_data(self.symbol, period="5d")
            
            if df is None:
                return None, None, {}
            
            # Get last valid bar
            last_idx = len(df) - 1
            
            # Skip if missing data
            if pd.isna(df['RSI'].iloc[last_idx]) or pd.isna(df['Distance%'].iloc[last_idx]):
                return None, None, {}
            
            # Check for signal
            distance = df['Distance%'].iloc[last_idx]
            rsi = df['RSI'].iloc[last_idx]
            
            signal_type = None
            if distance < -1.0 and rsi < 40:
                signal_type = 'LONG'
            elif distance > 1.0 and rsi > 60:
                signal_type = 'SHORT'
            
            if signal_type:
                score, details = self.improver.evaluate_entry_quality(df, last_idx)
                
                # Get market data
                market_data = {
                    'price': df['Close'].iloc[last_idx],
                    'sma': df['SMA20'].iloc[last_idx],
                    'distance': distance,
                    'rsi': rsi,
                    'volume_ratio': df['Volume_Ratio'].iloc[last_idx] if 'Volume_Ratio' in df else 1.0,
                    'atr_pct': (df['ATR'].iloc[last_idx] / df['Close'].iloc[last_idx] * 100) if 'ATR' in df else 0,
                    'timestamp': df.index[last_idx]
                }
                
                return score, details, market_data
            
            return None, None, {}
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None, {}
    
    def format_score_visual(self, score):
        """Create visual representation of score"""
        if score is None:
            return "No Signal"
            
        # Create meter
        meter_width = 50
        filled = int(score * meter_width / 100)
        
        meter = '█' * filled + '░' * (meter_width - filled)
        
        # Color based on score
        if score >= 70:
            meter = colored(meter, 'green')
            status = colored("STRONG SIGNAL", 'green', attrs=['bold'])
        elif score >= 50:
            meter = colored(meter, 'yellow')
            status = colored("MODERATE SIGNAL", 'yellow')
        else:
            meter = colored(meter, 'red')
            status = colored("WEAK SIGNAL", 'red')
        
        return f"[{meter}] {score}/100 - {status}"
    
    def display_live(self):
        """Display live score"""
        score, details, market_data = self.get_live_score()
        
        # Clear screen
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(colored(f"🎯 LIVE QUALITY SCORE - {self.symbol}", 'cyan', attrs=['bold']))
        print(colored("VXX Mean Reversion 15 Strategy", 'white', attrs=['bold']))
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if score is not None:
            print(f"\nCurrent Price: ${market_data['price']:.2f}")
            print(f"Distance from SMA: {market_data['distance']:.1f}%")
            print(f"RSI: {market_data['rsi']:.1f}")
            print(f"Volume Ratio: {market_data['volume_ratio']:.1f}x")
            
            print(f"\n{self.format_score_visual(score)}")
            
            # Score breakdown
            print("\nScore Components:")
            components = [
                ('Distance', details.get('distance_score', 0), 20),
                ('RSI', details.get('rsi_score', 0), 20),
                ('Volume', details.get('volume_score', 0), 15),
                ('Momentum', details.get('momentum_score', 0), 15),
                ('Higher TF', details.get('htf_score', 0), 10),
                ('Microstructure', details.get('micro_score', 0), 10),
                ('Time of Day', details.get('time_score', 0), 10)
            ]
            
            for name, value, max_val in components:
                bar_width = 20
                filled = int(value * bar_width / max_val) if max_val > 0 else 0
                bar = '▓' * filled + '░' * (bar_width - filled)
                
                if value >= max_val * 0.8:
                    bar = colored(bar, 'green')
                elif value >= max_val * 0.5:
                    bar = colored(bar, 'yellow')
                else:
                    bar = colored(bar, 'red')
                    
                print(f"  {name:<15} [{bar}] {value:>2}/{max_val}")
            
            # Add to history
            self.history.append({
                'time': datetime.now(),
                'score': score,
                'price': market_data['price']
            })
            
            # Show recent history
            if len(self.history) > 1:
                print("\nRecent History:")
                for h in self.history[-5:]:
                    time_str = h['time'].strftime('%H:%M:%S')
                    if h['score'] >= 70:
                        score_str = colored(f"{h['score']}", 'green')
                    elif h['score'] >= 50:
                        score_str = colored(f"{h['score']}", 'yellow')
                    else:
                        score_str = colored(f"{h['score']}", 'red')
                    print(f"  {time_str}: Score {score_str}/100 @ ${h['price']:.2f}")
            
        else:
            print("\nNo entry signal present")
            print("\nWaiting for:")
            print("  • LONG: Distance < -1% AND RSI < 40")
            print("  • SHORT: Distance > 1% AND RSI > 60")
        
        print("\nPress Ctrl+C to exit")
    
    def run_continuous(self, refresh_seconds=5):
        """Run continuous monitoring"""
        print(f"Starting live quality monitoring for {self.symbol}...")
        print(f"Refreshing every {refresh_seconds} seconds...")
        time.sleep(2)
        
        try:
            while True:
                self.display_live()
                time.sleep(refresh_seconds)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            
            # Show summary
            if self.history:
                scores = [h['score'] for h in self.history]
                print(f"\nSession Summary:")
                print(f"  • Signals seen: {len(scores)}")
                print(f"  • Average score: {sum(scores)/len(scores):.1f}")
                print(f"  • Max score: {max(scores)}")
                print(f"  • High quality (70+): {len([s for s in scores if s >= 70])}")


def main():
    print("=" * 60)
    print("🎯 LIVE QUALITY SCORE MONITOR")
    print("=" * 60)
    
    # Get symbol
    symbol = input("Enter symbol to monitor (default VXX): ").strip().upper()
    if not symbol:
        symbol = 'VXX'
    
    # Get refresh rate
    refresh = input("Refresh interval in seconds (default 5): ").strip()
    refresh_seconds = int(refresh) if refresh.isdigit() else 5
    
    # Create and run monitor
    monitor = LiveQualityScore(symbol)
    monitor.run_continuous(refresh_seconds)


if __name__ == "__main__":
    main()