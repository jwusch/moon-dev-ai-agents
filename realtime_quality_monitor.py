"""
🎯 Real-Time Entry Quality Score Monitor
Live dashboard showing entry quality scores for multiple symbols

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import time
import os
from termcolor import colored
from yfinance_cache_demo import YFinanceCache
from improved_entry_timing import ImprovedEntryTiming

class RealtimeQualityMonitor:
    def __init__(self, symbols):
        self.symbols = symbols
        self.cache = YFinanceCache()
        self.improver = ImprovedEntryTiming()
        self.last_scores = {}
        
    def get_current_data(self, symbol):
        """Get fresh data for symbol"""
        try:
            # Use the improver's prepare_data method to get all required indicators
            df = self.improver.prepare_data(symbol, period="5d")
            
            if df is None or len(df) < 50:
                return None
                
            return df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def format_score_bar(self, score, width=20):
        """Create visual progress bar for score"""
        filled = int(score * width / 100)
        bar = '█' * filled + '░' * (width - filled)
        
        # Color based on score
        if score >= 70:
            return colored(bar, 'green')
        elif score >= 50:
            return colored(bar, 'yellow')
        else:
            return colored(bar, 'red')
    
    def format_score_details(self, details):
        """Format score breakdown"""
        components = []
        
        # Define max scores for each component
        max_scores = {
            'distance_score': 20,
            'rsi_score': 20,
            'volume_score': 15,
            'momentum_score': 15,
            'htf_score': 10,
            'micro_score': 10,
            'time_score': 10
        }
        
        # Order for display
        order = ['distance_score', 'rsi_score', 'volume_score', 'momentum_score', 
                'htf_score', 'micro_score', 'time_score']
        
        for key in order:
            if key in details:
                value = details.get(key, 0)
                max_val = max_scores.get(key, 10)
                
                # Short labels
                labels = {
                    'distance_score': 'DST',
                    'rsi_score': 'RSI',
                    'volume_score': 'VOL',
                    'momentum_score': 'MOM',
                    'htf_score': 'HTF',
                    'micro_score': 'MCR',
                    'time_score': 'TIM'
                }
                
                label = labels.get(key, key[:3].upper())
                components.append(f"{label}:{value}/{max_val}")
        
        return " | ".join(components)
    
    def check_entry_conditions(self, df, idx):
        """Check if basic entry conditions are met"""
        if pd.isna(df['RSI'].iloc[idx]) or pd.isna(df['Distance%'].iloc[idx]):
            return None
            
        distance = df['Distance%'].iloc[idx]
        rsi = df['RSI'].iloc[idx]
        
        # Long conditions
        if distance < -1.0 and rsi < 40:
            return 'LONG'
        # Short conditions
        elif distance > 1.0 and rsi > 60:
            return 'SHORT'
            
        return None
    
    def display_dashboard(self):
        """Display real-time dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 100)
        print(colored("🎯 REAL-TIME ENTRY QUALITY MONITOR", 'cyan', attrs=['bold']))
        print(colored("VXX Mean Reversion 15 Strategy", 'white', attrs=['bold']))
        print("=" * 100)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nQuality Score Components:")
        print("DST: Distance from SMA (20) | RSI: RSI Level (20) | VOL: Volume Ratio (15)")
        print("MOM: Momentum Shift (15) | HTF: Higher TF Align (10) | MCR: Microstructure (10) | TIM: Time of Day (10)")
        print("-" * 100)
        print(f"{'Symbol':<8} {'Price':>8} {'SMA':>8} {'Dist%':>7} {'RSI':>5} {'Signal':<7} {'Score':>6} {'Quality Bar':<22} Components")
        print("-" * 100)
        
        rows = []
        
        for symbol in self.symbols:
            df = self.get_current_data(symbol)
            
            if df is None or len(df) == 0:
                continue
                
            # Get last bar during market hours
            last_idx = len(df) - 1
            
            # Skip if outside market hours
            current_time = df.index[last_idx]
            if current_time.hour < 9 or current_time.hour >= 16:
                # Find last market hours bar
                for i in range(len(df)-1, max(0, len(df)-10), -1):
                    if 9 <= df.index[i].hour < 16:
                        last_idx = i
                        break
            
            # Current values
            price = df['Close'].iloc[last_idx]
            sma = df['SMA20'].iloc[last_idx] if not pd.isna(df['SMA20'].iloc[last_idx]) else 0
            distance = df['Distance%'].iloc[last_idx] if not pd.isna(df['Distance%'].iloc[last_idx]) else 0
            rsi = df['RSI'].iloc[last_idx] if not pd.isna(df['RSI'].iloc[last_idx]) else 50
            
            # Check entry conditions
            signal_type = self.check_entry_conditions(df, last_idx)
            
            # Calculate quality score
            if signal_type:
                score, details = self.improver.evaluate_entry_quality(df, last_idx)
                signal_str = colored(signal_type, 'green' if signal_type == 'LONG' else 'red')
                
                # Track score changes
                prev_score = self.last_scores.get(symbol, 0)
                if score > prev_score:
                    score_str = colored(f"{score:>3}/100", 'green') + " ↑"
                elif score < prev_score:
                    score_str = colored(f"{score:>3}/100", 'red') + " ↓"
                else:
                    score_str = f"{score:>3}/100  "
                self.last_scores[symbol] = score
            else:
                score = 0
                details = {}
                signal_str = "   -   "
                score_str = "   -    "
            
            # Create row
            row = {
                'symbol': symbol,
                'price': price,
                'sma': sma,
                'distance': distance,
                'rsi': rsi,
                'signal': signal_str,
                'score': score,
                'score_str': score_str,
                'bar': self.format_score_bar(score) if score > 0 else ' ' * 20,
                'details': self.format_score_details(details) if details else '-'
            }
            rows.append(row)
        
        # Sort by score (highest first)
        rows.sort(key=lambda x: x['score'], reverse=True)
        
        # Display rows
        for row in rows:
            print(f"{row['symbol']:<8} ${row['price']:>7.2f} ${row['sma']:>7.2f} "
                  f"{row['distance']:>6.1f}% {row['rsi']:>5.1f} {row['signal']:<7} "
                  f"{row['score_str']:<8} {row['bar']:<22} {row['details']}")
        
        print("-" * 100)
        
        # Show top opportunities
        high_quality = [r for r in rows if r['score'] >= 60]
        if high_quality:
            print(colored("\n⭐ HIGH QUALITY SIGNALS (60+):", 'yellow'))
            for row in high_quality[:3]:
                print(f"  • {row['symbol']}: Score {row['score']}/100 - "
                      f"Price ${row['price']:.2f} ({row['distance']:+.1f}% from SMA)")
        
        # Legend
        print("\n" + colored("Legend:", 'cyan'))
        print(f"  {colored('█' * 5, 'green')} High Quality (70+)")
        print(f"  {colored('█' * 5, 'yellow')} Medium Quality (50-69)")
        print(f"  {colored('█' * 5, 'red')} Low Quality (<50)")
        print("\nPress Ctrl+C to exit")
    
    def run(self, refresh_seconds=15):
        """Run continuous monitoring"""
        print(f"Starting real-time monitoring of {len(self.symbols)} symbols...")
        print(f"Refreshing every {refresh_seconds} seconds...")
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_seconds)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            return


def main():
    # Default symbols to monitor
    default_symbols = [
        'VXX', 'SQQQ', 'AMD', 'NVDA', 'TSLA',
        'VIXY', 'UVXY', 'SPXS', 'SPY', 'QQQ',
        'TQQQ', 'XLF', 'AAPL', 'MSFT', 'META'
    ]
    
    print("=" * 70)
    print("🎯 REAL-TIME ENTRY QUALITY MONITOR")
    print("=" * 70)
    
    # Allow custom symbol input
    custom = input("\nUse custom symbols? (y/N): ").strip().lower()
    
    if custom == 'y':
        symbols_input = input("Enter symbols separated by commas: ").strip().upper()
        symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
    else:
        symbols = default_symbols
    
    # Get refresh rate
    refresh_input = input("\nRefresh interval in seconds (default 15): ").strip()
    refresh_seconds = int(refresh_input) if refresh_input.isdigit() else 15
    
    # Create and run monitor
    monitor = RealtimeQualityMonitor(symbols)
    monitor.run(refresh_seconds)


if __name__ == "__main__":
    main()