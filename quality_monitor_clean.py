"""
🎯 Clean Real-Time Quality Monitor
Enhanced UI for VXX Mean Reversion 15 Strategy

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import time
import os
import sys
from termcolor import colored
from yfinance_cache_demo import YFinanceCache
from improved_entry_timing import ImprovedEntryTiming

# Suppress yfinance cache messages
class SilentCache(YFinanceCache):
    def __init__(self):
        super().__init__()
        # Redirect cache messages during data fetch
        self._original_stdout = sys.stdout
        
    def get_data(self, *args, **kwargs):
        # Temporarily suppress output
        sys.stdout = open(os.devnull, 'w')
        try:
            result = super().get_data(*args, **kwargs)
            return result
        finally:
            sys.stdout = self._original_stdout

class CleanQualityMonitor:
    def __init__(self, symbols):
        self.symbols = symbols
        self.cache = SilentCache()
        self.improver = ImprovedEntryTiming()
        self.improver.cache = self.cache  # Use silent cache
        self.last_update = {}
        self.signal_history = {}
        
    def get_current_data(self, symbol):
        """Get fresh data for symbol"""
        try:
            # Temporarily suppress all output
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            df = self.improver.prepare_data(symbol, period="5d")
            
            sys.stdout = original_stdout
            
            if df is None or len(df) < 50:
                return None
                
            return df
        except Exception as e:
            sys.stdout = original_stdout
            return None
    
    def check_entry_conditions(self, df, idx):
        """Check if basic entry conditions are met"""
        if pd.isna(df['RSI'].iloc[idx]) or pd.isna(df['Distance%'].iloc[idx]):
            return None
            
        distance = df['Distance%'].iloc[idx]
        rsi = df['RSI'].iloc[idx]
        
        if distance < -1.0 and rsi < 40:
            return 'LONG'
        elif distance > 1.0 and rsi > 60:
            return 'SHORT'
            
        return None
    
    def format_price_change(self, current, previous):
        """Format price change with color"""
        if previous == 0:
            return ""
        change = ((current - previous) / previous) * 100
        if change > 0:
            return colored(f"+{change:.1f}%", 'green')
        elif change < 0:
            return colored(f"{change:.1f}%", 'red')
        else:
            return "0.0%"
    
    def create_score_meter(self, score, width=20):
        """Create visual meter for score"""
        if score == 0:
            return "━" * width
            
        filled = int(score * width / 100)
        
        # Create gradient effect
        meter = ""
        for i in range(width):
            if i < filled:
                if score >= 70:
                    meter += colored("█", 'green')
                elif score >= 50:
                    meter += colored("█", 'yellow')
                else:
                    meter += colored("█", 'red')
            else:
                meter += "─"
                
        return meter
    
    def display_dashboard(self):
        """Display clean dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Header
        print(colored("╔" + "═" * 98 + "╗", 'cyan'))
        print(colored("║", 'cyan') + " " * 30 + colored("VXX MEAN REVERSION 15", 'white', attrs=['bold']) + " " * 30 + colored("║", 'cyan'))
        print(colored("║", 'cyan') + " " * 28 + "Real-Time Quality Monitor" + " " * 28 + colored("║", 'cyan'))
        print(colored("╚" + "═" * 98 + "╝", 'cyan'))
        
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Market Hours: ", end="")
        hour = datetime.now().hour
        if 9 <= hour < 16:
            print(colored("OPEN", 'green', attrs=['bold']))
        else:
            print(colored("CLOSED", 'red', attrs=['bold']))
        
        # Collect data for all symbols
        rows = []
        signals_found = 0
        
        for symbol in self.symbols:
            df = self.get_current_data(symbol)
            
            if df is None or len(df) == 0:
                continue
                
            # Get last bar
            last_idx = len(df) - 1
            
            # Skip if outside market hours
            current_time = df.index[last_idx]
            if current_time.hour < 9 or current_time.hour >= 16:
                for i in range(len(df)-1, max(0, len(df)-10), -1):
                    if 9 <= df.index[i].hour < 16:
                        last_idx = i
                        break
            
            # Current values
            price = df['Close'].iloc[last_idx]
            sma = df['SMA20'].iloc[last_idx] if not pd.isna(df['SMA20'].iloc[last_idx]) else 0
            distance = df['Distance%'].iloc[last_idx] if not pd.isna(df['Distance%'].iloc[last_idx]) else 0
            rsi = df['RSI'].iloc[last_idx] if not pd.isna(df['RSI'].iloc[last_idx]) else 50
            volume_ratio = df['Volume_Ratio'].iloc[last_idx] if 'Volume_Ratio' in df and not pd.isna(df['Volume_Ratio'].iloc[last_idx]) else 1.0
            
            # Price change
            prev_price = self.last_update.get(symbol, price)
            price_change = self.format_price_change(price, prev_price)
            self.last_update[symbol] = price
            
            # Check entry conditions
            signal_type = self.check_entry_conditions(df, last_idx)
            
            # Calculate quality score
            if signal_type:
                score, details = self.improver.evaluate_entry_quality(df, last_idx)
                signals_found += 1
                
                # Track signal
                if symbol not in self.signal_history:
                    self.signal_history[symbol] = []
                self.signal_history[symbol].append({
                    'time': current_time,
                    'score': score,
                    'type': signal_type
                })
            else:
                score = 0
                details = {}
            
            row = {
                'symbol': symbol,
                'price': price,
                'price_change': price_change,
                'sma': sma,
                'distance': distance,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'signal': signal_type,
                'score': score,
                'details': details
            }
            rows.append(row)
        
        # Sort by score
        rows.sort(key=lambda x: x['score'], reverse=True)
        
        # Display table
        print("\n┌─────────┬──────────┬────────┬────────┬────────┬────────┬─────────┬───────────────────────┬─────────┐")
        print("│ Symbol  │  Price   │ Change │  SMA   │ Dist % │  RSI   │  Vol x  │    Quality Score      │  Signal │")
        print("├─────────┼──────────┼────────┼────────┼────────┼────────┼─────────┼───────────────────────┼─────────┤")
        
        for row in rows[:15]:  # Show top 15
            symbol = row['symbol']
            
            # Format values
            symbol_str = f"{symbol:<7}"
            price_str = f"${row['price']:>8.2f}"
            sma_str = f"${row['sma']:>6.2f}"
            dist_str = f"{row['distance']:>6.1f}%"
            rsi_str = f"{row['rsi']:>6.1f}"
            vol_str = f"{row['volume_ratio']:>7.1f}"
            
            # Color distance
            if row['distance'] < -2:
                dist_str = colored(dist_str, 'green')
            elif row['distance'] > 2:
                dist_str = colored(dist_str, 'red')
            
            # Color RSI
            if row['rsi'] < 30:
                rsi_str = colored(rsi_str, 'green')
            elif row['rsi'] > 70:
                rsi_str = colored(rsi_str, 'red')
                
            # Score meter
            meter = self.create_score_meter(row['score'])
            
            # Signal
            if row['signal'] == 'LONG':
                signal_str = colored("  LONG ↑", 'green', attrs=['bold'])
            elif row['signal'] == 'SHORT':
                signal_str = colored(" SHORT ↓", 'red', attrs=['bold'])
            else:
                signal_str = "    -    "
            
            print(f"│ {symbol_str} │{price_str} │{row['price_change']:>7} │{sma_str} │{dist_str} │{rsi_str} │{vol_str} │ {meter} │{signal_str}│")
        
        print("└─────────┴──────────┴────────┴────────┴────────┴────────┴─────────┴───────────────────────┴─────────┘")
        
        # High quality signals
        high_quality = [r for r in rows if r['score'] >= 60]
        if high_quality:
            print(f"\n{colored('⭐ HIGH QUALITY SIGNALS', 'yellow', attrs=['bold'])} (Score 60+):")
            for i, row in enumerate(high_quality[:5]):
                signal_type = colored("LONG", 'green') if row['signal'] == 'LONG' else colored("SHORT", 'red')
                print(f"  {i+1}. {colored(row['symbol'], 'white', attrs=['bold'])}: "
                      f"Score {colored(str(row['score']), 'yellow')}/100 | "
                      f"{signal_type} @ ${row['price']:.2f} | "
                      f"Distance {row['distance']:+.1f}% | RSI {row['rsi']:.1f}")
                
                # Show score breakdown
                if row['details']:
                    components = []
                    for key, val in [('distance_score', 'DST'), ('rsi_score', 'RSI'), 
                                   ('volume_score', 'VOL'), ('momentum_score', 'MOM')]:
                        if key in row['details']:
                            components.append(f"{val}:{row['details'][key]}")
                    print(f"     └─ {' | '.join(components)}")
        
        # Summary stats
        print(f"\n{colored('📊 SUMMARY', 'cyan')}:")
        print(f"  • Active Signals: {signals_found}")
        print(f"  • High Quality (60+): {len(high_quality)}")
        print(f"  • Symbols Monitored: {len(self.symbols)}")
        
        # Legend
        print(f"\nScore Legend: ", end="")
        print(colored("█" * 5, 'green') + " 70-100 (Strong) | ", end="")
        print(colored("█" * 5, 'yellow') + " 50-69 (Moderate) | ", end="")
        print(colored("█" * 5, 'red') + " 0-49 (Weak)")
        
        print(f"\nPress Ctrl+C to exit")
    
    def run(self, refresh_seconds=10):
        """Run continuous monitoring"""
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_seconds)
                
        except KeyboardInterrupt:
            print("\n\n" + colored("Monitoring stopped.", 'yellow'))
            
            # Show session summary
            if self.signal_history:
                print(f"\n{colored('SESSION SUMMARY:', 'cyan')}")
                total_signals = sum(len(signals) for signals in self.signal_history.values())
                print(f"  • Total signals seen: {total_signals}")
                
                # Best signals
                best_signals = []
                for symbol, signals in self.signal_history.items():
                    for signal in signals:
                        best_signals.append((symbol, signal['score'], signal['type']))
                best_signals.sort(key=lambda x: x[1], reverse=True)
                
                if best_signals:
                    print(f"\n  Top signals this session:")
                    for symbol, score, signal_type in best_signals[:5]:
                        print(f"    • {symbol}: {score}/100 ({signal_type})")


def main():
    # Default symbols
    default_symbols = [
        'VXX', 'SQQQ', 'AMD', 'NVDA', 'TSLA', 'VIXY', 
        'UVXY', 'SPXS', 'SPY', 'QQQ', 'TQQQ', 'XLF'
    ]
    
    print(colored("╔" + "═" * 60 + "╗", 'cyan'))
    print(colored("║", 'cyan') + " " * 15 + colored("VXX MEAN REVERSION 15", 'white', attrs=['bold']) + " " * 16 + colored("║", 'cyan'))
    print(colored("║", 'cyan') + " " * 13 + "Real-Time Quality Monitor" + " " * 14 + colored("║", 'cyan'))
    print(colored("╚" + "═" * 60 + "╝", 'cyan'))
    
    # Symbol selection
    print(f"\nDefault symbols: {', '.join(default_symbols[:6])}...")
    custom = input("Use custom symbols? (y/N): ").strip().lower()
    
    if custom == 'y':
        symbols_input = input("Enter symbols (comma-separated): ").strip().upper()
        symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
    else:
        symbols = default_symbols
    
    # Refresh rate
    refresh = input("Refresh interval in seconds (default 10): ").strip()
    refresh_seconds = int(refresh) if refresh.isdigit() else 10
    
    print(f"\n{colored('Starting monitor...', 'green')}")
    time.sleep(1)
    
    # Create and run monitor
    monitor = CleanQualityMonitor(symbols)
    monitor.run(refresh_seconds)


if __name__ == "__main__":
    main()