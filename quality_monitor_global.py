"""
🌍 Global Quality Monitor
Monitors multiple asset classes including 24/7 crypto markets

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
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

class GlobalQualityMonitor:
    def __init__(self):
        # Curated symbol list - removed symbols that consistently fail alpha discovery
        # Based on targeted backtest results, focusing on proven performers
        self.symbols = {
            'Crypto (24/7)': [
                'BTC-USD',    # Bitcoin
                'ETH-USD',    # Ethereum
                'BNB-USD',    # Binance Coin
                'SOL-USD',    # Solana
                'XRP-USD',    # Ripple
                'ADA-USD',    # Cardano
                'DOGE-USD',   # Dogecoin
                'AVAX-USD',   # Avalanche
                'DOT-USD',    # Polkadot
                # Removed: 'MATIC-USD' (delisting issues)
            ],
            'US ETFs': [
                'VXX', 'SQQQ', 'TQQQ', 'UVXY', 'VIXY',  # Keep volatility ETFs
                'SPY', 'QQQ', 'IWM', 'DIA'              # Remove ARKK (poor performer)
            ],
            'Proven Winners': [  # New category for tested winners
                'SH',      # Inverse S&P (+31.8% excess)
                'LQD',     # Investment Grade Bonds (+20.3% excess)  
                'FXY',     # Japanese Yen (+6.0% excess)
                'TLT',     # 20+ Year Treasury (+10.1% from earlier test)
                'USO'      # Oil ETF (+22.5% from earlier test)
            ],
            'Commodities': [
                'GLD',     # Gold ETF
                'SLV',     # Silver ETF
                'USO',     # Oil ETF (also in winners)
                'GDX',     # Gold Miners
                'XLE'      # Energy Sector
                # Removed: UNG, DBA (failed alpha discovery)
            ],
            'International ETFs': [
                'EWJ',     # Japan
                'EWG',     # Germany
                'EWU',     # UK
                'FXI',     # China
                'EWZ',     # Brazil
                'EWY',     # South Korea
                'EWT',     # Taiwan
                'INDA',    # India
                'EWA'      # Australia
                # Removed: RSX (geopolitical risks + delisting issues)
            ],
            'Volatility & Inverse': [
                'SVXY',    # Short VIX (tested)
                'UVIX',    # Ultra VIX
                'VXZ',     # Mid-term VIX
                'VIXM',    # VIX Mid-term
                'SH'       # Inverse S&P (proven winner)
                # Removed: VIX (data issues)
            ]
        }
        
        self.improver = QuietImprover()
        self.all_symbols = list(set([symbol for category in self.symbols.values() for symbol in category]))  # Remove duplicates
        
        # Track failed symbols to avoid repeated processing
        self.failed_symbols = set()
        
        # Similar symbol mappings for AI discovery
        self.symbol_categories = {
            'volatility': ['VXX', 'VIXY', 'UVXY', 'SVXY', 'UVIX', 'VXZ', 'VIXM'],
            'bonds': ['TLT', 'IEF', 'LQD', 'SHY', 'HYG', 'TBT'],
            'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'GDX'],
            'currencies': ['FXY', 'FXE', 'UUP'],
            'inverse': ['SH', 'SQQQ', 'PSQ', 'SVXY'],
            'crypto': [s for s in self.all_symbols if s.endswith('-USD')],
            'international': ['EWJ', 'EWG', 'EWU', 'FXI', 'EWZ', 'EWY', 'EWT', 'INDA', 'EWA']
        }
        
    def get_market_status(self, symbol):
        """Determine if market is open for symbol"""
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        
        # Crypto is always open
        if symbol.endswith('-USD'):
            return True, "24/7"
            
        # US market hours (14:30-21:00 UTC)
        if 14 <= hour < 21:
            return True, "US Open"
        
        # Asian markets (varies by country)
        # This is simplified - actual hours vary
        if symbol in ['EWJ', 'FXI', 'EWY', 'EWT'] and (0 <= hour < 7):
            return True, "Asia Open"
            
        # European markets (8:00-16:30 UTC)
        if symbol in ['EWG', 'EWU'] and (8 <= hour < 17):
            return True, "EU Open"
            
        return False, "Closed"
    
    def get_symbol_data(self, symbol):
        """Get current data for symbol"""
        try:
            df = self.improver.prepare_data(symbol, period="5d")
            if df is None or len(df) < 50:
                return None
                
            # Get last bar
            idx = len(df) - 1
            
            # Get values
            price = df['Close'].iloc[idx]
            sma = df['SMA20'].iloc[idx] if not pd.isna(df['SMA20'].iloc[idx]) else price
            distance = df['Distance%'].iloc[idx] if not pd.isna(df['Distance%'].iloc[idx]) else 0
            rsi = df['RSI'].iloc[idx] if not pd.isna(df['RSI'].iloc[idx]) else 50
            volume_ratio = df['Volume_Ratio'].iloc[idx] if 'Volume_Ratio' in df and not pd.isna(df['Volume_Ratio'].iloc[idx]) else 1.0
            timestamp = df.index[idx]
            
            # Check entry conditions
            signal = None
            score = 0
            
            if not pd.isna(distance) and not pd.isna(rsi):
                if distance < -1.0 and rsi < 40:
                    signal = 'LONG'
                    score, _ = self.improver.evaluate_entry_quality(df, idx)
                elif distance > 1.0 and rsi > 60:
                    signal = 'SHORT'
                    score, _ = self.improver.evaluate_entry_quality(df, idx)
            
            # Get market status
            is_open, market_status = self.get_market_status(symbol)
            
            return {
                'symbol': symbol,
                'price': price,
                'sma': sma,
                'distance': distance,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'signal': signal,
                'score': score,
                'timestamp': timestamp,
                'is_open': is_open,
                'market_status': market_status
            }
            
        except:
            return None
    
    def display(self):
        """Display current signals across all markets"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(colored("╔" + "═" * 90 + "╗", 'cyan'))
        print(colored("║", 'cyan') + " " * 30 + colored("GLOBAL VXX MEAN REVERSION 15", 'white', attrs=['bold']) + " " * 31 + colored("║", 'cyan'))
        print(colored("║", 'cyan') + " " * 28 + "Multi-Market Quality Monitor" + " " * 34 + colored("║", 'cyan'))
        print(colored("╚" + "═" * 90 + "╝", 'cyan'))
        
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Collect all data
        print("\nScanning markets...", end='', flush=True)
        
        all_signals = []
        market_data = {}
        
        for category, symbols in self.symbols.items():
            market_data[category] = []
            for symbol in symbols:
                data = self.get_symbol_data(symbol)
                if data:
                    market_data[category].append(data)
                    if data['signal']:
                        all_signals.append(data)
                print('.', end='', flush=True)
        
        print(" Done!")
        
        # Display active signals first
        if all_signals:
            print(f"\n{colored('🎯 ACTIVE SIGNALS', 'green', attrs=['bold'])}:")
            print("─" * 90)
            
            all_signals.sort(key=lambda x: x['score'], reverse=True)
            
            for sig in all_signals[:10]:  # Show top 10
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
                
                market_color = 'green' if sig['is_open'] else 'red'
                market_str = colored(sig['market_status'], market_color)
                
                # Get similar symbols using AI
                similar_symbols = self.find_similar_symbols(sig['symbol'])
                similar_str = f" | Similar: {', '.join(similar_symbols[:3])}" if similar_symbols else ""
                
                print(f"{sig['symbol']:<10} {signal_str} @ ${sig['price']:>10.2f} | "
                      f"Score: {sig['score']:>3}/100 [{bar}] {quality}")
                print(f"           Distance: {sig['distance']:>6.1f}% | RSI: {sig['rsi']:>5.1f} | "
                      f"Volume: {sig['volume_ratio']:>4.1f}x | {market_str}{similar_str}")
                print()
        
        # Show summary by category
        print(f"\n{colored('📊 MARKET OVERVIEW', 'yellow', attrs=['bold'])}:")
        print("─" * 90)
        
        for category, data_list in market_data.items():
            if data_list:
                signals_in_cat = [d for d in data_list if d['signal']]
                open_markets = [d for d in data_list if d['is_open']]
                
                print(f"\n{colored(category, 'white', attrs=['bold'])}: "
                      f"{len(signals_in_cat)} signals | "
                      f"{len(open_markets)}/{len(data_list)} markets open")
                
                # Show top movers
                sorted_by_distance = sorted(data_list, key=lambda x: abs(x['distance']), reverse=True)[:3]
                for data in sorted_by_distance:
                    dist_color = 'green' if data['distance'] < -1 else 'red' if data['distance'] > 1 else 'white'
                    dist_str = colored(f"{data['distance']:+.1f}%", dist_color)
                    
                    rsi_color = 'green' if data['rsi'] < 30 else 'red' if data['rsi'] > 70 else 'white'
                    rsi_str = colored(f"{data['rsi']:.0f}", rsi_color)
                    
                    status = f"[{colored('SIGNAL', 'green')}]" if data['signal'] else ""
                    
                    print(f"  {data['symbol']:<10} ${data['price']:>10.2f} | "
                          f"Dist: {dist_str} | RSI: {rsi_str} {status}")
        
        # Show failed symbols count
        if self.failed_symbols:
            print(f"\n⚠️  Excluded {len(self.failed_symbols)} symbols with data issues: {', '.join(list(self.failed_symbols)[:5])}{'...' if len(self.failed_symbols) > 5 else ''}")
        
        print("\n" + "─" * 90)
        print("Entry Conditions: LONG (Distance < -1% AND RSI < 40) | SHORT (Distance > 1% AND RSI > 60)")
        print(f"Symbols: {len(self.all_symbols) - len(self.failed_symbols)} active | Similar symbol discovery enabled")
        print(f"\nRefreshing in 30 seconds... Press Ctrl+C to exit")
    
    def run(self):
        """Run continuous monitoring"""
        try:
            while True:
                self.display()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n✅ Global Quality Monitor stopped.")
            if self.failed_symbols:
                print(f"📝 Excluded symbols: {', '.join(sorted(self.failed_symbols))}")
    
    def find_similar_symbols(self, trigger_symbol):
        """Find similar symbols to the one that triggered using AI reasoning"""
        similar_symbols = []
        
        # Find category of trigger symbol
        trigger_category = None
        for category, symbols in self.symbol_categories.items():
            if trigger_symbol in symbols:
                trigger_category = category
                break
        
        if trigger_category:
            # Get other symbols from same category
            category_symbols = [s for s in self.symbol_categories[trigger_category] 
                             if s != trigger_symbol and s not in self.failed_symbols]
            similar_symbols.extend(category_symbols[:3])
        
        # AI-based similar symbol discovery
        ai_similar = self._ai_discover_similar(trigger_symbol)
        for symbol in ai_similar:
            if symbol not in similar_symbols and symbol not in self.failed_symbols:
                similar_symbols.append(symbol)
        
        return similar_symbols[:5]  # Return max 5
    
    def _ai_discover_similar(self, symbol):
        """Use AI reasoning to find similar symbols"""
        # Simple rule-based AI for now (can be enhanced with LLM later)
        similar_map = {
            # Volatility relationships
            'VXX': ['VIXY', 'UVXY', 'SVXY'],
            'VIXY': ['VXX', 'UVXY'],
            'UVXY': ['VXX', 'VIXY'],
            'SVXY': ['VXX', 'VIXY', 'UVXY'],  # Inverse relationship
            
            # Bond relationships
            'TLT': ['IEF', 'TBT'],  # TBT is inverse
            'IEF': ['TLT', 'LQD'],
            'LQD': ['IEF', 'HYG'],
            'SH': ['SQQQ', 'PSQ'],  # Inverse equity relationships
            
            # Currency relationships  
            'FXY': ['FXE', 'UUP'],
            'FXE': ['FXY', 'UUP'],
            'UUP': ['FXE', 'FXY'],
            
            # Commodity relationships
            'GLD': ['SLV', 'GDX'],
            'SLV': ['GLD', 'GDX'],
            'USO': ['XLE', 'GLD'],  # Energy-commodity correlation
            
            # Crypto relationships
            'BTC-USD': ['ETH-USD', 'SOL-USD'],
            'ETH-USD': ['BTC-USD', 'SOL-USD'],
            'SOL-USD': ['BTC-USD', 'ETH-USD']
        }
        
        return similar_map.get(symbol, [])


if __name__ == "__main__":
    monitor = GlobalQualityMonitor()
    monitor.run()