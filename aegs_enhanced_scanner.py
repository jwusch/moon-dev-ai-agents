"""
🔥💎 AEGS ENHANCED SCANNER WITH AUTO-REGISTRY 💎🔥
Automatically includes all proven goldmine symbols and allows adding new ones

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored

class AEGSEnhancedScanner:
    """Enhanced scanner with goldmine registry integration"""
    
    def __init__(self):
        # Load goldmine registry
        self.registry_file = 'aegs_goldmine_registry.json'
        self.load_registry()
        
        # Build symbol list from registry
        self.build_symbol_list()
        
        # Results storage
        self.buy_signals = []
        self.near_signals = []
        
    def load_registry(self):
        """Load goldmine registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
            print(f"✅ Loaded {self.registry['metadata']['total_symbols']} symbols from registry")
        else:
            print("❌ Registry not found, using default symbols")
            self.registry = {"goldmine_symbols": {}, "categories": {}}
    
    def save_registry(self):
        """Save updated registry"""
        self.registry['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d")
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        print(f"✅ Registry saved with {self.registry['metadata']['total_symbols']} symbols")
    
    def build_symbol_list(self):
        """Build comprehensive symbol list from registry"""
        self.all_symbols = []
        self.symbol_info = {}
        
        # Add goldmine symbols by tier
        for tier, data in self.registry['goldmine_symbols'].items():
            for symbol, info in data.get('symbols', {}).items():
                if info.get('active', True):
                    self.all_symbols.append(symbol)
                    self.symbol_info[symbol] = {
                        'tier': tier,
                        'excess_return': info.get('excess_return', 0),
                        'category': info.get('category', 'Unknown')
                    }
        
        # Remove duplicates
        self.all_symbols = list(set(self.all_symbols))
        
        # Organize by priority (sort by excess return)
        self.all_symbols.sort(key=lambda x: self.symbol_info.get(x, {}).get('excess_return', 0), reverse=True)
        
        print(f"📊 Scanning {len(self.all_symbols)} proven goldmine symbols")
    
    def add_new_symbol(self, symbol, excess_return, category, tier=None):
        """Add a new successfully backtested symbol to registry"""
        
        # Determine tier based on excess return
        if tier is None:
            if excess_return > 1000:
                tier = "extreme_goldmines"
            elif excess_return > 100:
                tier = "high_potential"
            elif excess_return > 0:
                tier = "positive"
            else:
                print(f"❌ Symbol {symbol} has negative excess return, not adding")
                return False
        
        # Add to registry
        if tier not in self.registry['goldmine_symbols']:
            self.registry['goldmine_symbols'][tier] = {"symbols": {}}
        
        self.registry['goldmine_symbols'][tier]['symbols'][symbol] = {
            'excess_return': excess_return,
            'category': category,
            'active': True
        }
        
        # Add to category
        if category not in self.registry['categories']:
            self.registry['categories'][category] = []
        if symbol not in self.registry['categories'][category]:
            self.registry['categories'][category].append(symbol)
        
        # Update metadata
        self.registry['metadata']['total_symbols'] = len(self.all_symbols) + 1
        
        # Save registry
        self.save_registry()
        
        print(colored(f"✅ Added {symbol} to {tier} (excess: {excess_return:+.0f}%)", 'green'))
        return True
    
    def scan_all_symbols(self):
        """Scan all registry symbols for signals"""
        
        print(colored("🔥💎 AEGS ENHANCED SCANNER - GOLDMINE REGISTRY 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Scanning {len(self.all_symbols)} proven goldmine symbols...")
        print("=" * 80)
        
        # Scan by tier for organization
        tiers = ["extreme_goldmines", "high_potential", "positive"]
        
        for tier in tiers:
            tier_data = self.registry['goldmine_symbols'].get(tier, {})
            tier_symbols = [s for s in self.all_symbols if self.symbol_info.get(s, {}).get('tier') == tier]
            
            if tier_symbols:
                print(f"\n📊 {tier_data.get('description', tier).upper()}:")
                print("-" * 60)
                
                for symbol in tier_symbols:
                    info = self.symbol_info[symbol]
                    try:
                        self._scan_symbol(symbol, info['category'], info['excess_return'])
                    except Exception as e:
                        print(f"   ❌ Error scanning {symbol}: {str(e)[:50]}...")
        
        # Display comprehensive results
        self._display_results()
    
    def _scan_symbol(self, symbol, category, historical_excess):
        """Scan individual symbol"""
        
        print(f"   🔍 {symbol} (excess: {historical_excess:+.0f}%)...", end='', flush=True)
        
        # Get recent data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='200d')
        
        if len(df) < 50:
            print(" ❌ Insufficient data")
            return
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Check signals
        signal_strength = 0
        signals_triggered = []
        
        # RSI check
        if pd.notna(latest.get('RSI', np.nan)):
            if latest['RSI'] < 30:
                signal_strength += 35
                signals_triggered.append(f"RSI={latest['RSI']:.0f}")
            elif latest['RSI'] < 35:
                signal_strength += 20
                signals_triggered.append(f"RSI={latest['RSI']:.0f}")
        
        # Bollinger Band check
        if pd.notna(latest.get('BB_Position', np.nan)):
            if latest['BB_Position'] < 0:
                signal_strength += 35
                signals_triggered.append("BB_Below")
            elif latest['BB_Position'] < 0.2:
                signal_strength += 20
                signals_triggered.append(f"BB={latest['BB_Position']:.2f}")
        
        # Volume check
        if pd.notna(latest.get('Volume_Ratio', np.nan)):
            if latest['Volume_Ratio'] > 2.0 and latest.get('Daily_Change', 0) < -0.02:
                signal_strength += 30
                signals_triggered.append(f"Vol={latest['Volume_Ratio']:.1f}x")
            elif latest['Volume_Ratio'] > 1.5:
                signal_strength += 10
        
        # Price drop check
        daily_change = latest.get('Daily_Change', 0) * 100
        if daily_change < -10:
            signal_strength += 35
            signals_triggered.append(f"Drop={daily_change:.1f}%")
        elif daily_change < -5:
            signal_strength += 20
            signals_triggered.append(f"Drop={daily_change:.1f}%")
        
        # Determine status
        if signal_strength >= 70:
            self.buy_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'historical_excess': historical_excess,
                'daily_change': daily_change
            })
            print(colored(f" 🚀 STRONG BUY! Score: {signal_strength}/100", 'green', attrs=['bold']))
            
        elif signal_strength >= 50:
            self.buy_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'historical_excess': historical_excess,
                'daily_change': daily_change
            })
            print(colored(f" ✅ BUY! Score: {signal_strength}/100", 'green'))
            
        elif signal_strength >= 30:
            self.near_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'historical_excess': historical_excess
            })
            print(colored(f" ⚡ Near: {signal_strength}/100", 'yellow'))
            
        else:
            print(" ⏸️  No signal")
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change
        df['Daily_Change'] = df['Close'].pct_change()
        
        return df
    
    def _display_results(self):
        """Display comprehensive results"""
        
        print("\n" + "=" * 80)
        print(colored("🎯 AEGS SCAN RESULTS - GOLDMINE OPPORTUNITIES", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if self.buy_signals:
            # Sort by signal strength
            self.buy_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            print(colored(f"\n🚀 IMMEDIATE BUY SIGNALS ({len(self.buy_signals)}):", 'green', attrs=['bold']))
            print("=" * 80)
            
            for i, signal in enumerate(self.buy_signals[:10], 1):  # Top 10
                symbol = signal['symbol']
                strength = signal['signal_strength']
                price = signal['price']
                historical = signal['historical_excess']
                triggered = ', '.join(signal['signals'])
                
                # Priority based on historical performance
                if historical > 1000:
                    priority = "💎 EXTREME GOLDMINE"
                    priority_color = 'red'
                elif historical > 100:
                    priority = "🚀 HIGH PRIORITY"
                    priority_color = 'yellow'
                else:
                    priority = "✅ GOOD"
                    priority_color = 'green'
                
                print(colored(f"\n#{i}. {symbol} - {priority}", priority_color, attrs=['bold']))
                print(f"   Price: ${price:.2f} | Signal: {strength}/100")
                print(f"   Historical Excess: {historical:+.0f}%")
                print(f"   Triggers: {triggered}")
                
                # Investment potential
                if historical > 1000:
                    potential = 10000 * (1 + historical/100)
                    print(colored(f"   💰 Potential: $10k → ${potential:,.0f}", 'cyan'))
        
        else:
            print(colored("\n⏸️  NO IMMEDIATE BUY SIGNALS", 'blue'))
            print("All goldmine symbols currently between entry points")
        
        # Near signals summary
        if self.near_signals:
            print(colored(f"\n⚡ APPROACHING BUY ZONE ({len(self.near_signals)}):", 'yellow'))
            for sig in self.near_signals[:5]:
                print(f"   {sig['symbol']}: ${sig['price']:.2f} (Score: {sig['signal_strength']}/100)")
        
        # Category summary
        if self.buy_signals:
            category_counts = {}
            for sig in self.buy_signals:
                cat = sig['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(colored("\n📊 HOT CATEGORIES:", 'cyan'))
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {cat}: {count} signals")
        
        # Summary stats
        print(f"\n📈 MARKET SUMMARY:")
        print(f"   Total Scanned: {len(self.all_symbols)}")
        print(f"   Buy Signals: {len(self.buy_signals)}")
        print(f"   Near Signals: {len(self.near_signals)}")
        
        print("\n💎 AEGS Enhanced Scanner Complete!")


def main():
    """Run enhanced AEGS scanner"""
    
    scanner = AEGSEnhancedScanner()
    
    # Example: Add a new symbol (comment out after first run)
    # scanner.add_new_symbol("NEW_SYMBOL", 500, "New Category")
    
    # Run scan
    scanner.scan_all_symbols()
    
    # Show how to add new symbols
    print("\n💡 To add newly discovered goldmines:")
    print("   scanner.add_new_symbol('SYMBOL', excess_return, 'Category')")
    print("   Example: scanner.add_new_symbol('SOL-USD', 39496, 'Cryptocurrency')")


if __name__ == "__main__":
    main()