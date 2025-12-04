"""
🔥💎 AEGS QUICK MARKET SCANNER 💎🔥
Simplified scanner with robust data handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored

class QuickAEGSScanner:
    def __init__(self):
        # Top priority goldmine symbols 
        self.symbols = [
            'WULF', 'EQT', 'NOK', 'WKHS',  # Proven goldmines
            'MARA', 'RIOT', 'CLSK', 'CIFR',  # Crypto mining
            'SAVA', 'BIIB', 'VKTX', 'EDIT',  # Biotech
            'BB', 'GME', 'AMC', 'SOFI',  # Memes
            'LCID', 'RIVN', 'NKLA',  # SPACs
            'TNA', 'LABU', 'NUGT', 'SQQQ'  # Leveraged
        ]
        
        self.buy_signals = []
    
    def scan_symbol(self, symbol):
        """Scan single symbol with robust error handling"""
        try:
            # Download data
            data = yf.download(symbol, period='100d', progress=False, interval='1d')
            
            if data.empty:
                return None
                
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == symbol else col[1] for col in data.columns]
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                # Try alternative column names
                col_mapping = {
                    'Adj Close': 'Close',
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                data = data.rename(columns=col_mapping)
                
            if len(data) < 20:
                return None
                
            # Calculate simple indicators
            data = data.copy()
            
            # RSI (simple version)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 0.0001)  # Avoid division by zero
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple moving average
            data['SMA20'] = data['Close'].rolling(20).mean()
            data['Distance'] = (data['Close'] - data['SMA20']) / data['SMA20'] * 100
            
            # Volume ratio
            data['VolMA'] = data['Volume'].rolling(10).mean()
            data['VolRatio'] = data['Volume'] / (data['VolMA'] + 1)
            
            # Daily change
            data['Change'] = data['Close'].pct_change() * 100
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Calculate signal score
            score = 0
            triggers = []
            
            # RSI oversold
            if latest['RSI'] < 30:
                score += 40
                triggers.append(f"RSI={latest['RSI']:.1f}")
            elif latest['RSI'] < 35:
                score += 20
                triggers.append(f"RSI={latest['RSI']:.1f}")
            
            # Below SMA
            if latest['Distance'] < -5:
                score += 30
                triggers.append(f"Below_SMA={latest['Distance']:.1f}%")
            elif latest['Distance'] < -2:
                score += 15
                triggers.append(f"Near_SMA={latest['Distance']:.1f}%")
            
            # Volume spike
            if latest['VolRatio'] > 2:
                score += 20
                triggers.append(f"Vol={latest['VolRatio']:.1f}x")
            
            # Big drop
            if latest['Change'] < -8:
                score += 30
                triggers.append(f"Drop={latest['Change']:.1f}%")
            elif latest['Change'] < -4:
                score += 15
                triggers.append(f"Down={latest['Change']:.1f}%")
            
            return {
                'symbol': symbol,
                'price': latest['Close'],
                'score': score,
                'triggers': triggers,
                'change': latest['Change'],
                'rsi': latest['RSI'],
                'distance': latest['Distance'],
                'volume': latest['VolRatio']
            }
            
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {str(e)[:30]}")
            return None
    
    def run_scan(self):
        """Run quick scan on all symbols"""
        print(colored("🔥💎 AEGS QUICK SCANNER 💎🔥", 'cyan', attrs=['bold']))
        print("="*60)
        print(f"Scanning {len(self.symbols)} symbols...")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        results = []
        
        for symbol in self.symbols:
            print(f"📊 {symbol}...", end='')
            result = self.scan_symbol(symbol)
            
            if result:
                if result['score'] >= 60:
                    print(colored(f" 🚀 STRONG BUY ({result['score']})", 'green', attrs=['bold']))
                    self.buy_signals.append(result)
                elif result['score'] >= 40:
                    print(colored(f" ✅ BUY ({result['score']})", 'green'))
                    self.buy_signals.append(result)
                elif result['score'] >= 25:
                    print(colored(f" ⚡ Watch ({result['score']})", 'yellow'))
                    results.append(result)
                else:
                    print(" ⏸️ No signal")
            else:
                print(" ❌ No data")
        
        # Display results
        print("\n" + "="*60)
        print(colored("🎯 SCAN RESULTS", 'yellow', attrs=['bold']))
        print("="*60)
        
        if self.buy_signals:
            print(colored(f"\n🚀 BUY SIGNALS ({len(self.buy_signals)}):", 'green', attrs=['bold']))
            
            # Sort by score
            self.buy_signals.sort(key=lambda x: x['score'], reverse=True)
            
            for i, signal in enumerate(self.buy_signals, 1):
                s = signal['symbol']
                price = signal['price']
                score = signal['score']
                triggers = ', '.join(signal['triggers'][:3])
                change = signal['change']
                
                print(f"\n{i}. {s} @ ${price:.2f}")
                print(f"   Score: {score}/100")
                print(f"   Daily: {change:+.1f}%")
                print(f"   Triggers: {triggers}")
                
                if score >= 80:
                    print(colored("   🔥 EXTREME OPPORTUNITY!", 'red', attrs=['bold']))
                elif score >= 60:
                    print(colored("   ⚡ STRONG BUY", 'yellow'))
                else:
                    print(colored("   ✅ Good entry", 'green'))
            
            # Top pick
            top = self.buy_signals[0]
            print("\n" + "="*60)
            print(colored("🎯 TOP PICK:", 'red', attrs=['bold']))
            print(colored(f"{top['symbol']} @ ${top['price']:.2f}", 'red', attrs=['bold']))
            print(colored(f"Score: {top['score']}/100", 'red'))
            
        else:
            print(colored("\n⏸️ NO BUY SIGNALS", 'blue'))
            print("Market in consolidation - check again later")
        
        print(f"\n⏰ Scan completed: {datetime.now().strftime('%H:%M:%S')}")
        print("🔥 AEGS hunting for goldmine opportunities!")

def main():
    scanner = QuickAEGSScanner()
    scanner.run_scan()

if __name__ == "__main__":
    main()