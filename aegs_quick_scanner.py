"""
🔥💎 AEGS QUICK SCANNER - Find Buy Signals NOW! 💎🔥
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored

class AEGSQuickScanner:
    def __init__(self):
        # Focus on proven winners and high volatility
        self.symbols = [
            'WULF', 'EQT', 'NOK', 'WKHS',  # Proven goldmines
            'MARA', 'RIOT', 'CLSK', 'CORZ',  # Crypto mining
            'SAVA', 'BIIB', 'EDIT', 'NTLA',  # Biotech
            'BB', 'GME', 'AMC', 'SOFI',  # Meme cycles
            'SH', 'SQQQ', 'TZA', 'SVXY',  # Inverse/volatility
            'TNA', 'LABU', 'NUGT', 'FNGU',  # Leveraged
            'LCID', 'RIVN', 'NKLA', 'GOEV',  # SPACs
            'FANG', 'DVN', 'OXY', 'AR'  # Energy
        ]
        
    def scan_for_signals(self):
        print(colored("🔥💎 AEGS QUICK SCANNER - REAL-TIME BUY SIGNALS 💎🔥", 'cyan', attrs=['bold']))
        print(f"Scanning {len(self.symbols)} symbols at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        buy_signals = []
        near_signals = []
        
        for symbol in self.symbols:
            try:
                # Get recent data
                df = yf.download(symbol, period='50d', progress=False)
                
                if len(df) < 20:
                    continue
                
                # Simple indicators
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Volume
                df['Volume_MA'] = df['Volume'].rolling(20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
                
                # Daily change
                df['Daily_Change%'] = df['Close'].pct_change() * 100
                
                # Latest values
                latest = df.iloc[-1]
                prev_close = df['Close'].iloc[-2]
                
                # Signal scoring
                score = 0
                triggers = []
                
                # Check conditions
                if latest['RSI'] < 30:
                    score += 35
                    triggers.append(f"RSI={latest['RSI']:.0f}")
                elif latest['RSI'] < 35:
                    score += 20
                    triggers.append(f"RSI={latest['RSI']:.0f}")
                
                if latest['Distance%'] < -5:
                    score += 30
                    triggers.append(f"Oversold={latest['Distance%']:.1f}%")
                elif latest['Distance%'] < -3:
                    score += 20
                    triggers.append(f"Below_SMA={latest['Distance%']:.1f}%")
                
                if latest['Volume_Ratio'] > 2.0:
                    score += 25
                    triggers.append(f"Volume={latest['Volume_Ratio']:.1f}x")
                elif latest['Volume_Ratio'] > 1.5:
                    score += 15
                    triggers.append(f"Vol={latest['Volume_Ratio']:.1f}x")
                
                if latest['Daily_Change%'] < -5:
                    score += 20
                    triggers.append(f"Drop={latest['Daily_Change%']:.1f}%")
                
                # Display results
                if score >= 60:
                    buy_signals.append({
                        'symbol': symbol,
                        'price': latest['Close'],
                        'score': score,
                        'triggers': triggers
                    })
                    print(colored(f"🚀 {symbol}: STRONG BUY @ ${latest['Close']:.2f} | Score: {score}/100", 'green', attrs=['bold']))
                    print(f"   Signals: {', '.join(triggers)}")
                    
                elif score >= 40:
                    near_signals.append({
                        'symbol': symbol,
                        'price': latest['Close'],
                        'score': score,
                        'triggers': triggers
                    })
                    print(colored(f"⚡ {symbol}: Near buy @ ${latest['Close']:.2f} | Score: {score}/100", 'yellow'))
                    print(f"   Signals: {', '.join(triggers)}")
                
            except Exception as e:
                pass
        
        # Summary
        print("\n" + "=" * 80)
        print(colored("📊 SCAN SUMMARY", 'yellow', attrs=['bold']))
        print(f"✅ Buy Signals: {len(buy_signals)}")
        print(f"⚡ Near Signals: {len(near_signals)}")
        
        if buy_signals:
            print(colored("\n🔥 IMMEDIATE ACTION REQUIRED:", 'red', attrs=['bold', 'blink']))
            for sig in buy_signals:
                print(f"   BUY {sig['symbol']} @ ${sig['price']:.2f} (Score: {sig['score']}/100)")
        elif near_signals:
            print(colored("\n💡 WATCH LIST:", 'yellow'))
            for sig in near_signals[:3]:
                print(f"   {sig['symbol']} @ ${sig['price']:.2f} (Score: {sig['score']}/100)")
        else:
            print("\n⏸️  No immediate signals - market is consolidating")

if __name__ == "__main__":
    scanner = AEGSQuickScanner()
    scanner.scan_for_signals()