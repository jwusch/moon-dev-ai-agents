"""
🔥💎 AEGS LIVE MARKET SCANNER 💎🔥
Real-time scanning for immediate buy signals using Alpha Ensemble Goldmine Strategy

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored
from AEGS_complete_strategy import AlphaEnsembleGoldmineStrategy

class AEGSLiveScanner:
    """
    Scan markets in real-time for AEGS buy signals
    """
    
    def __init__(self):
        # High-priority goldmine symbols to scan
        self.priority_symbols = {
            'Proven Goldmines': ['WULF', 'EQT', 'NOK', 'WKHS'],
            'Crypto Mining': ['MARA', 'RIOT', 'CLSK', 'CORZ', 'CIFR', 'BTDR'],
            'Biotech Rockets': ['SAVA', 'BIIB', 'VKTX', 'BLUE', 'EDIT', 'NTLA', 'CRSP'],
            'Meme Cycles': ['BB', 'GME', 'AMC', 'KOSS', 'CLOV', 'SOFI'],
            'Energy Volatility': ['FANG', 'DVN', 'MRO', 'OXY', 'SWN', 'AR'],
            'SPAC Recovery': ['LCID', 'RIVN', 'NKLA', 'GOEV', 'RIDE', 'HYLN'],
            'Inverse/Defense': ['SH', 'SQQQ', 'TZA', 'SPXU', 'PSQ'],
            'Leveraged ETFs': ['TNA', 'LABU', 'NUGT', 'GUSH', 'FNGU', 'NAIL']
        }
        
        self.all_symbols = []
        for symbols in self.priority_symbols.values():
            self.all_symbols.extend(symbols)
        self.all_symbols = list(set(self.all_symbols))  # Remove duplicates
        
        self.buy_signals = []
        self.near_signals = []
        
    def scan_all_symbols(self):
        """Scan all symbols for immediate buy signals"""
        
        print(colored("🔥💎 AEGS LIVE MARKET SCANNER 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        print(f"Scanning {len(self.all_symbols)} goldmine symbols for immediate buy signals...")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for category, symbols in self.priority_symbols.items():
            print(f"\n📊 Scanning {category}...")
            
            for symbol in symbols:
                try:
                    self._scan_symbol(symbol, category)
                except Exception as e:
                    print(f"   ❌ Error scanning {symbol}: {str(e)[:50]}...")
                    continue
        
        # Display results
        self._display_scan_results()
        
    def _scan_symbol(self, symbol: str, category: str):
        """Scan individual symbol for signals"""
        
        print(f"   🔍 {symbol}...", end='', flush=True)
        
        # Initialize strategy
        strategy = AlphaEnsembleGoldmineStrategy(symbol)
        
        # Get recent data (last 200 days for indicators)
        df = yf.download(symbol, period='200d', progress=False)
        
        if len(df) < 100:
            print(" ❌ Insufficient data")
            return
        
        # Calculate indicators
        df = strategy.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate signal strength
        signal_strength = 0
        signals_triggered = []
        
        # Check RSI
        if pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                signal_strength += 30
                signals_triggered.append(f"RSI={latest['RSI']:.1f}")
            elif latest['RSI'] < 35:
                signal_strength += 15
                signals_triggered.append(f"RSI={latest['RSI']:.1f}")
        
        # Check Bollinger Bands
        if pd.notna(latest['BB_Position']):
            if latest['BB_Position'] < 0:  # Below lower band
                signal_strength += 30
                signals_triggered.append(f"BB_Below")
            elif latest['BB_Position'] < 0.1:
                signal_strength += 20
                signals_triggered.append(f"BB_Low={latest['BB_Position']:.2f}")
        
        # Check Distance from SMA
        if pd.notna(latest['Distance_Pct']):
            if latest['Distance_Pct'] < -5:
                signal_strength += 25
                signals_triggered.append(f"Dist={latest['Distance_Pct']:.1f}%")
            elif latest['Distance_Pct'] < -3:
                signal_strength += 15
                signals_triggered.append(f"Dist={latest['Distance_Pct']:.1f}%")
        
        # Check Volume expansion
        if pd.notna(latest['Volume_Ratio']):
            if latest['Volume_Ratio'] > 2.0 and latest['Daily_Change_Pct'] < -2:
                signal_strength += 30
                signals_triggered.append(f"VolSpike={latest['Volume_Ratio']:.1f}x")
            elif latest['Volume_Ratio'] > 1.5:
                signal_strength += 10
                signals_triggered.append(f"Vol={latest['Volume_Ratio']:.1f}x")
        
        # Check extreme moves
        if pd.notna(latest['Daily_Change_Pct']):
            if latest['Daily_Change_Pct'] < -10:
                signal_strength += 30
                signals_triggered.append(f"Crash={latest['Daily_Change_Pct']:.1f}%")
            elif latest['Daily_Change_Pct'] < -5:
                signal_strength += 15
                signals_triggered.append(f"Drop={latest['Daily_Change_Pct']:.1f}%")
        
        # Check MACD
        if pd.notna(latest['MACD']) and pd.notna(prev['MACD']):
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signal_strength += 20
                signals_triggered.append("MACD_Cross")
        
        # Determine signal status
        if signal_strength >= 70:
            # STRONG BUY SIGNAL
            self.buy_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'volume': latest['Volume'],
                'daily_change': latest.get('Daily_Change_Pct', 0)
            })
            print(colored(f" 🚀 STRONG BUY! Score: {signal_strength}/100", 'green', attrs=['bold']))
            
        elif signal_strength >= 50:
            # MODERATE BUY SIGNAL
            self.buy_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'volume': latest['Volume'],
                'daily_change': latest.get('Daily_Change_Pct', 0)
            })
            print(colored(f" ✅ BUY SIGNAL! Score: {signal_strength}/100", 'green'))
            
        elif signal_strength >= 30:
            # NEAR SIGNAL
            self.near_signals.append({
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'signal_strength': signal_strength,
                'signals': signals_triggered,
                'volume': latest['Volume'],
                'daily_change': latest.get('Daily_Change_Pct', 0)
            })
            print(colored(f" ⚡ Near signal: {signal_strength}/100", 'yellow'))
            
        else:
            print(" ⏸️  No signal")
    
    def _display_scan_results(self):
        """Display comprehensive scan results"""
        
        print("\n" + "=" * 80)
        print(colored("📊 AEGS SCAN RESULTS - IMMEDIATE OPPORTUNITIES", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if self.buy_signals:
            # Sort by signal strength
            self.buy_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            print(colored(f"\n🚀 IMMEDIATE BUY SIGNALS ({len(self.buy_signals)}):", 'green', attrs=['bold']))
            print("=" * 80)
            
            for i, signal in enumerate(self.buy_signals, 1):
                symbol = signal['symbol']
                strength = signal['signal_strength']
                price = signal['price']
                category = signal['category']
                triggered = ', '.join(signal['signals'])
                daily_change = signal['daily_change']
                
                # Determine urgency
                if strength >= 80:
                    urgency = "🔥 EXTREME"
                    urgency_color = 'red'
                elif strength >= 70:
                    urgency = "⚡ STRONG"
                    urgency_color = 'yellow'
                else:
                    urgency = "✅ MODERATE"
                    urgency_color = 'green'
                
                print(colored(f"\n#{i}. {symbol} - {urgency} BUY OPPORTUNITY", urgency_color, attrs=['bold']))
                print(f"   Category: {category}")
                print(f"   Current Price: ${price:.2f}")
                print(f"   Signal Strength: {strength}/100")
                print(f"   Triggers: {triggered}")
                print(f"   Daily Change: {daily_change:.1f}%")
                
                # Position sizing recommendation
                if strength >= 80:
                    print(colored("   💰 Recommended Position: LARGE (3-5% of portfolio)", 'green'))
                    print(colored("   🎯 Target Profit: +20-50% (extreme volatility expected)", 'cyan'))
                elif strength >= 70:
                    print(colored("   💰 Recommended Position: MEDIUM (2-3% of portfolio)", 'green'))
                    print(colored("   🎯 Target Profit: +15-30%", 'cyan'))
                else:
                    print(colored("   💰 Recommended Position: SMALL (1-2% of portfolio)", 'green'))
                    print(colored("   🎯 Target Profit: +10-20%", 'cyan'))
                
                print(colored("   ⏱️  Entry Timing: IMMEDIATE (signal is active NOW)", 'yellow'))
            
            # Best opportunity
            best = self.buy_signals[0]
            print("\n" + "=" * 80)
            print(colored("🎯 TOP PRIORITY BUY:", 'red', attrs=['bold']))
            print(colored(f"   {best['symbol']} @ ${best['price']:.2f}", 'red', attrs=['bold']))
            print(colored(f"   Signal Strength: {best['signal_strength']}/100", 'red'))
            print(colored(f"   Category: {best['category']}", 'red'))
            print(colored("   Action: BUY IMMEDIATELY", 'red', attrs=['bold', 'blink']))
            
        else:
            print(colored("\n⏸️  NO IMMEDIATE BUY SIGNALS", 'blue'))
            print("All goldmine symbols are currently between entry points")
        
        # Near signals
        if self.near_signals:
            print(colored(f"\n⚡ SYMBOLS APPROACHING BUY ZONE ({len(self.near_signals)}):", 'yellow'))
            
            self.near_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            for signal in self.near_signals[:5]:  # Top 5
                symbol = signal['symbol']
                strength = signal['signal_strength']
                price = signal['price']
                triggered = ', '.join(signal['signals'][:2])  # First 2 triggers
                
                print(f"   {symbol}: ${price:.2f} | Score: {strength}/100 | {triggered}")
            
            print(colored("\n   💡 Monitor these for entry in next 1-3 days", 'yellow'))
        
        # Market summary
        print("\n" + "=" * 80)
        print(colored("📈 MARKET SUMMARY:", 'cyan'))
        print(f"   Total Symbols Scanned: {len(self.all_symbols)}")
        print(f"   Active Buy Signals: {len(self.buy_signals)}")
        print(f"   Near Buy Signals: {len(self.near_signals)}")
        print(f"   No Signal: {len(self.all_symbols) - len(self.buy_signals) - len(self.near_signals)}")
        
        if self.buy_signals:
            categories_with_signals = {}
            for signal in self.buy_signals:
                cat = signal['category']
                categories_with_signals[cat] = categories_with_signals.get(cat, 0) + 1
            
            print(colored("\n   🔥 Hottest Categories:", 'cyan'))
            for cat, count in sorted(categories_with_signals.items(), key=lambda x: x[1], reverse=True):
                print(f"      {cat}: {count} signals")
        
        print("\n" + "=" * 80)
        print(colored("💡 TRADING RECOMMENDATIONS:", 'white', attrs=['bold']))
        
        if self.buy_signals:
            print("   1. IMMEDIATE ACTION: Deploy capital on top buy signals")
            print("   2. Start with highest signal strength symbols")
            print("   3. Use recommended position sizes based on strength")
            print("   4. Set stop losses at -20% for risk management")
            print("   5. Take profits at +20-50% depending on volatility")
        else:
            print("   1. WAIT: No immediate opportunities present")
            print("   2. Monitor near-signal symbols for entry")
            print("   3. Check again in 4-6 hours for new signals")
            print("   4. Market may be in consolidation phase")
        
        print("\n🔥 AEGS is actively hunting for goldmine opportunities!")
        print(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Run AEGS live market scan"""
    
    print("🚀 Starting AEGS Live Market Scanner...")
    print("🎯 Hunting for immediate buy signals across all goldmine categories...")
    
    scanner = AEGSLiveScanner()
    scanner.scan_all_symbols()
    
    print("\n✅ AEGS scan complete!")
    print("💎 Deploy capital on identified opportunities for maximum returns!")


if __name__ == "__main__":
    main()