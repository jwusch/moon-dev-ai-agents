"""
🔥💎 AEGS LIVE MARKET SCANNER (FIXED) 💎🔥
Real-time scanning for immediate buy signals using Alpha Ensemble Goldmine Strategy

Author: Claude (Anthropic)
Fixed: DataFrame assignment issue
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored
import sys

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
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with DataFrame fix"""
        
        # Ensure we have a single-level column index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)  # Avoid division by zero
        df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-period, 2 std)
        df.loc[:, 'BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df.loc[:, 'BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df.loc[:, 'BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Avoid division by zero in BB_Position
        bb_width = df['BB_Upper'] - df['BB_Lower']
        df.loc[:, 'BB_Position'] = np.where(
            bb_width != 0,
            (df['Close'] - df['BB_Lower']) / bb_width,
            0
        )
        
        # Moving Averages
        df.loc[:, 'SMA20'] = df['Close'].rolling(20).mean()
        df.loc[:, 'SMA50'] = df['Close'].rolling(50).mean()
        
        # Distance from SMA20
        df.loc[:, 'Distance_Pct'] = np.where(
            df['SMA20'] != 0,
            (df['Close'] - df['SMA20']) / df['SMA20'] * 100,
            0
        )
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df.loc[:, 'MACD'] = ema12 - ema26
        df.loc[:, 'MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df.loc[:, 'MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators
        df.loc[:, 'Volume_MA'] = df['Volume'].rolling(20).mean()
        df.loc[:, 'Volume_Ratio'] = np.where(
            df['Volume_MA'] != 0,
            df['Volume'] / df['Volume_MA'],
            1
        )
        
        # Price change indicators
        df.loc[:, 'Daily_Change_Pct'] = df['Close'].pct_change() * 100
        df.loc[:, '3Day_Change_Pct'] = df['Close'].pct_change(3) * 100
        
        return df
        
    def scan_all_symbols(self):
        """Scan all symbols for immediate buy signals"""
        
        print(colored("🔥💎 AEGS LIVE MARKET SCANNER (FIXED) 💎🔥", 'cyan', attrs=['bold']))
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
        
        # Get recent data (last 200 days for indicators)
        try:
            df = yf.download(symbol, period='200d', progress=False)
            
            if df.empty or len(df) < 100:
                print(" ❌ Insufficient data")
                return
                
        except Exception as e:
            print(f" ❌ Data download error: {str(e)[:30]}")
            return
        
        # Calculate indicators with fix
        df = self.calculate_indicators(df)
        
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
    
    print("🚀 Starting AEGS Live Market Scanner (FIXED)...")
    print("🎯 Hunting for immediate buy signals across all goldmine categories...")
    
    scanner = AEGSLiveScanner()
    scanner.scan_all_symbols()
    
    print("\n✅ AEGS scan complete!")
    print("💎 Deploy capital on identified opportunities for maximum returns!")


if __name__ == "__main__":
    main()