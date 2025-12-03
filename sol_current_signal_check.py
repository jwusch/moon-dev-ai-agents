"""
🔥💎 SOL-USD CURRENT SIGNAL CHECK 💎🔥
Check if we should be in a SOL position right now based on AEGS
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored

class SOLSignalChecker:
    def __init__(self):
        self.symbol = 'SOL-USD'
        
    def check_current_signals(self):
        """Check if SOL has active buy signals"""
        
        print(colored("🔥💎 SOL-USD AEGS SIGNAL CHECK 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        print(f"Checking: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Download recent data (need 200 days for indicators)
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period='200d')
        
        if len(df) < 100:
            print("❌ Insufficient data")
            return
        
        # Calculate all indicators
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        print(f"\n📊 CURRENT SOL-USD STATS:")
        print(f"   Price: ${latest['Close']:.2f}")
        print(f"   24h Change: {(latest['Close'] - prev['Close']) / prev['Close'] * 100:+.1f}%")
        
        # Check each signal type
        signals_active = []
        
        # 1. RSI Signal
        if pd.notna(latest['RSI']):
            print(f"\n📈 RSI: {latest['RSI']:.1f}")
            if latest['RSI'] < 30:
                signals_active.append(("RSI_OVERSOLD", "STRONG BUY"))
                print(colored("   🚀 RSI OVERSOLD - STRONG BUY SIGNAL!", 'green', attrs=['bold']))
            elif latest['RSI'] < 35:
                signals_active.append(("RSI_LOW", "BUY"))
                print(colored("   ⚡ RSI LOW - Buy signal approaching", 'yellow'))
            else:
                print("   ⏸️  RSI neutral")
        
        # 2. Bollinger Band Signal
        if pd.notna(latest['BB_Position']):
            print(f"\n📊 Bollinger Position: {latest['BB_Position']:.2f}")
            print(f"   Price: ${latest['Close']:.2f}")
            print(f"   Lower Band: ${latest['BB_Lower']:.2f}")
            print(f"   Upper Band: ${latest['BB_Upper']:.2f}")
            
            if latest['BB_Position'] < 0:
                signals_active.append(("BB_BELOW", "STRONG BUY"))
                print(colored("   🚀 BELOW LOWER BAND - STRONG BUY SIGNAL!", 'green', attrs=['bold']))
            elif latest['BB_Position'] < 0.2:
                signals_active.append(("BB_LOW", "BUY"))
                print(colored("   ⚡ Near lower band - Buy zone", 'yellow'))
            else:
                print("   ⏸️  Within bands")
        
        # 3. Volume Expansion Signal
        if pd.notna(latest['Volume_Ratio']) and pd.notna(latest['Daily_Change_Pct']):
            print(f"\n📊 Volume: {latest['Volume_Ratio']:.1f}x average")
            if latest['Volume_Ratio'] > 2.0 and latest['Daily_Change_Pct'] < -2:
                signals_active.append(("VOL_PANIC", "STRONG BUY"))
                print(colored("   🚀 VOLUME SPIKE + DECLINE - PANIC SELLING BUY!", 'green', attrs=['bold']))
            elif latest['Volume_Ratio'] > 1.5:
                print(colored("   ⚡ High volume detected", 'yellow'))
            else:
                print("   ⏸️  Normal volume")
        
        # 4. MACD Signal
        if pd.notna(latest['MACD']) and pd.notna(prev['MACD']):
            print(f"\n📊 MACD: {latest['MACD']:.2f}")
            print(f"   Signal: {latest['MACD_Signal']:.2f}")
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals_active.append(("MACD_CROSS", "BUY"))
                print(colored("   🚀 MACD BULLISH CROSSOVER - BUY SIGNAL!", 'green', attrs=['bold']))
            elif latest['MACD'] > latest['MACD_Signal']:
                print("   ⚡ MACD bullish")
            else:
                print("   ⏸️  MACD bearish")
        
        # 5. Extreme Reversion Signal
        if pd.notna(latest['Daily_Change_Pct']):
            print(f"\n📊 Daily Change: {latest['Daily_Change_Pct']:.1f}%")
            if latest['Daily_Change_Pct'] < -10:
                signals_active.append(("EXTREME_DROP", "STRONG BUY"))
                print(colored("   🔥💎 EXTREME DROP - GOLDMINE BUY SIGNAL!", 'red', attrs=['bold', 'blink']))
            elif latest['Daily_Change_Pct'] < -5:
                signals_active.append(("BIG_DROP", "BUY"))
                print(colored("   🚀 BIG DROP - Strong buy signal!", 'green', attrs=['bold']))
            
            # Check 3-day decline
            if len(df) >= 3:
                three_day_return = (latest['Close'] - df['Close'].iloc[-4]) / df['Close'].iloc[-4] * 100
                if three_day_return < -15:
                    signals_active.append(("3DAY_CRASH", "STRONG BUY"))
                    print(colored(f"   🔥 3-DAY CRASH: {three_day_return:.1f}% - BUY NOW!", 'red', attrs=['bold']))
        
        # FINAL VERDICT
        print("\n" + "=" * 80)
        print(colored("🎯 AEGS SIGNAL SUMMARY:", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if len(signals_active) >= 2:
            print(colored("\n🔥💎 STRONG BUY SIGNAL ACTIVE! 💎🔥", 'green', attrs=['bold', 'blink']))
            print(f"   {len(signals_active)} signals triggered:")
            for signal, strength in signals_active:
                print(f"   ✅ {signal}: {strength}")
            print(colored("\n   💰 ACTION: BUY SOL IMMEDIATELY!", 'green', attrs=['bold']))
            print(colored("   🎯 Based on backtesting: +39,496% excess return potential", 'cyan'))
            
        elif len(signals_active) == 1:
            print(colored("\n⚡ MODERATE BUY SIGNAL", 'yellow', attrs=['bold']))
            print(f"   1 signal triggered: {signals_active[0][0]}")
            print("   💡 Consider buying on additional confirmation")
            
        else:
            print(colored("\n⏸️  NO ACTIVE BUY SIGNALS", 'blue'))
            print("   SOL is not in oversold territory")
            print("   Wait for better entry opportunity")
        
        # Additional context
        print(f"\n📊 RECENT PERFORMANCE:")
        # Calculate various timeframe returns
        if len(df) >= 5:
            five_day_return = (latest['Close'] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
            print(f"   5-day return: {five_day_return:+.1f}%")
        
        if len(df) >= 20:
            twenty_day_return = (latest['Close'] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100
            print(f"   20-day return: {twenty_day_return:+.1f}%")
        
        # Show key levels
        print(f"\n📍 KEY LEVELS:")
        print(f"   20-day SMA: ${latest['SMA_20']:.2f}")
        print(f"   Distance from SMA: {latest['Distance_Pct']:.1f}%")
        
        # Position recommendation
        print(f"\n💡 POSITION SIZING:")
        if len(signals_active) >= 2:
            print("   Recommended: 3-5% of portfolio (strong signals)")
        elif len(signals_active) == 1:
            print("   Recommended: 1-2% of portfolio (single signal)")
        else:
            print("   Recommended: 0% - wait for signals")
        
        return signals_active
    
    def calculate_indicators(self, df):
        """Calculate all AEGS indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Price changes
        df['Daily_Change_Pct'] = df['Close'].pct_change() * 100
        df['Distance_Pct'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        
        return df


def main():
    """Check current SOL signals"""
    
    checker = SOLSignalChecker()
    signals = checker.check_current_signals()
    
    print("\n✅ SOL signal check complete!")
    
    if signals and len(signals) >= 2:
        print(colored("\n🚨 ALERT: Strong buy opportunity on SOL! 🚨", 'red', attrs=['bold', 'blink']))


if __name__ == "__main__":
    main()