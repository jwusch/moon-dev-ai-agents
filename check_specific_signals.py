"""
Check specific goldmine symbols for AEGS signals
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from termcolor import colored

# Check our top goldmines
symbols = ['WULF', 'EQT', 'MARA', 'NOK', 'RIOT']

print(colored("🔍 CHECKING TOP GOLDMINE SYMBOLS", 'cyan', attrs=['bold']))
print("=" * 70)

for symbol in symbols:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='5d')
    
    if len(df) > 0:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Quick RSI calc (simplified)
        gains = []
        losses = []
        for i in range(1, len(df)):
            change = df['Close'].iloc[i] - df['Close'].iloc[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Distance from 5-day average
        sma5 = df['Close'].mean()
        distance_pct = (latest['Close'] - sma5) / sma5 * 100
        
        # Daily change
        daily_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        
        # Volume spike
        avg_vol = df['Volume'].mean()
        vol_ratio = latest['Volume'] / avg_vol if avg_vol > 0 else 1
        
        print(f"\n{symbol}:")
        print(f"  Price: ${latest['Close']:.2f}")
        print(f"  Daily Change: {daily_change:+.1f}%")
        print(f"  RSI: {rsi:.0f}")
        print(f"  Distance from MA: {distance_pct:+.1f}%")
        print(f"  Volume Ratio: {vol_ratio:.1f}x")
        
        # AEGS signal conditions
        if rsi < 30:
            print(colored("  🚀 RSI OVERSOLD - Strong buy signal!", 'green'))
        elif rsi < 40:
            print(colored("  ⚡ RSI approaching oversold", 'yellow'))
        
        if distance_pct < -3:
            print(colored("  🚀 PRICE OVERSOLD - Below moving average!", 'green'))
        
        if vol_ratio > 2.0 and daily_change < -2:
            print(colored("  🚀 VOLUME SPIKE on decline - Panic selling!", 'green'))
        
        if daily_change < -5:
            print(colored("  🚀 BIG DROP - Extreme reversion opportunity!", 'green'))

print("\n" + "=" * 70)
print("💡 AEGS SIGNAL TRIGGERS:")
print("  • RSI < 30 = Strong oversold")
print("  • Distance < -3% = Below average")  
print("  • Volume > 2x + decline = Panic selling")
print("  • Daily drop > 5% = Extreme reversion")