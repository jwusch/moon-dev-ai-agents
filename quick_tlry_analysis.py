#!/usr/bin/env python3
"""
📊 QUICK TLRY 15-MINUTE ANALYSIS 📊
Fast TLRY analysis without chart display
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def analyze_tlry_15min():
    """Quick TLRY 15-minute analysis"""
    
    print("📊🌿 TLRY 15-MINUTE QUICK ANALYSIS 🌿📊")
    print("=" * 60)
    
    try:
        # Download last 3 days for context
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        print(f"📅 Downloading TLRY 15-min data: {start_date.date()} to {end_date.date()}")
        
        ticker = yf.Ticker("TLRY")
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="15m",
            auto_adjust=True,
            prepost=True
        )
        
        if df is None or df.empty:
            print("❌ No TLRY data available")
            return
        
        print(f"✅ TLRY data loaded: {len(df)} bars")
        
        # Current stats
        latest = df.iloc[-1]
        first = df.iloc[0]
        
        current_price = latest['Close']
        period_open = first['Open']
        period_high = df['High'].max()
        period_low = df['Low'].min()
        period_change = ((current_price - period_open) / period_open) * 100
        
        print(f"\n💰 TLRY PRICE ACTION:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Period Open: ${period_open:.2f}")
        print(f"   Period High: ${period_high:.2f}")
        print(f"   Period Low: ${period_low:.2f}")
        print(f"   Period Change: {period_change:+.2f}%")
        
        # Recent action (last 4 bars = 1 hour)
        recent_bars = df.tail(4)
        recent_high = recent_bars['High'].max()
        recent_low = recent_bars['Low'].min()
        recent_volume = recent_bars['Volume'].mean()
        avg_volume = df['Volume'].mean()
        
        print(f"\n⚡ RECENT HOUR ACTION:")
        print(f"   Recent High: ${recent_high:.2f}")
        print(f"   Recent Low: ${recent_low:.2f}")
        print(f"   Recent Volume: {recent_volume:,.0f}")
        print(f"   Avg Volume: {avg_volume:,.0f}")
        
        # AEGS-style analysis
        print(f"\n🎯 AEGS-STYLE ANALYSIS:")
        
        # Last 15-min bar analysis
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Price movement
        price_change_15min = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
        price_change_1hour = ((latest['Close'] - recent_bars.iloc[0]['Close']) / recent_bars.iloc[0]['Close']) * 100
        
        # Volume analysis
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # Position relative to recent range
        range_position = ((latest['Close'] - recent_low) / (recent_high - recent_low)) * 100 if recent_high != recent_low else 50
        
        print(f"   15-min Change: {price_change_15min:+.2f}%")
        print(f"   1-hour Change: {price_change_1hour:+.2f}%")
        print(f"   Volume Ratio: {volume_ratio:.2f}x avg")
        print(f"   Range Position: {range_position:.1f}% (0%=low, 100%=high)")
        
        # AEGS signal indicators
        is_oversold = range_position < 20  # In bottom 20% of recent range
        is_dropping = price_change_15min < -1.0  # Dropping >1% in 15min
        is_volume_surge = volume_ratio > 1.5  # Volume >1.5x average
        
        print(f"\n🔍 AEGS SIGNAL INDICATORS:")
        print(f"   Oversold Position: {'✅' if is_oversold else '❌'} (<20% of range)")
        print(f"   Recent Drop: {'✅' if is_dropping else '❌'} (>1% in 15min)")
        print(f"   Volume Surge: {'✅' if is_volume_surge else '❌'} (>1.5x avg)")
        
        # Signal strength
        signal_count = sum([is_oversold, is_dropping, is_volume_surge])
        
        if signal_count >= 2:
            signal_strength = "🎯 STRONG OVERSOLD SIGNAL"
        elif signal_count == 1:
            signal_strength = "⏳ WEAK SIGNAL - MONITOR"
        else:
            signal_strength = "❌ NO CLEAR SIGNAL"
        
        print(f"\n🎯 AEGS SIGNAL: {signal_strength} ({signal_count}/3 conditions)")
        
        # Show last 8 bars (2 hours) for pattern
        print(f"\n📊 RECENT 15-MIN BARS (Last 2 Hours):")
        print("   Time             Open    High     Low   Close   Volume    Change")
        print("   " + "-" * 70)
        
        for i, (timestamp, row) in enumerate(df.tail(8).iterrows()):
            if i > 0:
                prev_close = df.iloc[df.index.get_loc(timestamp) - 1]['Close']
                change_pct = ((row['Close'] - prev_close) / prev_close) * 100
            else:
                change_pct = 0
            
            time_str = timestamp.strftime('%m/%d %H:%M')
            print(f"   {time_str:<15} ${row['Open']:6.2f} ${row['High']:6.2f} ${row['Low']:6.2f} ${row['Close']:6.2f} {row['Volume']:8,.0f} {change_pct:+5.1f}%")
        
        print(f"\n🌿 TLRY 15-MINUTE ANALYSIS COMPLETE!")
        print(f"   💰 Current: ${current_price:.2f}")
        print(f"   📈 Period: {period_change:+.2f}%")
        print(f"   🎯 AEGS: {signal_count}/3 signals")
        
    except Exception as e:
        print(f"❌ Error analyzing TLRY: {e}")

if __name__ == "__main__":
    analyze_tlry_15min()