#!/usr/bin/env python3
"""
📊 TLRY 15-MINUTE CHART VIEWER 📊
Display TLRY's 15-minute OHLCV chart with technical analysis
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

def download_tlry_15min_data():
    """Download TLRY 15-minute data"""
    
    print("📊 DOWNLOADING TLRY 15-MINUTE DATA")
    print("=" * 50)
    
    # Get last 7 days for context
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"⏰ Timeframe: 15-minute intervals")
    
    try:
        ticker = yf.Ticker("TLRY")
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="15m",
            auto_adjust=True,
            prepost=True
        )
        
        if df is not None and not df.empty:
            print(f"✅ TLRY data downloaded: {len(df)} bars")
            print(f"📊 Date range: {df.index[0]} to {df.index[-1]}")
            print(f"💰 Current price: ${df.iloc[-1]['Close']:.2f}")
            
            # Add some basic technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            # Calculate daily change
            df['Price_Change'] = df['Close'].pct_change() * 100
            
            return df
        else:
            print("❌ No TLRY data received")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading TLRY data: {e}")
        return None

def create_tlry_chart(df):
    """Create comprehensive TLRY 15-minute chart"""
    
    print("\n📈 CREATING TLRY 15-MINUTE CHART")
    print("=" * 50)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main price chart
    ax1.plot(df.index, df['Close'], color='#2E8B57', linewidth=2, label='TLRY Close Price')
    ax1.plot(df.index, df['SMA_20'], color='#FF6B35', linewidth=1, alpha=0.8, label='SMA 20')
    
    # Add high/low bands
    ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.1, color='gray', label='High-Low Range')
    
    # Price chart formatting
    ax1.set_title(f'TLRY 15-Minute Chart - Last 7 Days\nCurrent: ${df.iloc[-1]["Close"]:.2f}', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Volume chart
    colors = ['red' if df.iloc[i]['Close'] < df.iloc[i]['Open'] else 'green' 
              for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6, width=0.01)
    ax2.plot(df.index, df['Volume_MA'], color='blue', linewidth=1, label='Volume MA 20')
    
    # Volume chart formatting
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format volume numbers
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    # Format x-axis dates for volume chart
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    chart_filename = f"tlry_15min_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    
    print(f"📊 Chart saved as: {chart_filename}")
    
    # Show the chart
    plt.show()
    
    return chart_filename

def analyze_tlry_data(df):
    """Analyze TLRY's recent 15-minute performance"""
    
    print("\n📊 TLRY 15-MINUTE ANALYSIS")
    print("=" * 50)
    
    # Current stats
    current_price = df.iloc[-1]['Close']
    open_price = df.iloc[0]['Open']
    high_price = df['High'].max()
    low_price = df['Low'].min()
    
    # Calculate period performance
    period_change = ((current_price - open_price) / open_price) * 100
    
    # Volume analysis
    avg_volume = df['Volume'].mean()
    max_volume = df['Volume'].max()
    recent_volume = df['Volume'].tail(4).mean()  # Last hour average
    
    print(f"💰 PRICE ANALYSIS:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Period Open: ${open_price:.2f}")
    print(f"   Period High: ${high_price:.2f}")
    print(f"   Period Low: ${low_price:.2f}")
    print(f"   Period Change: {period_change:+.2f}%")
    
    print(f"\n📊 VOLUME ANALYSIS:")
    print(f"   Average Volume: {avg_volume:,.0f}")
    print(f"   Max Volume: {max_volume:,.0f}")
    print(f"   Recent Hour Volume: {recent_volume:,.0f}")
    
    # Recent price action (last 2 hours = 8 bars)
    recent_bars = df.tail(8)
    recent_high = recent_bars['High'].max()
    recent_low = recent_bars['Low'].min()
    recent_volatility = ((recent_high - recent_low) / recent_low) * 100
    
    print(f"\n⚡ RECENT ACTION (Last 2 Hours):")
    print(f"   Recent High: ${recent_high:.2f}")
    print(f"   Recent Low: ${recent_low:.2f}")
    print(f"   Recent Volatility: {recent_volatility:.2f}%")
    
    # AEGS signal check
    print(f"\n🎯 AEGS SIGNAL CHECK:")
    
    # Simple AEGS-like conditions for latest bar
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Price drop check
    price_drop = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
    
    # Volume surge check
    if avg_volume > 0:
        volume_ratio = latest['Volume'] / avg_volume
    else:
        volume_ratio = 1
    
    # Simple oversold conditions
    is_dropping = price_drop < -2.0  # 2% drop in 15 minutes
    is_volume_surge = volume_ratio > 1.5  # 50% above average
    is_near_low = latest['Close'] <= (recent_low * 1.05)  # Within 5% of recent low
    
    print(f"   15-min Price Drop: {price_drop:+.2f}% ({'✅' if is_dropping else '❌'} <-2%)")
    print(f"   Volume Ratio: {volume_ratio:.1f}x ({'✅' if is_volume_surge else '❌'} >1.5x)")
    print(f"   Near Recent Low: {'✅' if is_near_low else '❌'}")
    
    # AEGS signal summary
    aegs_signals = sum([is_dropping, is_volume_surge, is_near_low])
    
    if aegs_signals >= 2:
        print(f"   🎯 AEGS SIGNAL: ⚡ POSSIBLE OVERSOLD BOUNCE SETUP ({aegs_signals}/3 conditions)")
    elif aegs_signals == 1:
        print(f"   🎯 AEGS SIGNAL: ⏳ WATCH ZONE ({aegs_signals}/3 conditions)")
    else:
        print(f"   🎯 AEGS SIGNAL: ❌ NO CLEAR SIGNAL ({aegs_signals}/3 conditions)")
    
    return {
        'current_price': current_price,
        'period_change': period_change,
        'recent_volatility': recent_volatility,
        'aegs_signals': aegs_signals
    }

def main():
    """Show TLRY 15-minute chart with analysis"""
    
    print("📊🌿 TLRY 15-MINUTE CHART ANALYZER 🌿📊")
    print("=" * 70)
    print("🎯 Downloading and analyzing TLRY's latest 15-minute action")
    
    # Download TLRY data
    df = download_tlry_15min_data()
    
    if df is None or df.empty:
        print("❌ Unable to get TLRY data")
        return
    
    # Analyze the data
    analysis = analyze_tlry_data(df)
    
    # Create and show chart
    chart_file = create_tlry_chart(df)
    
    print(f"\n🌿 TLRY 15-MINUTE CHART COMPLETE!")
    print(f"   📊 Chart: {chart_file}")
    print(f"   💰 Current: ${analysis['current_price']:.2f}")
    print(f"   📈 Period: {analysis['period_change']:+.2f}%")
    print(f"   ⚡ Volatility: {analysis['recent_volatility']:.2f}%")
    print(f"   🎯 AEGS Signals: {analysis['aegs_signals']}/3")

if __name__ == "__main__":
    main()