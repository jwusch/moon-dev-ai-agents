#!/usr/bin/env python3
"""
🕰️ Time-Based Pattern Analysis Demo
Demonstrates all time-based fractal indicators working together
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from termcolor import colored

# Import our time-based indicators
from src.fractal_alpha.indicators.volume_structure.volume_bars import VolumeBarsIndicator
from src.fractal_alpha.indicators.time_patterns.intraday_seasonality import IntradaySeasonalityIndicator
from src.fractal_alpha.indicators.time_patterns.renko_bars import RenkoBarsIndicator


def fetch_intraday_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch intraday data for analysis"""
    
    print(f"Fetching {days} days of intraday data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    
    # Get 5-minute data
    data = ticker.history(period=f"{days}d", interval="5m")
    
    if data.empty:
        print(colored("❌ No data retrieved", "red"))
        return pd.DataFrame()
    
    # Clean column names
    data.columns = [col.lower() for col in data.columns]
    
    # Remove any rows with NaN values
    data = data.dropna()
    
    print(f"✅ Retrieved {len(data)} data points")
    
    return data


def analyze_time_patterns(symbol: str = "SPY"):
    """Demonstrate all time-based pattern indicators"""
    
    print(colored(f"\n🕰️ TIME-BASED PATTERN ANALYSIS FOR {symbol}", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Fetch data
    data = fetch_intraday_data(symbol, days=30)
    
    if data.empty:
        print(colored("Unable to proceed without data", "red"))
        return
    
    # 1. Volume-Based Bars Analysis
    print(colored("\n📊 1. VOLUME-BASED BARS (VOLUME CLOCK)", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    volume_bars = VolumeBarsIndicator(
        bar_size=None,  # Auto-calculate
        lookback_bars=50,
        volume_factor=1.5
    )
    
    volume_result = volume_bars.calculate(data, symbol)
    
    print(f"Bar Size: {volume_result.metadata.get('bar_size', 0):,.0f} shares")
    print(f"Total Bars Created: {volume_result.metadata.get('total_bars', 0)}")
    print(f"Compression Ratio: {volume_result.metadata.get('compression_ratio', 0):.1f}:1")
    print(f"Current Trend: {volume_result.metadata.get('current_trend', 'Unknown')}")
    print(f"\nSignal: {colored(volume_result.signal.value, 'green' if 'BUY' in volume_result.signal.value else 'red' if 'SELL' in volume_result.signal.value else 'white')}")
    print(f"Confidence: {volume_result.confidence:.1f}%")
    
    if 'volume_profile' in volume_result.metadata:
        profile = volume_result.metadata['volume_profile']
        print(f"\nVolume Profile:")
        print(f"  High Activity Zone: ${profile.get('high_activity_price', 0):.2f} ({profile.get('high_activity_pct', 0):.1f}% of volume)")
        print(f"  Low Activity Zone: ${profile.get('low_activity_price', 0):.2f} ({profile.get('low_activity_pct', 0):.1f}% of volume)")
    
    # 2. Intraday Seasonality Analysis
    print(colored("\n🌞 2. INTRADAY SEASONALITY PATTERNS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    seasonality = IntradaySeasonalityIndicator(
        lookback_days=20,
        time_buckets=26,  # 15-minute buckets for market hours
        detect_anomalies=True
    )
    
    seasonality_result = seasonality.calculate(data, symbol)
    
    print(f"Opportunity Score: {seasonality_result.metadata.get('opportunity_score', 0):.1f}/100")
    print(f"Patterns Found: {seasonality_result.metadata.get('patterns_found', 0)}")
    print(f"Anomalies Detected: {seasonality_result.metadata.get('anomalies_detected', 0)}")
    
    # Show current time pattern
    if 'current_pattern' in seasonality_result.metadata:
        pattern = seasonality_result.metadata['current_pattern']
        print(f"\nCurrent Time Window ({pattern['time_range']}):")
        print(f"  Expected Return: {pattern['expected_return']*100:.3f}%")
        print(f"  Historical Win Rate: {pattern['historical_win_rate']*100:.1f}%")
        print(f"  Sample Size: {pattern['sample_size']} periods")
    
    # Show best/worst times
    if 'best_times' in seasonality_result.metadata:
        print("\nBest Trading Times:")
        for period in seasonality_result.metadata['best_times'][:3]:
            print(f"  {period['time']}: {period['avg_return']*100:.3f}% return, {period['win_rate']*100:.1f}% win rate")
    
    # Show anomalies
    if 'anomalies' in seasonality_result.metadata:
        print("\nCurrent Anomalies:")
        for anomaly_type, details in seasonality_result.metadata['anomalies'].items():
            print(f"  {anomaly_type.replace('_', ' ').title()}: {details['type']} (Z-score: {details['zscore']:.2f})")
    
    # 3. Renko Bars Analysis
    print(colored("\n🧱 3. RENKO BARS (PRICE-BASED)", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    # ATR-based Renko
    renko = RenkoBarsIndicator(
        brick_size=None,  # Use ATR
        atr_period=14,
        atr_multiplier=1.5,
        reversal_bricks=2,
        detect_patterns=True
    )
    
    renko_result = renko.calculate(data, symbol)
    
    print(f"Brick Size: ${renko_result.metadata.get('brick_size', 0):.2f}")
    print(f"Total Bricks: {renko_result.metadata.get('total_bricks', 0)}")
    print(f"Compression Ratio: {renko_result.metadata.get('compression_ratio', 0):.1f}:1")
    print(f"Current Direction: {colored(renko_result.metadata.get('current_direction', 'Unknown').upper(), 'green' if 'up' in str(renko_result.metadata.get('current_direction', '')).lower() else 'red')}")
    print(f"Trend Strength: {renko_result.metadata.get('trend_strength', 0):.1f}/100")
    
    if 'consecutive_up' in renko_result.metadata:
        print(f"\nConsecutive Movements:")
        print(f"  Up Bricks: {renko_result.metadata['consecutive_up']}")
        print(f"  Down Bricks: {renko_result.metadata['consecutive_down']}")
        print(f"  Up/Down Ratio: {renko_result.metadata.get('up_ratio', 0.5):.1%}")
    
    # Show patterns
    if 'recent_patterns' in renko_result.metadata:
        print("\nPatterns Detected:")
        for pattern in renko_result.metadata['recent_patterns']:
            print(f"  {pattern['type'].replace('_', ' ').title()} ({pattern['strength']}) - {pattern['bricks_ago']} bricks ago")
    
    # Show levels
    if 'resistance_level' in renko_result.metadata:
        print(f"\nKey Levels:")
        print(f"  Resistance: ${renko_result.metadata['resistance_level']:.2f}")
    if 'support_level' in renko_result.metadata:
        print(f"  Support: ${renko_result.metadata['support_level']:.2f}")
    
    print(f"\n{renko_result.metadata.get('recommendation', '')}")
    
    # Combined Analysis
    print(colored("\n🎯 COMBINED TIME PATTERN ANALYSIS", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Calculate consensus
    signals = [
        (volume_result.signal.value, volume_result.confidence),
        (seasonality_result.signal.value, seasonality_result.confidence),
        (renko_result.signal.value, renko_result.confidence)
    ]
    
    buy_confidence = sum(conf for sig, conf in signals if sig == "BUY") / len(signals)
    sell_confidence = sum(conf for sig, conf in signals if sig == "SELL") / len(signals)
    
    if buy_confidence > sell_confidence and buy_confidence > 30:
        consensus = "BUY"
        consensus_confidence = buy_confidence
    elif sell_confidence > buy_confidence and sell_confidence > 30:
        consensus = "SELL"
        consensus_confidence = sell_confidence
    else:
        consensus = "HOLD"
        consensus_confidence = 0
    
    print(f"Consensus Signal: {colored(consensus, 'green' if consensus == 'BUY' else 'red' if consensus == 'SELL' else 'yellow')}")
    print(f"Consensus Confidence: {consensus_confidence:.1f}%")
    
    print("\nIndividual Signals:")
    print(f"  Volume Bars: {colored(volume_result.signal.value, 'white')} ({volume_result.confidence:.1f}%)")
    print(f"  Seasonality: {colored(seasonality_result.signal.value, 'white')} ({seasonality_result.confidence:.1f}%)")
    print(f"  Renko Bars: {colored(renko_result.signal.value, 'white')} ({renko_result.confidence:.1f}%)")
    
    # Key insights
    print(colored("\n💡 KEY TIME PATTERN INSIGHTS", "yellow"))
    print("-" * 60)
    
    insights = []
    
    # Volume insights
    if volume_result.metadata.get('volume_anomaly', False):
        if volume_result.metadata['volume_anomaly']['type'] == 'spike':
            insights.append("📊 Volume spike detected - potential breakout")
        else:
            insights.append("📊 Low volume detected - wait for confirmation")
    
    # Seasonality insights  
    current_hour = datetime.now().hour
    if 9 <= current_hour < 10:
        insights.append("🌞 Opening hour - expect high volatility")
    elif 12 <= current_hour < 13:
        insights.append("🌞 Lunch hour - typically low activity")
    elif 15 <= current_hour < 16:
        insights.append("🌞 Power hour - increased trading activity")
    
    if seasonality_result.metadata.get('anomalies_detected', 0) > 0:
        insights.append("🌞 Unusual patterns detected vs historical norms")
    
    # Renko insights
    if renko_result.metadata.get('consecutive_up', 0) >= 5:
        insights.append("🧱 Extended uptrend in Renko - momentum strong")
    elif renko_result.metadata.get('consecutive_down', 0) >= 5:
        insights.append("🧱 Extended downtrend in Renko - bearish momentum")
    
    if renko_result.metadata.get('bricks_since_reversal', 100) <= 2:
        insights.append("🧱 Recent Renko reversal - new trend starting")
    
    for insight in insights:
        print(f"• {insight}")
    
    # Trading recommendations
    print(colored("\n📈 TRADING RECOMMENDATIONS", "green"))
    print("-" * 60)
    
    if consensus == "BUY" and consensus_confidence > 60:
        print("✅ Strong BUY signal across time patterns")
        print("   - Enter long position with confidence")
        print("   - Use Renko support as stop loss")
    elif consensus == "SELL" and consensus_confidence > 60:
        print("❌ Strong SELL signal across time patterns")
        print("   - Consider short position or exit longs")
        print("   - Use Renko resistance as stop loss")
    else:
        print("⏸️ Mixed signals - wait for clarity")
        print("   - Monitor for pattern alignment")
        print("   - Consider smaller position size")
    
    print("\n🔍 Time Pattern Strategies:")
    print("• Volume Bars: Trade breakouts with volume confirmation")
    print("• Seasonality: Time entries during favorable periods")
    print("• Renko: Follow trend until reversal patterns appear")


if __name__ == "__main__":
    # Analyze SPY by default, but can change to any symbol
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    try:
        analyze_time_patterns(symbol)
    except Exception as e:
        print(colored(f"\n❌ Error: {e}", "red"))
        print("Make sure you have yfinance installed: pip install yfinance")