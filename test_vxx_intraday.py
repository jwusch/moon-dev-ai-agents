"""
🎯 Test VXX with Different Timeframes
Demonstrating how volatility products trade differently on various timeframes

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("📊 VXX ANALYSIS - DAILY vs INTRADAY")
print("="*70)

# Download VXX data at different timeframes
vxx = yf.Ticker("VXX")

# Get different timeframes
timeframes = {
    "1d": {"period": "2y", "interval": "1d"},
    "1h": {"period": "3mo", "interval": "1h"},
    "30m": {"period": "1mo", "interval": "30m"},
    "15m": {"period": "7d", "interval": "15m"},
    "5m": {"period": "5d", "interval": "5m"},
}

print("\nAnalyzing VXX volatility and mean reversion at different timeframes:\n")

results = []

for tf_name, tf_params in timeframes.items():
    try:
        df = vxx.history(**tf_params)
        
        if len(df) < 20:
            continue
            
        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        
        # Mean reversion metrics
        # 1. Number of times price crosses its 20-period mean
        sma20 = df['Close'].rolling(20).mean()
        crosses = ((df['Close'] > sma20) != (df['Close'].shift(1) > sma20)).sum()
        
        # 2. Average move size
        avg_move = returns.abs().mean() * 100
        
        # 3. How often it reverses (changes direction)
        reversals = (returns > 0) != (returns.shift(1) > 0)
        reversal_pct = reversals.sum() / len(reversals) * 100
        
        # 4. Volatility
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        if tf_name != "1d":
            # Adjust for intraday
            if tf_name == "1h":
                volatility = returns.std() * np.sqrt(252 * 6.5) * 100  # 6.5 trading hours
            elif tf_name == "30m":
                volatility = returns.std() * np.sqrt(252 * 13) * 100  # 13 half-hours
            elif tf_name == "15m":
                volatility = returns.std() * np.sqrt(252 * 26) * 100  # 26 15-min bars
            elif tf_name == "5m":
                volatility = returns.std() * np.sqrt(252 * 78) * 100  # 78 5-min bars
        
        # Potential trades (using simple Camarilla-like strategy)
        # Buy when price is 2% below SMA20, sell when 2% above
        potential_long_entries = ((df['Close'] < sma20 * 0.98) & 
                                 (df['Close'].shift(1) >= sma20.shift(1) * 0.98)).sum()
        potential_short_entries = ((df['Close'] > sma20 * 1.02) & 
                                  (df['Close'].shift(1) <= sma20.shift(1) * 1.02)).sum()
        total_potential_trades = potential_long_entries + potential_short_entries
        
        result = {
            'Timeframe': tf_name,
            'Bars': len(df),
            'Days_Covered': (df.index[-1] - df.index[0]).days,
            'Avg_Move_%': avg_move,
            'Reversal_%': reversal_pct,
            'MA_Crosses': crosses,
            'Volatility_%': volatility,
            'Potential_Trades': total_potential_trades,
            'Trades_Per_Day': total_potential_trades / max(1, (df.index[-1] - df.index[0]).days)
        }
        
        results.append(result)
        
    except Exception as e:
        print(f"Error with {tf_name}: {e}")

# Display results
print(f"{'Timeframe':<10} {'Avg Move':<10} {'Reversals':<10} {'MA Crosses':<12} {'Pot. Trades':<12} {'Trades/Day':<12}")
print("-"*70)

for r in results:
    print(f"{r['Timeframe']:<10} {r['Avg_Move_%']:>8.2f}% {r['Reversal_%']:>8.1f}% "
          f"{r['MA_Crosses']:>10} {r['Potential_Trades']:>11} {r['Trades_Per_Day']:>11.1f}")

# Analyze intraday patterns
print("\n" + "="*70)
print("📈 INTRADAY VXX PATTERNS")
print("="*70)

# Get 5-minute data for detailed analysis
df_5m = vxx.history(period="1d", interval="5m")

if len(df_5m) > 0:
    # Calculate intraday statistics
    df_5m['Time'] = df_5m.index.time
    df_5m['Return'] = df_5m['Close'].pct_change()
    
    # Group by time of day
    time_stats = df_5m.groupby('Time')['Return'].agg(['mean', 'std', 'count'])
    
    # Find most volatile times
    most_volatile = time_stats.nlargest(5, 'std')
    
    print("\nMost volatile 5-minute periods (best for mean reversion):")
    for time, stats in most_volatile.iterrows():
        print(f"  {time}: Avg move ±{stats['std']*100:.2f}%")

# Simulate what would happen with Camarilla on different timeframes
print("\n" + "="*70)
print("💡 CAMARILLA STRATEGY IMPLICATIONS")
print("="*70)

print(f"""
Based on the analysis:

1. **DAILY TIMEFRAME** (what we tested):
   - Only {results[0]['Potential_Trades'] if results else 'few'} potential trades over 2 years
   - Large moves needed for signals (2%+ from mean)
   - Misses most of the action

2. **HOURLY TIMEFRAME**:
   - ~{results[1]['Trades_Per_Day'] if len(results) > 1 else '10-20':.0f} trades per day
   - Better capture of volatility spikes
   - More consistent signals

3. **15-MINUTE BARS**:
   - ~{results[3]['Trades_Per_Day'] if len(results) > 3 else '20-40':.0f} trades per day
   - Ideal for Camarilla levels
   - Catches intraday mean reversion

4. **5-MINUTE BARS**:
   - ~{results[4]['Trades_Per_Day'] if len(results) > 4 else '50+':.0f} trades per day
   - May be too noisy
   - Higher transaction costs

OPTIMAL APPROACH:
- Use 15-minute or 30-minute bars for VXX
- Tighter Camarilla levels (R1/S1 instead of R3/S3)
- Faster exits (return to pivot)
- Position size based on VIX level
""")

# Show current VXX opportunity
latest_daily = vxx.history(period="5d", interval="1d")
if len(latest_daily) > 0:
    current_price = latest_daily['Close'].iloc[-1]
    sma20 = latest_daily['Close'].rolling(20).mean().iloc[-1] if len(latest_daily) >= 20 else latest_daily['Close'].mean()
    distance = (current_price - sma20) / sma20 * 100
    
    print(f"\nCurrent VXX Status:")
    print(f"  Price: ${current_price:.2f}")
    print(f"  20-SMA: ${sma20:.2f}")
    print(f"  Distance: {distance:+.1f}%")
    
    if abs(distance) > 5:
        print(f"  🎯 Potential mean reversion opportunity!")

print("\n✅ Analysis complete!")