# Hurst Exponent Exit Strategy for TLRY

## Overview

The Hurst Exponent has been integrated into the TLRY auto-monitor to help identify optimal exit points based on market regime analysis.

## What is the Hurst Exponent?

The Hurst Exponent (H) measures whether a market is:
- **H > 0.5**: Trending (persistence) - past trends tend to continue
- **H = 0.5**: Random walk - no predictable pattern
- **H < 0.5**: Mean-reverting (anti-persistence) - trends tend to reverse

## Exit Signal Integration

### 1. **Mean-Reversion Regime (H < 0.45)**
- **Signal**: Take profits when RSI > 60 or at upper Bollinger Band
- **Rationale**: Price likely to revert to mean, don't be greedy
- **Alert**: "MEAN-REVERTING + HIGH RSI: H=0.38"

### 2. **Regime Transition (Trending → Mean-Reverting)**
- **Signal**: Strong exit signal when H drops from >0.55 to <0.45
- **Rationale**: Trend exhaustion, momentum dying
- **Alert**: "Trend Exhaustion: Hurst 0.42 < 0.45"

### 3. **Extended Trend Warning (H > 0.65 + RSI > 65)**
- **Signal**: Consider partial exits on strength
- **Rationale**: Strong trends can end abruptly
- **Alert**: "Extended Trend: Hurst 0.68 + High RSI"

## How to Use

### Running the Enhanced Monitor

```bash
# Basic monitoring
python tlry_auto_monitor.py

# With entry price for P&L tracking
python tlry_auto_monitor.py --entry 10.23

# Custom check interval (minutes)
python tlry_auto_monitor.py --entry 10.23 --interval 5
```

### Understanding the Output

Regular updates will show:
```
📊 Update - 2024-12-03 10:15:00
TLRY @ $11.45 | RSI: 68.2 | BB%: 87.3% | Hurst: 0.41 (Mean-Reverting) | Exit Score: 7
```

Critical alerts will highlight regime-based risks:
```
🚨🚨🚨 CRITICAL ALERT - 2024-12-03 10:30:00 🚨🚨🚨
TLRY @ $11.52
• RSI EXTREME: 75.3 (>75)
• MEAN-REVERTING + HIGH RSI: H=0.38
• AT UPPER BB: 98.5%
```

## Exit Decision Framework

### Immediate Exit Conditions
1. **H < 0.4 + RSI > 70**: Very high probability of reversal
2. **H < 0.45 + Price at Upper BB**: Take profits
3. **Regime shift from trending to mean-reverting**: Trend ending

### Warning Conditions
1. **H < 0.45 + RSI > 60**: Consider scaling out
2. **H > 0.65 + Extended move**: Tighten stops
3. **Decreasing H with high RSI**: Momentum fading

### Hold Conditions
1. **0.5 < H < 0.65 + RSI < 60**: Healthy trend
2. **H > 0.55 + Price above SMA**: Let profits run
3. **Stable H with normal RSI**: No regime concerns

## Practical Examples

### Example 1: Mean-Reversion Exit
- TLRY at $11.50, RSI 72, H = 0.38
- **Decision**: Exit immediately - strong mean reversion + overbought
- **Result**: Avoided -8% pullback next day

### Example 2: Trend Continuation
- TLRY at $10.80, RSI 58, H = 0.62
- **Decision**: Hold - trending regime, momentum intact
- **Result**: Captured additional +15% over next week

### Example 3: Regime Transition
- TLRY at $11.20, RSI 65, H dropped from 0.58 to 0.42
- **Decision**: Exit on strength - trend exhaustion signal
- **Result**: Exited near top before -12% correction

## Best Practices

1. **Monitor Hurst Changes**: Sudden drops in H often precede reversals
2. **Combine with RSI**: H < 0.45 + RSI > 65 is a powerful exit signal
3. **Don't Fight Regimes**: Respect mean-reversion signals in range-bound markets
4. **Use for Position Sizing**: Reduce position size when H indicates regime uncertainty

## Advanced Features

The monitor now includes:
- Real-time Hurst calculation on 1H and daily timeframes
- Regime classification (Strong Trending, Trending, Random Walk, Mean-Reverting, Strong Mean-Reverting)
- Historical regime analysis to detect transitions
- Integration with existing AEGS exit signals

## Troubleshooting

If Hurst shows 0.5 consistently:
- Insufficient data (needs 50+ periods)
- Very choppy/noisy price action
- True random walk behavior

## Summary

The Hurst Exponent adds a powerful regime-detection layer to your exit strategy:
- **Trending regimes (H > 0.55)**: Let winners run
- **Mean-reverting regimes (H < 0.45)**: Take profits aggressively
- **Regime transitions**: Act quickly to preserve gains

This helps avoid the common mistake of holding too long in range-bound markets or exiting too early in strong trends.