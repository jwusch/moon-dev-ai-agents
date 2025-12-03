# 📊 TLRY Position Card

## Position Details
- **Shares**: 100
- **Entry Price**: $7.18
- **Entry Date**: December 2, 2025
- **Investment**: $718

## Current Status (Live)
- **Current Price**: $7.70
- **Position Value**: $770
- **Profit/Loss**: +$52 (+7.2%)
- **Recent High**: $10.60
- **Drawdown from High**: -27.4%

## Exit Strategy Signals
| Signal | Status | Value |
|--------|--------|-------|
| RSI (1H) | ✅ Normal | < 70 |
| Bollinger Band | ✅ Safe | Not at upper band |
| Volume | ✅ Normal | Adequate |
| Momentum | ✅ Positive | Holding |
| Distance from SMA20 | ✅ Safe | < 10% |

## 🎯 Action Plan

### Current Recommendation: **HOLD** 
No exit signals present. Position is profitable and healthy.

### Exit Triggers (Sell When):
1. **🔴 RSI > 70** → Immediate exit signal
2. **🟡 RSI > 60 + Low volume** → Prepare to exit
3. **🟡 At Upper Bollinger Band (>95%)** → Take profits
4. **🟢 Profit > 30%** → Consider partial exit (50%)
5. **🔴 Loss > 10%** → Stop loss triggered

### Quick Commands
```bash
# Check current status
python tlry_exit_tracker.py --entry 7.18

# Run continuous monitoring (15-min intervals)
python tlry_auto_monitor.py --entry 7.18 --interval 15

# Run aggressive monitoring (5-min intervals) 
python tlry_auto_monitor.py --entry 7.18 --interval 5
```

### AEGS Model Notes
- TLRY showed only +8.1% excess return in backtesting
- Average hold time: 45 days
- Win rate: 53.1%
- Strategy: Vol_Expansion (exits on high volatility)

### Risk Management
- **Position Size**: 100 shares = ~$770
- **Max Risk**: -10% = -$77
- **Target**: +20-30% = +$144-216
- **Time Stop**: 60 days (AEGS average is 45)

---
*Updated: December 2, 2025 at 9:58 AM*