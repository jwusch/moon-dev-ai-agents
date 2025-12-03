# 📊 TLRY Exit Strategy Guide - AEGS Model

## Current Status (Dec 2, 2025)
- **Current Price**: $7.42
- **RSI (1H)**: ~50-60 range
- **Bollinger Band Position**: 83.1% (near upper band)
- **Exit Score**: 3/10 (HOLD - Monitor closely)

## AEGS Strategy Overview for TLRY
The AEGS backtest found TLRY to be a **marginal performer** with only +8.1% excess return over buy & hold. The winning strategy is **Vol_Expansion** which:
- Enters on volatility expansion from low periods
- Exits when volatility regime becomes "High"
- Has a 53.1% win rate with 49 trades over 10.5 years

## 🎯 Key Exit Signals to Watch

### 1. **IMMEDIATE EXIT SIGNALS** (Sell Now)
- [ ] RSI > 70 on 1-hour chart
- [ ] Price touches upper Bollinger Band (>95%)
- [ ] Rapid spike >15% in 4 hours
- [ ] If you're up >30% from entry

### 2. **WARNING SIGNALS** (Prepare to Exit)
- [x] Near upper Bollinger Band (currently 83.1%)
- [ ] RSI > 60 on hourly chart
- [ ] Price >5% above 20-day SMA
- [ ] Volume < 50% of average with RSI > 60
- [x] Momentum weakening (ROC negative)

### 3. **MONITORING SIGNALS** (Watch Closely)
- [ ] Volatility expansion detected
- [ ] Multiple timeframe divergence
- [ ] Support level breakdown

## 📈 Historical AEGS Exit Patterns

From the backtest, TLRY typically exits after:
- **Average hold time**: 45 days
- **Best exits**: When RSI > 70 or at volatility regime changes
- **Warning**: Stock is highly volatile with -57.7% max drawdown

## 🛠️ Tools to Track Your Position

### 1. **One-Time Check**
```bash
python tlry_exit_tracker.py --entry YOUR_ENTRY_PRICE
```

### 2. **Continuous Monitoring**
```bash
# Check every 15 minutes with alerts
python tlry_auto_monitor.py --entry YOUR_ENTRY_PRICE --interval 15
```

### 3. **Quick Status Check**
```bash
python tlry_exit_tracker.py
```

## 📊 AEGS Exit Rules Summary

1. **RSI > 70 (1H)** → Strong exit signal
2. **RSI > 50 + Price > SMA20** → Consider exit
3. **At Upper Bollinger Band** → Take profits
4. **Volume < 50% average + RSI > 60** → Weakness signal
5. **10%+ above SMA20** → Overextended

## ⚠️ Risk Management

Given TLRY's history:
- **Volatility**: Extremely high (Cannabis sector)
- **Max Drawdown**: -57.7% historically
- **Recommendation**: Only 0.5-1% portfolio allocation

## 🎯 Current Recommendation

**HOLD - Monitor Closely**
- Near upper Bollinger Band (83.1%)
- Momentum weakening
- Watch for RSI > 70 or BB% > 95% to exit

## 📱 Setting Up Alerts

For automated alerts, the monitoring script can be extended to send:
- Email alerts
- SMS via Twilio
- Discord/Telegram notifications

Just modify the `send_alert()` function in `tlry_auto_monitor.py`.

## 💡 Pro Tips

1. **Partial Exit Strategy**: Consider selling 50% on first strong signal
2. **Trailing Stop**: Set at -10% from recent high
3. **Time-Based Exit**: AEGS average hold is 45 days
4. **Volatility Exit**: When ATR% exceeds 150% of 5-day average

Remember: TLRY was only a marginal performer in AEGS testing (+8.1% excess return), so don't be greedy - take profits when signals align!