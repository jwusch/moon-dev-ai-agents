# 📊 Daily Trading Monitor Guide

## 🚀 Quick Start - Essential Daily Programs

### 1. **All-in-One Daily Monitor Script**
```bash
python daily_monitors.py
```
This runs all essential monitors in sequence.

---

## 🎯 Individual Monitors (Run as Needed)

### 📈 **TLRY Position Monitor** (If holding TLRY)
Monitor your TLRY position with Hurst regime analysis:
```bash
# Single check
python tlry_auto_monitor_hurst.py

# Continuous monitoring (every 5 minutes)
python tlry_auto_monitor_hurst.py --continuous --interval 300
```
**Shows**: Current P&L, exit signals, Hurst regime analysis

### 🔍 **AEGS Live Scanner** 
Find oversold bounce opportunities:
```bash
python aegs_live_scanner.py
```
**Shows**: Real-time buy signals across goldmine symbols

### 🤖 **AEGS Swarm Coordinator**
Run multiple AI agents for comprehensive analysis:
```bash
python src/agents/aegs_swarm_coordinator.py
```
**Shows**: Backtests, discoveries, and multi-agent insights

### 📊 **Data Quality Monitor**
Ensure your data feeds are healthy:
```bash
python src/fractal_alpha/monitoring/data_quality_monitor.py
```
**Shows**: Data quality scores, missing data alerts

### 🏃 **Main Trading System** (Full automation)
```bash
python src/main.py
```
Runs all configured agents in continuous loop

---

## ⏰ Recommended Daily Schedule

### **Pre-Market (8:30-9:30 AM)**
1. Run `daily_monitors.py` for overview
2. Check TLRY position status
3. Review AEGS scanner results
4. Check data quality

### **Market Hours (9:30 AM - 4:00 PM)**
- Keep TLRY monitor running if holding position
- Run AEGS scanner every 2-3 hours
- Check swarm coordinator at lunch for insights

### **After Market (4:00+ PM)**
- Final position review
- Run swarm coordinator for EOD analysis
- Check data quality report

---

## 💡 Pro Tips

### Create Aliases in `.bashrc`:
```bash
alias tlry='python tlry_auto_monitor_hurst.py'
alias aegs='python aegs_live_scanner.py'
alias daily='python daily_monitors.py'
alias swarm='python src/agents/aegs_swarm_coordinator.py'
```

### Run in Background with Logging:
```bash
# Run TLRY monitor in background
nohup python tlry_auto_monitor_hurst.py --continuous > tlry_monitor.log 2>&1 &

# Check the log
tail -f tlry_monitor.log
```

### Set Up Cron Jobs:
```bash
# Edit crontab
crontab -e

# Add morning check at 9:00 AM
0 9 * * 1-5 cd /mnt/c/Users/jwusc/moon-dev-ai-agents && python daily_monitors.py

# AEGS scan every 3 hours during market
0 10,13,15 * * 1-5 cd /mnt/c/Users/jwusc/moon-dev-ai-agents && python aegs_live_scanner.py
```

---

## 🔧 Troubleshooting

### Multi-Column DataFrame Error:
Already fixed in the latest versions

### Module Not Found Errors:
Ensure you're in the conda environment:
```bash
conda activate tflow
```

### API Rate Limits:
Check your `.env` file for valid API keys

---

## 📝 Current Alerts

Based on your latest run:
- **TLRY**: Showing +401% gain - consider taking profits!
- **EDIT**: AEGS showing strong buy signal (75/100)

---

## 🎉 Key Takeaway

Just run this each morning:
```bash
python daily_monitors.py
```

And keep TLRY monitor running during market hours if you have positions!