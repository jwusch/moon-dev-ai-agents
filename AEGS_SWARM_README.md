# 🔥💎 AEGS AI SWARM SYSTEM 💎🔥

## Overview

The AEGS AI Swarm is an autonomous system that continuously discovers, tests, and tracks potential "goldmine" trading symbols - those with the potential for >1,000% excess returns using the Alpha Ensemble Goldmine Strategy (AEGS).

## Key Features

- **Automated Discovery**: AI-powered symbol discovery using multiple strategies
- **Parallel Backtesting**: Test multiple symbols simultaneously for efficiency  
- **Auto-Registration**: Successful symbols automatically added to tracking
- **Continuous Monitoring**: 24/7 operation with configurable schedules
- **Real-time Alerts**: Immediate notification when goldmines are discovered

## Quick Start

### 1. Run Once (Manual Discovery)
```bash
python run_aegs_swarm.py
```

### 2. Run Continuously (Automated Daily)
```bash
python run_aegs_swarm.py --continuous --time 06:00
```

### 3. Test Mode (Limited Discovery)
```bash
python run_aegs_swarm.py --test
```

## System Architecture

### Agents

1. **Discovery Agent** (`aegs_discovery_agent.py`)
   - Scans for volatility explosions
   - Detects volume anomalies
   - Finds sector rotation opportunities
   - Uses AI pattern matching

2. **Backtest Agent** (`aegs_backtest_agent.py`)
   - Processes candidates in parallel
   - Runs comprehensive AEGS backtests
   - Auto-registers successful symbols
   - Handles errors gracefully

3. **Swarm Coordinator** (`aegs_swarm_coordinator.py`)
   - Orchestrates the discovery pipeline
   - Manages agent communication
   - Generates alerts and reports
   - Handles scheduling

## Discovery Strategies

### 1. Volatility Explosion Scanner
Finds symbols with sudden volatility increases (2x+ historical volatility)

### 2. Volume Anomaly Detection  
Identifies panic selling opportunities (3x+ volume with price decline)

### 3. Recovery Candidates
Locates beaten-down stocks ready for mean reversion (50%+ drawdown)

### 4. Sector Rotation
Discovers opportunities in underperforming sectors

### 5. AI Pattern Analysis
Uses LLM to find symbols similar to proven goldmines

## Workflow

```
Discovery Agent → Backtest Agent → Auto-Registration → Scanner Integration
     ↓                ↓                   ↓                    ↓
Find candidates  Test with AEGS    Add to registry    Monitor for signals
```

## Output Files

- `aegs_discoveries_YYYYMMDD_HHMMSS.json` - Discovered candidates
- `aegs_backtest_results_YYYYMMDD_HHMMSS.json` - Detailed test results
- `aegs_summary_YYYYMMDD_HHMMSS.json` - Quick reference summary
- `aegs_goldmine_alert_YYYYMMDD_HHMMSS.json` - Goldmine notifications

## Integration with AEGS Scanner

New goldmines are automatically:
1. Added to `aegs_goldmine_registry.json`
2. Included in daily signal scans
3. Monitored for entry opportunities

Check current signals:
```bash
python aegs_enhanced_scanner.py
```

## Configuration

Edit settings in `aegs_swarm_coordinator.py`:
```python
self.config = {
    'discovery_schedule': 'daily',
    'discovery_time': '06:00',
    'max_candidates_per_run': 50,
    'min_excess_for_alert': 1000
}
```

## Performance Expectations

- **Discovery Rate**: 20-50 candidates per run
- **Success Rate**: 10-20% meet goldmine criteria
- **Processing Time**: 20-40 minutes per full cycle
- **Goldmines Found**: 1-5 per week (average)

## Current Proven Goldmines

| Symbol | Excess Return | Category |
|--------|--------------|----------|
| SOL-USD | +39,496% | Cryptocurrency |
| WULF | +13,041% | Crypto Mining |
| NOK | +3,355% | Meme Potential |
| MARA | +1,457% | Crypto Mining |
| EQT | +1,038% | Energy Cycles |

## Troubleshooting

### No candidates found
- Markets may be stable (low volatility)
- Try running during market hours
- Check internet connection for data access

### Backtests failing
- Ensure sufficient historical data (500+ days)
- Check API rate limits
- Verify symbol validity

### Memory issues
- Reduce parallel workers in backtest agent
- Process fewer candidates per run

## Future Enhancements

1. **Web Dashboard** - Real-time monitoring interface
2. **Mobile Alerts** - Push notifications for goldmines
3. **ML Optimization** - Learn from successful patterns
4. **Cloud Deployment** - AWS/GCP integration
5. **Community Sharing** - Goldmine leaderboard

## Safety & Disclaimers

- This is an experimental system for educational purposes
- Past performance does not guarantee future results
- Always perform your own due diligence
- Start with small position sizes
- Use stop losses for risk management

## Support

- Check logs in output JSON files
- Review AEGS_Strategy_Documentation.md
- Ensure all dependencies installed: `pip install -r requirements.txt`

---

**Ready to discover the next millionaire-making opportunity? Run the swarm! 🚀💎**