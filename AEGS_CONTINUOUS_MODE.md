# 🔄 AEGS Continuous Mode Documentation

## Overview
The AEGS Swarm now supports flexible continuous operation modes to discover goldmine opportunities 24/7.

## Running Modes

### 1. Single Run (Default)
Run one discovery cycle and exit:
```bash
python run_aegs_swarm.py
```

### 2. Continuous with Interval
Run continuously with a specified interval between cycles:
```bash
# Run every 4 hours (default)
python run_aegs_swarm.py --continuous

# Run every 2 hours
python run_aegs_swarm.py --continuous --interval 2

# Run every 30 minutes (0.5 hours)
python run_aegs_swarm.py --continuous --interval 0.5

# Run every 6 hours
python run_aegs_swarm.py --continuous --interval 6
```

### 3. Continuous with Daily Schedule
Run once per day at a specific time:
```bash
# Run daily at 6:00 AM
python run_aegs_swarm.py --continuous --schedule --time 06:00

# Run daily at 9:00 PM
python run_aegs_swarm.py --continuous --schedule --time 21:00
```

## Additional Options

### Test Mode
Limit the number of candidates for faster testing:
```bash
python run_aegs_swarm.py --continuous --test --interval 0.25
```

### Use Validated Symbols
Use pre-validated active symbols:
```bash
python run_aegs_swarm.py --continuous --use-validated --interval 4
```

## What Happens in Continuous Mode

1. **Discovery Phase**: AI agents search for new volatile candidates
2. **Backtest Phase**: Tests candidates with AEGS strategy
3. **Analysis Phase**: Ranks results and identifies goldmines
4. **Alert Phase**: Notifies when goldmines are found (>1000% excess return)
5. **Wait Phase**: Shows countdown to next cycle
6. **Repeat**: Automatically starts next cycle

## Features

- **Error Recovery**: If a cycle fails, waits 5 minutes and retries
- **Progress Updates**: Shows remaining time every 30 minutes
- **Graceful Shutdown**: Press Ctrl+C to stop cleanly
- **History Tracking**: Avoids retesting the same symbols
- **Automatic Registry**: Updates goldmine registry automatically

## Monitoring Output

During continuous mode, you'll see:
```
🔥💎 AEGS SWARM CYCLE #1 STARTING 💎🔥
Time: 2025-12-02 14:30:00
================================================================================

📚 Backtest History:
   Total symbols tested: 45
   Tested today: 12
   Tested this month: 45

📡 PHASE 1: DISCOVERY
✅ Discovered 20 candidates

🧪 PHASE 2: BACKTESTING
[Progress bars for backtesting]

📊 PHASE 3: ANALYSIS
💎 Found 2 goldmines!

⏰ Waiting 4 hours until next discovery cycle...
Next run: 2025-12-02 18:30:00
   210 minutes remaining...
```

## Recommended Settings

### For Active Trading
```bash
# Run every 2 hours during market hours
python run_aegs_swarm.py --continuous --interval 2
```

### For Daily Monitoring
```bash
# Run once per day before market open
python run_aegs_swarm.py --continuous --schedule --time 08:30
```

### For Testing/Development
```bash
# Run every 15 minutes with limited candidates
python run_aegs_swarm.py --continuous --interval 0.25 --test
```

### For Production 24/7
```bash
# Run every 4 hours with validated symbols
python run_aegs_swarm.py --continuous --interval 4 --use-validated
```

## Tips

1. **Resource Usage**: Each cycle can take 20-60 minutes depending on candidates
2. **API Limits**: Be mindful of Yahoo Finance rate limits
3. **Disk Space**: Results are saved to JSON files, monitor disk usage
4. **Background Running**: Use `nohup` or `screen` for server deployment:
   ```bash
   nohup python run_aegs_swarm.py --continuous --interval 4 > aegs_swarm.log 2>&1 &
   ```

## Stopping the Swarm

- **Interactive**: Press Ctrl+C
- **Background Process**: `kill -SIGINT <pid>`
- **Emergency Stop**: `kill -9 <pid>` (not recommended)

The swarm will complete the current cycle before shutting down gracefully.