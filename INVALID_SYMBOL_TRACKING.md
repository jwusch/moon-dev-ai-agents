# 🚫 Invalid Symbol Tracking System

## Overview
The AEGS system now tracks symbols that fail during discovery or backtesting to prevent wasting resources on retrying known-bad symbols.

## How It Works

### 1. Automatic Tracking
Symbols are automatically marked as invalid when:
- **No Data Found**: Symbol appears delisted or has no price data
- **Insufficient Data**: Less than 500 data points for meaningful backtest  
- **Backtest Errors**: Errors during the backtesting process
- **Discovery Failures**: Symbol not found on Yahoo Finance

### 2. Error Categories
- `no_data`: Symbol has no price data (likely delisted)
- `insufficient_data`: Not enough historical data
- `backtest_failed`: Failed during backtest execution
- `backtest_error`: Exception during backtesting
- `not_found`: Symbol not found on exchange
- `data_error`: Other data retrieval errors

### 3. Exclusion Chain
When the discovery agent runs, it excludes:
1. Previously tested symbols (from backtest history)
2. **Invalid symbols (from invalid tracker)** ← NEW!
3. Registered goldmine symbols
4. Commonly tested symbols

## Files

### Core Components
- `src/agents/invalid_symbol_tracker.py` - Main tracking class
- `aegs_invalid_symbols.json` - Persistent storage of invalid symbols

### Integration Points
- `aegs_enhanced_discovery.py` - Excludes invalid symbols
- `src/agents/aegs_discovery_agent.py` - Excludes invalid symbols
- `src/agents/aegs_backtest_agent.py` - Adds failed symbols to tracker

## Usage

### View Invalid Symbols
```bash
# Show summary
python manage_invalid_symbols.py --summary

# Show all invalid symbols
python manage_invalid_symbols.py --show

# Export to file
python manage_invalid_symbols.py --export invalid_report.json
```

### Manage Invalid Symbols
```bash
# Retry a symbol (remove from invalid list)
python manage_invalid_symbols.py --retry MULN

# Clean up old entries (older than 90 days)
python manage_invalid_symbols.py --cleanup 90
```

### Test the System
```bash
# Run test to verify tracking works
python test_invalid_tracking.py

# Run continuous mode - will skip invalid symbols
python run_aegs_swarm.py --continuous --interval 1
```

## Example Invalid Symbol Entry
```json
{
  "BRDS": {
    "reason": "Symbol delisted",
    "first_failed": "2025-12-02",
    "error_type": "no_data",
    "fail_count": 1
  }
}
```

## Benefits

1. **Efficiency**: No wasted API calls on known-bad symbols
2. **Speed**: Faster discovery cycles by skipping invalid symbols
3. **Tracking**: Know why symbols failed and when
4. **Flexibility**: Can manually retry symbols if needed
5. **Persistence**: Invalid symbols tracked across sessions

## Current Statistics

As of the last update:
- Total invalid symbols: 6
- Most common error: `no_data` (delisted symbols)
- Symbols: BRDS, FSR, WISH, SDC, ASTR, RIDE

## Continuous Mode Behavior

When running in continuous mode:
1. First cycle loads invalid symbols from file
2. Any new failures are added to the tracker
3. Subsequent cycles skip all invalid symbols
4. Manual intervention can retry specific symbols

This prevents the "symbol not found" errors you were seeing and ensures the swarm focuses only on viable trading opportunities.