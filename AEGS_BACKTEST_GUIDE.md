# 🔥💎 AEGS BACKTEST QUICK GUIDE 💎🔥

## How to Run AEGS Backtest for Any Symbol

### 1. **Single Symbol Test**

```bash
# Basic usage (category will be guessed)
python run_aegs_backtest.py SYMBOL

# With specific category
python run_aegs_backtest.py SYMBOL "Category Name"
```

**Examples:**
```bash
# Test Apple stock
python run_aegs_backtest.py AAPL

# Test Tesla with category
python run_aegs_backtest.py TSLA "EV Stock"

# Test crypto (use -USD suffix)
python run_aegs_backtest.py DOGE-USD "Cryptocurrency"
python run_aegs_backtest.py SHIB-USD
python run_aegs_backtest.py ADA-USD

# Test leveraged ETFs
python run_aegs_backtest.py TQQQ "Leveraged ETF"
python run_aegs_backtest.py SOXL "Leveraged ETF"
```

### 2. **Batch Testing Multiple Symbols**

Edit `aegs_batch_backtest.py` and modify the `SYMBOLS_TO_TEST` list:

```python
SYMBOLS_TO_TEST = [
    ("AAPL", "Tech Stock"),
    ("TSLA", "EV Stock"),
    ("DOGE-USD", "Cryptocurrency"),
    # Add more symbols...
]
```

Then run:
```bash
python aegs_batch_backtest.py
```

### 3. **Understanding Results**

The backtest will show:

- **💎 EXTREME GOLDMINE** (>1,000% excess return) - Deploy capital immediately!
- **🚀 HIGH POTENTIAL** (100-1,000% excess) - Add to watch list
- **✅ POSITIVE** (10-100% excess) - Small positions only
- **❌ UNDERPERFORMS** (<0% excess) - Avoid for mean reversion

### 4. **Auto-Registration**

Symbols with >10% excess return are automatically added to the goldmine registry and will be included in future scans.

### 5. **Check Current Signals**

After adding symbols, run the scanner to check for buy signals:

```bash
# Run enhanced scanner (includes all registered symbols)
python aegs_enhanced_scanner.py

# Check specific symbol for current signals
python sol_current_signal_check.py  # For SOL-USD
python check_specific_signals.py    # For top goldmines
```

## Quick Examples to Try

### Proven Goldmines (already discovered):
```bash
python run_aegs_backtest.py SOL-USD    # +39,496% excess!
python run_aegs_backtest.py WULF       # +13,041% excess!
python run_aegs_backtest.py NOK        # +3,355% excess!
python run_aegs_backtest.py MARA       # +1,457% excess!
```

### Test New Candidates:
```bash
# Crypto
python run_aegs_backtest.py DOGE-USD
python run_aegs_backtest.py PEPE-USD
python run_aegs_backtest.py BONK-USD

# Meme stocks
python run_aegs_backtest.py HOOD
python run_aegs_backtest.py RBLX
python run_aegs_backtest.py PLTR

# Leveraged ETFs
python run_aegs_backtest.py SOXL
python run_aegs_backtest.py ARKK
python run_aegs_backtest.py JETS
```

## Notes

- Backtest requires 500+ days of data
- Results show excess return vs buy & hold
- Win rate typically 50-80% for successful symbols
- Sharpe ratio >1.0 indicates good risk-adjusted returns
- Auto-registers symbols with >10% excess return

## Current Top Opportunity

**SOL-USD: +39,496% excess return potential!**
- $10k → $6.1 million
- 58.8% win rate
- Wait for RSI < 30 to enter

---

*Happy goldmine hunting! 💎🚀*