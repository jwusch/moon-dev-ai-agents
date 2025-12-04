# 🌋 AEGS Enhanced Volatility Scanner Commands

## Quick Reference

### Main Commands

```bash
# Quick volatility scan (25 stocks, ~30 seconds)
python aegs_volatility_scanner.py --quick

# Full volatility scan (50 stocks, ~60 seconds) 
python aegs_volatility_scanner.py

# Limit number of stocks to analyze
python aegs_volatility_scanner.py --limit 20
```

### Integration with Daily Monitors

The volatility scanner is integrated into your daily monitors but **disabled by default** to save time. To enable:

1. Edit `daily_monitors.py`
2. Change `'enabled': False` to `'enabled': True` for the volatility scanner
3. Run `python daily_monitors.py`

### What It Does

1. **🌋 PHASE 1: VOLATILE STOCK DISCOVERY**
   - Scans 123+ stocks from major indexes (S&P 500, NASDAQ, Russell 2000)
   - Analyzes daily volatility, volume spikes, price ranges
   - Filters by price ($1-$500), volume (>500K), market cap (>$50M)
   - Ranks by volatility score (combines daily move, volume, range)

2. **🎯 PHASE 2: AEGS TECHNICAL ANALYSIS**
   - Runs AEGS-style analysis on top volatile stocks
   - Calculates RSI, Bollinger Bands, technical signals
   - Scores stocks on oversold conditions + volatility
   - Minimum 40/100 signal strength to qualify

3. **💎 PHASE 3: TOP OPPORTUNITIES**
   - Combines AEGS signals with volatility metrics
   - Provides position size recommendations
   - Categories: Strong AEGS + High Vol, Extreme Volatility plays
   - Saves results to JSON file for review

### Stock Universe Coverage

- **S&P 500**: AAPL, TSLA, NVDA, AMD, META, GOOGL, AMZN, etc.
- **NASDAQ 100**: QQQ, ARKK, PLTR, SOFI, HOOD, RIVN, LCID, etc.
- **Biotech**: EDIT, CRSP, NTLA, SAVA, BIIB, MRNA, BNTX, etc.
- **Fintech**: PYPL, SQ, COIN, AFRM, UPST, HOOD, etc.
- **EV Stocks**: TSLA, RIVN, LCID, NIO, XPEV, LI, etc.
- **Meme Stocks**: GME, AMC, BB, KOSS, CLOV, etc.
- **Penny Volatility**: NOK, SIRI, SNDL, etc.

### Output Files

Results are automatically saved as:
```
aegs_volatility_scan_YYYYMMDD_HHMMSS.json
```

Contains:
- Top volatile stocks discovered
- AEGS signals found
- Trading opportunities with reasoning
- Position size recommendations

### Performance Tips

- Use `--quick` for daily monitoring (faster)
- Full scan provides more comprehensive coverage
- Best run during market hours for real-time data
- Some older symbols may show errors (normal - they get filtered out)

## Example Output

```
🎯 TOP VOLATILITY PICKS FOR AEGS ANALYSIS
1. EXPR: $2.24 | +6.2% | MODERATE Vol | Score: 33
2. PDD: $54.01 | +4.7% | MODERATE Vol | Score: 32
3. BB: $96.62 | +0.5% | LOW Vol | Score: 24

🎯 AEGS SIGNALS FOUND: 1
💎 TOP OPPORTUNITIES: Strong AEGS signal on volatile stock
```

This automatically discovers the most volatile stocks each day and runs them through AEGS analysis - perfect for finding new trading opportunities beyond the standard goldmine list! 🚀