#!/usr/bin/env python3
"""
Test symbols for data availability
"""
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Check symbols that failed in the backtest
test_symbols = ['SHY', 'TBT', 'HYG', 'UNG', 'DBA', 'PDBC', 'VIXY', 'XLB', 'REZ', 'PSQ', 'VTI', 'RSX']

print("Testing symbol data availability:")
print("=" * 50)

for symbol in test_symbols:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1mo')
        
        if len(hist) < 10:
            print(f'{symbol}: ❌ INSUFFICIENT DATA (only {len(hist)} days)')
        elif info.get('regularMarketPrice', 0) <= 0:
            price = info.get('regularMarketPrice', 'N/A')
            print(f'{symbol}: ❌ INVALID PRICE ({price})')
        else:
            price = info.get('regularMarketPrice', 0)
            print(f'{symbol}: ✅ OK ({len(hist)} days, ${price:.2f})')
    except Exception as e:
        print(f'{symbol}: ❌ ERROR - {str(e)[:50]}')

print("\n" + "=" * 50)
print("Checking global quality monitor symbols...")

# Test current symbols in global quality monitor that might be problematic
monitor_symbols = ['VXX', 'SQQQ', 'TQQQ', 'UVXY', 'VIXY', 'ARKK', 'UNG', 'RSX', 'UVIX', 'VXZ', 'VIXM']

for symbol in monitor_symbols:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1mo')
        
        if len(hist) < 10:
            print(f'{symbol}: ❌ INSUFFICIENT DATA (only {len(hist)} days)')
        elif info.get('regularMarketPrice', 0) <= 0:
            price = info.get('regularMarketPrice', 'N/A') 
            print(f'{symbol}: ❌ INVALID PRICE ({price})')
        else:
            price = info.get('regularMarketPrice', 0)
            print(f'{symbol}: ✅ OK ({len(hist)} days, ${price:.2f})')
    except Exception as e:
        print(f'{symbol}: ❌ ERROR - {str(e)[:50]}')