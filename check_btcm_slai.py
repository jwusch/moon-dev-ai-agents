#!/usr/bin/env python3
"""
Check BTCM vs SLAI symbol data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def check_symbol_data():
    # Check both symbols
    btcm = yf.Ticker('BTCM')
    slai = yf.Ticker('SLAI')
    
    print("=" * 60)
    print("SYMBOL DATA COMPARISON: BTCM vs SLAI")
    print("=" * 60)
    
    # Check BTCM info
    print("\nBTCM Company Info:")
    btcm_info = btcm.info
    print(f"  Name: {btcm_info.get('longName', 'N/A')}")
    print(f"  Symbol: {btcm_info.get('symbol', 'N/A')}")
    print(f"  Exchange: {btcm_info.get('exchange', 'N/A')}")
    print(f"  Currency: {btcm_info.get('currency', 'N/A')}")
    print(f"  Market Cap: ${btcm_info.get('marketCap', 0):,}")
    
    # Check SLAI info
    print("\nSLAI Company Info:")
    slai_info = slai.info
    print(f"  Name: {slai_info.get('longName', 'N/A')}")
    print(f"  Symbol: {slai_info.get('symbol', 'N/A')}")
    print(f"  Exchange: {slai_info.get('exchange', 'N/A')}")
    
    # Get historical data
    print("\nBTCM Historical Data:")
    btcm_hist = btcm.history(period='max')
    if not btcm_hist.empty:
        print(f"  Date range: {btcm_hist.index[0].date()} to {btcm_hist.index[-1].date()}")
        print(f"  Total trading days: {len(btcm_hist)}")
        print(f"  Latest close: ${btcm_hist['Close'].iloc[-1]:.2f}")
        print(f"  First close: ${btcm_hist['Close'].iloc[0]:.2f}")
        
        # Check if data is current
        latest_date = btcm_hist.index[-1].date()
        today = datetime.now().date()
        days_old = (today - latest_date).days
        print(f"  Data freshness: {days_old} days old")
        if days_old > 5:
            print("  ⚠️  WARNING: Data appears to be stale!")
    else:
        print("  No historical data available")
    
    print("\nSLAI Historical Data:")
    slai_hist = slai.history(period='max')
    if not slai_hist.empty:
        print(f"  Date range: {slai_hist.index[0].date()} to {slai_hist.index[-1].date()}")
        print(f"  Total trading days: {len(slai_hist)}")
        print(f"  Latest close: ${slai_hist['Close'].iloc[-1]:.2f}")
    else:
        print("  No historical data available")
    
    # Check what BIT Mining trades as
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    if btcm_info.get('longName', '').lower().find('bit mining') >= 0:
        print("❌ BTCM still shows as BIT Mining in yfinance")
        print("   This company changed to SLAI but yfinance hasn't updated")
    elif btcm_info.get('longName', '').lower().find('solai') >= 0:
        print("✅ BTCM now shows as SOLAI Limited in yfinance") 
        print("   The ticker symbol stayed BTCM but company name changed")
    
    if not slai_hist.empty:
        print("\n✅ SLAI has historical data - this might be a different company")
    else:
        print("\n❌ SLAI has no data - symbol may not be active yet")
        
    # Test if we can get data for "BIT Mining"
    print("\nSearching for BIT Mining under other symbols...")
    test_symbols = ['BTCM', 'SLAI', 'BIT', 'BITM']
    for symbol in test_symbols:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get('longName', '')
        if name:
            print(f"  {symbol}: {name}")

if __name__ == "__main__":
    check_symbol_data()