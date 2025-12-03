#!/usr/bin/env python3
"""
Test OTC symbol formats for potentially delisted stocks
"""

import yfinance as yf
import pandas as pd

def test_otc_symbols():
    symbols_to_check = ['DCFC', 'NAKD', 'GNUS', 'CEI', 'EXPR', 'PROG', 'APRN']
    otc_suffixes = ['', '.PK', '.OB', '.OTC']

    print("🔍 Testing OTC Symbol Formats:")
    print("=" * 50)

    for symbol in symbols_to_check:
        print(f'\n--- {symbol} ---')
        found_data = False
        
        for suffix in otc_suffixes:
            try:
                ticker_symbol = f'{symbol}{suffix}'
                ticker = yf.Ticker(ticker_symbol)
                
                # Try to get recent data
                hist = ticker.history(period="5d")
                info = ticker.info
                
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    company_name = info.get('longName', 'N/A')
                    print(f'✅ {ticker_symbol}: ${latest_price:.3f} - {company_name}')
                    found_data = True
                    break
                elif 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    price = info['regularMarketPrice']
                    company_name = info.get('longName', 'N/A')
                    print(f'✅ {ticker_symbol}: ${price} - {company_name}')
                    found_data = True
                    break
                    
            except Exception as e:
                continue
        
        if not found_data:
            print(f'❌ {symbol}: No data found in any format (likely delisted)')

if __name__ == "__main__":
    test_otc_symbols()