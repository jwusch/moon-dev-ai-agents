#!/usr/bin/env python3
"""
🌍 EXCHANGE SYMBOL FETCHER 🌍
Get comprehensive symbol lists from all major exchanges
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
import yfinance as yf

class ExchangeSymbolFetcher:
    """Fetch symbols from multiple exchanges"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_nasdaq_symbols(self):
        """Get all NASDAQ symbols from official API"""
        print("🔍 Fetching NASDAQ symbols...")
        
        try:
            # NASDAQ API endpoint
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '10000',
                'exchange': 'NASDAQ'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if 'data' in data and 'rows' in data['data']:
                symbols = [row['symbol'] for row in data['data']['rows']]
                print(f"✅ Found {len(symbols)} NASDAQ symbols")
                return symbols
            else:
                print("❌ Failed to parse NASDAQ data")
                return []
                
        except Exception as e:
            print(f"❌ NASDAQ fetch error: {e}")
            return []
    
    def get_nyse_symbols(self):
        """Get all NYSE symbols from official API"""
        print("🔍 Fetching NYSE symbols...")
        
        try:
            # NYSE via NASDAQ API
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '10000',
                'exchange': 'NYSE'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if 'data' in data and 'rows' in data['data']:
                symbols = [row['symbol'] for row in data['data']['rows']]
                print(f"✅ Found {len(symbols)} NYSE symbols")
                return symbols
            else:
                print("❌ Failed to parse NYSE data")
                return []
                
        except Exception as e:
            print(f"❌ NYSE fetch error: {e}")
            return []
    
    def get_amex_symbols(self):
        """Get all AMEX symbols"""
        print("🔍 Fetching AMEX symbols...")
        
        try:
            # AMEX via NASDAQ API
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '10000',
                'exchange': 'AMEX'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if 'data' in data and 'rows' in data['data']:
                symbols = [row['symbol'] for row in data['data']['rows']]
                print(f"✅ Found {len(symbols)} AMEX symbols")
                return symbols
            else:
                print("❌ Failed to parse AMEX data")
                return []
                
        except Exception as e:
            print(f"❌ AMEX fetch error: {e}")
            return []
    
    def get_sp500_symbols(self):
        """Get S&P 500 symbols (high-quality subset)"""
        print("🔍 Fetching S&P 500 symbols...")
        
        try:
            # Wikipedia S&P 500 list
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (some have periods, etc.)
            symbols = [symbol.replace('.', '-') for symbol in symbols]
            
            print(f"✅ Found {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            print(f"❌ S&P 500 fetch error: {e}")
            return []
    
    def get_russell_2000_symbols(self):
        """Get Russell 2000 symbols (small cap focus)"""
        print("🔍 Fetching Russell 2000 symbols...")
        
        try:
            # Using iShares Russell 2000 ETF (IWM) holdings
            # This is a proxy for Russell 2000 components
            ticker = yf.Ticker("IWM")
            holdings = ticker.funds_data
            
            # Note: This is simplified - in practice, you'd need 
            # a data provider for full Russell 2000 list
            print("⚠️  Russell 2000: Using proxy method (limited)")
            return []
            
        except Exception as e:
            print(f"❌ Russell 2000 fetch error: {e}")
            return []
    
    def filter_valid_symbols(self, symbols, min_price=1.0, max_symbols=None):
        """Filter symbols by basic criteria"""
        print(f"🔍 Filtering {len(symbols)} symbols...")
        
        valid_symbols = []
        invalid_symbols = []
        
        # Sample first to avoid overwhelming yfinance
        if max_symbols and len(symbols) > max_symbols:
            print(f"⚠️  Limiting to first {max_symbols} symbols for validation")
            symbols = symbols[:max_symbols]
        
        for i, symbol in enumerate(symbols, 1):
            if i % 100 == 0:
                print(f"   Validated {i}/{len(symbols)}...")
            
            try:
                # Quick validation
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Basic filters
                if info.get('regularMarketPrice', 0) >= min_price:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
                    
                # Rate limiting
                time.sleep(0.05)
                
            except:
                invalid_symbols.append(symbol)
                continue
        
        print(f"✅ Valid: {len(valid_symbols)}, Invalid: {len(invalid_symbols)}")
        return valid_symbols, invalid_symbols
    
    def get_all_exchange_symbols(self, validate=False, max_per_exchange=1000):
        """Get symbols from all major exchanges"""
        print("🌍 FETCHING ALL EXCHANGE SYMBOLS 🌍")
        print("=" * 50)
        
        all_symbols = {}
        
        # Fetch from each exchange
        exchanges = {
            'NASDAQ': self.get_nasdaq_symbols,
            'NYSE': self.get_nyse_symbols, 
            'AMEX': self.get_amex_symbols,
            'SP500': self.get_sp500_symbols,
        }
        
        for exchange_name, fetch_func in exchanges.items():
            symbols = fetch_func()
            
            if validate and symbols:
                symbols, _ = self.filter_valid_symbols(symbols, max_symbols=max_per_exchange)
            
            all_symbols[exchange_name] = symbols
            
            # Rate limiting between exchanges
            time.sleep(1)
        
        # Summary
        total_symbols = sum(len(symbols) for symbols in all_symbols.values())
        print(f"\n🎯 EXCHANGE SUMMARY:")
        for exchange, symbols in all_symbols.items():
            print(f"   {exchange:<8}: {len(symbols):>5} symbols")
        print(f"   {'TOTAL':<8}: {total_symbols:>5} symbols")
        
        return all_symbols
    
    def save_symbols_to_file(self, all_symbols, filename=None):
        """Save symbols to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"exchange_symbols_{timestamp}.json"
        
        # Add metadata
        output = {
            'fetch_date': datetime.now().isoformat(),
            'exchanges': all_symbols,
            'summary': {
                exchange: len(symbols) for exchange, symbols in all_symbols.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        total = sum(len(symbols) for symbols in all_symbols.values())
        print(f"\n💾 Saved {total} symbols to: {filename}")
        return filename

def main():
    """Demo comprehensive symbol fetching"""
    print("🚀 COMPREHENSIVE EXCHANGE SYMBOL FETCHER")
    print("=" * 60)
    
    fetcher = ExchangeSymbolFetcher()
    
    # Get symbols from all exchanges
    all_symbols = fetcher.get_all_exchange_symbols(
        validate=False,  # Skip validation for speed
        max_per_exchange=None
    )
    
    # Save to file
    filename = fetcher.save_symbols_to_file(all_symbols)
    
    # Show some examples
    print(f"\n📋 SAMPLE SYMBOLS BY EXCHANGE:")
    for exchange, symbols in all_symbols.items():
        if symbols:
            sample = symbols[:10]
            print(f"   {exchange}: {', '.join(sample)}...")
    
    print(f"\n✅ Ready for AEGS scanning across all exchanges!")
    print(f"💡 Use this symbol list for comprehensive market coverage")

if __name__ == "__main__":
    main()