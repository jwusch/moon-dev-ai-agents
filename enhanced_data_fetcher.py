"""
🔍💎 ENHANCED DATA FETCHER WITH FALLBACK SOURCES 💎🔍
Handles delisted stocks, ticker changes, and alternative data sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataFetcher:
    """
    Enhanced data fetcher with multiple fallback options
    """
    
    def __init__(self):
        # Known ticker mappings for common changes
        self.ticker_mappings = {
            # Mergers and acquisitions
            'SPRT': 'GREE',  # Support.com merged with Greenidge
            'RIDE': 'RIDEQ',  # Lordstown Motors bankruptcy
            'MULN': 'MULN',  # Try with .PK for pink sheets
            'FFIE': 'FFIEQ',  # Faraday Future bankruptcy
            'APRN': 'APRNQ',  # Blue Apron bankruptcy
            'EXPR': 'EXPR',  # Express bankruptcy
            'REV': 'REVVQ',  # Revlon bankruptcy
            'BBBY': 'BBBYQ',  # Bed Bath & Beyond bankruptcy
            'GNUS': 'GNUS',  # Genius Brands
            'PROG': 'PRVB',  # Progenity changed to Provention Bio
            'VIEW': 'VIEW',  # View Inc
            'BODY': 'BODY',  # The Beachbody Company
            'RDBX': 'RDBX',  # Redbox (delisted after acquisition)
            
            # SPAC ticker changes
            'DWAC': 'DJT',   # Digital World Acquisition Corp became Trump Media
            'CFVI': 'RUM',   # CF Acquisition Corp VI became Rumble
            
            # Crypto tickers
            'GBTC': 'GBTC',  # Grayscale Bitcoin Trust
            'ETHE': 'ETHE',  # Grayscale Ethereum Trust
            
            # Add .PK for pink sheets
            'CEI': 'CEIW',   # Camber Energy warrants
            'XELA': 'XELA',  # Exela Technologies
        }
        
        # Alternative suffixes to try
        self.suffixes = ['', 'Q', '.PK', 'W', '.OB', '.QB']
        
    def fetch_data(self, symbol, period='max', interval='1d'):
        """
        Fetch data with multiple fallback options
        """
        
        # 1. Try original symbol
        data = self._try_yfinance(symbol, period, interval)
        if data is not None and len(data) > 0:
            return data, symbol
        
        # 2. Check known mappings
        if symbol in self.ticker_mappings:
            mapped_symbol = self.ticker_mappings[symbol]
            print(f"   🔄 Trying mapped ticker: {mapped_symbol}")
            data = self._try_yfinance(mapped_symbol, period, interval)
            if data is not None and len(data) > 0:
                return data, mapped_symbol
        
        # 3. Try with different suffixes
        for suffix in self.suffixes:
            if suffix and not symbol.endswith(suffix):
                test_symbol = f"{symbol}{suffix}"
                data = self._try_yfinance(test_symbol, period, interval)
                if data is not None and len(data) > 0:
                    print(f"   ✅ Found as: {test_symbol}")
                    return data, test_symbol
        
        # 4. Try searching for the symbol
        search_result = self._search_symbol(symbol)
        if search_result:
            print(f"   🔍 Search found: {search_result}")
            data = self._try_yfinance(search_result, period, interval)
            if data is not None and len(data) > 0:
                return data, search_result
        
        # 5. Check if it's a crypto and needs USD suffix
        crypto_symbol = f"{symbol}-USD"
        data = self._try_yfinance(crypto_symbol, period, interval)
        if data is not None and len(data) > 0:
            print(f"   🪙 Found as crypto: {crypto_symbol}")
            return data, crypto_symbol
        
        # 6. Last resort - check alternative data sources
        data = self._check_alternative_sources(symbol)
        if data is not None:
            return data, f"{symbol}_ALT"
        
        return None, None
    
    def _try_yfinance(self, symbol, period, interval):
        """
        Try to fetch data from yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if len(data) > 0:
                return data
        except:
            pass
        return None
    
    def _search_symbol(self, symbol):
        """
        Search for symbol using yfinance search
        """
        try:
            # Try to get info and extract possible symbols
            search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', [])
                
                # Look for exact or close matches
                for quote in quotes:
                    if quote.get('symbol', '').upper().startswith(symbol.upper()):
                        return quote['symbol']
                    
                # If no exact match, return first result
                if quotes:
                    return quotes[0]['symbol']
        except:
            pass
        
        return None
    
    def _check_alternative_sources(self, symbol):
        """
        Check alternative data sources (placeholder for additional APIs)
        """
        # This could be expanded to include:
        # - Alpha Vantage API
        # - IEX Cloud
        # - Polygon.io
        # - CoinGecko for crypto
        # - etc.
        
        print(f"   ⚠️  {symbol}: No data found in any source")
        return None
    
    def get_active_symbols(self, symbol_list):
        """
        Filter list to only active, tradeable symbols
        """
        active_symbols = []
        delisted_symbols = []
        mapped_symbols = {}
        
        print(colored("🔍 Checking symbol availability...", 'yellow'))
        print("=" * 60)
        
        for symbol in symbol_list:
            data, found_symbol = self.fetch_data(symbol, period='5d')
            
            if data is not None:
                current_price = data['Close'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                
                active_symbols.append({
                    'original': symbol,
                    'active': found_symbol,
                    'price': current_price,
                    'volume': volume,
                    'changed': symbol != found_symbol
                })
                
                if symbol != found_symbol:
                    mapped_symbols[symbol] = found_symbol
                    
            else:
                delisted_symbols.append(symbol)
        
        # Report results
        print(colored(f"\n✅ Active symbols: {len(active_symbols)}", 'green'))
        for sym in active_symbols[:10]:  # Show first 10
            if sym['changed']:
                print(f"   {sym['original']} → {sym['active']} (${sym['price']:.2f})")
            else:
                print(f"   {sym['original']} (${sym['price']:.2f})")
        
        if mapped_symbols:
            print(colored(f"\n🔄 Ticker changes found: {len(mapped_symbols)}", 'yellow'))
            for old, new in list(mapped_symbols.items())[:5]:
                print(f"   {old} → {new}")
        
        if delisted_symbols:
            print(colored(f"\n❌ Delisted/Not found: {len(delisted_symbols)}", 'red'))
            print(f"   {', '.join(delisted_symbols[:10])}")
            if len(delisted_symbols) > 10:
                print(f"   ... and {len(delisted_symbols) - 10} more")
        
        return active_symbols, delisted_symbols, mapped_symbols


def update_symbol_universe():
    """
    Update and validate symbol universe
    """
    
    print(colored("🔍💎 VALIDATING SYMBOL UNIVERSE 💎🔍", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Test symbols that commonly have issues
    problem_symbols = [
        'MULN', 'FFIE', 'REV', 'APRN', 'EXPR', 'RIDE', 'GNUS',
        'PROG', 'VIEW', 'BODY', 'BBBY', 'RDBX', 'WEBR', 'DWAC',
        'CFVI', 'SPRT', 'IRNT', 'OPAD', 'TMC', 'CEI', 'XELA'
    ]
    
    fetcher = EnhancedDataFetcher()
    
    # Check each symbol
    active, delisted, mapped = fetcher.get_active_symbols(problem_symbols)
    
    # Create updated mapping file
    mapping_data = {
        'last_updated': datetime.now().isoformat(),
        'ticker_mappings': fetcher.ticker_mappings,
        'active_symbols': [s['active'] for s in active],
        'delisted_symbols': delisted,
        'discovered_mappings': mapped
    }
    
    # Save mapping data
    with open('symbol_mappings.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\n💾 Symbol mappings saved to symbol_mappings.json")
    
    # Create helper function for AEGS
    helper_code = f'''"""
Enhanced symbol validation for AEGS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from enhanced_data_fetcher import EnhancedDataFetcher

# Known good symbols after validation
ACTIVE_SYMBOLS = {json.dumps([s['active'] for s in active], indent=2)}

# Delisted symbols to avoid
DELISTED_SYMBOLS = {json.dumps(delisted, indent=2)}

# Ticker mappings
TICKER_MAPPINGS = {json.dumps(mapped, indent=2)}

def get_valid_symbol(symbol):
    """Get valid trading symbol"""
    if symbol in TICKER_MAPPINGS:
        return TICKER_MAPPINGS[symbol]
    if symbol in DELISTED_SYMBOLS:
        return None
    return symbol

def filter_valid_symbols(symbols):
    """Filter to only valid symbols"""
    valid = []
    for symbol in symbols:
        valid_symbol = get_valid_symbol(symbol)
        if valid_symbol:
            valid.append(valid_symbol)
    return valid
'''
    
    with open('validated_symbols.py', 'w') as f:
        f.write(helper_code)
    
    print("💾 Created validated_symbols.py helper")
    
    return active, delisted, mapped


def demo_enhanced_fetcher():
    """
    Demo the enhanced fetcher
    """
    
    print(colored("🔍 ENHANCED DATA FETCHER DEMO", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Test problematic symbols
    test_symbols = ['MULN', 'DWAC', 'BBBY', 'SPRT', 'PROG', 'BTC']
    
    fetcher = EnhancedDataFetcher()
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        data, found_symbol = fetcher.fetch_data(symbol, period='5d')
        
        if data is not None:
            current_price = data['Close'].iloc[-1]
            print(colored(f"   ✅ Found as: {found_symbol}", 'green'))
            print(f"   Price: ${current_price:.2f}")
            print(f"   Data points: {len(data)}")
        else:
            print(colored(f"   ❌ Not found in any source", 'red'))


if __name__ == "__main__":
    # Run validation
    update_symbol_universe()
    
    print("\n" + "=" * 60)
    
    # Run demo
    demo_enhanced_fetcher()