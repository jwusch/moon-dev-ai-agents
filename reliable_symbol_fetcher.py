#!/usr/bin/env python3
"""
🎯 RELIABLE SYMBOL FETCHER 🎯
Get symbols using reliable methods that don't depend on external APIs
"""

import yfinance as yf
import json
import time
import string
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class ReliableSymbolFetcher:
    """Reliable symbol fetching using yfinance discovery"""
    
    def __init__(self):
        self.discovered_symbols = {
            'NASDAQ': [],
            'NYSE': [],
            'AMEX': [],
            'OTHER': []
        }
    
    def generate_symbol_candidates(self):
        """Generate likely symbol candidates"""
        candidates = []
        
        # 1-4 letter combinations (most common)
        for length in range(1, 5):
            for combo in self._generate_letter_combinations(length):
                candidates.append(combo)
        
        # Common patterns
        common_suffixes = ['X', 'Y', 'Z']
        for suffix in common_suffixes:
            for i in range(1, 4):
                for prefix in string.ascii_uppercase:
                    candidates.append(prefix * i + suffix)
        
        return candidates[:2000]  # Limit to prevent overwhelming
    
    def _generate_letter_combinations(self, length):
        """Generate letter combinations of given length"""
        import itertools
        return [''.join(combo) for combo in itertools.product(string.ascii_uppercase, repeat=length)][:500]
    
    def test_symbol_validity(self, symbol):
        """Test if a symbol is valid and get exchange info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if symbol has valid data
            if 'symbol' in info and info.get('regularMarketPrice'):
                exchange = info.get('exchange', 'OTHER').upper()
                price = info.get('regularMarketPrice', 0)
                
                # Map exchange names
                if 'NASDAQ' in exchange or 'NMS' in exchange:
                    exchange = 'NASDAQ'
                elif 'NYSE' in exchange or 'NYQ' in exchange:
                    exchange = 'NYSE'
                elif 'AMEX' in exchange or 'ASE' in exchange:
                    exchange = 'AMEX'
                else:
                    exchange = 'OTHER'
                
                return {
                    'symbol': symbol,
                    'exchange': exchange,
                    'price': price,
                    'valid': True
                }
            
            return None
            
        except Exception:
            return None
    
    def discover_symbols_by_scanning(self, max_workers=10, max_symbols=1000):
        """Discover symbols by testing common patterns"""
        print(f"🔍 Discovering symbols by pattern scanning...")
        
        candidates = self.generate_symbol_candidates()[:max_symbols]
        print(f"📊 Testing {len(candidates)} symbol candidates...")
        
        valid_symbols = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.test_symbol_validity, symbol): symbol
                for symbol in candidates
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1
                
                if result and result['valid']:
                    valid_symbols.append(result)
                    exchange = result['exchange']
                    symbol = result['symbol']
                    price = result['price']
                    print(f"✅ {symbol} ({exchange}) - ${price:.2f}")
                
                if completed % 50 == 0:
                    print(f"📊 Progress: {completed}/{len(candidates)} tested, {len(valid_symbols)} valid found")
        
        # Organize by exchange
        for result in valid_symbols:
            exchange = result['exchange']
            if exchange in self.discovered_symbols:
                self.discovered_symbols[exchange].append(result['symbol'])
            else:
                self.discovered_symbols['OTHER'].append(result['symbol'])
        
        return self.discovered_symbols
    
    def get_known_major_symbols(self):
        """Get known major symbols from major indices"""
        major_symbols = {
            'NASDAQ': [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'ADBE',
                'CRM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'COST', 'TMUS', 'CMCSA', 'PEP',
                'AMAT', 'INTU', 'ISRG', 'MU', 'BKNG', 'ADP', 'MDLZ', 'GILD', 'REGN', 'VRTX',
                'FISV', 'CSX', 'ATVI', 'PYPL', 'LRCX', 'KLAC', 'MCHP', 'PAYX', 'CTAS', 'FAST'
            ],
            'NYSE': [
                'JPM', 'JNJ', 'WMT', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC',
                'ADBE', 'NFLX', 'CRM', 'ABT', 'TMO', 'CVX', 'LLY', 'ABBV', 'PFE', 'KO',
                'ORCL', 'COST', 'DHR', 'VZ', 'ACN', 'MRK', 'PEP', 'T', 'BMY', 'LIN',
                'WFC', 'PM', 'RTX', 'NEE', 'HON', 'UPS', 'LOW', 'MS', 'IBM', 'CAT'
            ],
            'SP500_TECH': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'ORCL',
                'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO', 'IBM', 'TXN', 'INTU'
            ]
        }
        
        return major_symbols
    
    def expand_symbol_ranges(self, base_symbols):
        """Expand symbol list by testing similar patterns"""
        expanded = []
        
        print(f"🔍 Expanding from {len(base_symbols)} base symbols...")
        
        for symbol in base_symbols:
            # Test variations
            variations = [
                symbol + 'A', symbol + 'B',  # Class variations
                symbol + '-A', symbol + '-B',  # Dash variations
                symbol[:-1] if len(symbol) > 1 else symbol,  # Shorter version
            ]
            
            for variation in variations:
                result = self.test_symbol_validity(variation)
                if result and result['valid']:
                    expanded.append(result)
                    print(f"✅ Found variation: {variation}")
                
                time.sleep(0.1)  # Rate limiting
        
        return expanded
    
    def comprehensive_symbol_discovery(self):
        """Comprehensive symbol discovery using multiple methods"""
        print("🚀 COMPREHENSIVE SYMBOL DISCOVERY")
        print("=" * 50)
        
        all_symbols = {}
        
        # Method 1: Known major symbols
        print("\n1️⃣ Loading known major symbols...")
        major_symbols = self.get_known_major_symbols()
        for exchange, symbols in major_symbols.items():
            all_symbols[exchange] = symbols
            print(f"   {exchange}: {len(symbols)} symbols")
        
        # Method 2: Pattern-based discovery (limited)
        print("\n2️⃣ Pattern-based discovery...")
        discovered = self.discover_symbols_by_scanning(max_symbols=500)
        
        # Merge discoveries
        for exchange, symbols in discovered.items():
            if exchange in all_symbols:
                # Remove duplicates
                new_symbols = set(symbols) - set(all_symbols[exchange])
                all_symbols[exchange].extend(list(new_symbols))
            else:
                all_symbols[exchange] = symbols
        
        # Method 3: Alphabet scanning (like our original approach)
        print("\n3️⃣ Systematic alphabet scanning...")
        alphabet_symbols = self.scan_alphabet_systematically()
        
        for exchange, symbols in alphabet_symbols.items():
            if exchange in all_symbols:
                new_symbols = set(symbols) - set(all_symbols[exchange])
                all_symbols[exchange].extend(list(new_symbols))
            else:
                all_symbols[exchange] = symbols
        
        return all_symbols
    
    def scan_alphabet_systematically(self):
        """Systematic alphabet scanning for symbols"""
        print("🔤 Systematic alphabet scanning...")
        
        symbols_by_exchange = {'NASDAQ': [], 'NYSE': [], 'OTHER': []}
        
        # Test common 3-4 letter combinations
        test_symbols = []
        
        # Major patterns
        for first in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for second in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                # 2-letter
                test_symbols.append(first + second)
                
                # 3-letter with common endings
                for third in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    test_symbols.append(first + second + third)
        
        # Limit to reasonable size
        test_symbols = test_symbols[:1000]
        
        print(f"Testing {len(test_symbols)} systematic combinations...")
        
        valid_count = 0
        for i, symbol in enumerate(test_symbols):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(test_symbols)} ({valid_count} valid found)")
            
            result = self.test_symbol_validity(symbol)
            if result and result['valid']:
                exchange = result['exchange']
                if exchange in symbols_by_exchange:
                    symbols_by_exchange[exchange].append(symbol)
                else:
                    symbols_by_exchange['OTHER'].append(symbol)
                
                valid_count += 1
                print(f"   ✅ {symbol} ({exchange})")
            
            time.sleep(0.05)  # Rate limiting
        
        return symbols_by_exchange

def main():
    """Demo reliable symbol fetching"""
    print("🎯 RELIABLE SYMBOL DISCOVERY SYSTEM")
    print("=" * 60)
    
    fetcher = ReliableSymbolFetcher()
    
    # Comprehensive discovery
    all_symbols = fetcher.comprehensive_symbol_discovery()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reliable_symbols_{timestamp}.json"
    
    output = {
        'discovery_date': datetime.now().isoformat(),
        'method': 'Reliable Multi-Method Discovery',
        'exchanges': all_symbols,
        'summary': {
            exchange: len(symbols) for exchange, symbols in all_symbols.items()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Summary
    total = sum(len(symbols) for symbols in all_symbols.values())
    print(f"\n🎉 DISCOVERY COMPLETE!")
    print(f"📊 RESULTS BY EXCHANGE:")
    for exchange, symbols in all_symbols.items():
        print(f"   {exchange:<10}: {len(symbols):>4} symbols")
    print(f"   {'TOTAL':<10}: {total:>4} symbols")
    
    print(f"\n💾 Saved to: {filename}")
    print(f"🚀 Ready for multi-exchange AEGS scanning!")

if __name__ == "__main__":
    main()