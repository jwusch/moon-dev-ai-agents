#!/usr/bin/env python3
"""
Quick Symbol Validator for AEGS Discovery
Filters out delisted/dead symbols before backtesting
"""

import yfinance as yf
import pandas as pd
from typing import List, Set
import json
from datetime import datetime

class QuickSymbolValidator:
    """Validates symbols are active before backtesting"""
    
    def __init__(self, cache_file='validated_symbols_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load validation cache"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'valid_symbols': {},
                'invalid_symbols': {},
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
    
    def _save_cache(self):
        """Save validation cache"""
        self.cache['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def is_symbol_valid(self, symbol: str) -> bool:
        """Check if symbol has recent trading data"""
        
        # Check cache first
        if symbol in self.cache['valid_symbols']:
            return True
        if symbol in self.cache['invalid_symbols']:
            return False
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get very recent data (5 days)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                # Try getting basic info
                info = ticker.info
                if not info or 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                    self.cache['invalid_symbols'][symbol] = datetime.now().strftime('%Y-%m-%d')
                    return False
            
            # Has recent data - mark as valid
            self.cache['valid_symbols'][symbol] = datetime.now().strftime('%Y-%m-%d')
            return True
            
        except Exception:
            # Any error = invalid
            self.cache['invalid_symbols'][symbol] = datetime.now().strftime('%Y-%m-%d')
            return False
    
    def validate_symbol_list(self, symbols: List[str]) -> tuple[List[str], List[str]]:
        """Validate a list of symbols, return (valid, invalid)"""
        valid = []
        invalid = []
        
        print(f"🔍 Validating {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if self.is_symbol_valid(symbol):
                valid.append(symbol)
                print(f"[{i+1}/{len(symbols)}] ✅ {symbol}: Valid")
            else:
                invalid.append(symbol)
                print(f"[{i+1}/{len(symbols)}] ❌ {symbol}: Invalid/Delisted")
        
        # Save cache after validation
        self._save_cache()
        
        return valid, invalid

def test_validator():
    """Test the validator with known symbols"""
    validator = QuickSymbolValidator()
    
    # Test with mix of valid and invalid symbols
    test_symbols = [
        'AAPL',    # Valid
        'TSLA',    # Valid  
        'DCFC',    # Invalid (delisted)
        'NAKD',    # Invalid (delisted)
        'NVDA',    # Valid
        'GNUS',    # Invalid (delisted)
        'QQQ',     # Valid
        'CEI'      # Invalid (delisted)
    ]
    
    valid, invalid = validator.validate_symbol_list(test_symbols)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Valid: {valid}")
    print(f"   ❌ Invalid: {invalid}")
    print(f"   📈 Success Rate: {len(valid)}/{len(test_symbols)} ({len(valid)/len(test_symbols)*100:.1f}%)")

if __name__ == "__main__":
    test_validator()