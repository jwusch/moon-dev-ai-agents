"""
Thread-safe version of Invalid Symbol Tracker
"""

import json
import os
import threading
from datetime import datetime
from termcolor import colored

class InvalidSymbolTracker:
    """Track invalid symbols with thread safety"""
    
    def __init__(self):
        self.invalid_file = 'aegs_invalid_symbols.json'
        self.invalid_symbols = {}
        self._lock = threading.Lock()  # Thread safety
        self.load_invalid_symbols()
    
    def load_invalid_symbols(self):
        """Load invalid symbols from file"""
        if os.path.exists(self.invalid_file):
            try:
                with open(self.invalid_file, 'r') as f:
                    data = json.load(f)
                self.invalid_symbols = data.get('invalid_symbols', {})
            except Exception as e:
                print(f"Error loading invalid symbols: {e}")
                self.invalid_symbols = {}
    
    def is_invalid(self, symbol):
        """Check if a symbol is marked as invalid"""
        with self._lock:
            return symbol in self.invalid_symbols
    
    def add_invalid_symbol(self, symbol, reason, error_type='unknown'):
        """Add a symbol to the invalid list (thread-safe)"""
        with self._lock:
            if symbol not in self.invalid_symbols:
                print(colored(f"🚫 Added {symbol} to invalid symbols: {reason}", 'yellow'))
                self.invalid_symbols[symbol] = {
                    'reason': reason,
                    'first_failed': datetime.now().strftime('%Y-%m-%d'),
                    'error_type': error_type,
                    'fail_count': 1,
                    'last_failed': datetime.now().strftime('%Y-%m-%d')
                }
            else:
                # Update fail count
                self.invalid_symbols[symbol]['fail_count'] += 1
                self.invalid_symbols[symbol]['last_failed'] = datetime.now().strftime('%Y-%m-%d')
                self.invalid_symbols[symbol]['reason'] = reason
            
        # Save after releasing the lock
        self.save_invalid_symbols()
    
    def save_invalid_symbols(self):
        """Save invalid symbols to file (thread-safe)"""
        # Create a copy while holding the lock
        with self._lock:
            invalid_symbols_copy = dict(self.invalid_symbols)
        
        data = {
            'invalid_symbols': invalid_symbols_copy,
            'metadata': {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_invalid': len(invalid_symbols_copy)
            }
        }
        
        try:
            # Write to a temporary file first
            temp_file = self.invalid_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.invalid_file)
            
        except Exception as e:
            print(colored(f"⚠️ Error saving invalid symbols: {e}", 'red'))
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def remove_invalid_symbol(self, symbol):
        """Remove a symbol from the invalid list"""
        with self._lock:
            if symbol in self.invalid_symbols:
                del self.invalid_symbols[symbol]
                print(colored(f"✅ Removed {symbol} from invalid list", 'green'))
        
        self.save_invalid_symbols()
    
    def get_invalid_symbols(self):
        """Get a copy of all invalid symbols"""
        with self._lock:
            return dict(self.invalid_symbols)
