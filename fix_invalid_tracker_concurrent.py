#!/usr/bin/env python3
"""
🔧 Fix Invalid Symbol Tracker Concurrent Modification Error
"""

import json
import os
import shutil
from datetime import datetime
from termcolor import colored

def fix_invalid_tracker():
    """Fix the invalid symbol tracker to handle concurrent modifications"""
    
    print(colored("🔧 Fixing Invalid Symbol Tracker", 'cyan', attrs=['bold']))
    print("="*60)
    
    tracker_file = 'src/agents/invalid_symbol_tracker.py'
    
    # First, backup the current file
    backup_file = tracker_file + '.backup'
    shutil.copy(tracker_file, backup_file)
    print(f"✅ Created backup: {backup_file}")
    
    # Read the current file
    with open(tracker_file, 'r') as f:
        content = f.read()
    
    # Find the save_invalid_symbols method
    print("\n🔍 Locating save_invalid_symbols method...")
    
    # Replace the problematic iteration with a copy
    old_pattern = """def save_invalid_symbols(self):
        \"\"\"Save invalid symbols to file\"\"\"
        data = {
            'invalid_symbols': self.invalid_symbols,
            'metadata': {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_invalid': len(self.invalid_symbols)
            }
        }
        
        with open(self.invalid_file, 'w') as f:
            json.dump(data, f, indent=2)"""
    
    new_pattern = """def save_invalid_symbols(self):
        \"\"\"Save invalid symbols to file\"\"\"
        # Create a copy to avoid concurrent modification errors
        invalid_symbols_copy = dict(self.invalid_symbols)
        
        data = {
            'invalid_symbols': invalid_symbols_copy,
            'metadata': {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_invalid': len(invalid_symbols_copy)
            }
        }
        
        try:
            with open(self.invalid_file, 'w') as f:
                json.dump(data, f, indent=2)
        except RuntimeError as e:
            if "dictionary changed size during iteration" in str(e):
                print(colored(f"⚠️  Concurrent modification detected, retrying...", 'yellow'))
                # Retry with another copy
                self.save_invalid_symbols()
            else:
                raise"""
    
    # Check if the pattern exists
    if "def save_invalid_symbols(self):" in content:
        print("✅ Found save_invalid_symbols method")
        
        # Find the method and replace it
        import re
        
        # Pattern to match the entire method
        method_pattern = r'def save_invalid_symbols\(self\):(.*?)(?=\n    def|\n\n|\Z)'
        
        # Find the method
        match = re.search(method_pattern, content, re.DOTALL)
        if match:
            old_method = match.group(0)
            print(f"📊 Current method length: {len(old_method)} chars")
            
            # Create the new method
            new_method = '''def save_invalid_symbols(self):
        """Save invalid symbols to file"""
        # Create a copy to avoid concurrent modification errors
        invalid_symbols_copy = dict(self.invalid_symbols)
        
        data = {
            'invalid_symbols': invalid_symbols_copy,
            'metadata': {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_invalid': len(invalid_symbols_copy)
            }
        }
        
        try:
            with open(self.invalid_file, 'w') as f:
                json.dump(data, f, indent=2)
        except RuntimeError as e:
            if "dictionary changed size during iteration" in str(e):
                print(colored(f"⚠️  Concurrent modification detected, retrying...", 'yellow'))
                # Retry with another copy
                time.sleep(0.1)  # Small delay before retry
                self.save_invalid_symbols()
            else:
                raise'''
            
            # Replace the method
            new_content = content.replace(old_method, new_method)
            
            # Add time import if not present
            if "import time" not in new_content and "time.sleep" in new_method:
                # Find the imports section
                import_lines = new_content.split('\n')
                for i, line in enumerate(import_lines):
                    if line.startswith('from datetime import'):
                        import_lines.insert(i + 1, 'import time')
                        break
                new_content = '\n'.join(import_lines)
            
            # Write the updated file
            with open(tracker_file, 'w') as f:
                f.write(new_content)
            
            print(colored("✅ Fixed save_invalid_symbols method!", 'green'))
            print("\nChanges made:")
            print("1. Create a copy of invalid_symbols dict before saving")
            print("2. Added try-except to handle concurrent modifications")
            print("3. Added retry logic with small delay")
            
        else:
            print("❌ Could not find method pattern")
    else:
        print("❌ save_invalid_symbols method not found")
    
    # Also check add_invalid_symbol method for thread safety
    print("\n🔍 Checking add_invalid_symbol method...")
    
    # Look for potential race conditions
    if "self.invalid_symbols[symbol] = {" in content:
        print("⚠️  Found potential concurrent modification point")
        print("💡 Recommendation: Use threading.Lock() for thread safety")
        
        # Create a thread-safe version
        create_thread_safe_version(tracker_file)

def create_thread_safe_version(tracker_file):
    """Create a thread-safe version of the tracker"""
    
    print("\n📝 Creating thread-safe version...")
    
    thread_safe_file = tracker_file.replace('.py', '_threadsafe.py')
    
    thread_safe_content = '''"""
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
'''
    
    with open(thread_safe_file, 'w') as f:
        f.write(thread_safe_content)
    
    print(f"✅ Created thread-safe version: {thread_safe_file}")
    print("\nFeatures:")
    print("- Uses threading.Lock() for all operations")
    print("- Atomic file writes with temp file")
    print("- Safe concurrent access from multiple threads")

def main():
    """Run the fix"""
    fix_invalid_tracker()
    
    print(colored("\n✅ Fix complete!", 'green', attrs=['bold']))
    print("\nNext steps:")
    print("1. The tracker should now handle concurrent modifications")
    print("2. Consider using the thread-safe version for production")
    print("3. Monitor for any further concurrent access issues")

if __name__ == "__main__":
    main()