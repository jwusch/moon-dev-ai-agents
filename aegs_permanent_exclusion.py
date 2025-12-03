#!/usr/bin/env python3
"""
AEGS Permanent Exclusion Manager
Prevents repeated retries of symbols with persistent errors
"""

import json
import os
from datetime import datetime
from termcolor import colored

class PermanentExclusionManager:
    def __init__(self, threshold=10):
        """
        Initialize exclusion manager
        
        Args:
            threshold: Number of failures before permanent exclusion
        """
        self.threshold = threshold
        self.invalid_file = 'aegs_invalid_symbols.json'
        self.permanent_file = 'aegs_permanent_exclusions.json'
        self.load_data()
        
    def load_data(self):
        """Load invalid symbols and permanent exclusions"""
        # Load invalid symbols
        if os.path.exists(self.invalid_file):
            with open(self.invalid_file, 'r') as f:
                self.invalid_data = json.load(f)
        else:
            self.invalid_data = {'invalid_symbols': {}, 'metadata': {}}
            
        # Load permanent exclusions
        if os.path.exists(self.permanent_file):
            with open(self.permanent_file, 'r') as f:
                self.permanent_data = json.load(f)
        else:
            self.permanent_data = {
                'excluded_symbols': {},
                'metadata': {
                    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'threshold': self.threshold
                }
            }
    
    def update_permanent_exclusions(self):
        """Move symbols with too many failures to permanent exclusion"""
        moved_count = 0
        
        for symbol, info in self.invalid_data['invalid_symbols'].items():
            fail_count = info.get('fail_count', 0)
            
            if fail_count >= self.threshold and symbol not in self.permanent_data['excluded_symbols']:
                # Move to permanent exclusions
                self.permanent_data['excluded_symbols'][symbol] = {
                    'reason': info.get('reason', 'Unknown'),
                    'fail_count': fail_count,
                    'first_failed': info.get('first_failed', 'Unknown'),
                    'excluded_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'error_type': info.get('error_type', 'Unknown')
                }
                moved_count += 1
                
        if moved_count > 0:
            self.save_permanent_exclusions()
            print(colored(f"✅ Moved {moved_count} symbols to permanent exclusion", 'green'))
            
        return moved_count
    
    def save_permanent_exclusions(self):
        """Save permanent exclusions to file"""
        self.permanent_data['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.permanent_data['metadata']['total_excluded'] = len(self.permanent_data['excluded_symbols'])
        
        with open(self.permanent_file, 'w') as f:
            json.dump(self.permanent_data, f, indent=2)
    
    def analyze_failures(self):
        """Analyze failure patterns"""
        json_errors = {}
        insufficient_data = {}
        other_errors = {}
        
        for symbol, info in self.invalid_data['invalid_symbols'].items():
            reason = info.get('reason', '')
            fail_count = info.get('fail_count', 0)
            
            if 'Extra data: line' in reason:
                json_errors[symbol] = fail_count
            elif 'insufficient data' in reason.lower():
                insufficient_data[symbol] = fail_count
            else:
                other_errors[symbol] = fail_count
                
        return json_errors, insufficient_data, other_errors
    
    def generate_report(self):
        """Generate a comprehensive report"""
        json_errors, insufficient_data, other_errors = self.analyze_failures()
        
        print(colored("\n📊 AEGS FAILURE ANALYSIS REPORT", 'yellow', attrs=['bold']))
        print("="*60)
        
        # JSON parsing errors (the problematic ones)
        print(colored("\n🔴 JSON PARSING ERRORS (Persistent):", 'red'))
        print(f"Total: {len(json_errors)} symbols")
        
        # Show worst offenders
        worst_json = sorted(json_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        for symbol, count in worst_json:
            print(f"  {symbol}: {count} failures")
            
        # Insufficient data errors
        print(colored("\n🟡 INSUFFICIENT DATA:", 'yellow'))
        print(f"Total: {len(insufficient_data)} symbols")
        
        # Other errors
        print(colored("\n🟠 OTHER ERRORS:", 'yellow'))
        print(f"Total: {len(other_errors)} symbols")
        
        # Recommendations
        print(colored("\n💡 RECOMMENDATIONS:", 'cyan'))
        print(f"1. Permanently exclude symbols with {self.threshold}+ failures")
        print(f"2. {len([c for c in json_errors.values() if c >= self.threshold])} symbols ready for permanent exclusion")
        print(f"3. These symbols are wasting computational resources")
        
        # Already excluded
        if os.path.exists(self.permanent_file):
            print(colored(f"\n✅ Already Permanently Excluded: {len(self.permanent_data['excluded_symbols'])} symbols", 'green'))
    
    def clean_invalid_list(self):
        """Remove permanently excluded symbols from invalid list"""
        if not os.path.exists(self.permanent_file):
            return 0
            
        removed = 0
        symbols_to_remove = []
        
        for symbol in self.permanent_data['excluded_symbols']:
            if symbol in self.invalid_data['invalid_symbols']:
                symbols_to_remove.append(symbol)
                removed += 1
                
        for symbol in symbols_to_remove:
            del self.invalid_data['invalid_symbols'][symbol]
            
        if removed > 0:
            # Save cleaned invalid symbols
            self.invalid_data['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.invalid_data['metadata']['total_invalid'] = len(self.invalid_data['invalid_symbols'])
            
            with open(self.invalid_file, 'w') as f:
                json.dump(self.invalid_data, f, indent=2)
                
            print(colored(f"🧹 Cleaned {removed} permanently excluded symbols from invalid list", 'green'))
            
        return removed


def main():
    """Run the permanent exclusion manager"""
    manager = PermanentExclusionManager(threshold=10)
    
    # Generate report
    manager.generate_report()
    
    # Update permanent exclusions
    moved = manager.update_permanent_exclusions()
    
    # Clean invalid list
    cleaned = manager.clean_invalid_list()
    
    print(colored(f"\n✅ Summary: Moved {moved} to permanent, cleaned {cleaned} from invalid list", 'green'))
    
    # Show the symbols that should be excluded from AEGS runs
    if os.path.exists(manager.permanent_file):
        print(colored("\n🚫 Symbols to NEVER retry:", 'red'))
        for symbol in sorted(manager.permanent_data['excluded_symbols'].keys()):
            info = manager.permanent_data['excluded_symbols'][symbol]
            print(f"  {symbol}: {info['fail_count']} failures - {info['reason'][:50]}...")


if __name__ == "__main__":
    main()