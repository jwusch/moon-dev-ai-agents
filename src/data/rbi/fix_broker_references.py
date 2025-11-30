#!/usr/bin/env python3
"""
🌙 Moon Dev's Broker Reference Fixer
Fixes incorrect broker references in RBI backtest files
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_broker_references(content: str) -> Tuple[bool, str]:
    """Fix incorrect broker references in backtesting files"""
    
    patterns_to_fix = [
        # Fix self.broker.equity to self.equity
        (r'self\.broker\.equity', r'self.equity'),
        
        # Fix self.broker.balance to self.equity
        (r'self\.broker\.balance', r'self.equity'),
        
        # Fix self.broker.margin to self.equity
        (r'self\.broker\.margin', r'self.equity'),
        
        # Fix self.broker.cash to self.equity
        (r'self\.broker\.cash', r'self.equity'),
        
        # Fix strategy.broker to strategy._broker (if accessing internal)
        (r'strategy\.broker\.', r'strategy._broker.'),
    ]
    
    modified = False
    original_content = content
    
    for pattern, replacement in patterns_to_fix:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    return modified, content

def process_file(filepath: Path) -> Tuple[bool, str]:
    """Process a single file to fix broker references"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if no broker references
        if 'broker' not in content:
            return False, "No broker references found"
        
        # Apply fixes
        modified, fixed_content = fix_broker_references(content)
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, "Fixed broker references"
        
        return False, "No incorrect broker references found"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main function to fix broker references in all RBI files"""
    print("🌙 Moon Dev's Broker Reference Fixer")
    print("=" * 60)
    
    # Find all Python files in RBI directories
    rbi_base = Path(".")
    patterns = ["**/backtests/*.py", "**/backtests_final/*.py", "**/backtests_package/*.py"]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(rbi_base.glob(pattern))
    
    print(f"📁 Found {len(all_files)} backtest files")
    
    # Process files
    fixed = 0
    skipped = 0
    failed = 0
    
    for filepath in all_files:
        success, message = process_file(filepath)
        
        if success:
            fixed += 1
            print(f"✅ Fixed: {filepath.name} - {message}")
        elif "No broker references" in message:
            skipped += 1
        else:
            if "No incorrect" not in message:
                failed += 1
                print(f"❌ Failed: {filepath.name} - {message}")
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  ✅ Fixed: {fixed}")
    print(f"  ⏭️  Skipped (no broker refs): {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Total: {len(all_files)}")

if __name__ == "__main__":
    main()