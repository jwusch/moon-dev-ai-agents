#!/usr/bin/env python3
"""
🌙 Moon Dev's VWAP Error Fixer
Fixes VWAP indicator issues in RBI backtest files
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_vwap_indicator(content: str) -> Tuple[bool, str]:
    """Fix VWAP indicator issues in backtesting files"""
    
    # Pattern 1: Fix pandas_ta VWAP calls that might return None
    pattern1 = r'(vwap_values\s*=\s*ta\.vwap\([^)]+\))\s*\n\s*(self\.I\(vwap_values[^)]*\))'
    
    replacement1 = r'''\1
        
        # If VWAP returns None or empty, use typical price as fallback
        if vwap_values is None or (hasattr(vwap_values, '__len__') and len(vwap_values) == 0):
            vwap_values = (self.data.High + self.data.Low + self.data.Close) / 3
        elif hasattr(vwap_values, 'fillna'):
            vwap_values = vwap_values.ffill().fillna((self.data.High + self.data.Low + self.data.Close) / 3).values
        
        self.vwap = self.I(lambda: vwap_values, name='VWAP')'''
    
    # Pattern 2: Fix self.data.VWAP references
    pattern2 = r'self\.data\.VWAP'
    replacement2 = r'self.vwap'
    
    # Pattern 3: Fix direct VWAP assignments without proper checks
    pattern3 = r'self\.vwap\s*=\s*self\.I\(ta\.vwap[^)]+\)'
    
    replacement3 = r'''# Calculate VWAP with fallback
        vwap_result = ta.vwap(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume
        )
        
        if vwap_result is None or (hasattr(vwap_result, '__len__') and len(vwap_result) == 0):
            vwap_values = (self.data.High + self.data.Low + self.data.Close) / 3
        else:
            vwap_values = vwap_result.ffill().fillna((self.data.High + self.data.Low + self.data.Close) / 3).values
            
        self.vwap = self.I(lambda: vwap_values, name='VWAP')'''
    
    # Pattern 4: Fix pandas_ta.vwap calls in I() without proper handling
    pattern4 = r'self\.I\(pandas_ta\.vwap[^)]+\)'
    pattern4_alt = r'self\.I\(ta\.vwap[^)]+\)'
    
    # Apply fixes
    modified = False
    original_content = content
    
    # Apply pattern 1
    if re.search(pattern1, content, re.MULTILINE):
        content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
        modified = True
    
    # Apply pattern 2
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        modified = True
    
    # Apply pattern 3
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content)
        modified = True
    
    # Fix deprecated fillna syntax
    if 'fillna(method=' in content:
        content = content.replace(".fillna(method='ffill')", ".ffill()")
        content = content.replace('.fillna(method="ffill")', '.ffill()')
        content = content.replace(".fillna(method='bfill')", ".bfill()")
        content = content.replace('.fillna(method="bfill")', '.bfill()')
        modified = True
    
    return modified, content

def process_file(filepath: Path) -> Tuple[bool, str]:
    """Process a single file to fix VWAP issues"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if no VWAP usage
        if 'vwap' not in content.lower():
            return False, "No VWAP usage found"
        
        # Apply fixes
        modified, fixed_content = fix_vwap_indicator(content)
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, "Fixed VWAP issues"
        
        return False, "No VWAP issues found"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main function to fix VWAP issues in all RBI files"""
    print("🌙 Moon Dev's VWAP Error Fixer")
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
        elif "No VWAP" in message:
            skipped += 1
        else:
            if "No VWAP issues" not in message:
                failed += 1
                print(f"❌ Failed: {filepath.name} - {message}")
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  ✅ Fixed: {fixed}")
    print(f"  ⏭️  Skipped (no VWAP): {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Total: {len(all_files)}")

if __name__ == "__main__":
    main()