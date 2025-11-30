#!/usr/bin/env python3
"""
🌙 Moon Dev Comprehensive Syntax Error Fixer
Fixes all common syntax errors in backtest files across all directories
"""
import os
import re
import ast

def fix_syntax_errors_in_file(filepath):
    """Fix common syntax errors in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = []
        
        # Fix 1: Missing space after 'and' or 'or' operators
        if re.search(r'\b(and|or)(?=[a-zA-Z_])', content):
            content = re.sub(r'\b(and|or)(?=[a-zA-Z_])', r'\1 ', content)
            fixes_applied.append("Fixed missing spaces after and/or operators")
        
        # Fix 2: Invalid decimal literals (e.g., .3f instead of 0.3f)
        if re.search(r'(?<!\d)\.(\d+)', content):
            content = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', content)
            fixes_applied.append("Fixed invalid decimal literals")
        
        # Fix 3: Unterminated string literals
        # This is harder to fix automatically, but we can detect them
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Count quotes in the line
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            if single_quotes % 2 != 0 or double_quotes % 2 != 0:
                # This line might have an unterminated string
                # For now, just flag it
                fixes_applied.append(f"Warning: Potential unterminated string on line {i+1}")
        
        # Only write if we made changes
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, fixes_applied
        
        # Try to parse the file to check for remaining syntax errors
        try:
            ast.parse(content)
            return True, ["No syntax errors found"]
        except SyntaxError as e:
            return False, [f"Remaining syntax error: {str(e)}"]
            
    except Exception as e:
        return False, [f"Error processing file: {str(e)}"]

def scan_and_fix_directory(directory):
    """Scan directory for Python files and fix syntax errors"""
    fixed_files = []
    error_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and 'BT' in file:
                filepath = os.path.join(root, file)
                success, fixes = fix_syntax_errors_in_file(filepath)
                
                if success and fixes != ["No syntax errors found"]:
                    fixed_files.append((filepath, fixes))
                    print(f"✅ Fixed: {filepath}")
                    for fix in fixes:
                        print(f"   - {fix}")
                elif not success:
                    error_files.append((filepath, fixes))
                    print(f"❌ Error in: {filepath}")
                    for error in fixes:
                        print(f"   - {error}")
    
    return fixed_files, error_files

def main():
    """Main function to fix all backtest files"""
    base_dir = '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi'
    
    # Directories to check
    directories = [
        '03_13_2025/backtests_final',
        '03_14_2025/backtests',
        '03_15_2025/backtests',
        '11_30_2025/backtests'
    ]
    
    all_fixed = []
    all_errors = []
    
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"\n🔍 Scanning {dir_path}...")
            fixed, errors = scan_and_fix_directory(full_path)
            all_fixed.extend(fixed)
            all_errors.extend(errors)
    
    print(f"\n🌙 Summary:")
    print(f"✅ Fixed {len(all_fixed)} files")
    print(f"❌ {len(all_errors)} files still have errors")
    
    if all_errors:
        print("\n⚠️ Files that still need manual fixing:")
        for filepath, errors in all_errors:
            print(f"\n{filepath}:")
            for error in errors:
                print(f"  - {error}")

if __name__ == "__main__":
    main()