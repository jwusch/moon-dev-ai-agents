#!/usr/bin/env python3
"""
🔍 Diagnose JSON Parsing Error in AEGS Backtest
Find which JSON file has errors at line 311 and line 424
"""

import json
import os
from termcolor import colored

def check_json_file(filepath):
    """Check if a JSON file is valid and count lines"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.count('\n') + 1
            
        # Try to parse it
        try:
            json.loads(content)
            return True, lines, None
        except json.JSONDecodeError as e:
            return False, lines, str(e)
    except Exception as e:
        return False, 0, str(e)

def find_problematic_json_files():
    """Find JSON files with parsing errors"""
    print(colored("🔍 Searching for JSON files with parsing errors...", 'cyan', attrs=['bold']))
    print("="*80)
    
    problematic_files = []
    
    # Search for all JSON files
    for root, dirs, files in os.walk('.'):
        # Skip cache directories
        if 'cache' in root or 'node_modules' in root or '.git' in root:
            continue
            
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                valid, lines, error = check_json_file(filepath)
                
                if not valid and error:
                    # Check if it's one of our target errors
                    if "line 311" in error or "line 424" in error:
                        problematic_files.append((filepath, lines, error))
                        print(colored(f"❌ Found problematic file: {filepath}", 'red'))
                        print(f"   Lines: {lines}")
                        print(f"   Error: {error}")
                        print()
    
    return problematic_files

def check_specific_files():
    """Check specific files that might have issues"""
    print(colored("\n📊 Checking specific files...", 'yellow'))
    print("-"*80)
    
    files_to_check = [
        'aegs_goldmine_registry.json',
        'QQQ_ensemble_strategy.json', 
        'SPY_ensemble_strategy.json',
        'aegs_backtest_history.json',
        'aegs_invalid_symbols.json'
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            valid, lines, error = check_json_file(filename)
            
            print(f"\n{filename}:")
            print(f"  Lines: {lines}")
            print(f"  Valid: {'✅' if valid else '❌'}")
            
            if not valid:
                print(f"  Error: {error}")
                
                # Check specific lines if possible
                if "line 311" in str(error) or "line 424" in str(error):
                    print(colored(f"  ⚠️  This file has the error we're looking for!", 'yellow'))
                    
                    # Show content around the error line
                    with open(filename, 'r') as f:
                        lines_list = f.readlines()
                        
                    if "line 311" in str(error) and len(lines_list) >= 311:
                        print(f"\n  Content around line 311:")
                        for i in range(max(0, 310-3), min(len(lines_list), 310+3)):
                            print(f"    {i+1}: {lines_list[i].rstrip()}")
                    
                    if "line 424" in str(error) and len(lines_list) >= 424:
                        print(f"\n  Content around line 424:")
                        for i in range(max(0, 423-3), min(len(lines_list), 423+3)):
                            print(f"    {i+1}: {lines_list[i].rstrip()}")

def check_for_duplicate_json():
    """Check if any file has duplicate JSON objects"""
    print(colored("\n🔍 Checking for files with duplicate JSON objects...", 'cyan'))
    print("-"*80)
    
    # The error "Extra data" usually means there are multiple JSON objects
    for filename in ['aegs_goldmine_registry.json', 'aegs_backtest_history.json']:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            
            # Try to find multiple JSON objects
            import re
            json_objects = re.findall(r'\{[^{}]*\}', content, re.DOTALL)
            
            if len(json_objects) > 1:
                print(colored(f"⚠️  {filename} might have multiple JSON objects!", 'yellow'))
                
            # Check for duplicate closing braces
            if content.count('}}') > content.count('{{'):
                print(colored(f"⚠️  {filename} has extra closing braces!", 'yellow'))

def main():
    """Run diagnostics"""
    print(colored("🔍 AEGS JSON ERROR DIAGNOSTICS", 'green', attrs=['bold']))
    print("="*80)
    
    # Find problematic files
    problematic = find_problematic_json_files()
    
    if problematic:
        print(colored(f"\n❌ Found {len(problematic)} files with JSON errors", 'red'))
    else:
        print(colored("\n✅ No JSON files found with line 311/424 errors in current directory", 'green'))
    
    # Check specific files
    check_specific_files()
    
    # Check for duplicate JSON
    check_for_duplicate_json()
    
    print(colored("\n🎯 DIAGNOSIS COMPLETE", 'green', attrs=['bold']))
    print("="*80)
    
    if problematic:
        print("\n💡 Solution:")
        print("1. The error 'Extra data: line X' means there's extra content after the JSON ends")
        print("2. This usually happens when:")
        print("   - There are multiple JSON objects in one file")
        print("   - There's extra text or characters after the closing }")
        print("   - The file was corrupted during writing")
        print("\n3. To fix: Edit the problematic file and ensure it contains only one valid JSON object")

if __name__ == "__main__":
    main()