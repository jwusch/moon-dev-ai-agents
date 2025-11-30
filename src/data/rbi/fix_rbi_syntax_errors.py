#!/usr/bin/env python3
"""
🌙 Moon Dev's RBI Syntax Error Fixer
Fixes common syntax errors in RBI-generated backtest files
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import ast

def is_valid_python(content: str) -> bool:
    """Check if content is valid Python syntax"""
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False

def fix_unterminated_strings(content: str) -> str:
    """Fix unterminated string literals"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Count quotes
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        triple_single = line.count("'''")
        triple_double = line.count('"""')
        
        # Fix odd number of quotes (excluding triple quotes)
        if single_quotes % 2 == 1 and triple_single % 2 == 0:
            line += "'"
        elif double_quotes % 2 == 1 and triple_double % 2 == 0:
            line += '"'
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def remove_invalid_characters(content: str) -> str:
    """Remove invalid characters like emojis that cause syntax errors"""
    # Remove common emojis that cause issues
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Replace emojis in comments only
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # If line contains emoji and it's not in a string or comment, comment out the line
        if emoji_pattern.search(line):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Check if it's likely a comment that should be
                if any(keyword in line.lower() for keyword in ['moon dev', 'built with', 'strategy']):
                    line = '# ' + line
                else:
                    # Remove emojis from code
                    line = emoji_pattern.sub('', line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_line1_prose(content: str) -> str:
    """Fix files that start with prose instead of code"""
    lines = content.split('\n')
    if not lines:
        return content
    
    first_line = lines[0].strip()
    
    # Common patterns indicating prose instead of code
    prose_indicators = [
        "Here's", "Here is", "Let me", "I'll", "This is", "The following",
        "Below is", "Fixed", "Updated", "Corrected", "Please find"
    ]
    
    if any(first_line.startswith(indicator) for indicator in prose_indicators):
        # Find where the actual code starts
        code_start = -1
        for i, line in enumerate(lines):
            # Look for import statements or class definitions
            if line.strip().startswith(('import ', 'from ', 'class ', 'def ', '"""')):
                code_start = i
                break
        
        if code_start > 0:
            # Remove prose and keep only code
            return '\n'.join(lines[code_start:])
    
    return content

def extract_code_from_prose(content: str) -> str:
    """Extract Python code from a file that contains prose and code"""
    # Look for code blocks
    code_block_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_block_pattern, content, re.DOTALL)
    
    if matches:
        # Return the first code block found
        return matches[0]
    
    # Try to find code without markdown markers
    lines = content.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Start collecting when we see imports or class definitions
        if line.strip().startswith(('import ', 'from ', 'class ', '"""')) and not in_code:
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return content

def fix_file(filepath: Path) -> Tuple[bool, str]:
    """Fix a single file and return (success, error_message)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already valid
        if is_valid_python(content):
            return True, "Already valid"
        
        # Apply fixes in order
        original_content = content
        
        # Fix prose at beginning
        content = fix_line1_prose(content)
        
        # Extract code from prose if needed
        if not is_valid_python(content):
            content = extract_code_from_prose(original_content)
        
        # Remove invalid characters
        content = remove_invalid_characters(content)
        
        # Fix unterminated strings
        content = fix_unterminated_strings(content)
        
        # Final validation
        if is_valid_python(content):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        else:
            # If still not valid, try one more aggressive approach
            # Comment out everything and add a placeholder
            if 'class' in content and 'Strategy' in content:
                # Extract strategy name
                strategy_match = re.search(r'class\s+(\w+)\s*\(', content)
                if strategy_match:
                    strategy_name = strategy_match.group(1)
                    placeholder = f'''"""
Moon Dev's {strategy_name} Strategy
This file had syntax errors and needs manual fixing.
Original content has been preserved as comments below.
"""

# Original content (commented out due to syntax errors):
'''
                    commented_content = '\n'.join('# ' + line for line in original_content.split('\n'))
                    content = placeholder + '\n' + commented_content
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return True, "Commented out (needs manual fix)"
            
            return False, "Could not fix automatically"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main function to fix all RBI files with syntax errors"""
    print("🌙 Moon Dev's RBI Syntax Error Fixer")
    print("=" * 60)
    
    # Find all RBI directories
    rbi_base = Path(".")
    patterns = ["**/backtests/*.py", "**/backtests_final/*.py", "**/backtests_package/*.py"]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(rbi_base.glob(pattern))
    
    print(f"📁 Found {len(all_files)} backtest files")
    
    # Process files
    fixed = 0
    failed = 0
    already_valid = 0
    
    for filepath in all_files:
        success, message = fix_file(filepath)
        
        if success:
            if message == "Already valid":
                already_valid += 1
            else:
                fixed += 1
                print(f"✅ Fixed: {filepath.name} - {message}")
        else:
            failed += 1
            print(f"❌ Failed: {filepath.name} - {message}")
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  ✅ Fixed: {fixed}")
    print(f"  ✅ Already valid: {already_valid}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Total: {len(all_files)}")

if __name__ == "__main__":
    main()