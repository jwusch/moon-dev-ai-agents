#!/usr/bin/env python3
"""
Test the fixed regex parsing on existing AI responses
"""

import json
import re

def test_parser():
    """Test regex parsing on the JSON results"""
    
    # Load the results
    with open('quick_test_results.json', 'r') as f:
        results = json.load(f)
    
    # Test text parsing function
    def extract_fields_from_response(text: str):
        """Extract structured fields from AI response text"""
        
        fields = {}
        
        # Define field patterns (fixed regex)
        patterns = {
            'status': r'STATUS\s*:\s*([^\n\r]+)',
            'exchange': r'EXCHANGE\s*:\s*([^\n\r]+)', 
            'company': r'COMPANY\s*:\s*([^\n\r]+)',
            'new_symbol': r'NEW_SYMBOL\s*:\s*([^\n\r]+)',
            'action': r'ACTION\s*:\s*([^\n\r]+)',
            'tradable': r'TRADABLE\s*:\s*([^\n\r]+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                fields[field] = match.group(1).strip()
            else:
                fields[field] = "Not found"
        
        return fields
    
    # Test each AI response
    for symbol, data in results.items():
        print(f"\n=== {symbol} ===")
        
        for provider, response_data in data['individual_responses'].items():
            if 'raw_response' in response_data:
                print(f"\n{provider.upper()}:")
                raw = response_data['raw_response']
                
                # Test the parsing
                parsed = extract_fields_from_response(raw)
                
                for field, value in parsed.items():
                    if value != "Not found":
                        print(f"  {field}: {value}")
                
                # Show if anything was found
                found_count = len([v for v in parsed.values() if v != "Not found"])
                print(f"  → Found {found_count}/6 fields")

if __name__ == "__main__":
    test_parser()