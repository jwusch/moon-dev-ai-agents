#!/usr/bin/env python3
"""
🔍💎 SYMBOL CONSENSUS SWARM 💎🔍
AI Swarm system to verify stock symbols and current trading status

Uses multiple AI models to research and validate stock symbols, providing:
- Current trading status (active, delisted, OTC, merged, etc.)
- Correct ticker symbol and exchange
- Company name and basic info
- Consensus recommendation on tradability

Author: Claude (Anthropic)
Created: December 2, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.swarm_agent import SwarmAgent
from termcolor import colored, cprint
import json
from typing import List, Dict, Any
from datetime import datetime

class SymbolConsensusSwarm:
    """
    AI Swarm for researching and validating stock symbols
    """
    
    def __init__(self):
        self.name = "Symbol Consensus Swarm"
        self.swarm = SwarmAgent()
        
    def research_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Research a single symbol using AI swarm consensus
        
        Args:
            symbol: Stock symbol to research (e.g., 'DCFC', 'GNUS')
            
        Returns:
            Dict with consensus results about symbol status
        """
        
        print(colored(f"\n🔍 Researching symbol: {symbol}", 'cyan', attrs=['bold']))
        print("=" * 60)
        
        # Create research prompt for the swarm
        research_prompt = f"""
Research the stock symbol '{symbol}' and provide current trading status information.

Please analyze:
1. CURRENT STATUS: Is this symbol currently tradable? (Active/Delisted/Suspended/OTC/Other)
2. EXCHANGE: What exchange does it trade on? (NYSE/NASDAQ/OTC/Pink Sheets/None)
3. COMPANY NAME: Current or former company name
4. SYMBOL CHANGES: Has the ticker changed? If so, what's the new symbol?
5. CORPORATE ACTIONS: Any recent mergers, acquisitions, bankruptcies, or restructuring?
6. TRADABILITY: Can this symbol be traded today via standard brokers?

Format your response as:
STATUS: [Current status]
EXCHANGE: [Exchange or None]
COMPANY: [Company name]
NEW_SYMBOL: [New symbol if changed, or None]
ACTION: [Recent corporate action if any]
TRADABLE: [YES/NO with brief explanation]

Be accurate and current - check multiple sources if possible.
"""
        
        # Query the swarm
        swarm_result = self.swarm.query(research_prompt)
        
        # Parse individual AI responses
        parsed_responses = self._parse_ai_responses(swarm_result["responses"])
        
        # Create consensus analysis
        consensus = self._create_consensus(symbol, parsed_responses, swarm_result["consensus_summary"])
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'individual_responses': parsed_responses,
            'consensus_summary': swarm_result["consensus_summary"],
            'consensus_analysis': consensus,
            'model_mapping': swarm_result["model_mapping"]
        }
    
    def research_symbol_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Research multiple symbols
        
        Args:
            symbols: List of stock symbols to research
            
        Returns:
            Dict mapping symbol -> research results
        """
        
        print(colored(f"\\n🚀 Starting batch symbol research for {len(symbols)} symbols", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            print(colored(f"\\n[{i+1}/{len(symbols)}] Processing {symbol}...", 'cyan'))
            
            try:
                result = self.research_symbol(symbol)
                results[symbol] = result
                
                # Show quick summary
                consensus = result['consensus_analysis']
                status_color = 'green' if consensus['tradable'] else 'red'
                print(colored(f"   Result: {consensus['status']} - {consensus['exchange']} - {'TRADABLE' if consensus['tradable'] else 'NOT TRADABLE'}", status_color))
                
            except Exception as e:
                print(colored(f"   ❌ Error researching {symbol}: {str(e)}", 'red'))
                results[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return results
    
    def _parse_ai_responses(self, raw_responses: Dict[str, Dict]) -> Dict[str, Dict]:
        """Parse structured data from AI responses"""
        
        parsed = {}
        
        for provider, data in raw_responses.items():
            if not data.get("success"):
                parsed[provider] = {"error": data.get("error", "Unknown error")}
                continue
            
            response_text = data.get("response", "")
            
            # Try to extract structured fields
            extracted = self._extract_fields_from_response(response_text)
            
            parsed[provider] = {
                "raw_response": response_text,
                "extracted_fields": extracted,
                "model": data.get("model", "Unknown")
            }
        
        return parsed
    
    def _extract_fields_from_response(self, text: str) -> Dict[str, str]:
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
        
        import re
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                fields[field] = match.group(1).strip()
            else:
                fields[field] = "Not found"
        
        return fields
    
    def _create_consensus(self, symbol: str, parsed_responses: Dict, consensus_summary: str) -> Dict[str, Any]:
        """Create consensus analysis from individual AI responses"""
        
        # Collect all extracted fields
        statuses = []
        exchanges = []
        companies = []
        new_symbols = []
        tradables = []
        
        for provider, data in parsed_responses.items():
            if "extracted_fields" in data:
                fields = data["extracted_fields"]
                statuses.append(fields.get('status', '').lower())
                exchanges.append(fields.get('exchange', '').lower())
                companies.append(fields.get('company', ''))
                new_symbols.append(fields.get('new_symbol', '').lower())
                tradables.append(fields.get('tradable', '').lower())
        
        # Create consensus
        consensus = {
            'symbol': symbol,
            'status': self._get_consensus_value(statuses),
            'exchange': self._get_consensus_value(exchanges),
            'company': self._get_consensus_value(companies),
            'new_symbol': self._get_consensus_value(new_symbols),
            'tradable': self._determine_tradability(tradables),
            'confidence': self._calculate_confidence(parsed_responses),
            'ai_count': len([r for r in parsed_responses.values() if "extracted_fields" in r])
        }
        
        return consensus
    
    def _get_consensus_value(self, values: List[str]) -> str:
        """Get most common value from AI responses"""
        if not values:
            return "Unknown"
        
        # Filter out empty/invalid values
        valid_values = [v for v in values if v and v.lower() not in ['not found', 'unknown', 'none', 'n/a']]
        
        if not valid_values:
            return "Unknown"
        
        # Return most common value
        from collections import Counter
        counter = Counter(valid_values)
        return counter.most_common(1)[0][0]
    
    def _determine_tradability(self, tradable_responses: List[str]) -> bool:
        """Determine if symbol is tradable based on AI consensus"""
        if not tradable_responses:
            return False
        
        yes_count = sum(1 for r in tradable_responses if r and ('yes' in r.lower() or 'tradable' in r.lower()))
        total_count = len([r for r in tradable_responses if r and r.lower() != 'not found'])
        
        if total_count == 0:
            return False
        
        # Require majority consensus for tradability
        return yes_count / total_count > 0.5
    
    def _calculate_confidence(self, parsed_responses: Dict) -> str:
        """Calculate confidence based on AI agreement"""
        successful_responses = len([r for r in parsed_responses.values() if "extracted_fields" in r])
        
        if successful_responses >= 4:
            return "High"
        elif successful_responses >= 2:
            return "Medium"
        else:
            return "Low"
    
    def save_results(self, results: Dict[str, Dict], filename: str = None):
        """Save research results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"symbol_research_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(colored(f"\\n💾 Results saved to: {filename}", 'green'))
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print a summary of research results"""
        
        print(colored("\\n📊 SYMBOL RESEARCH SUMMARY", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        tradable_count = 0
        delisted_count = 0
        otc_count = 0
        unknown_count = 0
        
        for symbol, data in results.items():
            if 'error' in data:
                print(colored(f"{symbol:8} | ❌ ERROR: {data['error']}", 'red'))
                unknown_count += 1
                continue
            
            consensus = data.get('consensus_analysis', {})
            status = consensus.get('status', 'Unknown')
            exchange = consensus.get('exchange', 'Unknown')
            tradable = consensus.get('tradable', False)
            company = consensus.get('company', 'Unknown')
            
            if tradable:
                color = 'green'
                tradable_count += 1
                tradable_symbol = "✅ TRADABLE"
            elif 'otc' in status.lower() or 'pink' in exchange.lower():
                color = 'yellow' 
                otc_count += 1
                tradable_symbol = "🟡 OTC"
            elif 'delisted' in status.lower() or 'bankrupt' in status.lower():
                color = 'red'
                delisted_count += 1
                tradable_symbol = "❌ DELISTED"
            else:
                color = 'white'
                unknown_count += 1
                tradable_symbol = "❓ UNKNOWN"
            
            print(colored(f"{symbol:8} | {tradable_symbol:12} | {exchange:15} | {company[:30]}", color))
        
        print(colored("\\n📈 SUMMARY STATS:", 'cyan', attrs=['bold']))
        total = len(results)
        print(f"   ✅ Tradable: {tradable_count}/{total} ({tradable_count/total*100:.1f}%)")
        print(f"   🟡 OTC: {otc_count}/{total} ({otc_count/total*100:.1f}%)")
        print(f"   ❌ Delisted: {delisted_count}/{total} ({delisted_count/total*100:.1f}%)")
        print(f"   ❓ Unknown/Error: {unknown_count}/{total} ({unknown_count/total*100:.1f}%)")


def main():
    """Main function for testing symbol research"""
    
    print(colored("🔍💎 SYMBOL CONSENSUS SWARM 💎🔍", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    # Initialize swarm
    swarm = SymbolConsensusSwarm()
    
    # Test symbols (mix of questionable ones from your discovery)
    test_symbols = [
        'DCFC',    # Likely delisted
        'NAKD',    # Likely delisted  
        'GNUS',    # Likely delisted
        'CEI',     # Likely delisted
        'AAPL',    # Obviously valid (control)
        'TSLA',    # Obviously valid (control)
        'HMNY'     # The legendary trillion-gain symbol
    ]
    
    print(colored(f"Testing {len(test_symbols)} symbols with AI swarm consensus...", 'yellow'))
    
    # Research all symbols
    results = swarm.research_symbol_batch(test_symbols)
    
    # Print summary
    swarm.print_summary(results)
    
    # Save results
    swarm.save_results(results)
    
    print(colored("\\n🎉 Symbol research complete!", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()