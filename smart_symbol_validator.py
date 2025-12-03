"""
🔍💎 SMART SYMBOL VALIDATOR 💎🔍
Intelligently validates and updates symbol lists
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import json
from termcolor import colored

class SmartSymbolValidator:
    """
    Smart validation with proper symbol handling
    """
    
    def __init__(self):
        # Known good mappings only
        self.verified_mappings = {
            # Confirmed ticker changes
            'DWAC': 'DJT',    # Trump Media
            'CFVI': 'RUM',    # Rumble
            'SPRT': 'GREE',   # Greenidge
            
            # Still trading but often problematic
            'BBBY': 'BBBY',   # Still active
            'AMC': 'AMC',     # Still active
            'GME': 'GME',     # Still active
        }
        
        # Known delisted - don't waste time checking
        self.known_delisted = {
            'MULN', 'FFIE', 'REV', 'APRN', 'EXPR', 'RIDE', 
            'PROG', 'VIEW', 'BODY', 'RDBX', 'WEBR', 'IRNT',
            'OPAD', 'TMC', 'CEI', 'XELA', 'GNUS', 'NAKD',
            'OCUP', 'SESN', 'ATER', 'BBIG'
        }
    
    def validate_symbol(self, symbol):
        """
        Validate a single symbol efficiently
        """
        # Skip known delisted
        if symbol in self.known_delisted:
            return None, 'delisted'
        
        # Check verified mappings first
        if symbol in self.verified_mappings:
            mapped = self.verified_mappings[symbol]
            try:
                ticker = yf.Ticker(mapped)
                hist = ticker.history(period='5d')
                if len(hist) > 0:
                    return mapped, 'mapped'
            except:
                pass
        
        # Try original symbol
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            if len(hist) > 0:
                return symbol, 'active'
        except:
            pass
        
        return None, 'not_found'
    
    def bulk_validate(self, symbols):
        """
        Validate multiple symbols efficiently
        """
        results = {
            'active': [],
            'mapped': {},
            'delisted': [],
            'not_found': []
        }
        
        print(colored("🔍 Validating symbols...", 'yellow'))
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Checking {symbol}...", end='\r')
            
            valid_symbol, status = self.validate_symbol(symbol)
            
            if status == 'active':
                results['active'].append(symbol)
            elif status == 'mapped':
                results['mapped'][symbol] = valid_symbol
                results['active'].append(valid_symbol)
            elif status == 'delisted':
                results['delisted'].append(symbol)
            else:
                results['not_found'].append(symbol)
        
        print("\n")
        return results
    
    def update_discovery_lists(self):
        """
        Update all discovery lists with valid symbols
        """
        
        print(colored("🔧 UPDATING DISCOVERY LISTS", 'cyan', attrs=['bold']))
        print("=" * 60)
        
        # Common volatile penny stocks that might still be active
        volatile_pennies = [
            # Crypto/Blockchain
            'BTBT', 'CAN', 'SOS', 'EBON', 'NCTY', 'LGHL', 'BTCM',
            # EV/Clean Tech  
            'WKHS', 'HYLN', 'GOEV', 'NKLA', 'CHPT', 'BLNK', 'EVGO',
            # Biotech
            'BNGO', 'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS',
            # Tech/Space
            'ASTR', 'MNTS', 'BKSY', 'PL', 'ASTS', 'RKLB', 'IONQ',
            # Cannabis
            'SNDL', 'OGI', 'TLRY', 'CGC', 'CRON', 'ACB',
            # Recent volatility
            'CENN', 'BARK', 'TALK', 'OPEN', 'PSFE', 'SOFI', 'HOOD',
            # SPACs
            'LCID', 'RIVN', 'SPCE', 'STEM', 'QS', 'MVST', 'FSR',
            # Meme stocks
            'AMC', 'GME', 'BB', 'KOSS', 'BBBY', 'NOK', 'PLTR',
            # Others
            'WISH', 'CLOV', 'SDC', 'WKHS', 'RIDE', 'GOEV', 'HYLN'
        ]
        
        # Remove duplicates
        volatile_pennies = list(set(volatile_pennies))
        
        # Validate all
        results = self.bulk_validate(volatile_pennies)
        
        # Report results
        print(colored("\n📊 VALIDATION RESULTS:", 'green'))
        print(f"✅ Active symbols: {len(results['active'])}")
        print(f"🔄 Mapped symbols: {len(results['mapped'])}")
        print(f"❌ Delisted: {len(results['delisted'])}")
        print(f"❓ Not found: {len(results['not_found'])}")
        
        # Show active symbols
        if results['active']:
            print(colored("\n✅ ACTIVE TRADING SYMBOLS:", 'green'))
            # Get current prices
            for symbol in sorted(results['active'])[:20]:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d')
                    if len(hist) > 0:
                        price = hist['Close'].iloc[-1]
                        volume = hist['Volume'].iloc[-1]
                        print(f"   {symbol}: ${price:.2f} (Vol: {volume:,.0f})")
                except:
                    print(f"   {symbol}: Active")
        
        # Show mappings
        if results['mapped']:
            print(colored("\n🔄 SYMBOL MAPPINGS:", 'yellow'))
            for old, new in results['mapped'].items():
                print(f"   {old} → {new}")
        
        # Create updated lists
        self.save_validated_lists(results)
        
        return results
    
    def save_validated_lists(self, results):
        """
        Save validated symbol lists
        """
        
        # Create discovery file for AEGS
        discovery_data = {
            'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent': 'Smart Symbol Validator',
            'candidates_found': len(results['active']),
            'candidates': []
        }
        
        # Only include truly volatile penny stocks under $10
        for symbol in results['active']:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                if len(hist) > 0:
                    price = hist['Close'].iloc[-1]
                    volatility = hist['Close'].pct_change().std()
                    
                    if price < 10 and volatility > 0.03:  # Under $10, >3% volatility
                        discovery_data['candidates'].append({
                            'symbol': symbol,
                            'reason': f"Active penny stock ${price:.2f}, {volatility:.1%} volatility"
                        })
            except:
                pass
        
        if discovery_data['candidates']:
            filename = f"validated_pennies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(discovery_data, f, indent=2)
            print(f"\n💾 Saved {len(discovery_data['candidates'])} validated penny stocks to {filename}")
        
        # Save complete validation data
        validation_data = {
            'last_updated': datetime.now().isoformat(),
            'active_symbols': sorted(results['active']),
            'symbol_mappings': results['mapped'],
            'delisted_symbols': sorted(results['delisted']),
            'statistics': {
                'total_checked': len(results['active']) + len(results['delisted']) + len(results['not_found']),
                'active_count': len(results['active']),
                'mapped_count': len(results['mapped']),
                'delisted_count': len(results['delisted'])
            }
        }
        
        with open('symbol_validation_report.json', 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print("💾 Saved complete validation report to symbol_validation_report.json")


def main():
    """
    Run smart symbol validation
    """
    
    validator = SmartSymbolValidator()
    results = validator.update_discovery_lists()
    
    print(colored("\n✅ Symbol validation complete!", 'green'))
    print("\nNext steps:")
    print("1. Run AEGS backtest on validated symbols")
    print("2. Update discovery agents to use validated lists")
    print("3. Remove delisted symbols from scanners")


if __name__ == "__main__":
    main()