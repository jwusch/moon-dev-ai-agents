#!/usr/bin/env python
"""
🏆 AEGS DYNAMIC GOLDMINE REGISTRY
Automatically manages and updates the AEGS goldmine monitoring list
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set
from termcolor import colored
import yfinance as yf
import pandas as pd


class AEGSDynamicRegistry:
    """Manages dynamic goldmine symbol registry with auto-discovery"""
    
    def __init__(self):
        self.registry_file = 'aegs_dynamic_goldmines.json'
        self.registry = self._load_registry()
        
        # Performance thresholds for auto-promotion
        self.promotion_criteria = {
            'min_signal_strength': 60,        # Minimum AEGS signal strength
            'min_volatility_score': 30,       # Minimum volatility score
            'min_occurrences': 2,             # Must appear in scans at least X times
            'max_days_since_signal': 14,      # Must have had signal in last 14 days
            'min_volume': 1000000,            # Minimum daily volume
            'price_range': (2.0, 200.0)       # Price range filter
        }
        
        # Categories for dynamic symbols
        self.dynamic_categories = {
            'Volatile Discoveries': [],
            'High Performance': [],
            'Momentum Leaders': [],
            'Volume Breakouts': [],
            'Technical Patterns': []
        }
        
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(colored(f"Warning: Could not load registry: {e}", 'yellow'))
                
        # Default registry structure
        return {
            'last_updated': datetime.now().isoformat(),
            'dynamic_symbols': {},
            'categories': {
                'Volatile Discoveries': [],
                'High Performance': [],
                'Momentum Leaders': [],
                'Volume Breakouts': [],
                'Technical Patterns': []
            },
            'blacklist': [],  # Symbols to never add
            'performance_history': {}
        }
        
    def _save_registry(self):
        """Save registry to file"""
        self.registry['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            print(colored(f"Error saving registry: {e}", 'red'))
            
    def add_volatility_discoveries(self, volatility_results: Dict):
        """Process volatility scan results and add qualifying symbols"""
        
        print(colored("🔍 Processing volatility discoveries for AEGS promotion...", 'cyan'))
        
        promotions = []
        updates = []
        
        # Process AEGS signals from volatility scan
        if 'aegs_signals' in volatility_results:
            for signal in volatility_results['aegs_signals']:
                symbol = signal['symbol']
                
                # Check promotion criteria
                if self._meets_promotion_criteria(signal):
                    promotion = self._promote_to_goldmine(symbol, signal, 'Volatile Discoveries')
                    if promotion:
                        promotions.append(promotion)
                else:
                    # Still track it for future promotion
                    self._update_symbol_tracking(symbol, signal)
                    updates.append(symbol)
                    
        # Process top volatile stocks (even without AEGS signals)
        if 'volatile_picks' in volatility_results:
            for pick in volatility_results['volatile_picks'][:10]:  # Top 10 only
                symbol = pick['symbol']
                
                # Track high volatility stocks for pattern recognition
                if pick['overall_score'] >= 40:  # High volatility threshold
                    self._update_symbol_tracking(symbol, pick, track_type='volatility')
                    if symbol not in updates:
                        updates.append(symbol)
        
        self._save_registry()
        
        # Report results
        if promotions:
            print(colored(f"\n🎉 PROMOTED {len(promotions)} SYMBOLS TO AEGS GOLDMINE LIST:", 'green', attrs=['bold']))
            for promo in promotions:
                print(f"  • {colored(promo['symbol'], 'green', attrs=['bold'])}: "
                      f"{promo['category']} "
                      f"(Signal: {promo['signal_strength']}/100, "
                      f"Vol: {promo['volatility_score']:.0f})")
        
        if updates:
            print(colored(f"\n📊 TRACKING {len(updates)} SYMBOLS FOR FUTURE PROMOTION:", 'yellow'))
            for symbol in updates[:5]:  # Show first 5
                print(f"  • {symbol}")
            if len(updates) > 5:
                print(f"  ... and {len(updates) - 5} more")
                
        return {'promotions': promotions, 'tracked_updates': len(updates)}
        
    def _meets_promotion_criteria(self, signal: Dict) -> bool:
        """Check if signal meets promotion criteria"""
        
        symbol = signal['symbol']
        criteria = self.promotion_criteria
        
        # Already in blacklist
        if symbol in self.registry.get('blacklist', []):
            return False
            
        # Check signal strength
        if signal.get('signal_strength', 0) < criteria['min_signal_strength']:
            return False
            
        # Check volatility score
        if signal.get('volatility_score', 0) < criteria['min_volatility_score']:
            return False
            
        # Check price range
        price = signal.get('current_price', 0)
        if not (criteria['price_range'][0] <= price <= criteria['price_range'][1]):
            return False
            
        # Check occurrence frequency
        symbol_history = self.registry['dynamic_symbols'].get(symbol, {})
        occurrences = symbol_history.get('signal_count', 0)
        
        if occurrences < criteria['min_occurrences'] - 1:  # -1 because this is a new occurrence
            return False
            
        # Check recency
        last_signal = symbol_history.get('last_signal_date')
        if last_signal:
            last_signal_date = datetime.fromisoformat(last_signal)
            days_since = (datetime.now() - last_signal_date).days
            if days_since > criteria['max_days_since_signal']:
                return False
                
        return True
        
    def _promote_to_goldmine(self, symbol: str, signal: Dict, category: str) -> Dict:
        """Promote symbol to goldmine list"""
        
        # Add to registry category
        if category not in self.registry['categories']:
            self.registry['categories'][category] = []
            
        if symbol not in self.registry['categories'][category]:
            self.registry['categories'][category].append(symbol)
            
        # Record promotion details
        promotion = {
            'symbol': symbol,
            'category': category,
            'promotion_date': datetime.now().isoformat(),
            'signal_strength': signal.get('signal_strength', 0),
            'volatility_score': signal.get('volatility_score', 0),
            'current_price': signal.get('current_price', 0),
            'reason': f"Promoted from volatility scan - {signal.get('triggers', 'Strong signal')}"
        }
        
        # Update performance history
        if symbol not in self.registry['performance_history']:
            self.registry['performance_history'][symbol] = []
            
        self.registry['performance_history'][symbol].append(promotion)
        
        # Update the live AEGS scanner file
        self._update_aegs_live_scanner(category, symbol)
        
        return promotion
        
    def _update_symbol_tracking(self, symbol: str, data: Dict, track_type: str = 'signal'):
        """Update symbol tracking data"""
        
        if symbol not in self.registry['dynamic_symbols']:
            self.registry['dynamic_symbols'][symbol] = {
                'first_seen': datetime.now().isoformat(),
                'signal_count': 0,
                'volatility_count': 0,
                'last_signal_date': None,
                'last_volatility_date': None,
                'best_signal_strength': 0,
                'best_volatility_score': 0,
                'price_history': []
            }
            
        symbol_data = self.registry['dynamic_symbols'][symbol]
        
        if track_type == 'signal':
            symbol_data['signal_count'] += 1
            symbol_data['last_signal_date'] = datetime.now().isoformat()
            symbol_data['best_signal_strength'] = max(
                symbol_data['best_signal_strength'], 
                data.get('signal_strength', 0)
            )
        elif track_type == 'volatility':
            symbol_data['volatility_count'] += 1
            symbol_data['last_volatility_date'] = datetime.now().isoformat()
            symbol_data['best_volatility_score'] = max(
                symbol_data['best_volatility_score'],
                data.get('overall_score', 0)
            )
            
        # Track price
        current_price = data.get('current_price', 0)
        if current_price > 0:
            symbol_data['price_history'].append({
                'date': datetime.now().isoformat(),
                'price': current_price
            })
            
        # Keep only last 30 price points
        if len(symbol_data['price_history']) > 30:
            symbol_data['price_history'] = symbol_data['price_history'][-30:]
            
    def _update_aegs_live_scanner(self, category: str, symbol: str):
        """Update the live AEGS scanner file to include new symbol"""
        
        try:
            # Read current AEGS scanner file
            with open('aegs_live_scanner.py', 'r') as f:
                content = f.read()
                
            # Find the priority_symbols section
            lines = content.split('\n')
            updated_lines = []
            in_priority_symbols = False
            category_found = False
            
            for line in lines:
                if 'self.priority_symbols = {' in line:
                    in_priority_symbols = True
                    
                if in_priority_symbols and category in line and ':' in line:
                    # Found the category line
                    category_found = True
                    
                    # Extract current symbols
                    start_bracket = line.find('[')
                    end_bracket = line.rfind(']')
                    
                    if start_bracket != -1 and end_bracket != -1:
                        symbols_part = line[start_bracket+1:end_bracket]
                        
                        # Parse existing symbols
                        existing_symbols = []
                        if symbols_part.strip():
                            # Simple parsing - assumes format like 'SYM1', 'SYM2'
                            parts = [s.strip().strip("'\"") for s in symbols_part.split(',')]
                            existing_symbols = [s for s in parts if s and s != '']
                            
                        # Add new symbol if not already present
                        if symbol not in existing_symbols:
                            existing_symbols.append(symbol)
                            
                        # Rebuild line
                        symbols_str = ', '.join([f"'{s}'" for s in existing_symbols])
                        category_part = line[:start_bracket+1]
                        end_part = line[end_bracket:]
                        line = f"{category_part}{symbols_str}{end_part}"
                        
                elif in_priority_symbols and line.strip() == '}':
                    in_priority_symbols = False
                    
                updated_lines.append(line)
                
            # Write back the file if we found and updated the category
            if category_found:
                with open('aegs_live_scanner.py', 'w') as f:
                    f.write('\n'.join(updated_lines))
                    
                print(colored(f"  ✅ Added {symbol} to {category} in aegs_live_scanner.py", 'green'))
            else:
                print(colored(f"  ⚠️ Could not find category {category} in AEGS scanner", 'yellow'))
                
        except Exception as e:
            print(colored(f"  ❌ Error updating AEGS scanner: {str(e)}", 'red'))
            
    def get_current_goldmines(self) -> Dict:
        """Get current dynamic goldmine symbols"""
        return self.registry.get('categories', {})
        
    def remove_underperformers(self, days_threshold: int = 30):
        """Remove symbols that haven't performed well"""
        
        print(colored(f"🧹 Cleaning up underperformers (no signals in {days_threshold} days)...", 'cyan'))
        
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        removals = []
        
        for category, symbols in self.registry['categories'].items():
            symbols_to_remove = []
            
            for symbol in symbols:
                symbol_data = self.registry['dynamic_symbols'].get(symbol, {})
                last_signal = symbol_data.get('last_signal_date')
                
                if last_signal:
                    last_signal_date = datetime.fromisoformat(last_signal)
                    if last_signal_date < cutoff_date:
                        symbols_to_remove.append(symbol)
                        
            # Remove underperformers
            for symbol in symbols_to_remove:
                symbols.remove(symbol)
                removals.append({'symbol': symbol, 'category': category})
                
        if removals:
            print(colored(f"  🗑️ Removed {len(removals)} underperforming symbols:", 'yellow'))
            for removal in removals[:5]:  # Show first 5
                print(f"    • {removal['symbol']} from {removal['category']}")
                
        self._save_registry()
        return removals
        
    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        
        total_dynamic = sum(len(symbols) for symbols in self.registry['categories'].values())
        total_tracked = len(self.registry.get('dynamic_symbols', {}))
        
        category_counts = {
            category: len(symbols) 
            for category, symbols in self.registry['categories'].items()
        }
        
        return {
            'total_goldmine_symbols': total_dynamic,
            'total_tracked_symbols': total_tracked,
            'category_breakdown': category_counts,
            'last_updated': self.registry.get('last_updated'),
            'blacklist_count': len(self.registry.get('blacklist', []))
        }
        
    def display_status(self):
        """Display current registry status"""
        
        stats = self.get_registry_stats()
        
        print(colored("🏆 AEGS DYNAMIC GOLDMINE REGISTRY STATUS", 'cyan', attrs=['bold']))
        print("=" * 60)
        print(f"Total Goldmine Symbols: {stats['total_goldmine_symbols']}")
        print(f"Total Tracked Symbols: {stats['total_tracked_symbols']}")
        print(f"Last Updated: {stats['last_updated']}")
        
        print(colored("\n📊 CATEGORY BREAKDOWN:", 'yellow'))
        for category, count in stats['category_breakdown'].items():
            if count > 0:
                symbols = self.registry['categories'][category]
                print(f"  • {category}: {count} symbols")
                if count <= 5:
                    print(f"    {', '.join(symbols)}")
                else:
                    print(f"    {', '.join(symbols[:3])} ... and {count-3} more")
                    
        print(f"\nBlacklisted Symbols: {stats['blacklist_count']}")


def integrate_volatility_discoveries():
    """Main function to integrate volatility discoveries"""
    
    print(colored("🔗 INTEGRATING VOLATILITY DISCOVERIES WITH AEGS GOLDMINES", 'cyan', attrs=['bold']))
    print("=" * 70)
    
    # Initialize registry
    registry = AEGSDynamicRegistry()
    
    # Look for recent volatility scan results
    import glob
    scan_files = glob.glob('aegs_volatility_scan_*.json')
    
    if not scan_files:
        print(colored("❌ No volatility scan results found. Run the volatility scanner first.", 'red'))
        return
        
    # Use most recent scan
    latest_scan = max(scan_files, key=os.path.getctime)
    
    print(f"📂 Loading results from: {latest_scan}")
    
    try:
        with open(latest_scan, 'r') as f:
            volatility_results = json.load(f)
            
        # Process discoveries
        results = registry.add_volatility_discoveries(volatility_results)
        
        # Clean up old symbols
        removals = registry.remove_underperformers(days_threshold=21)
        
        # Display status
        print(f"\n" + "=" * 70)
        registry.display_status()
        
        print(colored(f"\n✅ INTEGRATION COMPLETE", 'green', attrs=['bold']))
        print(f"  • Promotions: {len(results['promotions'])}")
        print(f"  • Tracked Updates: {results['tracked_updates']}")
        print(f"  • Removals: {len(removals)}")
        
    except Exception as e:
        print(colored(f"❌ Error processing scan results: {str(e)}", 'red'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AEGS Dynamic Registry Manager')
    parser.add_argument('--integrate', '-i', action='store_true',
                       help='Integrate latest volatility discoveries')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show registry status')
    parser.add_argument('--cleanup', '-c', action='store_true',
                       help='Clean up underperformers')
    
    args = parser.parse_args()
    
    if args.integrate:
        integrate_volatility_discoveries()
    elif args.status:
        registry = AEGSDynamicRegistry()
        registry.display_status()
    elif args.cleanup:
        registry = AEGSDynamicRegistry()
        registry.remove_underperformers()
    else:
        integrate_volatility_discoveries()  # Default action