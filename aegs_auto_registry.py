"""
🔥💎 AEGS AUTO-REGISTRY SYSTEM 💎🔥
Automatically adds symbols to registry when backtesting succeeds

Author: Claude (Anthropic)
"""

import json
import os
from datetime import datetime
from termcolor import colored

class AEGSAutoRegistry:
    """Automatic registry management for AEGS goldmine symbols"""
    
    def __init__(self):
        self.registry_file = 'aegs_goldmine_registry.json'
        self.load_registry()
    
    def load_registry(self):
        """Load existing registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            # Initialize empty registry
            self.registry = {
                "goldmine_symbols": {
                    "extreme_goldmines": {
                        "description": "Symbols with >1000% excess return",
                        "symbols": {}
                    },
                    "high_potential": {
                        "description": "Symbols with 100-1000% excess return",
                        "symbols": {}
                    },
                    "positive": {
                        "description": "Symbols with 10-100% excess return",
                        "symbols": {}
                    }
                },
                "categories": {},
                "metadata": {
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "total_symbols": 0,
                    "goldmine_count": 0,
                    "high_potential_count": 0,
                    "strategy_name": "Alpha Ensemble Goldmine Strategy (AEGS)"
                }
            }
    
    def save_registry(self):
        """Save registry to file"""
        self.registry['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d")
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def auto_add_symbol(self, backtest_results):
        """
        Automatically add symbol based on backtest results
        
        Args:
            backtest_results: Dictionary with keys:
                - symbol: str
                - excess_return_pct: float
                - strategy_total_return_pct: float
                - win_rate: float
                - total_trades: int
                - category: str (optional)
        """
        
        symbol = backtest_results['symbol']
        excess_return = backtest_results['excess_return_pct']
        
        # Determine if symbol qualifies
        if excess_return <= 0:
            print(colored(f"❌ {symbol}: Negative excess return ({excess_return:.1f}%), not adding", 'red'))
            return False
        
        # Determine tier
        if excess_return > 1000:
            tier = "extreme_goldmines"
            tier_color = 'red'
            emoji = "💎"
        elif excess_return > 100:
            tier = "high_potential"
            tier_color = 'yellow'
            emoji = "🚀"
        elif excess_return > 10:
            tier = "positive"
            tier_color = 'green'
            emoji = "✅"
        else:
            print(colored(f"❌ {symbol}: Excess return too low ({excess_return:.1f}%), not adding", 'yellow'))
            return False
        
        # Check if already exists
        for t in self.registry['goldmine_symbols']:
            if symbol in self.registry['goldmine_symbols'][t].get('symbols', {}):
                print(f"ℹ️  {symbol} already in registry, updating...")
                # Update existing entry
                self.registry['goldmine_symbols'][t]['symbols'][symbol].update({
                    'excess_return': excess_return,
                    'last_updated': datetime.now().strftime("%Y-%m-%d"),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'total_trades': backtest_results.get('total_trades', 0)
                })
                self.save_registry()
                return True
        
        # Add new symbol
        category = backtest_results.get('category', 'Unknown')
        
        self.registry['goldmine_symbols'][tier]['symbols'][symbol] = {
            'excess_return': excess_return,
            'category': category,
            'active': True,
            'added_date': datetime.now().strftime("%Y-%m-%d"),
            'strategy_return': backtest_results.get('strategy_total_return_pct', 0),
            'win_rate': backtest_results.get('win_rate', 0),
            'total_trades': backtest_results.get('total_trades', 0)
        }
        
        # Add to category mapping
        if category not in self.registry['categories']:
            self.registry['categories'][category] = []
        if symbol not in self.registry['categories'][category]:
            self.registry['categories'][category].append(symbol)
        
        # Update metadata
        self.update_metadata()
        
        # Save
        self.save_registry()
        
        # Success message
        print(colored(f"{emoji} AUTO-ADDED {symbol} to {tier}!", tier_color, attrs=['bold']))
        print(f"   Excess Return: {excess_return:+.0f}%")
        print(f"   Strategy Return: {backtest_results.get('strategy_total_return_pct', 0):+.0f}%")
        print(f"   Win Rate: {backtest_results.get('win_rate', 0):.1f}%")
        print(f"   Category: {category}")
        
        # Show potential
        if excess_return > 1000:
            potential = 10000 * (1 + backtest_results.get('strategy_total_return_pct', 0)/100)
            print(colored(f"   💰 $10k → ${potential:,.0f} potential!", 'cyan'))
        
        return True
    
    def update_metadata(self):
        """Update registry metadata"""
        total = 0
        goldmine = 0
        high_pot = 0
        
        for tier, data in self.registry['goldmine_symbols'].items():
            symbols = data.get('symbols', {})
            total += len(symbols)
            
            if tier == "extreme_goldmines":
                goldmine = len(symbols)
            elif tier == "high_potential":
                high_pot = len(symbols)
        
        self.registry['metadata']['total_symbols'] = total
        self.registry['metadata']['goldmine_count'] = goldmine
        self.registry['metadata']['high_potential_count'] = high_pot
    
    def remove_symbol(self, symbol):
        """Remove symbol from registry"""
        found = False
        
        for tier in self.registry['goldmine_symbols']:
            if symbol in self.registry['goldmine_symbols'][tier].get('symbols', {}):
                del self.registry['goldmine_symbols'][tier]['symbols'][symbol]
                found = True
                print(f"✅ Removed {symbol} from {tier}")
                break
        
        if found:
            # Remove from categories
            for cat in self.registry['categories']:
                if symbol in self.registry['categories'][cat]:
                    self.registry['categories'][cat].remove(symbol)
            
            self.update_metadata()
            self.save_registry()
        else:
            print(f"❌ {symbol} not found in registry")
    
    def get_all_symbols(self):
        """Get all symbols in registry"""
        symbols = []
        for tier in self.registry['goldmine_symbols']:
            for symbol in self.registry['goldmine_symbols'][tier].get('symbols', {}):
                symbols.append(symbol)
        return symbols
    
    def display_registry(self):
        """Display current registry contents"""
        print(colored("\n🔥💎 AEGS GOLDMINE REGISTRY 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        for tier, data in self.registry['goldmine_symbols'].items():
            symbols = data.get('symbols', {})
            if symbols:
                print(f"\n📊 {data.get('description', tier).upper()}:")
                print("-" * 60)
                
                # Sort by excess return
                sorted_symbols = sorted(symbols.items(), 
                                      key=lambda x: x[1].get('excess_return', 0), 
                                      reverse=True)
                
                for symbol, info in sorted_symbols[:10]:  # Top 10
                    excess = info.get('excess_return', 0)
                    category = info.get('category', 'Unknown')
                    win_rate = info.get('win_rate', 0)
                    
                    print(f"   {symbol}: {excess:+.0f}% excess | "
                          f"{category} | "
                          f"Win: {win_rate:.1f}%")
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Symbols: {self.registry['metadata']['total_symbols']}")
        print(f"   Extreme Goldmines: {self.registry['metadata']['goldmine_count']}")
        print(f"   High Potential: {self.registry['metadata']['high_potential_count']}")
        print(f"   Last Updated: {self.registry['metadata']['last_updated']}")


# Integration function for backtesting
def register_backtest_result(symbol, results, category=None):
    """
    Helper function to automatically register successful backtests
    
    Args:
        symbol: str - The symbol that was backtested
        results: BacktestResults object with attributes:
            - excess_return_pct
            - strategy_total_return_pct
            - win_rate
            - total_trades
        category: str - Optional category for the symbol
    
    Returns:
        bool - True if successfully added
    """
    
    registry = AEGSAutoRegistry()
    
    # Prepare results dictionary
    backtest_data = {
        'symbol': symbol,
        'excess_return_pct': results.excess_return_pct,
        'strategy_total_return_pct': results.strategy_total_return_pct,
        'win_rate': results.win_rate,
        'total_trades': results.total_trades,
        'category': category or 'Unknown'
    }
    
    # Auto-add to registry
    return registry.auto_add_symbol(backtest_data)


def main():
    """Demo auto-registry system"""
    
    registry = AEGSAutoRegistry()
    
    # Display current registry
    registry.display_registry()
    
    # Example: Simulate adding a new backtest result
    print(colored("\n\n📊 DEMO: Adding new backtest result...", 'yellow'))
    
    demo_result = {
        'symbol': 'DEMO-SYMBOL',
        'excess_return_pct': 2500,
        'strategy_total_return_pct': 3000,
        'win_rate': 65.5,
        'total_trades': 150,
        'category': 'Demo Category'
    }
    
    registry.auto_add_symbol(demo_result)
    
    # Show how to integrate with backtesting
    print(colored("\n\n💡 INTEGRATION EXAMPLE:", 'cyan'))
    print("# In your backtest code:")
    print("from aegs_auto_registry import register_backtest_result")
    print("")
    print("# After successful backtest:")
    print("if results.excess_return_pct > 0:")
    print("    register_backtest_result(symbol, results, category)")


if __name__ == "__main__":
    main()