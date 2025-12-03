"""
Quick script to clean up demo and update SOL-USD with actual backtest data
"""

from aegs_auto_registry import AEGSAutoRegistry

# Initialize registry
registry = AEGSAutoRegistry()

# Remove demo symbol
registry.remove_symbol('DEMO-SYMBOL')

# Update SOL-USD with actual backtest data
sol_results = {
    'symbol': 'SOL-USD',
    'excess_return_pct': 39496,
    'strategy_total_return_pct': 61300,
    'win_rate': 58.8,
    'total_trades': 34,
    'category': 'Cryptocurrency'
}

registry.auto_add_symbol(sol_results)

# Display updated registry
registry.display_registry()

print("\n✅ Registry cleaned and updated!")