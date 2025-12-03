# Manual registration helper
from aegs_auto_registry import AEGSGoldmineRegistry

registry = AEGSGoldmineRegistry()

# Add symbols manually if needed
symbols_to_add = {
    "CENN": {"excess_return": 100, "category": "Penny Volatility"},
    "GOEV": {"excess_return": 500, "category": "SPAC Disaster"},
    "SPCE": {"excess_return": 300, "category": "Space SPAC"},
    # Add more as needed
}

for symbol, info in symbols_to_add.items():
    registry.add_symbol(
        symbol=symbol,
        excess_return=info["excess_return"],
        category=info["category"]
    )
    print(f"Added {symbol} to registry")

registry.display_summary()
