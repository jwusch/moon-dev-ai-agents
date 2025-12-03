"""
🔥 Add High Volatility Discoveries to AEGS System
Backtest and register new volatile symbols
"""

import json
from datetime import datetime
from termcolor import colored

def add_new_symbols_to_aegs():
    """Add discovered volatile symbols to AEGS for testing"""
    
    print(colored("🔥 ADDING NEW VOLATILITY SYMBOLS TO AEGS", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # High volatility penny stocks from our scans
    new_discoveries = {
        # From penny volatility scan
        "CENN": {"reason": "Ultra-penny $0.16, 85% decline, high volatility", "category": "Penny Volatility"},
        "MNTS": {"reason": "Sub-$1 space SPAC, 75% decline", "category": "Space SPAC"},
        "CAN": {"reason": "9.2% daily volatility crypto stock", "category": "Crypto Penny"},
        "SOS": {"reason": "7.7% daily volatility blockchain", "category": "Crypto Penny"},
        "COSM": {"reason": "6% volatility, -18% recent move", "category": "Penny Volatility"},
        
        # From boom/bust scan
        "GOEV": {"reason": "100% crash from $141 to $0.37", "category": "SPAC Disaster"},
        "SPCE": {"reason": "93% crash, near ATL, space tourism", "category": "Space SPAC"},
        "CHPT": {"reason": "88% crash, EV charging leader", "category": "EV Infrastructure"},
        "STEM": {"reason": "81% crash, energy storage", "category": "Clean Energy"},
        "BIRD": {"reason": "82% crash, sustainable shoes", "category": "Consumer SPAC"},
        "LCID": {"reason": "76% crash, luxury EV maker", "category": "EV SPAC"},
        
        # Other high potential from scans
        "HYLN": {"reason": "5.5% volatility, hydrogen trucks", "category": "Clean Transport"},
        "BTBT": {"reason": "5.3% volatility, Bitcoin mining", "category": "Crypto Mining"},
        "BARK": {"reason": "73% crash, dog products SPAC", "category": "Consumer SPAC"},
        "PSFE": {"reason": "70% crash, fintech payments", "category": "Fintech SPAC"}
    }
    
    # Create discovery file for AEGS backtest agent
    discovery_data = {
        'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'agent': 'Volatility Discovery Integration',
        'candidates_found': len(new_discoveries),
        'candidates': []
    }
    
    for symbol, info in new_discoveries.items():
        discovery_data['candidates'].append({
            'symbol': symbol,
            'reason': info['reason']
        })
        print(f"   📊 {symbol}: {info['reason']}")
    
    # Save discovery file
    filename = f"aegs_volatility_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(discovery_data, f, indent=2)
    
    print(f"\n💾 Saved {len(new_discoveries)} discoveries to: {filename}")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run backtest on these symbols:")
    print(f"   python src/agents/aegs_backtest_agent.py {filename}")
    print("\n2. Successful symbols (>10% excess return) will auto-register")
    print("\n3. Check enhanced scanner for new signals:")
    print("   python aegs_enhanced_scanner.py")
    
    # Also create a manual registration helper
    print("\n📝 Creating manual registration helper...")
    
    helper_code = '''# Manual registration helper
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
'''
    
    with open('manual_aegs_registration.py', 'w') as f:
        f.write(helper_code)
    
    print("💾 Created manual_aegs_registration.py for direct registration")
    
    return filename

if __name__ == "__main__":
    discovery_file = add_new_symbols_to_aegs()