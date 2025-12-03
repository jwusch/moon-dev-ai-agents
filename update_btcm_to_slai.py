#!/usr/bin/env python3
"""
Update BTCM to SLAI in database and create exit monitor
"""

import sqlite3
import json
from datetime import datetime
from termcolor import colored

def update_btcm_to_slai():
    """Update BTCM references to SLAI in the database"""
    
    # Connect to database
    conn = sqlite3.connect('aegs_data.db')
    cursor = conn.cursor()
    
    print(colored("🔄 Updating BTCM to SLAI in database...", 'cyan'))
    
    try:
        # Remove BTCM from invalid symbols
        cursor.execute("DELETE FROM invalid_symbols WHERE symbol = 'BTCM'")
        removed_btcm = cursor.rowcount
        
        # Update any discoveries from BTCM to SLAI
        cursor.execute("""
            UPDATE discoveries 
            SET symbol = 'SLAI' 
            WHERE symbol = 'BTCM'
        """)
        updated_discoveries = cursor.rowcount
        
        # Update backtest cache
        cursor.execute("""
            UPDATE backtest_cache 
            SET symbol = 'SLAI' 
            WHERE symbol = 'BTCM'
        """)
        updated_cache = cursor.rowcount
        
        conn.commit()
        
        print(colored(f"✅ Successfully updated database:", 'green'))
        print(f"   - Removed BTCM from invalid symbols: {removed_btcm}")
        print(f"   - Updated discoveries: {updated_discoveries}")
        print(f"   - Updated cache entries: {updated_cache}")
        
    except Exception as e:
        conn.rollback()
        print(colored(f"❌ Error updating database: {e}", 'red'))
        
    finally:
        conn.close()
    
    # Also update the JSON file
    try:
        with open('aegs_invalid_symbols.json', 'r') as f:
            data = json.load(f)
        
        if 'BTCM' in data.get('invalid_symbols', {}):
            del data['invalid_symbols']['BTCM']
            data['metadata']['total_invalid'] = len(data['invalid_symbols'])
            data['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open('aegs_invalid_symbols.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(colored("✅ Removed BTCM from JSON invalid symbols", 'green'))
    except Exception as e:
        print(colored(f"⚠️ Could not update JSON file: {e}", 'yellow'))

def create_slai_exit_monitor():
    """Create exit monitoring configuration for SLAI"""
    
    config = {
        "symbol": "SLAI",
        "entry_price": 1.38,
        "shares": 300,
        "position_value": 414.00,
        "entry_date": datetime.now().strftime('%Y-%m-%d'),
        "strategy": "AEGS Discovery",
        "exit_criteria": {
            "stop_loss": 1.24,  # -10% from entry
            "take_profit_1": 1.52,  # +10% from entry
            "take_profit_2": 1.66,  # +20% from entry
            "take_profit_3": 1.79,  # +30% from entry
            "time_stop": 30,  # Exit after 30 days
            "trailing_stop": 0.10  # 10% trailing stop
        },
        "notes": "Changed from BTCM ticker. Company is SOLAI Limited (formerly BIT Mining)"
    }
    
    # Save configuration
    with open('slai_position_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(colored("\n📊 SLAI Exit Monitor Configuration Created", 'yellow'))
    print("=" * 60)
    print(f"Symbol: SLAI")
    print(f"Entry: ${config['entry_price']} x {config['shares']} shares = ${config['position_value']}")
    print("\nExit Targets:")
    print(f"  🛑 Stop Loss: ${config['exit_criteria']['stop_loss']} (-10%)")
    print(f"  🎯 Target 1: ${config['exit_criteria']['take_profit_1']} (+10%)")
    print(f"  🎯 Target 2: ${config['exit_criteria']['take_profit_2']} (+20%)")
    print(f"  🎯 Target 3: ${config['exit_criteria']['take_profit_3']} (+30%)")
    print(f"  ⏱️ Time Stop: {config['exit_criteria']['time_stop']} days")
    print(f"  📉 Trailing Stop: {config['exit_criteria']['trailing_stop']*100}%")

if __name__ == "__main__":
    # Update database
    update_btcm_to_slai()
    
    # Create exit monitor
    create_slai_exit_monitor()
    
    print(colored("\n✅ All updates complete!", 'green'))
    print("You can now track SLAI instead of BTCM")