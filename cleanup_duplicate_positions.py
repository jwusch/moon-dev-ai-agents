#!/usr/bin/env python
"""
🧹 POSITION DATABASE CLEANUP UTILITY
Removes duplicate test positions and keeps only actual trades
"""

import sqlite3
from datetime import datetime
from termcolor import colored

def cleanup_positions_database():
    """Remove duplicate positions and keep only real trades"""
    
    print(colored("🧹 CLEANING UP POSITIONS DATABASE", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Connect to database
    conn = sqlite3.connect('src/data/positions.db')
    conn.row_factory = sqlite3.Row
    
    # Get current positions
    cursor = conn.execute("SELECT * FROM positions ORDER BY entry_date")
    positions = cursor.fetchall()
    
    print(f"Found {len(positions)} total positions")
    
    # Group by symbol to find duplicates
    symbol_groups = {}
    for pos in positions:
        symbol = pos['symbol']
        if symbol not in symbol_groups:
            symbol_groups[symbol] = []
        symbol_groups[symbol].append(pos)
    
    # Clean up duplicates
    for symbol, pos_list in symbol_groups.items():
        if len(pos_list) > 1:
            print(f"\n📊 {symbol}: Found {len(pos_list)} positions")
            
            # Keep only the most recent entry for each status
            open_positions = [p for p in pos_list if p['status'] == 'OPEN']
            closed_positions = [p for p in pos_list if p['status'] == 'CLOSED']
            
            # Keep only the latest open position
            if len(open_positions) > 1:
                open_positions.sort(key=lambda x: x['entry_date'], reverse=True)
                positions_to_delete = open_positions[1:]  # Delete all but the latest
                
                for pos in positions_to_delete:
                    conn.execute("DELETE FROM positions WHERE id = ?", (pos['id'],))
                    print(f"  🗑️  Deleted duplicate open position #{pos['id']}")
            
            # Keep only the latest closed position (if any)
            if len(closed_positions) > 1:
                closed_positions.sort(key=lambda x: x['exit_date'] or '1900-01-01', reverse=True)
                positions_to_delete = closed_positions[1:]  # Delete all but the latest
                
                for pos in positions_to_delete:
                    conn.execute("DELETE FROM positions WHERE id = ?", (pos['id'],))
                    print(f"  🗑️  Deleted duplicate closed position #{pos['id']}")
    
    conn.commit()
    
    # Show final count
    cursor = conn.execute("SELECT COUNT(*) as total FROM positions")
    final_count = cursor.fetchone()['total']
    
    print(f"\n✅ Cleanup complete!")
    print(f"Positions remaining: {final_count}")
    
    # Show remaining positions
    cursor = conn.execute("SELECT symbol, status, COUNT(*) as count FROM positions GROUP BY symbol, status")
    remaining = cursor.fetchall()
    
    print(f"\n📊 Current positions:")
    for pos in remaining:
        status_color = 'green' if pos['status'] == 'OPEN' else 'blue'
        print(colored(f"  {pos['symbol']}: {pos['count']} {pos['status'].lower()}", status_color))
    
    conn.close()

if __name__ == "__main__":
    cleanup_positions_database()