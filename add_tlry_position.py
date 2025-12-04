#!/usr/bin/env python3
"""
Add TLRY position to tracker
"""

from src.data.position_tracker import PositionTracker
from datetime import datetime

def add_tlry_position():
    """Add the new TLRY position"""
    
    print("🔥💎 Adding TLRY Position to Tracker 💎🔥")
    print("=" * 50)
    
    # Initialize tracker
    tracker = PositionTracker()
    
    # Add the position
    position_id = tracker.add_position(
        symbol='TLRY',
        entry_price=7.19,
        shares=100,
        entry_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        strategy='Manual Buy - Cannabis Oversold Bounce',
        entry_reason='AEGS analysis showed potential bounce at $7.19 support level'
    )
    
    print(f"📊 Position ID: {position_id}")
    print(f"💰 Position Size: ${7.19 * 100:.2f}")
    
    # Add a note about the position
    tracker.add_note(
        position_id, 
        "Entered based on AEGS backtest analysis. Cannabis sector oversold, looking for bounce to $8.50-9.00 range.",
        "entry_analysis"
    )
    
    # Display current holdings
    print("\n" + "=" * 50)
    tracker.display_current_holdings()
    
    # Show TLRY history
    print("\n" + "=" * 50)
    print("📈 TLRY Trading History:")
    tlry_history = tracker.get_symbol_history('TLRY')
    if not tlry_history.empty:
        for _, trade in tlry_history.iterrows():
            status_color = 'green' if trade['status'] == 'CLOSED' else 'yellow'
            print(f"  {trade['entry_date']}: {trade['shares']} shares @ ${trade['entry_price']:.2f}")
            print(f"    Status: {trade['status']} | Strategy: {trade['strategy']}")
            if trade['status'] == 'CLOSED':
                pnl = trade['profit_loss']
                pnl_pct = trade['profit_loss_pct']
                color = 'green' if pnl > 0 else 'red'
                print(f"    Exit: ${trade['exit_price']:.2f} on {trade['exit_date']}")
                print(f"    P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            print()
    
    return position_id

if __name__ == "__main__":
    add_tlry_position()