"""
📊 Position Tracker Database
Manages all trading positions and historical performance
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from termcolor import colored


class PositionTracker:
    """Database manager for trading positions"""
    
    def __init__(self, db_path: str = "positions.db"):
        self.db_path = Path(__file__).parent / db_path
        self.conn = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Read and execute schema
        schema_path = Path(__file__).parent / "positions_database.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = f.read()
                # Execute schema statements one by one
                for statement in schema.split(';'):
                    if statement.strip():
                        try:
                            self.conn.execute(statement)
                        except sqlite3.OperationalError as e:
                            # Skip if table/view already exists
                            if "already exists" not in str(e):
                                print(f"Schema error: {e}")
                self.conn.commit()
                
    def add_position(self, symbol: str, entry_price: float, shares: int,
                     entry_date: Optional[str] = None, strategy: str = "",
                     entry_reason: str = "") -> int:
        """Add a new position to the database"""
        if not entry_date:
            entry_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        position_size = entry_price * shares
        cursor = self.conn.execute("""
            INSERT INTO positions (symbol, entry_date, entry_price, shares, position_size, strategy, entry_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, entry_date, entry_price, shares, position_size, strategy, entry_reason))
        
        self.conn.commit()
        print(colored(f"✅ Added position: {shares} shares of {symbol} @ ${entry_price}", 'green'))
        return cursor.lastrowid
        
    def close_position(self, position_id: int, exit_price: float, 
                       exit_date: Optional[str] = None, exit_reason: str = "") -> Dict:
        """Close an open position"""
        if not exit_date:
            exit_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # Get position to calculate P&L
        pos = self.get_position(position_id)
        if not pos:
            print(colored(f"❌ Position #{position_id} not found", 'red'))
            return None
            
        exit_value = exit_price * pos['shares']
        profit_loss = (exit_price - pos['entry_price']) * pos['shares']
        profit_loss_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        
        self.conn.execute("""
            UPDATE positions 
            SET exit_date = ?, exit_price = ?, exit_value = ?, 
                profit_loss = ?, profit_loss_pct = ?, exit_reason = ?, status = 'CLOSED'
            WHERE id = ? AND status = 'OPEN'
        """, (exit_date, exit_price, exit_value, profit_loss, profit_loss_pct, 
              exit_reason, position_id))
        
        self.conn.commit()
        
        # Get the updated position details
        position = self.get_position(position_id)
        if position:
            pnl = position['profit_loss']
            pnl_pct = position['profit_loss_pct']
            color = 'green' if pnl > 0 else 'red'
            print(colored(f"✅ Closed position #{position_id}: {position['symbol']} "
                         f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)", color))
        return position
        
    def get_position(self, position_id: int) -> Optional[Dict]:
        """Get a specific position by ID"""
        cursor = self.conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
        
    def get_all_positions(self) -> pd.DataFrame:
        """Get all positions (both open and closed)"""
        query = """
            SELECT * FROM positions
            ORDER BY entry_date DESC
        """
        return pd.read_sql_query(query, self.conn)
        
    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        query = """
            SELECT 
                id, symbol, shares, entry_price, entry_date,
                position_size, 
                CAST(julianday('now') - julianday(entry_date) AS INTEGER) as days_held,
                strategy
            FROM positions
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
        """
        return pd.read_sql_query(query, self.conn)
        
    def get_symbol_history(self, symbol: str) -> pd.DataFrame:
        """Get all trades for a specific symbol"""
        query = """
            SELECT * FROM positions 
            WHERE symbol = ? 
            ORDER BY entry_date DESC
        """
        return pd.read_sql_query(query, self.conn, params=(symbol,))
        
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary by symbol"""
        return pd.read_sql_query("SELECT * FROM performance_summary", self.conn)
        
    def get_monthly_performance(self) -> pd.DataFrame:
        """Get monthly P&L summary"""
        return pd.read_sql_query("SELECT * FROM monthly_performance", self.conn)
        
    def add_note(self, position_id: int, note: str, note_type: str = "general"):
        """Add a note to a position"""
        self.conn.execute("""
            INSERT INTO position_notes (position_id, note_type, note)
            VALUES (?, ?, ?)
        """, (position_id, note_type, note))
        self.conn.commit()
        
    def import_json_position(self, json_path: str):
        """Import a position from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Check if it's an open or closed position
        if 'exit_price' in data and data.get('status') == 'CLOSED':
            # Closed position - calculate P&L
            entry_price = data['entry_price']
            exit_price = data['exit_price']
            shares = data['shares']
            position_size = entry_price * shares
            exit_value = exit_price * shares
            profit_loss = (exit_price - entry_price) * shares
            profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
            
            self.conn.execute("""
                INSERT INTO positions (
                    symbol, entry_date, entry_price, shares, position_size,
                    exit_date, exit_price, exit_value, profit_loss, profit_loss_pct,
                    status, strategy, entry_reason, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'], data['entry_date'], entry_price, shares, position_size,
                data.get('exit_date'), exit_price, exit_value, profit_loss, profit_loss_pct,
                'CLOSED', data.get('strategy', ''), 
                data.get('entry_reason', ''), data.get('exit_reason', 'Imported from JSON')
            ))
        else:
            # Open position
            self.add_position(
                data['symbol'], data['entry_price'], data['shares'],
                data['entry_date'], data.get('strategy', '')
            )
        
        self.conn.commit()
        print(colored(f"✅ Imported position from {json_path}", 'green'))
        
    def display_current_holdings(self):
        """Display current holdings in a nice format"""
        holdings = self.get_open_positions()
        
        if holdings.empty:
            print(colored("No open positions", 'yellow'))
            return
            
        print(colored("\n📊 CURRENT HOLDINGS", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        for _, pos in holdings.iterrows():
            print(f"\n{pos['symbol']}:")
            print(f"  Shares: {pos['shares']}")
            print(f"  Entry: ${pos['entry_price']:.2f} on {pos['entry_date']}")
            print(f"  Position Size: ${pos['position_size']:.2f}")
            print(f"  Days Held: {pos['days_held']}")
            if pos['strategy']:
                print(f"  Strategy: {pos['strategy']}")
                
    def display_performance_summary(self):
        """Display performance summary"""
        summary = self.get_performance_summary()
        
        if summary.empty:
            print(colored("No trading history", 'yellow'))
            return
            
        print(colored("\n📈 PERFORMANCE SUMMARY", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        for _, row in summary.iterrows():
            color = 'green' if row['total_profit_loss'] > 0 else 'red'
            win_rate = (row['winning_trades'] / row['closed_trades'] * 100) if row['closed_trades'] > 0 else 0
            
            print(f"\n{row['symbol']}:")
            print(f"  Total Trades: {row['total_trades']} (Closed: {row['closed_trades']}, Open: {row['open_positions']})")
            print(f"  Win Rate: {win_rate:.1f}% ({row['winning_trades']}W / {row['losing_trades']}L)")
            print(colored(f"  Total P&L: ${row['total_profit_loss']:.2f}", color))
            print(f"  Avg Return: {row['avg_return_pct']:.2f}%")
            print(f"  Best Trade: ${row['best_trade']:.2f}")
            print(f"  Worst Trade: ${row['worst_trade']:.2f}")


def main():
    """Demo the position tracker"""
    tracker = PositionTracker()
    
    # Import the closed TLRY position
    print("📊 Importing TLRY position...")
    tlry_closed_path = Path(__file__).parent.parent.parent / "tlry_position_card_closed.json"
    if tlry_closed_path.exists():
        tracker.import_json_position(str(tlry_closed_path))
    
    # Display current holdings
    tracker.display_current_holdings()
    
    # Display performance summary
    tracker.display_performance_summary()
    
    # Show monthly performance
    print(colored("\n📅 MONTHLY PERFORMANCE", 'cyan', attrs=['bold']))
    print("=" * 80)
    monthly = tracker.get_monthly_performance()
    if not monthly.empty:
        print(monthly.to_string(index=False))
    else:
        print("No monthly data yet")


if __name__ == "__main__":
    main()