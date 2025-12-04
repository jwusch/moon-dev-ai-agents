#!/usr/bin/env python
"""
💼 POSITION PORTAL
Interactive dashboard for managing trading positions
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.position_tracker import PositionTracker
from termcolor import colored
import pandas as pd


class PositionPortal:
    """Interactive position management portal"""
    
    def __init__(self):
        self.tracker = PositionTracker()
        
    def main_menu(self):
        """Display main menu"""
        while True:
            print(colored("\n" + "="*60, 'blue'))
            print(colored("💼 POSITION PORTAL", 'cyan', attrs=['bold']))
            print(colored("="*60, 'blue'))
            print("\n1. View Current Holdings")
            print("2. View Performance Summary")
            print("3. Add New Position")
            print("4. Close Position")
            print("5. View Symbol History")
            print("6. Monthly Performance")
            print("7. Import from JSON")
            print("8. Exit")
            
            choice = input(colored("\nSelect option (1-8): ", 'yellow'))
            
            if choice == '1':
                self.view_holdings()
            elif choice == '2':
                self.view_performance()
            elif choice == '3':
                self.add_position()
            elif choice == '4':
                self.close_position()
            elif choice == '5':
                self.view_symbol_history()
            elif choice == '6':
                self.view_monthly_performance()
            elif choice == '7':
                self.import_json()
            elif choice == '8':
                print(colored("\n👋 Goodbye!", 'green'))
                break
            else:
                print(colored("Invalid option. Please try again.", 'red'))
                
    def view_holdings(self):
        """View current holdings with real-time prices"""
        holdings = self.tracker.get_open_positions()
        
        print(colored("\n📊 CURRENT HOLDINGS", 'cyan', attrs=['bold']))
        print("="*80)
        
        if holdings.empty:
            print(colored("No open positions", 'yellow'))
        else:
            # Add current price lookup here if needed
            total_value = 0
            for _, pos in holdings.iterrows():
                value = pos['position_size']
                total_value += value
                
                print(f"\n{pos['symbol']}:")
                print(f"  ID: #{pos['id']}")
                print(f"  Shares: {pos['shares']}")
                print(f"  Entry: ${pos['entry_price']:.2f} on {pos['entry_date']}")
                print(f"  Position Value: ${value:,.2f}")
                print(f"  Days Held: {pos['days_held']}")
                if pos['strategy']:
                    print(f"  Strategy: {pos['strategy']}")
                    
            print(colored(f"\nTotal Portfolio Value: ${total_value:,.2f}", 'green', attrs=['bold']))
            
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def view_performance(self):
        """View performance summary"""
        self.tracker.display_performance_summary()
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def add_position(self):
        """Add a new position"""
        print(colored("\n➕ ADD NEW POSITION", 'cyan', attrs=['bold']))
        
        symbol = input("Symbol: ").upper()
        try:
            shares = int(input("Shares: "))
            entry_price = float(input("Entry price: $"))
            strategy = input("Strategy (optional): ")
            entry_reason = input("Entry reason (optional): ")
            
            position_id = self.tracker.add_position(
                symbol, entry_price, shares, 
                strategy=strategy, entry_reason=entry_reason
            )
            
            print(colored(f"✅ Position #{position_id} added successfully!", 'green'))
            
        except ValueError:
            print(colored("❌ Invalid input. Please enter numbers for shares and price.", 'red'))
            
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def close_position(self):
        """Close an open position"""
        holdings = self.tracker.get_open_positions()
        
        if holdings.empty:
            print(colored("No open positions to close", 'yellow'))
            input(colored("\nPress Enter to continue...", 'gray'))
            return
            
        print(colored("\n🔒 CLOSE POSITION", 'cyan', attrs=['bold']))
        print("\nOpen positions:")
        
        for _, pos in holdings.iterrows():
            print(f"  #{pos['id']}: {pos['shares']} shares of {pos['symbol']} @ ${pos['entry_price']}")
            
        try:
            position_id = int(input("\nEnter position ID to close: "))
            exit_price = float(input("Exit price: $"))
            exit_reason = input("Exit reason (optional): ")
            
            result = self.tracker.close_position(position_id, exit_price, exit_reason=exit_reason)
            
            if result:
                pnl = result['profit_loss']
                pnl_pct = result['profit_loss_pct']
                color = 'green' if pnl > 0 else 'red'
                print(colored(f"\n✅ Position closed! P&L: ${pnl:.2f} ({pnl_pct:.2f}%)", color))
                
        except ValueError:
            print(colored("❌ Invalid input.", 'red'))
            
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def view_symbol_history(self):
        """View all trades for a symbol"""
        print(colored("\n📈 SYMBOL HISTORY", 'cyan', attrs=['bold']))
        symbol = input("Enter symbol: ").upper()
        
        history = self.tracker.get_symbol_history(symbol)
        
        if history.empty:
            print(colored(f"No trades found for {symbol}", 'yellow'))
        else:
            print(f"\nTrade history for {symbol}:")
            print("="*80)
            
            for _, trade in history.iterrows():
                status_color = 'green' if trade['status'] == 'CLOSED' else 'yellow'
                print(f"\n#{trade['id']} - {colored(trade['status'], status_color)}")
                print(f"  Entry: {trade['shares']} shares @ ${trade['entry_price']} on {trade['entry_date']}")
                
                if trade['exit_price']:
                    pnl_color = 'green' if trade['profit_loss'] > 0 else 'red'
                    print(f"  Exit: ${trade['exit_price']} on {trade['exit_date']}")
                    print(colored(f"  P&L: ${trade['profit_loss']:.2f} ({trade['profit_loss_pct']:.2f}%)", pnl_color))
                    
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def view_monthly_performance(self):
        """View monthly P&L"""
        print(colored("\n📅 MONTHLY PERFORMANCE", 'cyan', attrs=['bold']))
        print("="*80)
        
        monthly = self.tracker.get_monthly_performance()
        
        if monthly.empty:
            print("No monthly data yet")
        else:
            total_pnl = 0
            for _, row in monthly.iterrows():
                pnl = row['monthly_pnl'] if row['monthly_pnl'] else 0
                total_pnl += pnl
                color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'white'
                
                print(f"{row['month']}: ", end='')
                print(colored(f"${pnl:>8.2f}", color), end='')
                print(f" | {row['trades_closed']:>2} trades | ", end='')
                print(f"{row['wins']}W/{row['losses']}L | ", end='')
                print(f"Avg: {row['avg_return_pct']:.2f}%")
                
            print("-"*40)
            total_color = 'green' if total_pnl > 0 else 'red'
            print(colored(f"TOTAL: ${total_pnl:>8.2f}", total_color, attrs=['bold']))
            
        input(colored("\nPress Enter to continue...", 'gray'))
        
    def import_json(self):
        """Import position from JSON file"""
        print(colored("\n📥 IMPORT FROM JSON", 'cyan', attrs=['bold']))
        
        json_path = input("Enter JSON file path: ")
        
        try:
            self.tracker.import_json_position(json_path)
        except Exception as e:
            print(colored(f"❌ Error importing: {str(e)}", 'red'))
            
        input(colored("\nPress Enter to continue...", 'gray'))


def quick_summary():
    """Show quick summary without menu"""
    tracker = PositionTracker()
    
    # Current holdings
    holdings = tracker.get_open_positions()
    
    print(colored("📊 POSITION SUMMARY", 'cyan', attrs=['bold']))
    print("="*60)
    
    if not holdings.empty:
        print(colored(f"\n✅ {len(holdings)} Open Positions:", 'green'))
        total_value = 0
        for _, pos in holdings.iterrows():
            value = pos['position_size']
            total_value += value
            print(f"  • {pos['symbol']}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${value:,.2f}")
        print(colored(f"\nTotal Value: ${total_value:,.2f}", 'green', attrs=['bold']))
    else:
        print(colored("\n❌ No open positions", 'yellow'))
        
    # Recent performance
    summary = tracker.get_performance_summary()
    
    if not summary.empty:
        print(colored("\n📈 Recent Performance:", 'cyan'))
        for _, row in summary.iterrows():
            if row['closed_trades'] > 0:
                color = 'green' if row['total_profit_loss'] > 0 else 'red'
                win_rate = (row['winning_trades'] / row['closed_trades'] * 100)
                print(f"  • {row['symbol']}: ", end='')
                print(colored(f"${row['total_profit_loss']:.2f}", color), end='')
                print(f" ({win_rate:.0f}% win rate)")


def main():
    parser = argparse.ArgumentParser(description='Position Portal - Manage your trading positions')
    parser.add_argument('--summary', '-s', action='store_true', 
                       help='Show quick summary without interactive menu')
    
    args = parser.parse_args()
    
    if args.summary:
        quick_summary()
    else:
        portal = PositionPortal()
        portal.main_menu()


if __name__ == "__main__":
    main()