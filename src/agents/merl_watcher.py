"""
Moon Dev's MERL Trade Watcher
Monitors Merlin Chain (MERL) trades in real-time

Usage:
    python src/agents/merl_watcher.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "MERL/USDT"
EXCHANGE = "okx"
CHECK_INTERVAL = 5
SHOW_LAST_TRADES = 20

class MERLWatcher:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.seen_trades = set()

        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Moon Dev's MERL Trade Watcher", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan")
        cprint(f"Exchange: {exchange_id.upper()}", "white")
        cprint(f"Symbol: {SYMBOL}", "white")

    def fetch_recent_trades(self, limit=50):
        try:
            trades = self.exchange.fetch_trades(SYMBOL, limit=limit)
            return trades
        except Exception as e:
            cprint(f"Error fetching trades: {e}", "red")
            return []

    def format_trade(self, trade, is_new=False):
        side = trade['side'].upper()
        price = trade['price']
        amount = trade['amount']
        cost = trade.get('cost', price * amount)
        timestamp = trade['datetime'][:19] if trade.get('datetime') else 'Unknown'

        side_color = "green" if side == "BUY" else "red"
        emoji = "🟢" if side == "BUY" else "🔴"

        if is_new:
            bg = "on_blue" if side == "BUY" else "on_magenta"
            return colored(f"{emoji} NEW {side} | {amount:.2f} MERL @ ${price:.4f} | ${cost:.2f} | {timestamp}", "white", bg)
        else:
            return f"{emoji} {colored(side, side_color):4} | {amount:.2f} MERL @ ${price:.4f} | ${cost:.2f} | {timestamp}"

    def show_recent_trades(self):
        cprint(f"\nLast {SHOW_LAST_TRADES} trades:", "yellow")
        cprint("-" * 70, "white")

        trades = self.fetch_recent_trades(SHOW_LAST_TRADES)
        for trade in trades[-SHOW_LAST_TRADES:]:
            print(self.format_trade(trade))
            self.seen_trades.add(trade['id'])

        cprint("-" * 70, "white")
        cprint("Watching for new trades...\n", "green")

    def monitor(self):
        while True:
            try:
                trades = self.fetch_recent_trades(50)
                new_trades = [t for t in trades if t['id'] not in self.seen_trades]

                for trade in new_trades:
                    print(self.format_trade(trade, is_new=True))
                    self.seen_trades.add(trade['id'])

                if len(self.seen_trades) > 1000:
                    self.seen_trades = set(list(self.seen_trades)[-500:])

            except KeyboardInterrupt:
                cprint("\nMERL Watcher shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"Error: {e}", "yellow")

            time.sleep(CHECK_INTERVAL)

def main():
    watcher = MERLWatcher()
    watcher.show_recent_trades()
    watcher.monitor()

if __name__ == "__main__":
    main()
