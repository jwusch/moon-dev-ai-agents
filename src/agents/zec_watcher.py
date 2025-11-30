"""
🌙 Moon Dev's ZEC Trade Watcher
Monitors Zcash (ZEC) trades in real-time from exchanges

Usage:
    python src/agents/zec_watcher.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SYMBOL = "ZEC/USDT"  # Trading pair to watch
EXCHANGE = "okx"      # Exchange to use (okx, kraken, coinbase)
CHECK_INTERVAL = 5    # Seconds between checks
SHOW_LAST_TRADES = 20 # Number of recent trades to show on startup

# Alternative symbols by exchange
EXCHANGE_SYMBOLS = {
    "okx": "ZEC/USDT",
    "kraken": "ZEC/USD",
    "coinbase": "ZEC/USD"
}

class ZECWatcher:
    def __init__(self, exchange_id=EXCHANGE, symbol=None):
        self.exchange_id = exchange_id
        self.symbol = symbol or EXCHANGE_SYMBOLS.get(exchange_id, SYMBOL)
        self.exchange = getattr(ccxt, exchange_id)()
        self.seen_trades = set()
        self.last_trade_id = None

        cprint(f"\n{'='*60}", "cyan")
        cprint(f"🌙 Moon Dev's ZEC Trade Watcher", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan")
        cprint(f"📊 Exchange: {exchange_id.upper()}", "white")
        cprint(f"💎 Symbol: {self.symbol}", "white")
        cprint(f"⏱️  Check interval: {CHECK_INTERVAL}s", "white")

    def fetch_recent_trades(self, limit=50):
        """Fetch recent trades from exchange"""
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            return trades
        except Exception as e:
            cprint(f"❌ Error fetching trades: {e}", "red")
            return []

    def format_trade(self, trade, is_new=False):
        """Format a trade for display"""
        side = trade['side'].upper()
        price = trade['price']
        amount = trade['amount']
        cost = trade.get('cost', price * amount)
        timestamp = trade['datetime'][:19] if trade.get('datetime') else 'Unknown'

        # Color based on side
        side_color = "green" if side == "BUY" else "red"
        emoji = "🟢" if side == "BUY" else "🔴"

        if is_new:
            bg = "on_blue" if side == "BUY" else "on_magenta"
            return colored(f"{emoji} NEW {side} | {amount:.4f} ZEC @ ${price:.2f} | ${cost:.2f} | {timestamp}", "white", bg)
        else:
            return f"{emoji} {colored(side, side_color):4} | {amount:.4f} ZEC @ ${price:.2f} | ${cost:.2f} | {timestamp}"

    def show_recent_trades(self):
        """Display recent trades on startup"""
        cprint(f"\n📜 Last {SHOW_LAST_TRADES} trades:", "yellow")
        cprint("-" * 70, "white")

        trades = self.fetch_recent_trades(SHOW_LAST_TRADES)
        if not trades:
            cprint("No trades found", "yellow")
            return

        for trade in trades[-SHOW_LAST_TRADES:]:
            print(self.format_trade(trade))
            self.seen_trades.add(trade['id'])

        if trades:
            self.last_trade_id = trades[-1]['id']

        cprint("-" * 70, "white")
        cprint(f"✅ Watching for new trades...\n", "green")

    def monitor(self):
        """Monitor for new trades"""
        while True:
            try:
                trades = self.fetch_recent_trades(50)

                new_trades = [t for t in trades if t['id'] not in self.seen_trades]

                for trade in new_trades:
                    print(self.format_trade(trade, is_new=True))
                    self.seen_trades.add(trade['id'])

                # Keep seen_trades from growing too large
                if len(self.seen_trades) > 1000:
                    self.seen_trades = set(list(self.seen_trades)[-500:])

            except KeyboardInterrupt:
                cprint("\n👋 ZEC Watcher shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"⚠️ Error: {e}", "yellow")

            time.sleep(CHECK_INTERVAL)

def main():
    watcher = ZECWatcher()
    watcher.show_recent_trades()
    watcher.monitor()

if __name__ == "__main__":
    main()
