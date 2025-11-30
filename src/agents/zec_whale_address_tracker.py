"""
🌙 Moon Dev's ZEC Whale Address Tracker
Tracks large ZEC wallet addresses and their movements

Approaches:
1. Monitor large transactions on the ZEC blockchain
2. Track known whale addresses
3. Detect new whales from exchange trades

Note: ZEC has transparent (t-addresses) and shielded (z-addresses)
      We can only track transparent addresses publicly

Usage:
    python src/agents/zec_whale_address_tracker.py
"""

import requests
import time
from datetime import datetime, timedelta
from termcolor import colored, cprint
import os
import sys
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
CHECK_INTERVAL = 60  # Seconds between checks
WHALE_THRESHOLD_ZEC = 100  # Minimum ZEC to be considered a whale transaction
DATA_DIR = PROJECT_ROOT / "src" / "data" / "zec_whales"

# Known exchange addresses (transparent addresses only)
KNOWN_ADDRESSES = {
    # Major exchanges (examples - you can add more)
    "t1Kef6MquYAGMDQTKKsWRwLfNLf4xREqVv1": "Binance Hot Wallet",
    "t1eXiVKk9VwLw57FZYaxS58h6N8qGEZwPK3": "Kraken",
    "t1Rj4Mb1E3kSu4nkjqLJz4p1CJuRr9v6jkv": "Bittrex",
}

class ZECWhaleAddressTracker:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.whale_addresses_file = DATA_DIR / "whale_addresses.json"
        self.transactions_file = DATA_DIR / "whale_transactions.json"

        # Load or initialize whale addresses
        self.whale_addresses = self._load_whale_addresses()
        self.recent_transactions = []

        cprint(f"\n{'='*70}", "cyan")
        cprint(f"🐋 Moon Dev's ZEC Whale Address Tracker", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")
        cprint(f"📊 Whale threshold: {WHALE_THRESHOLD_ZEC} ZEC", "white")
        cprint(f"💾 Data directory: {DATA_DIR}", "white")
        cprint(f"🐋 Tracked addresses: {len(self.whale_addresses)}", "white")

    def _load_whale_addresses(self):
        """Load saved whale addresses"""
        if self.whale_addresses_file.exists():
            try:
                with open(self.whale_addresses_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Start with known addresses
        return {addr: {"label": label, "total_volume": 0, "tx_count": 0, "first_seen": None, "last_seen": None}
                for addr, label in KNOWN_ADDRESSES.items()}

    def _save_whale_addresses(self):
        """Save whale addresses to file"""
        with open(self.whale_addresses_file, 'w') as f:
            json.dump(self.whale_addresses, f, indent=2, default=str)

    def fetch_recent_blocks(self):
        """Fetch recent ZEC blocks using Blockchair API"""
        try:
            # Blockchair provides free ZEC blockchain data
            url = "https://api.blockchair.com/zcash/blocks?limit=5"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                print(f"⚠️ Blockchair API returned status {response.status_code}")
                return []

        except Exception as e:
            print(f"⚠️ Error fetching blocks: {e}")
            return []

    def fetch_large_transactions(self):
        """Fetch large ZEC transactions"""
        try:
            # Blockchair API for large transactions
            # Note: Free tier has rate limits
            url = f"https://api.blockchair.com/zcash/transactions?q=output_total(gte:{int(WHALE_THRESHOLD_ZEC * 1e8)})&limit=10"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                return []

        except Exception as e:
            print(f"⚠️ Error fetching transactions: {e}")
            return []

    def fetch_richlist(self):
        """Fetch ZEC rich list (top holders)"""
        try:
            # Using zcashblockexplorer.com API
            url = "https://api.blockchair.com/zcash/addresses?limit=20&s=balance(desc)"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            return []

        except Exception as e:
            print(f"⚠️ Error fetching rich list: {e}")
            return []

    def add_whale_address(self, address, volume, label=None):
        """Add or update a whale address"""
        now = datetime.now().isoformat()

        if address not in self.whale_addresses:
            self.whale_addresses[address] = {
                "label": label or "Unknown Whale",
                "total_volume": 0,
                "tx_count": 0,
                "first_seen": now,
                "last_seen": now
            }
            cprint(f"🆕 New whale detected: {address[:16]}...", "yellow")

        self.whale_addresses[address]["total_volume"] += volume
        self.whale_addresses[address]["tx_count"] += 1
        self.whale_addresses[address]["last_seen"] = now

        self._save_whale_addresses()

    def display_dashboard(self):
        """Display whale tracking dashboard"""
        print("\n" + "═" * 70)
        print("🐋 ZEC Whale Address Dashboard")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)

        # Fetch latest data
        print("\n📊 Fetching blockchain data...")

        # Get rich list
        richlist = self.fetch_richlist()

        if richlist:
            print("\n💰 TOP ZEC HOLDERS (Rich List):")
            print("─" * 70)
            print(f"{'Rank':<6} {'Address':<40} {'Balance (ZEC)':<15}")
            print("─" * 70)

            for i, addr_data in enumerate(richlist[:10], 1):
                if isinstance(addr_data, dict):
                    address = addr_data.get('address', 'Unknown')
                    balance = addr_data.get('balance', 0) / 1e8  # Convert from zatoshi

                    # Check if known address
                    label = ""
                    if address in KNOWN_ADDRESSES:
                        label = f" ({KNOWN_ADDRESSES[address]})"

                    # Track this whale
                    if balance >= WHALE_THRESHOLD_ZEC:
                        self.add_whale_address(address, 0, KNOWN_ADDRESSES.get(address))

                    print(f"{i:<6} {address[:38]:<40} {balance:>12,.2f}{label}")

        # Show tracked whales
        print("\n🐋 TRACKED WHALE ADDRESSES:")
        print("─" * 70)

        # Sort by total volume
        sorted_whales = sorted(
            self.whale_addresses.items(),
            key=lambda x: x[1].get('total_volume', 0),
            reverse=True
        )[:15]

        if sorted_whales:
            print(f"{'Address':<40} {'Label':<20} {'TXs':<6}")
            print("─" * 70)
            for address, data in sorted_whales:
                label = data.get('label', 'Unknown')[:18]
                tx_count = data.get('tx_count', 0)
                print(f"{address[:38]:<40} {label:<20} {tx_count:<6}")
        else:
            print("No whale addresses tracked yet")

        # Show recent large transactions
        large_txs = self.fetch_large_transactions()

        if large_txs:
            print("\n🔥 RECENT LARGE TRANSACTIONS:")
            print("─" * 70)

            for tx in large_txs[:5]:
                if isinstance(tx, dict):
                    tx_hash = tx.get('hash', 'Unknown')[:16]
                    output_total = tx.get('output_total', 0) / 1e8
                    block_time = tx.get('time', 'Unknown')

                    if output_total >= WHALE_THRESHOLD_ZEC:
                        print(f"TX: {tx_hash}... | {output_total:,.2f} ZEC | {block_time}")

        print("\n" + "═" * 70)
        print(f"🐋 Total tracked addresses: {len(self.whale_addresses)}")
        print("═" * 70)

    def display_simple_stats(self):
        """Display simple stats without API calls (for faster refresh)"""
        print("\n" + "═" * 70)
        print("🐋 ZEC Whale Tracker - Quick Stats")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)

        print(f"\n📊 Tracked Whale Addresses: {len(self.whale_addresses)}")

        # Show top tracked whales
        sorted_whales = sorted(
            self.whale_addresses.items(),
            key=lambda x: x[1].get('tx_count', 0),
            reverse=True
        )[:10]

        if sorted_whales:
            print("\n🐋 Most Active Whales:")
            print("─" * 70)
            for address, data in sorted_whales:
                label = data.get('label', 'Unknown')[:20]
                tx_count = data.get('tx_count', 0)
                last_seen = data.get('last_seen', 'Never')
                if last_seen and last_seen != 'Never':
                    last_seen = last_seen[:19]
                print(f"  {address[:20]}... | {label:<20} | TXs: {tx_count} | Last: {last_seen}")

        print("\n" + "═" * 70)
        print("💡 Full refresh with blockchain data every 5 minutes")
        print("═" * 70)

    def monitor(self):
        """Continuously monitor whale addresses"""
        cprint("\n✅ Starting whale address monitoring...\n", "green")

        iteration = 0
        while True:
            try:
                # Full dashboard every 5 iterations (5 minutes with 60s interval)
                if iteration % 5 == 0:
                    self.display_dashboard()
                else:
                    self.display_simple_stats()

                iteration += 1
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n👋 Whale Address Tracker shutting down...", "yellow")
                self._save_whale_addresses()
                break
            except Exception as e:
                cprint(f"⚠️ Error: {e}", "yellow")
                time.sleep(10)

def main():
    tracker = ZECWhaleAddressTracker()
    tracker.monitor()

if __name__ == "__main__":
    main()
