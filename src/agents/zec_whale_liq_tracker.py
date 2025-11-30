"""
🌙 Moon Dev's ZEC Whale Liquidation Price Tracker
Tracks large trades and calculates their liquidation price levels

Shows where whale positions would get liquidated relative to current price
Inspired by Moon Dev's liquidation tracking approach

Usage:
    python src/agents/zec_whale_liq_tracker.py
"""

import ccxt
import time
from datetime import datetime, timedelta
from termcolor import colored, cprint
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SYMBOL = "ZEC/USDT:USDT"  # ZEC perpetual
EXCHANGE = "okx"
CHECK_INTERVAL = 10  # Seconds between updates
WHALE_THRESHOLD_USD = 5000  # Minimum USD value to be considered a whale trade
LEVERAGE_LEVELS = [3, 5, 10, 20, 25]  # Common leverage levels
TRACK_HOURS = 4  # How many hours of trades to track

class ZECWhaleLiqTracker:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })

        cprint(f"\n{'='*70}", "cyan")
        cprint(f"🐋 Moon Dev's ZEC Whale Liquidation Tracker", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")
        cprint(f"📊 Loading {exchange_id.upper()} markets...", "white")

        self.exchange.load_markets()

        # Find ZEC perpetual symbol
        self.symbol = self._find_zec_perp()
        if not self.symbol:
            cprint(f"❌ No ZEC perpetual found on {exchange_id}", "red")
            return

        cprint(f"💎 Symbol: {self.symbol}", "white")
        cprint(f"🐋 Whale threshold: ${WHALE_THRESHOLD_USD:,}", "white")
        cprint(f"⏱️  Tracking last {TRACK_HOURS} hours of trades", "white")

        # Store whale entries: {price: {'longs': volume, 'shorts': volume}}
        self.whale_entries = defaultdict(lambda: {'longs': 0, 'shorts': 0, 'count': 0})
        self.seen_trades = set()

    def _find_zec_perp(self):
        """Find the ZEC perpetual futures symbol"""
        for symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            if 'ZEC' in symbol and market.get('swap', False):
                return symbol
        candidates = ["ZEC/USDT:USDT", "ZEC-USDT-SWAP", "ZECUSDT"]
        for c in candidates:
            if c in self.exchange.markets:
                return c
        return None

    def fetch_recent_trades(self, limit=500):
        """Fetch recent trades"""
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            return trades
        except Exception as e:
            cprint(f"❌ Error fetching trades: {e}", "red")
            return []

    def fetch_ticker(self):
        """Get current price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker
        except Exception as e:
            cprint(f"❌ Error fetching ticker: {e}", "red")
            return None

    def calculate_liquidation_price(self, entry_price, leverage, is_long=True):
        """Calculate liquidation price for a position"""
        maint_margin = 0.005  # ~0.5% maintenance margin
        if is_long:
            liq_price = entry_price * (1 - (1/leverage) + maint_margin)
        else:
            liq_price = entry_price * (1 + (1/leverage) - maint_margin)
        return liq_price

    def update_whale_entries(self):
        """Update whale entry tracking from recent trades"""
        trades = self.fetch_recent_trades(500)

        cutoff_time = datetime.utcnow() - timedelta(hours=TRACK_HOURS)

        new_whales = 0
        for trade in trades:
            trade_id = trade['id']
            if trade_id in self.seen_trades:
                continue

            self.seen_trades.add(trade_id)

            # Check if trade is within our time window
            trade_time = datetime.fromisoformat(trade['datetime'].replace('Z', '+00:00')).replace(tzinfo=None)
            if trade_time < cutoff_time:
                continue

            # Check if it's a whale trade
            cost = trade.get('cost', trade['price'] * trade['amount'])
            if cost < WHALE_THRESHOLD_USD:
                continue

            # Round price to nearest $1 for bucketing
            price_bucket = round(trade['price'])
            side = trade['side'].upper()

            if side == 'BUY':
                self.whale_entries[price_bucket]['longs'] += cost
            else:
                self.whale_entries[price_bucket]['shorts'] += cost

            self.whale_entries[price_bucket]['count'] += 1
            new_whales += 1

        return new_whales

    def get_liquidation_levels(self, current_price):
        """Calculate liquidation levels for all tracked whale entries"""
        long_liq_levels = defaultdict(float)  # {liq_price: total_volume}
        short_liq_levels = defaultdict(float)

        for entry_price, data in self.whale_entries.items():
            for leverage in LEVERAGE_LEVELS:
                # Long liquidation prices (below entry)
                if data['longs'] > 0:
                    liq_price = self.calculate_liquidation_price(entry_price, leverage, is_long=True)
                    long_liq_levels[round(liq_price)] += data['longs'] / len(LEVERAGE_LEVELS)

                # Short liquidation prices (above entry)
                if data['shorts'] > 0:
                    liq_price = self.calculate_liquidation_price(entry_price, leverage, is_long=False)
                    short_liq_levels[round(liq_price)] += data['shorts'] / len(LEVERAGE_LEVELS)

        return long_liq_levels, short_liq_levels

    def display_liquidation_map(self):
        """Display the whale liquidation price map"""
        ticker = self.fetch_ticker()
        if not ticker:
            return

        current_price = ticker['last']

        # Update whale entries
        new_whales = self.update_whale_entries()

        # Get liquidation levels
        long_liq_levels, short_liq_levels = self.get_liquidation_levels(current_price)

        # Find liquidation levels near current price
        price_range = current_price * 0.15  # +/- 15% from current price

        # Filter levels within range
        nearby_long_liqs = {p: v for p, v in long_liq_levels.items()
                           if current_price - price_range <= p <= current_price}
        nearby_short_liqs = {p: v for p, v in short_liq_levels.items()
                            if current_price <= p <= current_price + price_range}

        # Header
        print("\n" + "═" * 70)
        print("🐋 ZEC Whale Liquidation Price Map")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 70)

        # Current price
        print(f"💰 Current Price: ${current_price:,.2f}")
        whale_count = sum(d['count'] for d in self.whale_entries.values())
        print(f"🐋 Whale trades tracked: {whale_count}")

        # Short liquidation levels (ABOVE current price - where shorts get rekt)
        print("─" * 70)
        print("🔺 SHORT LIQ LEVELS (price UP = shorts liquidated)")

        if nearby_short_liqs:
            sorted_shorts = sorted(nearby_short_liqs.items(), key=lambda x: x[0])[:6]
            for liq_price, volume in sorted_shorts:
                dist = ((liq_price - current_price) / current_price) * 100
                bar_len = min(int(volume / 5000), 25)
                bar = "█" * bar_len
                print(f"     ${liq_price:>6,.0f} (+{dist:>4.1f}%) ${volume:>8,.0f} {bar}")
        else:
            print("     No significant short liquidation levels nearby")

        # Visual separator with current price marker
        print("─" * 70)
        print(f"         >>> 💰 CURRENT PRICE: ${current_price:,.2f} <<< ")
        print("─" * 70)

        # Long liquidation levels (BELOW current price - where longs get rekt)
        print("🔻 LONG LIQ LEVELS (price DOWN = longs liquidated)")

        if nearby_long_liqs:
            sorted_longs = sorted(nearby_long_liqs.items(), key=lambda x: x[0], reverse=True)[:6]
            for liq_price, volume in sorted_longs:
                dist = ((current_price - liq_price) / current_price) * 100
                bar_len = min(int(volume / 5000), 25)
                bar = "█" * bar_len
                print(f"     ${liq_price:>6,.0f} (-{dist:>4.1f}%) ${volume:>8,.0f} {bar}")
        else:
            print("     No significant long liquidation levels nearby")

        # Summary
        print("─" * 70)
        total_long_liq = sum(nearby_long_liqs.values()) if nearby_long_liqs else 0
        total_short_liq = sum(nearby_short_liqs.values()) if nearby_short_liqs else 0

        print("📊 SUMMARY:")
        print(f"     Long liq exposure (below):  ${total_long_liq:>10,.0f}")
        print(f"     Short liq exposure (above): ${total_short_liq:>10,.0f}")

        if total_long_liq > total_short_liq * 1.5:
            print("     ⚠️  More longs at risk - watch for downside cascade!")
        elif total_short_liq > total_long_liq * 1.5:
            print("     ⚠️  More shorts at risk - watch for upside squeeze!")

        print("═" * 70)

    def monitor(self):
        """Continuously monitor whale liquidation levels"""
        cprint("\n✅ Starting whale liquidation tracking...\n", "green")

        while True:
            try:
                self.display_liquidation_map()
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n👋 Whale Liquidation Tracker shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"⚠️ Error: {e}", "yellow")
                time.sleep(5)

def main():
    tracker = ZECWhaleLiqTracker()
    if tracker.symbol:
        tracker.monitor()

if __name__ == "__main__":
    main()
