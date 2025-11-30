"""
Moon Dev's MERL Order Book Monitor
Monitors MERL order books across multiple exchanges

Usage:
    python src/agents/merl_orderbook_monitor.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECK_INTERVAL = 10
ORDERBOOK_DEPTH = 20
WHALE_ORDER_USD = 5000  # Lower threshold for MERL

EXCHANGES = {
    "okx": {"symbol": "MERL/USDT", "name": "OKX"},
    "kraken": {"symbol": "MERL/USD", "name": "Kraken"},
    "kucoin": {"symbol": "MERL/USDT", "name": "KuCoin"},
    "gate": {"symbol": "MERL/USDT", "name": "Gate.io"},
}

class MERLOrderBookMonitor:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's MERL Order Book Monitor", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")

        self.exchanges = {}
        self.symbols = {}

        for ex_id, config in EXCHANGES.items():
            try:
                exchange = getattr(ccxt, ex_id)({'enableRateLimit': True})
                exchange.load_markets()
                symbol = config["symbol"]

                if symbol in exchange.markets:
                    self.exchanges[ex_id] = exchange
                    self.symbols[ex_id] = symbol
                    cprint(f"  {config['name']}: {symbol}", "green")
                else:
                    cprint(f"  {config['name']}: MERL not found", "yellow")
            except Exception as e:
                cprint(f"  {config['name']}: Failed - {str(e)[:40]}", "red")

        if self.exchanges:
            cprint(f"\nMonitoring {len(self.exchanges)} exchanges", "white")
            cprint(f"Whale order threshold: ${WHALE_ORDER_USD:,}", "white")

    def fetch_orderbook(self, exchange_id):
        try:
            exchange = self.exchanges[exchange_id]
            symbol = self.symbols[exchange_id]
            return exchange.fetch_order_book(symbol, limit=ORDERBOOK_DEPTH)
        except:
            return None

    def fetch_ticker(self, exchange_id):
        try:
            exchange = self.exchanges[exchange_id]
            symbol = self.symbols[exchange_id]
            return exchange.fetch_ticker(symbol)
        except:
            return None

    def analyze_orderbook(self, orderbook, price):
        if not orderbook or not price:
            return None

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not bids or not asks:
            return None

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0

        bid_volume = sum(b[1] for b in bids)
        ask_volume = sum(a[1] for a in asks)
        bid_value = sum(b[0] * b[1] for b in bids)
        ask_value = sum(a[0] * a[1] for a in asks)

        whale_bids = [(b[0], b[1], b[0]*b[1]) for b in bids if b[0]*b[1] >= WHALE_ORDER_USD]
        whale_asks = [(a[0], a[1], a[0]*a[1]) for a in asks if a[0]*a[1] >= WHALE_ORDER_USD]

        imbalance = (bid_value - ask_value) / (bid_value + ask_value) * 100 if (bid_value + ask_value) > 0 else 0

        return {
            'best_bid': best_bid, 'best_ask': best_ask,
            'spread': spread, 'spread_pct': spread_pct,
            'bid_volume': bid_volume, 'ask_volume': ask_volume,
            'bid_value': bid_value, 'ask_value': ask_value,
            'whale_bids': whale_bids, 'whale_asks': whale_asks,
            'imbalance': imbalance,
            'bids': bids[:5], 'asks': asks[:5],
        }

    def display_dashboard(self):
        print("\n" + "=" * 80)
        print("MERL ORDER BOOK MONITOR")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        all_analyses = {}
        for ex_id in self.exchanges:
            ticker = self.fetch_ticker(ex_id)
            orderbook = self.fetch_orderbook(ex_id)
            price = ticker['last'] if ticker else None
            analysis = self.analyze_orderbook(orderbook, price)
            if analysis:
                all_analyses[ex_id] = {'analysis': analysis, 'price': price}

        if not all_analyses:
            print("No orderbook data available")
            return

        print("\n EXCHANGE COMPARISON")
        print("-" * 80)
        print(f"{'Exchange':<12} {'Price':>10} {'Spread':>10} {'Bid Vol':>12} {'Ask Vol':>12} {'Imbalance':>10}")
        print("-" * 80)

        for ex_id, data in all_analyses.items():
            a = data['analysis']
            price = data['price']
            name = EXCHANGES[ex_id]['name']
            imb = a['imbalance']
            if imb > 10:
                imb_str = colored(f"+{imb:.1f}%", "green")
            elif imb < -10:
                imb_str = colored(f"{imb:.1f}%", "red")
            else:
                imb_str = f"{imb:.1f}%"
            print(f"{name:<12} ${price:>9,.4f} {a['spread_pct']:>9.3f}% {a['bid_volume']:>11,.0f} {a['ask_volume']:>11,.0f} {imb_str:>10}")

        # Whale orders
        print("\n WHALE ORDERS (>$5k)")
        print("-" * 80)

        all_whale_bids = []
        all_whale_asks = []
        for ex_id, data in all_analyses.items():
            a = data['analysis']
            name = EXCHANGES[ex_id]['name']
            for wb in a['whale_bids']:
                all_whale_bids.append((name, wb[0], wb[1], wb[2]))
            for wa in a['whale_asks']:
                all_whale_asks.append((name, wa[0], wa[1], wa[2]))

        if all_whale_asks:
            print(colored("  WHALE SELLS (resistance):", "red"))
            for ex, price, size, value in sorted(all_whale_asks, key=lambda x: x[1])[:8]:
                print(f"    {ex:<10} ${price:>8,.4f} | {size:>12,.0f} MERL | ${value:>10,.0f}")

        if all_whale_bids:
            print(colored("\n  WHALE BUYS (support):", "green"))
            for ex, price, size, value in sorted(all_whale_bids, key=lambda x: -x[1])[:8]:
                print(f"    {ex:<10} ${price:>8,.4f} | {size:>12,.0f} MERL | ${value:>10,.0f}")

        if not all_whale_bids and not all_whale_asks:
            print("  No whale orders detected")

        # Market pressure
        total_bid_value = sum(d['analysis']['bid_value'] for d in all_analyses.values())
        total_ask_value = sum(d['analysis']['ask_value'] for d in all_analyses.values())

        if total_bid_value + total_ask_value > 0:
            buy_pressure = total_bid_value / (total_bid_value + total_ask_value) * 100
            sell_pressure = 100 - buy_pressure

            print("\n MARKET PRESSURE")
            print("-" * 80)
            bar_len = 40
            buy_bars = int(buy_pressure / 100 * bar_len)
            buy_bar = colored("█" * buy_bars, "green")
            sell_bar = colored("█" * (bar_len - buy_bars), "red")
            print(f"  {buy_bar}{sell_bar}")
            print(f"  {buy_pressure:.1f}% BUY | ${total_bid_value:,.0f} vs ${total_ask_value:,.0f} | {sell_pressure:.1f}% SELL")

        print("\n" + "=" * 80)

    def monitor(self):
        if not self.exchanges:
            return
        cprint("\nStarting orderbook monitoring...\n", "green")
        while True:
            try:
                self.display_dashboard()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                cprint("\nOrderbook Monitor shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"Error: {e}", "yellow")
                time.sleep(5)

def main():
    monitor = MERLOrderBookMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main()
