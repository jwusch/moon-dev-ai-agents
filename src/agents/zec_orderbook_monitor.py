"""
Moon Dev's ZEC Order Book Monitor
Monitors ZEC order books across multiple exchanges (Coinbase, Binance, OKX)
Shows bid/ask spreads, depth, and whale walls

No API keys required - uses public data via CCXT

Usage:
    python src/agents/zec_orderbook_monitor.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
CHECK_INTERVAL = 10  # Seconds between updates
ORDERBOOK_DEPTH = 20  # Levels to fetch
WHALE_ORDER_USD = 10000  # Minimum USD to highlight as whale order

# Exchanges to monitor
# Note: Using binanceus for US users (binance.com is geo-restricted)
EXCHANGES = {
    "binanceus": {"symbol": "ZEC/USD", "name": "Binance.US"},
    "coinbase": {"symbol": "ZEC/USD", "name": "Coinbase"},
    "okx": {"symbol": "ZEC/USDT", "name": "OKX"},
    "kraken": {"symbol": "ZEC/USD", "name": "Kraken"},
}

class ZECOrderBookMonitor:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's ZEC Order Book Monitor", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")

        self.exchanges = {}
        self.symbols = {}

        # Initialize exchanges
        for ex_id, config in EXCHANGES.items():
            try:
                exchange = getattr(ccxt, ex_id)({'enableRateLimit': True})
                exchange.load_markets()

                # Check if ZEC is available
                symbol = config["symbol"]
                if symbol in exchange.markets:
                    self.exchanges[ex_id] = exchange
                    self.symbols[ex_id] = symbol
                    cprint(f"  {config['name']}: {symbol}", "green")
                else:
                    # Try alternative symbols
                    alternatives = ["ZEC/USDT", "ZEC/USD", "ZEC/BUSD"]
                    found = False
                    for alt in alternatives:
                        if alt in exchange.markets:
                            self.exchanges[ex_id] = exchange
                            self.symbols[ex_id] = alt
                            cprint(f"  {config['name']}: {alt}", "green")
                            found = True
                            break
                    if not found:
                        cprint(f"  {config['name']}: ZEC not found", "yellow")
            except Exception as e:
                cprint(f"  {config['name']}: Failed - {str(e)[:40]}", "red")

        if not self.exchanges:
            cprint("No exchanges available!", "red")
            return

        cprint(f"\nMonitoring {len(self.exchanges)} exchanges", "white")
        cprint(f"Whale order threshold: ${WHALE_ORDER_USD:,}", "white")

    def fetch_orderbook(self, exchange_id):
        """Fetch orderbook from an exchange"""
        try:
            exchange = self.exchanges[exchange_id]
            symbol = self.symbols[exchange_id]
            orderbook = exchange.fetch_order_book(symbol, limit=ORDERBOOK_DEPTH)
            return orderbook
        except Exception as e:
            return None

    def fetch_ticker(self, exchange_id):
        """Fetch current price"""
        try:
            exchange = self.exchanges[exchange_id]
            symbol = self.symbols[exchange_id]
            ticker = exchange.fetch_ticker(symbol)
            return ticker
        except:
            return None

    def analyze_orderbook(self, orderbook, price):
        """Analyze orderbook for key metrics"""
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

        # Calculate total bid/ask volume
        bid_volume = sum(b[1] for b in bids)
        ask_volume = sum(a[1] for a in asks)
        bid_value = sum(b[0] * b[1] for b in bids)
        ask_value = sum(a[0] * a[1] for a in asks)

        # Find whale orders (large orders)
        whale_bids = [(b[0], b[1], b[0]*b[1]) for b in bids if b[0]*b[1] >= WHALE_ORDER_USD]
        whale_asks = [(a[0], a[1], a[0]*a[1]) for a in asks if a[0]*a[1] >= WHALE_ORDER_USD]

        # Bid/ask imbalance
        imbalance = (bid_value - ask_value) / (bid_value + ask_value) * 100 if (bid_value + ask_value) > 0 else 0

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': spread_pct,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_value': bid_value,
            'ask_value': ask_value,
            'whale_bids': whale_bids,
            'whale_asks': whale_asks,
            'imbalance': imbalance,
            'bids': bids[:5],  # Top 5 levels
            'asks': asks[:5],
        }

    def display_dashboard(self):
        """Display the orderbook dashboard"""
        print("\n" + "=" * 80)
        print("ZEC ORDER BOOK MONITOR")
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

        # Summary table
        print("\n EXCHANGE COMPARISON")
        print("-" * 80)
        print(f"{'Exchange':<12} {'Price':>10} {'Spread':>10} {'Bid Vol':>12} {'Ask Vol':>12} {'Imbalance':>10}")
        print("-" * 80)

        for ex_id, data in all_analyses.items():
            a = data['analysis']
            price = data['price']
            name = EXCHANGES[ex_id]['name']

            # Color imbalance
            imb = a['imbalance']
            if imb > 10:
                imb_str = colored(f"+{imb:.1f}%", "green")
            elif imb < -10:
                imb_str = colored(f"{imb:.1f}%", "red")
            else:
                imb_str = f"{imb:.1f}%"

            print(f"{name:<12} ${price:>9,.2f} {a['spread_pct']:>9.3f}% {a['bid_volume']:>11,.2f} {a['ask_volume']:>11,.2f} {imb_str:>10}")

        print("-" * 80)

        # Detailed view for each exchange
        for ex_id, data in all_analyses.items():
            a = data['analysis']
            price = data['price']
            name = EXCHANGES[ex_id]['name']

            print(f"\n {name} - ${price:,.2f}")
            print("-" * 40)

            # Show top 5 asks (sells) - in reverse order so lowest is at bottom
            print("  ASKS (Sells):")
            for ask in reversed(a['asks']):
                ask_price, ask_size = ask[0], ask[1]
                ask_value = ask_price * ask_size
                dist = ((ask_price - price) / price) * 100

                if ask_value >= WHALE_ORDER_USD:
                    print(colored(f"    ${ask_price:>8,.2f} | {ask_size:>10,.4f} ZEC | ${ask_value:>10,.0f} | +{dist:.2f}%", "red", attrs=['bold']))
                else:
                    print(f"    ${ask_price:>8,.2f} | {ask_size:>10,.4f} ZEC | ${ask_value:>10,.0f} | +{dist:.2f}%")

            print(f"  {'─'*38}")
            print(colored(f"    SPREAD: ${a['spread']:.4f} ({a['spread_pct']:.3f}%)", "yellow"))
            print(f"  {'─'*38}")

            # Show top 5 bids (buys)
            print("  BIDS (Buys):")
            for bid in a['bids']:
                bid_price, bid_size = bid[0], bid[1]
                bid_value = bid_price * bid_size
                dist = ((price - bid_price) / price) * 100

                if bid_value >= WHALE_ORDER_USD:
                    print(colored(f"    ${bid_price:>8,.2f} | {bid_size:>10,.4f} ZEC | ${bid_value:>10,.0f} | -{dist:.2f}%", "green", attrs=['bold']))
                else:
                    print(f"    ${bid_price:>8,.2f} | {bid_size:>10,.4f} ZEC | ${bid_value:>10,.0f} | -{dist:.2f}%")

        # Whale orders summary
        print("\n WHALE ORDERS (>$10k)")
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

        if all_whale_bids or all_whale_asks:
            if all_whale_asks:
                print(colored("  WHALE SELLS (resistance):", "red"))
                for ex, price, size, value in sorted(all_whale_asks, key=lambda x: x[1]):
                    print(f"    {ex:<10} ${price:>8,.2f} | {size:>10,.2f} ZEC | ${value:>10,.0f}")

            if all_whale_bids:
                print(colored("\n  WHALE BUYS (support):", "green"))
                for ex, price, size, value in sorted(all_whale_bids, key=lambda x: -x[1]):
                    print(f"    {ex:<10} ${price:>8,.2f} | {size:>10,.2f} ZEC | ${value:>10,.0f}")
        else:
            print("  No whale orders detected")

        # Market pressure indicator
        print("\n MARKET PRESSURE")
        print("-" * 80)

        total_bid_value = sum(d['analysis']['bid_value'] for d in all_analyses.values())
        total_ask_value = sum(d['analysis']['ask_value'] for d in all_analyses.values())

        if total_bid_value + total_ask_value > 0:
            buy_pressure = total_bid_value / (total_bid_value + total_ask_value) * 100
            sell_pressure = 100 - buy_pressure

            bar_len = 40
            buy_bars = int(buy_pressure / 100 * bar_len)
            sell_bars = bar_len - buy_bars

            buy_bar = colored("█" * buy_bars, "green")
            sell_bar = colored("█" * sell_bars, "red")

            print(f"  Buy Pressure:  {buy_bar}{sell_bar} Sell Pressure")
            print(f"  {buy_pressure:.1f}% BUY | ${total_bid_value:,.0f} vs ${total_ask_value:,.0f} | {sell_pressure:.1f}% SELL")

            if buy_pressure > 60:
                print(colored("  Bullish orderbook - more buyers than sellers", "green"))
            elif sell_pressure > 60:
                print(colored("  Bearish orderbook - more sellers than buyers", "red"))
            else:
                print("  Neutral orderbook - balanced")

        print("\n" + "=" * 80)

    def monitor(self):
        """Continuously monitor orderbooks"""
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
    monitor = ZECOrderBookMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main()
