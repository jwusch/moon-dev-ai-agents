"""
Moon Dev's MERL Liquidation Agent
Monitors MERL derivatives for liquidation levels, open interest, and funding rates

Usage:
    python src/agents/merl_liquidation_agent.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "MERL/USDT:USDT"
EXCHANGE = "okx"
CHECK_INTERVAL = 30
LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 25, 50]

class MERLLiquidationAgent:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})

        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's MERL Liquidation Agent", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")
        cprint(f"Loading {exchange_id.upper()} markets...", "white")

        self.exchange.load_markets()
        self.symbol = self._find_merl_perp()

        if not self.symbol:
            cprint(f"No MERL perpetual found on {exchange_id}", "red")
            return

        cprint(f"Symbol: {self.symbol}", "white")
        self.last_price = None
        self.last_oi = None

    def _find_merl_perp(self):
        for symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            if 'MERL' in symbol and market.get('swap', False):
                return symbol
        candidates = ["MERL/USDT:USDT", "MERL-USDT-SWAP"]
        for c in candidates:
            if c in self.exchange.markets:
                return c
        return None

    def fetch_ticker(self):
        try:
            return self.exchange.fetch_ticker(self.symbol)
        except Exception as e:
            cprint(f"Error fetching ticker: {e}", "red")
            return None

    def fetch_funding_rate(self):
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                return self.exchange.fetch_funding_rate(self.symbol)
            return None
        except:
            return None

    def fetch_open_interest(self):
        try:
            if hasattr(self.exchange, 'fetch_open_interest'):
                return self.exchange.fetch_open_interest(self.symbol)
            return None
        except:
            return None

    def calculate_liquidation_prices(self, entry_price, is_long=True):
        liq_prices = {}
        for lev in LEVERAGE_LEVELS:
            maint_margin = 0.005
            if is_long:
                liq_price = entry_price * (1 - (1/lev) + maint_margin)
            else:
                liq_price = entry_price * (1 + (1/lev) - maint_margin)
            liq_prices[lev] = liq_price
        return liq_prices

    def display_data(self):
        ticker = self.fetch_ticker()
        if not ticker:
            return

        current_price = ticker['last']
        price_change = ""
        if self.last_price:
            if current_price > self.last_price:
                price_change = colored("▲", "green")
            elif current_price < self.last_price:
                price_change = colored("▼", "red")
        self.last_price = current_price

        print("\n" + "=" * 70)
        print("MERL Derivatives Dashboard")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        change_24h = ticker.get('percentage', 0) or 0
        change_color = "green" if change_24h >= 0 else "red"

        print(f"Price: ${current_price:.4f} {price_change}")
        print(f"24h Change: {colored(f'{change_24h:+.2f}%', change_color)}")
        high_24h = ticker.get('high') or 0
        low_24h = ticker.get('low') or 0
        print(f"24h High: ${high_24h:.4f}  |  24h Low: ${low_24h:.4f}")
        quote_vol = ticker.get('quoteVolume') or ticker.get('baseVolume') or 0
        print(f"24h Volume: ${quote_vol:,.0f}")

        print("-" * 70)
        oi_data = self.fetch_open_interest()
        if oi_data:
            oi_value = oi_data.get('openInterestValue', 0) or oi_data.get('openInterest', 0) * current_price
            print(f"Open Interest: ${oi_value:,.0f}")

        funding = self.fetch_funding_rate()
        if funding:
            fund_rate = funding.get('fundingRate', 0) * 100
            fund_color = "green" if fund_rate < 0 else "red"
            print(f"Funding Rate: {colored(f'{fund_rate:.4f}%', fund_color)} (8h)")

        print("-" * 70)
        print(colored("LONG Liquidation Prices:", "green"))
        long_liqs = self.calculate_liquidation_prices(current_price, is_long=True)
        for lev, liq_price in long_liqs.items():
            dist = ((current_price - liq_price) / current_price) * 100
            print(f"  {lev:2}x: ${liq_price:.4f} ({dist:.1f}% below)")

        print("-" * 70)
        print(colored("SHORT Liquidation Prices:", "red"))
        short_liqs = self.calculate_liquidation_prices(current_price, is_long=False)
        for lev, liq_price in short_liqs.items():
            dist = ((liq_price - current_price) / current_price) * 100
            print(f"  {lev:2}x: ${liq_price:.4f} ({dist:.1f}% above)")

        print("=" * 70)

    def monitor(self):
        cprint("\nStarting MERL liquidation monitoring...\n", "green")
        while True:
            try:
                self.display_data()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                cprint("\nMERL Liquidation Agent shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"Error: {e}", "yellow")
                time.sleep(5)

def main():
    agent = MERLLiquidationAgent()
    if agent.symbol:
        agent.monitor()

if __name__ == "__main__":
    main()
