"""
🌙 Moon Dev's ZEC Liquidation Agent
Monitors Zcash (ZEC) derivatives for liquidation levels, open interest, and funding rates

Usage:
    python src/agents/zec_liquidation_agent.py
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
SYMBOL = "ZEC/USDT:USDT"  # ZEC perpetual futures
EXCHANGE = "okx"
CHECK_INTERVAL = 30  # Seconds between checks
LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 25, 50]  # Common leverage levels to calculate liq prices

class ZECLiquidationAgent:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })

        # Load markets
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"🌙 Moon Dev's ZEC Liquidation Agent", "cyan", attrs=['bold'])
        cprint(f"{'='*70}", "cyan")
        cprint(f"📊 Loading {exchange_id.upper()} markets...", "white")

        self.exchange.load_markets()

        # Find ZEC perpetual symbol
        self.symbol = self._find_zec_perp()
        if not self.symbol:
            cprint(f"❌ No ZEC perpetual found on {exchange_id}", "red")
            return

        cprint(f"💎 Symbol: {self.symbol}", "white")
        cprint(f"⏱️  Check interval: {CHECK_INTERVAL}s", "white")

        self.last_price = None
        self.last_oi = None

    def _find_zec_perp(self):
        """Find the ZEC perpetual futures symbol"""
        for symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            if 'ZEC' in symbol and market.get('swap', False):
                return symbol
        # Try common formats
        candidates = ["ZEC/USDT:USDT", "ZEC-USDT-SWAP", "ZECUSDT"]
        for c in candidates:
            if c in self.exchange.markets:
                return c
        return None

    def fetch_ticker(self):
        """Fetch current price and 24h stats"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker
        except Exception as e:
            cprint(f"❌ Error fetching ticker: {e}", "red")
            return None

    def fetch_funding_rate(self):
        """Fetch current funding rate"""
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = self.exchange.fetch_funding_rate(self.symbol)
                return funding
            return None
        except Exception as e:
            cprint(f"⚠️ Funding rate not available: {e}", "yellow")
            return None

    def fetch_open_interest(self):
        """Fetch open interest"""
        try:
            if hasattr(self.exchange, 'fetch_open_interest'):
                oi = self.exchange.fetch_open_interest(self.symbol)
                return oi
            return None
        except Exception as e:
            cprint(f"⚠️ Open interest not available: {e}", "yellow")
            return None

    def calculate_liquidation_prices(self, entry_price, is_long=True):
        """Calculate liquidation prices for different leverage levels"""
        liq_prices = {}
        for lev in LEVERAGE_LEVELS:
            # Simplified liquidation calculation
            # Real calculation depends on maintenance margin, but this gives approximation
            # For longs: liq_price = entry * (1 - 1/leverage + maintenance_margin)
            # For shorts: liq_price = entry * (1 + 1/leverage - maintenance_margin)
            maint_margin = 0.005  # ~0.5% maintenance margin approximation

            if is_long:
                liq_price = entry_price * (1 - (1/lev) + maint_margin)
            else:
                liq_price = entry_price * (1 + (1/lev) - maint_margin)

            liq_prices[lev] = liq_price
        return liq_prices

    def display_data(self):
        """Fetch and display all ZEC derivatives data"""
        ticker = self.fetch_ticker()
        if not ticker:
            return

        current_price = ticker['last']

        # Price change indicator
        price_change = ""
        if self.last_price:
            if current_price > self.last_price:
                price_change = colored("▲", "green")
            elif current_price < self.last_price:
                price_change = colored("▼", "red")
        self.last_price = current_price

        # Header
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + colored("  🌙 ZEC Derivatives Dashboard", "cyan").center(77) + "║")
        print("║" + colored(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white").center(77) + "║")
        print("╠" + "═" * 68 + "╣")

        # Price Info
        change_24h = ticker.get('percentage', 0) or 0
        change_color = "green" if change_24h >= 0 else "red"

        print(f"║  💰 Price: ${current_price:,.2f} {price_change}".ljust(70) + "║")
        print(f"║  📈 24h Change: {colored(f'{change_24h:+.2f}%', change_color)}".ljust(79) + "║")
        high_24h = ticker.get('high') or 0
        low_24h = ticker.get('low') or 0
        print(f"║  📊 24h High: ${high_24h:,.2f}  |  24h Low: ${low_24h:,.2f}".ljust(70) + "║")
        quote_vol = ticker.get('quoteVolume') or ticker.get('baseVolume') or 0
        print(f"║  💵 24h Volume: ${quote_vol:,.0f}".ljust(70) + "║")

        # Open Interest
        print("╠" + "═" * 68 + "╣")
        oi_data = self.fetch_open_interest()
        if oi_data:
            oi_value = oi_data.get('openInterestValue', 0) or oi_data.get('openInterest', 0) * current_price
            oi_amount = oi_data.get('openInterest', 0)

            oi_change = ""
            if self.last_oi:
                oi_diff = oi_value - self.last_oi
                if oi_diff > 0:
                    oi_change = colored(f"+${oi_diff:,.0f}", "green")
                elif oi_diff < 0:
                    oi_change = colored(f"-${abs(oi_diff):,.0f}", "red")
            self.last_oi = oi_value

            print(f"║  📂 Open Interest: ${oi_value:,.0f} ({oi_amount:,.2f} ZEC) {oi_change}".ljust(79) + "║")
        else:
            print(f"║  📂 Open Interest: Not available".ljust(70) + "║")

        # Funding Rate
        funding = self.fetch_funding_rate()
        if funding:
            fund_rate = funding.get('fundingRate', 0) * 100  # Convert to percentage
            fund_color = "green" if fund_rate < 0 else "red"  # Negative = shorts pay longs
            next_funding = funding.get('fundingDatetime', 'Unknown')
            print(f"║  💸 Funding Rate: {colored(f'{fund_rate:.4f}%', fund_color)} (8h)".ljust(79) + "║")
            if fund_rate > 0:
                print(f"║     → Longs pay shorts (bullish sentiment)".ljust(70) + "║")
            else:
                print(f"║     → Shorts pay longs (bearish sentiment)".ljust(70) + "║")
        else:
            print(f"║  💸 Funding Rate: Not available".ljust(70) + "║")

        # Liquidation Levels
        print("╠" + "═" * 68 + "╣")
        print(f"║  🔻 " + colored("LONG Liquidation Prices (if longing at current price):", "green").ljust(65) + "║")
        long_liqs = self.calculate_liquidation_prices(current_price, is_long=True)
        for lev, liq_price in long_liqs.items():
            dist = ((current_price - liq_price) / current_price) * 100
            print(f"║     {lev:2}x leverage: ${liq_price:,.2f} ({dist:.1f}% below)".ljust(70) + "║")

        print("╠" + "═" * 68 + "╣")
        print(f"║  🔺 " + colored("SHORT Liquidation Prices (if shorting at current price):", "red").ljust(65) + "║")
        short_liqs = self.calculate_liquidation_prices(current_price, is_long=False)
        for lev, liq_price in short_liqs.items():
            dist = ((liq_price - current_price) / current_price) * 100
            print(f"║     {lev:2}x leverage: ${liq_price:,.2f} ({dist:.1f}% above)".ljust(70) + "║")

        print("╚" + "═" * 68 + "╝")

    def monitor(self):
        """Continuously monitor ZEC derivatives"""
        cprint("\n✅ Starting ZEC liquidation monitoring...\n", "green")

        while True:
            try:
                self.display_data()
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n👋 ZEC Liquidation Agent shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"⚠️ Error: {e}", "yellow")
                time.sleep(5)

def main():
    agent = ZECLiquidationAgent()
    if agent.symbol:
        agent.monitor()

if __name__ == "__main__":
    main()
