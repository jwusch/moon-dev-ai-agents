"""
Moon Dev's Funding Arbitrage Bot
Delta-neutral funding rate arbitrage for US traders on Kraken

Features:
1. Calculator - Calculate profits for delta-neutral trades
2. Verifier - Verify live rates from Kraken
3. Monitor - Watch for entry/exit signals

Usage:
    python src/agents/funding_arb_bot.py                    # Full monitoring mode
    python src/agents/funding_arb_bot.py --calc DASH 1000   # Calculate for $1000 DASH
    python src/agents/funding_arb_bot.py --verify           # Verify current rates
    python src/agents/funding_arb_bot.py --scan             # Quick scan for opportunities
"""

import ccxt
import time
import csv
import argparse
from datetime import datetime, timedelta
from termcolor import colored, cprint
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directory
DATA_DIR = PROJECT_ROOT / "src" / "data" / "funding_arb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fee structure (Kraken)
KRAKEN_FUTURES_TAKER_FEE = 0.0005  # 0.05%
KRAKEN_FUTURES_MAKER_FEE = 0.0002  # 0.02%
KRAKEN_SPOT_TAKER_FEE = 0.0026    # 0.26%
KRAKEN_SPOT_MAKER_FEE = 0.0016    # 0.16%
KRAKEN_MARGIN_OPEN_FEE = 0.0002   # 0.02% to open
KRAKEN_MARGIN_ROLLOVER = 0.0002   # 0.02% per 4 hours

# Thresholds (CORRECTED - using realistic historical rates)
MIN_FUNDING_FOR_ENTRY = 0.02     # Minimum %/hr to consider entering (0.02% = 0.48%/day)
MIN_PROFIT_AFTER_FEES = 0.01     # Minimum %/hr profit after fees
EXIT_FUNDING_THRESHOLD = 0.01    # Exit if funding drops below this

# Historical lookback for accurate rates
FUNDING_HISTORY_HOURS = 24       # Hours of history to average

# Monitoring
CHECK_INTERVAL = 60              # Seconds between checks


@dataclass
class FundingRate:
    """Funding rate data"""
    symbol: str
    rate_hourly: float           # % per hour
    rate_daily: float            # % per day
    predicted_rate: float        # predicted next rate
    price_perp: float
    price_spot: float
    spread_pct: float
    open_interest: float
    volume_24h: float
    timestamp: datetime


@dataclass
class TradeCalculation:
    """Trade profitability calculation"""
    symbol: str
    position_size: float         # USD
    direction: str               # LONG_PERP or SHORT_PERP
    funding_rate_hourly: float   # %

    # Costs
    entry_fees: float            # USD
    hourly_margin_cost: float    # USD
    spread_cost: float           # USD
    total_entry_cost: float      # USD

    # Revenue
    hourly_funding: float        # USD
    daily_funding: float         # USD

    # Net
    hourly_profit: float         # USD
    daily_profit: float          # USD
    break_even_hours: float      # Hours to break even

    # Rates
    hourly_profit_pct: float     # %
    daily_profit_pct: float      # %
    apy: float                   # %


class FundingArbBot:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's Funding Arbitrage Bot", "cyan", attrs=['bold'])
        cprint(f"Delta-Neutral Funding Rate Arbitrage for US Traders", "cyan")
        cprint(f"{'='*70}", "cyan")

        self.kraken_futures = None
        self.kraken_spot = None
        self.rates: Dict[str, FundingRate] = {}
        self.active_positions: List[dict] = []
        self.start_time = datetime.now()

        # Log file
        self.log_file = DATA_DIR / f"arb_log_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_log_file()

        # Initialize exchanges
        self._init_exchanges()

    def _init_log_file(self):
        """Initialize CSV log"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'funding_rate', 'price_perp',
                    'price_spot', 'spread', 'signal'
                ])

    def _init_exchanges(self):
        """Initialize Kraken connections"""
        cprint("\nInitializing Kraken...", "white")

        try:
            self.kraken_futures = ccxt.krakenfutures({'enableRateLimit': True})
            self.kraken_futures.load_markets()
            perps = len([m for m in self.kraken_futures.markets if self.kraken_futures.markets[m].get('type') == 'swap'])
            cprint(f"  Kraken Futures: {perps} perpetuals", "green")
        except Exception as e:
            cprint(f"  Kraken Futures: Failed - {e}", "red")

        try:
            self.kraken_spot = ccxt.kraken({'enableRateLimit': True})
            self.kraken_spot.load_markets()
            margin = len([m for m in self.kraken_spot.markets if self.kraken_spot.markets[m].get('margin')])
            cprint(f"  Kraken Spot: {margin} margin-enabled markets", "green")
        except Exception as e:
            cprint(f"  Kraken Spot: Failed - {e}", "red")

    # =========================================================================
    # HISTORICAL FUNDING RATES (ACCURATE)
    # =========================================================================

    def get_historical_funding_rate(self, symbol: str, hours: int = FUNDING_HISTORY_HOURS) -> Tuple[float, List[float]]:
        """
        Get ACTUAL historical funding rate (not the misleading 'current' rate)

        Returns:
            Tuple of (average_rate_pct_per_hour, list_of_rates)
        """
        perp_symbol = f"{symbol}/USD:USD"

        try:
            history = self.kraken_futures.fetch_funding_rate_history(perp_symbol, limit=hours)

            if not history:
                return 0.0, []

            # Extract rates (raw values are decimals, multiply by 100 for %)
            rates = [h.get('fundingRate', 0) * 100 for h in history]
            avg_rate = sum(rates) / len(rates) if rates else 0

            return avg_rate, rates

        except Exception as e:
            cprint(f"Error fetching historical rates for {symbol}: {e}", "yellow")
            return 0.0, []

    # =========================================================================
    # CALCULATOR
    # =========================================================================

    def calculate_trade(self, symbol: str, position_size: float,
                       funding_rate: float = None) -> Optional[TradeCalculation]:
        """
        Calculate profitability for a delta-neutral funding trade

        USES HISTORICAL RATES for accuracy (not misleading 'current' rate)

        Args:
            symbol: Trading pair (e.g., "DASH", "BTC", "ETH")
            position_size: Position size in USD
            funding_rate: Override funding rate (% per hour), or fetch from history
        """
        # Normalize symbol
        perp_symbol = f"{symbol}/USD:USD"
        spot_symbol = f"{symbol}/USD"

        # Get current prices and HISTORICAL rates
        try:
            if funding_rate is None:
                # Use HISTORICAL average, not misleading current rate
                funding_rate, rate_history = self.get_historical_funding_rate(symbol)
                cprint(f"  Using {len(rate_history)}h historical avg: {funding_rate:.4f}%/hr", "cyan")

            perp_ticker = self.kraken_futures.fetch_ticker(perp_symbol)
            spot_ticker = self.kraken_spot.fetch_ticker(spot_symbol)

            price_perp = perp_ticker['last']
            price_spot = spot_ticker['last']
            spread_pct = abs(price_perp - price_spot) / price_spot * 100

        except Exception as e:
            cprint(f"Error fetching data for {symbol}: {e}", "red")
            return None

        # Determine direction
        if funding_rate < 0:
            direction = "LONG_PERP"  # Long perp, short spot
        else:
            direction = "SHORT_PERP"  # Short perp, long spot

        # Calculate costs
        # Entry: Futures taker + Spot margin taker + Margin open fee
        futures_entry_fee = position_size * KRAKEN_FUTURES_TAKER_FEE
        spot_entry_fee = position_size * KRAKEN_SPOT_TAKER_FEE
        margin_open_fee = position_size * KRAKEN_MARGIN_OPEN_FEE
        entry_fees = futures_entry_fee + spot_entry_fee + margin_open_fee

        # Spread cost (difference between perp and spot)
        spread_cost = position_size * spread_pct / 100

        # Hourly margin rollover (0.02% per 4 hours = 0.005% per hour)
        hourly_margin_cost = position_size * KRAKEN_MARGIN_ROLLOVER / 4

        total_entry_cost = entry_fees + spread_cost

        # Calculate funding revenue
        hourly_funding = position_size * abs(funding_rate) / 100
        daily_funding = hourly_funding * 24

        # Net profit
        hourly_profit = hourly_funding - hourly_margin_cost
        daily_profit = hourly_profit * 24

        # Break-even
        if hourly_profit > 0:
            break_even_hours = total_entry_cost / hourly_profit
        else:
            break_even_hours = float('inf')

        # Percentages
        hourly_profit_pct = (hourly_profit / position_size) * 100
        daily_profit_pct = hourly_profit_pct * 24
        apy = daily_profit_pct * 365

        return TradeCalculation(
            symbol=symbol,
            position_size=position_size,
            direction=direction,
            funding_rate_hourly=funding_rate,
            entry_fees=entry_fees,
            hourly_margin_cost=hourly_margin_cost,
            spread_cost=spread_cost,
            total_entry_cost=total_entry_cost,
            hourly_funding=hourly_funding,
            daily_funding=daily_funding,
            hourly_profit=hourly_profit,
            daily_profit=daily_profit,
            break_even_hours=break_even_hours,
            hourly_profit_pct=hourly_profit_pct,
            daily_profit_pct=daily_profit_pct,
            apy=apy
        )

    def display_calculation(self, calc: TradeCalculation):
        """Display trade calculation"""
        print("\n" + "=" * 70)
        print(colored(f"DELTA-NEUTRAL TRADE CALCULATOR: {calc.symbol}", "cyan", attrs=['bold']))
        print("=" * 70)

        # Position info
        print(f"\n Position Size: ${calc.position_size:,.2f}")
        print(f" Direction: {colored(calc.direction, 'green' if 'LONG' in calc.direction else 'red')}")
        print(f" Funding Rate: {calc.funding_rate_hourly:+.4f}%/hour")

        # Trade setup
        print(f"\n" + colored(" TRADE SETUP:", "yellow", attrs=['bold']))
        print("-" * 70)
        if "LONG" in calc.direction:
            print(f"  1. LONG  ${calc.position_size:,.0f} {calc.symbol} perpetual on Kraken Futures")
            print(f"  2. SHORT ${calc.position_size:,.0f} {calc.symbol} spot on Kraken (margin)")
        else:
            print(f"  1. SHORT ${calc.position_size:,.0f} {calc.symbol} perpetual on Kraken Futures")
            print(f"  2. LONG  ${calc.position_size:,.0f} {calc.symbol} spot on Kraken")

        # Costs
        print(f"\n" + colored(" COSTS:", "yellow", attrs=['bold']))
        print("-" * 70)
        print(f"  Entry fees (futures + spot):  ${calc.entry_fees:.2f}")
        print(f"  Spread cost:                  ${calc.spread_cost:.2f}")
        print(f"  Hourly margin cost:           ${calc.hourly_margin_cost:.4f}")
        print(f"  " + "-" * 40)
        print(f"  Total entry cost:             ${calc.total_entry_cost:.2f}")

        # Revenue
        print(f"\n" + colored(" REVENUE:", "yellow", attrs=['bold']))
        print("-" * 70)
        print(f"  Hourly funding received:      ${calc.hourly_funding:.2f}")
        print(f"  Daily funding received:       ${calc.daily_funding:.2f}")

        # Profit
        profit_color = "green" if calc.hourly_profit > 0 else "red"
        print(f"\n" + colored(" NET PROFIT:", "yellow", attrs=['bold']))
        print("-" * 70)
        print(f"  Hourly profit:   {colored(f'${calc.hourly_profit:.2f}', profit_color)} ({calc.hourly_profit_pct:+.3f}%)")
        print(f"  Daily profit:    {colored(f'${calc.daily_profit:.2f}', profit_color)} ({calc.daily_profit_pct:+.2f}%)")
        print(f"  APY:             {colored(f'{calc.apy:.0f}%', profit_color)}")
        print(f"  Break-even:      {calc.break_even_hours:.1f} hours")

        # Summary
        print(f"\n" + colored(" SUMMARY:", "cyan", attrs=['bold']))
        print("-" * 70)
        if calc.hourly_profit > 0:
            print(colored(f"  PROFITABLE - ${calc.daily_profit:.2f}/day on ${calc.position_size:,.0f}", "green", attrs=['bold']))
            print(f"  Recover entry costs in {calc.break_even_hours:.1f} hours")
        else:
            print(colored(f"  NOT PROFITABLE at current rates", "red", attrs=['bold']))

        print("=" * 70)

    # =========================================================================
    # VERIFIER
    # =========================================================================

    def verify_rates(self, symbols: List[str] = None) -> Dict[str, FundingRate]:
        """
        Verify current funding rates from Kraken

        USES HISTORICAL RATES for accuracy

        Args:
            symbols: List of symbols to check, or None for top opportunities
        """
        if symbols is None:
            symbols = ['DASH', 'YFI', 'BTC', 'ETH', 'ZEC', 'SOL', 'XMR', 'LTC', 'AVAX', 'AAVE']

        rates = {}

        print("\n" + "=" * 80)
        print(colored("KRAKEN FUNDING RATE VERIFICATION (HISTORICAL)", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(colored("Using 24h historical average (accurate)", "yellow"))
        print("=" * 80)

        print(f"\n{'Symbol':<10} {'Avg Rate/hr':>12} {'Daily':>10} {'Perp':>12} {'Spot':>12} {'Spread':>10} {'Action':>12}")
        print("-" * 80)

        for symbol in symbols:
            try:
                perp_symbol = f"{symbol}/USD:USD"
                spot_symbol = f"{symbol}/USD"

                # Use HISTORICAL funding rate (accurate)
                rate_hourly, rate_history = self.get_historical_funding_rate(symbol)

                # Get additional info from current data
                funding = self.kraken_futures.fetch_funding_rate(perp_symbol)
                info = funding.get('info', {})
                predicted = rate_hourly  # Use historical as prediction too
                oi = float(info.get('openInterest', 0))
                vol = float(info.get('volumeQuote', 0))

                # Fetch prices
                perp_ticker = self.kraken_futures.fetch_ticker(perp_symbol)
                spot_ticker = self.kraken_spot.fetch_ticker(spot_symbol)

                price_perp = perp_ticker['last']
                price_spot = spot_ticker['last']
                spread = (price_perp - price_spot) / price_spot * 100

                rate = FundingRate(
                    symbol=symbol,
                    rate_hourly=rate_hourly,
                    rate_daily=rate_hourly * 24,
                    predicted_rate=predicted,
                    price_perp=price_perp,
                    price_spot=price_spot,
                    spread_pct=spread,
                    open_interest=oi,
                    volume_24h=vol,
                    timestamp=datetime.now()
                )
                rates[symbol] = rate

                # Display
                if rate_hourly < -MIN_FUNDING_FOR_ENTRY:
                    action = colored("LONG PERP", "green")
                    rate_str = colored(f"{rate_hourly:+.4f}%", "green")
                elif rate_hourly > MIN_FUNDING_FOR_ENTRY:
                    action = colored("SHORT PERP", "red")
                    rate_str = colored(f"{rate_hourly:+.4f}%", "red")
                else:
                    action = "-"
                    rate_str = f"{rate_hourly:+.4f}%"

                daily = rate_hourly * 24
                print(f"{symbol:<10} {rate_str:>20} {daily:>+9.1f}% ${price_perp:>11.2f} ${price_spot:>11.2f} {spread:>+9.2f}% {action:>12}")

            except Exception as e:
                print(f"{symbol:<10} {'ERROR':>12} - {str(e)[:40]}")

        print("-" * 80)

        # Log
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for symbol, rate in rates.items():
                signal = "LONG" if rate.rate_hourly < -MIN_FUNDING_FOR_ENTRY else "SHORT" if rate.rate_hourly > MIN_FUNDING_FOR_ENTRY else "NONE"
                writer.writerow([
                    rate.timestamp.isoformat(), symbol, f"{rate.rate_hourly:.4f}",
                    f"{rate.price_perp:.4f}", f"{rate.price_spot:.4f}",
                    f"{rate.spread_pct:.4f}", signal
                ])

        self.rates = rates
        return rates

    # =========================================================================
    # MONITOR
    # =========================================================================

    def check_entry_signals(self) -> List[Tuple[str, str, float]]:
        """Check for entry signals"""
        signals = []

        for symbol, rate in self.rates.items():
            if abs(rate.rate_hourly) >= MIN_FUNDING_FOR_ENTRY:
                direction = "LONG_PERP" if rate.rate_hourly < 0 else "SHORT_PERP"
                signals.append((symbol, direction, rate.rate_hourly))

        return signals

    def check_exit_signals(self) -> List[Tuple[str, str]]:
        """Check for exit signals on active positions"""
        signals = []

        for position in self.active_positions:
            symbol = position['symbol']
            if symbol in self.rates:
                rate = self.rates[symbol]

                # Exit if funding dropped significantly
                if abs(rate.rate_hourly) < EXIT_FUNDING_THRESHOLD:
                    signals.append((symbol, "FUNDING_LOW"))

                # Exit if funding flipped
                if position['direction'] == "LONG_PERP" and rate.rate_hourly > 0:
                    signals.append((symbol, "FUNDING_FLIPPED"))
                elif position['direction'] == "SHORT_PERP" and rate.rate_hourly < 0:
                    signals.append((symbol, "FUNDING_FLIPPED"))

        return signals

    def display_monitor(self):
        """Display monitoring dashboard"""
        print("\n" + "=" * 80)
        print(colored("FUNDING ARBITRAGE MONITOR", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Entry signals
        entry_signals = self.check_entry_signals()

        print(f"\n" + colored(" ENTRY SIGNALS (funding > {:.1f}%/hr):".format(MIN_FUNDING_FOR_ENTRY), "yellow", attrs=['bold']))
        print("-" * 80)

        if entry_signals:
            for symbol, direction, rate in sorted(entry_signals, key=lambda x: abs(x[2]), reverse=True):
                color = "green" if "LONG" in direction else "red"
                daily = abs(rate) * 24
                print(colored(f"  {symbol:<8} {direction:<12} {rate:+.4f}%/hr ({daily:.1f}%/day)", color))

                # Quick profit estimate
                est_daily = 1000 * daily / 100 * 0.8  # Rough estimate after fees
                print(f"           Est. profit: ${est_daily:.0f}/day per $1000")
        else:
            print("  No entry signals")

        # Exit signals
        exit_signals = self.check_exit_signals()

        if exit_signals:
            print(f"\n" + colored(" EXIT SIGNALS:", "red", attrs=['bold']))
            print("-" * 80)
            for symbol, reason in exit_signals:
                print(colored(f"  {symbol}: {reason}", "red"))

        # Best opportunity detail
        if entry_signals:
            best = max(entry_signals, key=lambda x: abs(x[2]))
            print(f"\n" + colored(" BEST OPPORTUNITY:", "green", attrs=['bold']))
            print("-" * 80)
            calc = self.calculate_trade(best[0], 1000)
            if calc:
                print(f"  Symbol:     {calc.symbol}")
                print(f"  Direction:  {calc.direction}")
                print(f"  Funding:    {calc.funding_rate_hourly:+.4f}%/hr")
                print(f"  Daily:      ${calc.daily_profit:.2f} profit on $1000")
                print(f"  Break-even: {calc.break_even_hours:.1f} hours")

        # Runtime
        runtime = datetime.now() - self.start_time
        print(f"\n  Runtime: {runtime} | Next check in {CHECK_INTERVAL}s")
        print("=" * 80)

    def run_monitor(self):
        """Run continuous monitoring"""
        cprint(f"\nStarting Funding Arbitrage Monitor...", "green")
        cprint(f"  Check interval: {CHECK_INTERVAL}s", "white")
        cprint(f"  Entry threshold: {MIN_FUNDING_FOR_ENTRY}%/hr", "white")
        cprint(f"  Exit threshold: {EXIT_FUNDING_THRESHOLD}%/hr", "white")
        cprint(f"\nPress Ctrl+C to stop\n", "yellow")

        while True:
            try:
                # Verify rates
                self.verify_rates()

                # Display monitor
                self.display_monitor()

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n\nShutting down monitor...", "yellow")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(30)

    # =========================================================================
    # QUICK SCAN
    # =========================================================================

    def quick_scan(self):
        """Quick scan of all Kraken Futures for opportunities using HISTORICAL rates"""
        print("\n" + "=" * 80)
        print(colored("QUICK SCAN - ALL KRAKEN FUTURES (HISTORICAL RATES)", "cyan", attrs=['bold']))
        print(colored("Using 24h historical average (accurate)", "yellow"))
        print("=" * 80)

        # Top coins to check with historical rates
        top_coins = ['DASH', 'YFI', 'BTC', 'ETH', 'ZEC', 'SOL', 'XMR', 'LTC',
                     'AVAX', 'AAVE', 'LINK', 'UNI', 'DOGE', 'ADA', 'DOT',
                     'MATIC', 'ATOM', 'FIL', 'NEAR', 'APT']

        opportunities = []
        print(f"\nScanning {len(top_coins)} major markets with historical rates...")

        for base in top_coins:
            try:
                sym = f"{base}/USD:USD"

                # Get HISTORICAL rate (accurate)
                rate, history = self.get_historical_funding_rate(base)

                if abs(rate) > 0.01:  # > 0.01%/hr (realistic threshold)
                    ticker = self.kraken_futures.fetch_ticker(sym)
                    price = ticker.get('last') or ticker.get('close') or 0
                    if price is None or price == 0:
                        continue

                    opportunities.append({
                        'symbol': base,
                        'rate': rate,
                        'price': price,
                        'volume': 0,  # Skip volume for speed
                        'direction': 'LONG_PERP' if rate < 0 else 'SHORT_PERP',
                        'history_hours': len(history)
                    })
            except:
                pass

        # Sort by absolute rate
        opportunities.sort(key=lambda x: abs(x['rate']), reverse=True)

        # Display
        print(f"\nFound {len(opportunities)} opportunities (> 0.1%/hr):\n")

        longs = [o for o in opportunities if 'LONG' in o['direction']]
        shorts = [o for o in opportunities if 'SHORT' in o['direction']]

        print(colored(" GO LONG (shorts pay you):", "green", attrs=['bold']))
        print("-" * 60)
        for o in longs[:10]:
            daily = abs(o['rate']) * 24
            print(f"  {o['symbol']:<8} {o['rate']:+.4f}%/hr  ({daily:.0f}%/day)  ${o['price']:.2f}")

        print(f"\n" + colored(" GO SHORT (longs pay you):", "red", attrs=['bold']))
        print("-" * 60)
        for o in shorts[:10]:
            daily = abs(o['rate']) * 24
            print(f"  {o['symbol']:<8} {o['rate']:+.4f}%/hr  ({daily:.0f}%/day)  ${o['price']:.2f}")

        print("\n" + "=" * 80)

        return opportunities


def main():
    parser = argparse.ArgumentParser(description="Moon Dev's Funding Arbitrage Bot")
    parser.add_argument("--calc", nargs=2, metavar=('SYMBOL', 'SIZE'),
                       help="Calculate trade for SYMBOL with SIZE USD")
    parser.add_argument("--verify", action="store_true",
                       help="Verify current funding rates")
    parser.add_argument("--scan", action="store_true",
                       help="Quick scan all markets")
    parser.add_argument("--monitor", action="store_true",
                       help="Run continuous monitoring")

    args = parser.parse_args()

    bot = FundingArbBot()

    if args.calc:
        symbol = args.calc[0].upper()
        size = float(args.calc[1])
        calc = bot.calculate_trade(symbol, size)
        if calc:
            bot.display_calculation(calc)

    elif args.verify:
        bot.verify_rates()

    elif args.scan:
        bot.quick_scan()

    elif args.monitor:
        bot.run_monitor()

    else:
        # Default: run full monitoring
        bot.run_monitor()


if __name__ == "__main__":
    main()
