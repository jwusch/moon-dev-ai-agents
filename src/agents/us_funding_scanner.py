"""
Moon Dev's US Funding Rate Scanner
Scans US-accessible exchanges for extreme funding rate opportunities

Supported Exchanges:
- Kraken Futures (available in most US states except NY)
- Hyperliquid (decentralized, US accessible)

Usage:
    python src/agents/us_funding_scanner.py
    python src/agents/us_funding_scanner.py --interval 300  # Check every 5 min
    python src/agents/us_funding_scanner.py --min-rate 0.5  # Only show >0.5%/hr
"""

import ccxt
import time
import csv
import argparse
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directory
DATA_DIR = PROJECT_ROOT / "src" / "data" / "funding_scanner"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECK_INTERVAL = 300  # 5 minutes between scans
MIN_FUNDING_RATE = 0.1  # Minimum funding rate % to show (per hour)
TOP_N_OPPORTUNITIES = 15  # Show top N opportunities

# Alert thresholds (will highlight these)
ALERT_THRESHOLD_HIGH = 1.0  # >= 1%/hour = HIGH opportunity
ALERT_THRESHOLD_EXTREME = 5.0  # >= 5%/hour = EXTREME opportunity


@dataclass
class FundingOpportunity:
    """Represents a funding rate opportunity"""
    exchange: str
    symbol: str
    funding_rate: float  # % per hour
    predicted_rate: float  # predicted next rate
    price: float
    open_interest: float
    volume_24h: float
    direction: str  # LONG or SHORT
    daily_yield: float  # % per day
    timestamp: datetime


class USFundingScanner:
    def __init__(self, check_interval: int = CHECK_INTERVAL, min_rate: float = MIN_FUNDING_RATE):
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's US Funding Rate Scanner", "cyan", attrs=['bold'])
        cprint(f"Scanning US-Accessible Exchanges for Funding Opportunities", "cyan")
        cprint(f"{'='*70}", "cyan")

        self.check_interval = check_interval
        self.min_rate = min_rate
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.opportunities: List[FundingOpportunity] = []
        self.start_time = datetime.now()
        self.scans_performed = 0

        # CSV log file
        self.log_file = DATA_DIR / f"funding_scan_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_log_file()

        # Initialize exchanges
        self._init_exchanges()

    def _init_log_file(self):
        """Initialize CSV log file"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'exchange', 'symbol', 'funding_rate_pct',
                    'predicted_rate', 'price', 'open_interest', 'volume_24h',
                    'direction', 'daily_yield'
                ])
        cprint(f"  Logging to: {self.log_file.name}", "white")

    def _init_exchanges(self):
        """Initialize exchange connections"""
        cprint("\nInitializing exchanges...", "white")

        # Kraken Futures
        try:
            kf = ccxt.krakenfutures({'enableRateLimit': True})
            kf.load_markets()
            perps = [m for m in kf.markets if kf.markets[m].get('type') == 'swap']
            self.exchanges['krakenfutures'] = kf
            cprint(f"  Kraken Futures: {len(perps)} perpetual markets", "green")
        except Exception as e:
            cprint(f"  Kraken Futures: Failed - {str(e)[:40]}", "red")

        # Hyperliquid
        try:
            hl = ccxt.hyperliquid({'enableRateLimit': True})
            hl.load_markets()
            perps = [m for m in hl.markets if hl.markets[m].get('type') == 'swap']
            self.exchanges['hyperliquid'] = hl
            cprint(f"  Hyperliquid: {len(perps)} perpetual markets", "green")
        except Exception as e:
            cprint(f"  Hyperliquid: Failed - {str(e)[:40]}", "red")

        cprint(f"\n  Total exchanges: {len(self.exchanges)}", "white")

    def scan_kraken_futures(self) -> List[FundingOpportunity]:
        """Scan Kraken Futures for funding opportunities"""
        opportunities = []

        if 'krakenfutures' not in self.exchanges:
            return opportunities

        kf = self.exchanges['krakenfutures']
        perps = [m for m in kf.markets if kf.markets[m].get('type') == 'swap']

        for sym in perps:
            try:
                funding = kf.fetch_funding_rate(sym)
                info = funding.get('info', {})

                # Get real rate from raw info (Kraken uses different format)
                real_rate = float(info.get('fundingRate', 0)) * 100  # Convert to %
                predicted = float(info.get('fundingRatePrediction', 0)) * 100

                if abs(real_rate) < self.min_rate:
                    continue

                ticker = kf.fetch_ticker(sym)

                # Determine direction
                if real_rate < 0:
                    direction = "LONG"  # Shorts pay longs
                else:
                    direction = "SHORT"  # Longs pay shorts

                opp = FundingOpportunity(
                    exchange="Kraken Futures",
                    symbol=sym,
                    funding_rate=real_rate,
                    predicted_rate=predicted,
                    price=ticker['last'],
                    open_interest=float(info.get('openInterest', 0)),
                    volume_24h=float(info.get('volumeQuote', 0)),
                    direction=direction,
                    daily_yield=abs(real_rate) * 24,
                    timestamp=datetime.now()
                )
                opportunities.append(opp)

            except Exception as e:
                continue

        return opportunities

    def scan_hyperliquid(self) -> List[FundingOpportunity]:
        """Scan Hyperliquid for funding opportunities"""
        opportunities = []

        if 'hyperliquid' not in self.exchanges:
            return opportunities

        hl = self.exchanges['hyperliquid']
        perps = [m for m in hl.markets if hl.markets[m].get('type') == 'swap']

        for sym in perps[:50]:  # Limit to first 50 to avoid rate limits
            try:
                funding = hl.fetch_funding_rate(sym)
                rate = funding.get('fundingRate', 0)

                if rate is None:
                    continue

                # Hyperliquid returns 8h rate, convert to hourly
                rate_hourly = float(rate) * 100 / 8

                if abs(rate_hourly) < self.min_rate:
                    continue

                ticker = hl.fetch_ticker(sym)

                direction = "LONG" if rate_hourly < 0 else "SHORT"

                opp = FundingOpportunity(
                    exchange="Hyperliquid",
                    symbol=sym,
                    funding_rate=rate_hourly,
                    predicted_rate=rate_hourly,  # No prediction available
                    price=ticker['last'],
                    open_interest=0,  # Would need separate call
                    volume_24h=ticker.get('quoteVolume', 0) or 0,
                    direction=direction,
                    daily_yield=abs(rate_hourly) * 24,
                    timestamp=datetime.now()
                )
                opportunities.append(opp)

            except Exception as e:
                continue

        return opportunities

    def scan_all(self) -> List[FundingOpportunity]:
        """Scan all exchanges"""
        all_opportunities = []

        cprint("  Scanning Kraken Futures...", "cyan")
        all_opportunities.extend(self.scan_kraken_futures())

        cprint("  Scanning Hyperliquid...", "cyan")
        all_opportunities.extend(self.scan_hyperliquid())

        # Sort by absolute funding rate (most extreme first)
        all_opportunities.sort(key=lambda x: abs(x.funding_rate), reverse=True)

        return all_opportunities

    def log_opportunities(self, opportunities: List[FundingOpportunity]):
        """Log opportunities to CSV"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for opp in opportunities:
                writer.writerow([
                    opp.timestamp.isoformat(),
                    opp.exchange,
                    opp.symbol,
                    f"{opp.funding_rate:.4f}",
                    f"{opp.predicted_rate:.4f}",
                    f"{opp.price:.6f}",
                    f"{opp.open_interest:.2f}",
                    f"{opp.volume_24h:.2f}",
                    opp.direction,
                    f"{opp.daily_yield:.2f}"
                ])

    def display_dashboard(self, opportunities: List[FundingOpportunity]):
        """Display the scanner dashboard"""
        print("\n" + "=" * 85)
        print(colored("US FUNDING RATE SCANNER", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 85)

        if not opportunities:
            print(colored("\nNo opportunities found above minimum threshold", "yellow"))
            return

        # Separate LONG and SHORT opportunities
        long_opps = [o for o in opportunities if o.direction == "LONG"]
        short_opps = [o for o in opportunities if o.direction == "SHORT"]

        # Display LONG opportunities (shorts pay longs - negative funding)
        print("\n" + colored(" GO LONG (Shorts Pay You)", "green", attrs=['bold']))
        print("-" * 85)
        print(f"{'Exchange':<16} {'Symbol':<18} {'Rate/hr':>10} {'Daily':>10} {'Price':>12} {'Volume 24h':>14}")
        print("-" * 85)

        for opp in long_opps[:TOP_N_OPPORTUNITIES//2]:
            # Color based on rate
            if abs(opp.funding_rate) >= ALERT_THRESHOLD_EXTREME:
                rate_str = colored(f"{opp.funding_rate:+.4f}%", "green", attrs=['bold'])
                yield_str = colored(f"{opp.daily_yield:.1f}%", "green", attrs=['bold'])
            elif abs(opp.funding_rate) >= ALERT_THRESHOLD_HIGH:
                rate_str = colored(f"{opp.funding_rate:+.4f}%", "green")
                yield_str = colored(f"{opp.daily_yield:.1f}%", "green")
            else:
                rate_str = f"{opp.funding_rate:+.4f}%"
                yield_str = f"{opp.daily_yield:.1f}%"

            print(f"{opp.exchange:<16} {opp.symbol:<18} {rate_str:>18} {yield_str:>10} ${opp.price:>11.4f} ${opp.volume_24h:>13,.0f}")

        # Display SHORT opportunities (longs pay shorts - positive funding)
        print("\n" + colored(" GO SHORT (Longs Pay You)", "red", attrs=['bold']))
        print("-" * 85)
        print(f"{'Exchange':<16} {'Symbol':<18} {'Rate/hr':>10} {'Daily':>10} {'Price':>12} {'Volume 24h':>14}")
        print("-" * 85)

        for opp in short_opps[:TOP_N_OPPORTUNITIES//2]:
            if abs(opp.funding_rate) >= ALERT_THRESHOLD_EXTREME:
                rate_str = colored(f"{opp.funding_rate:+.4f}%", "red", attrs=['bold'])
                yield_str = colored(f"{opp.daily_yield:.1f}%", "red", attrs=['bold'])
            elif abs(opp.funding_rate) >= ALERT_THRESHOLD_HIGH:
                rate_str = colored(f"{opp.funding_rate:+.4f}%", "red")
                yield_str = colored(f"{opp.daily_yield:.1f}%", "red")
            else:
                rate_str = f"{opp.funding_rate:+.4f}%"
                yield_str = f"{opp.daily_yield:.1f}%"

            print(f"{opp.exchange:<16} {opp.symbol:<18} {rate_str:>18} {yield_str:>10} ${opp.price:>11.4f} ${opp.volume_24h:>13,.0f}")

        # Best opportunities summary
        print("\n" + colored(" BEST OPPORTUNITIES", "yellow", attrs=['bold']))
        print("-" * 85)

        if long_opps:
            best_long = long_opps[0]
            print(colored(f"  BEST LONG:  {best_long.symbol} on {best_long.exchange}", "green"))
            print(f"              Rate: {best_long.funding_rate:+.4f}%/hr | Daily: {best_long.daily_yield:.1f}%")
            print(f"              $100 position = ${100 * best_long.daily_yield / 100:.2f}/day")

        if short_opps:
            best_short = short_opps[0]
            print(colored(f"  BEST SHORT: {best_short.symbol} on {best_short.exchange}", "red"))
            print(f"              Rate: {best_short.funding_rate:+.4f}%/hr | Daily: {best_short.daily_yield:.1f}%")
            print(f"              $100 position = ${100 * best_short.daily_yield / 100:.2f}/day")

        # Alerts for extreme opportunities
        extreme = [o for o in opportunities if abs(o.funding_rate) >= ALERT_THRESHOLD_EXTREME]
        if extreme:
            print("\n" + colored(" EXTREME ALERTS", "magenta", attrs=['bold', 'blink']))
            print("-" * 85)
            for opp in extreme:
                alert_color = "green" if opp.direction == "LONG" else "red"
                print(colored(f"  !! {opp.symbol} on {opp.exchange}: {opp.funding_rate:+.4f}%/hr ({opp.daily_yield:.0f}%/day) - GO {opp.direction}", alert_color, attrs=['bold']))

        # Session stats
        print("\n" + colored(" SESSION STATS", "white", attrs=['bold']))
        print("-" * 85)
        runtime = datetime.now() - self.start_time
        self.scans_performed += 1
        print(f"  Runtime: {runtime}")
        print(f"  Scans: {self.scans_performed}")
        print(f"  Opportunities found: {len(opportunities)}")
        print(f"  Log file: {self.log_file.name}")
        print(f"  Next scan in: {self.check_interval}s")

        print("\n" + "=" * 85)

    def run(self):
        """Main scanner loop"""
        cprint(f"\nStarting US Funding Rate Scanner...", "green")
        cprint(f"  Check interval: {self.check_interval}s", "white")
        cprint(f"  Minimum rate: {self.min_rate}%/hr", "white")
        cprint(f"\nPress Ctrl+C to stop\n", "yellow")

        while True:
            try:
                cprint(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning exchanges...", "cyan")

                # Scan all exchanges
                opportunities = self.scan_all()

                # Log to CSV
                if opportunities:
                    self.log_opportunities(opportunities)

                # Store latest
                self.opportunities = opportunities

                # Display dashboard
                self.display_dashboard(opportunities)

                # Wait for next scan
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                cprint("\n\nShutting down US Funding Scanner...", "yellow")
                self._print_summary()
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                import traceback
                traceback.print_exc()
                time.sleep(30)

    def _print_summary(self):
        """Print session summary"""
        print("\n" + "=" * 60)
        print(colored("SESSION SUMMARY", "cyan", attrs=['bold']))
        print("=" * 60)
        print(f"  Runtime: {datetime.now() - self.start_time}")
        print(f"  Total scans: {self.scans_performed}")
        print(f"  Log file: {self.log_file}")

        if self.opportunities:
            print(f"\n  Last scan found {len(self.opportunities)} opportunities")
            best = max(self.opportunities, key=lambda x: abs(x.funding_rate))
            print(f"  Best opportunity: {best.symbol} ({best.funding_rate:+.4f}%/hr)")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Moon Dev's US Funding Rate Scanner")
    parser.add_argument(
        "--interval",
        type=int,
        default=CHECK_INTERVAL,
        help=f"Scan interval in seconds (default: {CHECK_INTERVAL})"
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=MIN_FUNDING_RATE,
        help=f"Minimum funding rate %/hr to display (default: {MIN_FUNDING_RATE})"
    )

    args = parser.parse_args()

    scanner = USFundingScanner(
        check_interval=args.interval,
        min_rate=args.min_rate
    )
    scanner.run()


if __name__ == "__main__":
    main()
