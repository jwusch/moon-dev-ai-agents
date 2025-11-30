"""
Moon Dev's MERL Funding Rate Tracker
Monitors the MERL perp vs spot funding rate arbitrage opportunity

Discovery: 2025-11-26
- OKX Perp trading 12% below spot
- Funding rate at -1.5%/hour cap (shorts paying longs)
- 24h funding yield: ~16%
- Open interest: $19M

This agent tracks when the opportunity opens/closes.

Usage:
    python src/agents/merl_funding_tracker.py
    python src/agents/merl_funding_tracker.py --interval 60  # Check every 60s
"""

import ccxt
import time
import csv
import argparse
from datetime import datetime, timedelta
from termcolor import colored, cprint
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import os
load_dotenv()

# Data directory for logs
DATA_DIR = PROJECT_ROOT / "src" / "data" / "merl_funding"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECK_INTERVAL = 30  # seconds between checks

# Alert thresholds
FUNDING_ALERT_THRESHOLD = -0.5  # Alert if funding rises above this (opportunity weakening)
SPREAD_ALERT_THRESHOLD = 5.0    # Alert if spread drops below this %
FUNDING_CLOSE_THRESHOLD = 0.0   # Opportunity closed if funding goes positive

# Exchanges to monitor
SPOT_EXCHANGES = {
    "okx": "MERL/USDT",
    "gate": "MERL/USDT",
    "mexc": "MERL/USDT",
    "kucoin": "MERL/USDT",
}

PERP_EXCHANGES = {
    "okx": "MERL/USDT:USDT",
    "gate": "MERL/USDT:USDT",
}


@dataclass
class FundingSnapshot:
    """Snapshot of funding rate data"""
    timestamp: datetime
    exchange: str
    funding_rate: float  # Current funding rate %
    funding_rate_annualized: float
    next_funding_time: Optional[datetime]
    spot_price: float
    perp_price: float
    spread_pct: float
    open_interest: float
    spot_volume_24h: float
    perp_volume_24h: float


class MERLFundingTracker:
    def __init__(self, check_interval: int = CHECK_INTERVAL):
        """Initialize the funding tracker"""
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's MERL Funding Rate Tracker", "cyan", attrs=['bold'])
        cprint(f"Monitoring Funding Rate Arbitrage Opportunity", "cyan")
        cprint(f"{'='*70}", "cyan")

        self.check_interval = check_interval
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.snapshots: List[FundingSnapshot] = []
        self.start_time = datetime.now()
        self.checks_performed = 0

        # Alert state
        self.last_alert_time = None
        self.opportunity_open = True

        # CSV log file
        self.log_file = DATA_DIR / f"funding_log_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_log_file()

        # Initialize exchanges
        self._init_exchanges()

    def _init_log_file(self):
        """Initialize CSV log file"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'exchange', 'funding_rate_pct', 'funding_annualized',
                    'spot_price', 'perp_price', 'spread_pct', 'open_interest',
                    'spot_volume_24h', 'perp_volume_24h'
                ])
        cprint(f"  Logging to: {self.log_file}", "white")

    def _init_exchanges(self):
        """Initialize exchange connections"""
        cprint("\nInitializing exchanges...", "white")

        all_exchanges = set(SPOT_EXCHANGES.keys()) | set(PERP_EXCHANGES.keys())

        for ex_id in all_exchanges:
            try:
                exchange = getattr(ccxt, ex_id)({'enableRateLimit': True})
                exchange.load_markets()
                self.exchanges[ex_id] = exchange

                # Check what's available
                spot_sym = SPOT_EXCHANGES.get(ex_id)
                perp_sym = PERP_EXCHANGES.get(ex_id)

                has_spot = spot_sym and spot_sym in exchange.markets
                has_perp = perp_sym and perp_sym in exchange.markets

                status = []
                if has_spot:
                    status.append("spot")
                if has_perp:
                    status.append("perp")

                if status:
                    cprint(f"  {ex_id.upper()}: {', '.join(status)}", "green")
                else:
                    cprint(f"  {ex_id.upper()}: no MERL markets", "yellow")

            except Exception as e:
                cprint(f"  {ex_id.upper()}: Failed - {str(e)[:40]}", "red")

    def fetch_funding_data(self, exchange_id: str) -> Optional[FundingSnapshot]:
        """Fetch funding rate and price data from an exchange"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                return None

            spot_sym = SPOT_EXCHANGES.get(exchange_id)
            perp_sym = PERP_EXCHANGES.get(exchange_id)

            if not perp_sym or perp_sym not in exchange.markets:
                return None

            # Fetch perp ticker
            perp_ticker = exchange.fetch_ticker(perp_sym)
            perp_price = perp_ticker['last']
            perp_volume = perp_ticker.get('quoteVolume') or 0

            # Fetch spot ticker
            spot_price = 0
            spot_volume = 0
            if spot_sym and spot_sym in exchange.markets:
                spot_ticker = exchange.fetch_ticker(spot_sym)
                spot_price = spot_ticker['last']
                spot_volume = spot_ticker.get('quoteVolume') or 0

            # Calculate spread
            spread_pct = 0
            if perp_price > 0 and spot_price > 0:
                spread_pct = ((spot_price - perp_price) / perp_price) * 100

            # Fetch funding rate
            funding_rate = 0
            next_funding = None
            try:
                if hasattr(exchange, 'fetch_funding_rate'):
                    funding = exchange.fetch_funding_rate(perp_sym)
                    funding_rate = (funding.get('fundingRate') or 0) * 100
                    if funding.get('fundingDatetime'):
                        next_funding = datetime.fromisoformat(funding['fundingDatetime'].replace('Z', '+00:00'))
            except:
                pass

            # Fetch open interest
            open_interest = 0
            try:
                if hasattr(exchange, 'fetch_open_interest'):
                    oi = exchange.fetch_open_interest(perp_sym)
                    open_interest = oi.get('openInterestValue') or (oi.get('openInterest', 0) * perp_price)
            except:
                pass

            # Annualized funding (assuming hourly settlement)
            # -1.5% per hour = -1.5 * 24 * 365 = -13,140% annualized
            funding_annualized = funding_rate * 24 * 365

            return FundingSnapshot(
                timestamp=datetime.now(),
                exchange=exchange_id,
                funding_rate=funding_rate,
                funding_rate_annualized=funding_annualized,
                next_funding_time=next_funding,
                spot_price=spot_price,
                perp_price=perp_price,
                spread_pct=spread_pct,
                open_interest=open_interest,
                spot_volume_24h=spot_volume,
                perp_volume_24h=perp_volume,
            )

        except Exception as e:
            return None

    def log_snapshot(self, snapshot: FundingSnapshot):
        """Log snapshot to CSV"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                snapshot.timestamp.isoformat(),
                snapshot.exchange,
                f"{snapshot.funding_rate:.4f}",
                f"{snapshot.funding_rate_annualized:.2f}",
                f"{snapshot.spot_price:.6f}",
                f"{snapshot.perp_price:.6f}",
                f"{snapshot.spread_pct:.4f}",
                f"{snapshot.open_interest:.0f}",
                f"{snapshot.spot_volume_24h:.0f}",
                f"{snapshot.perp_volume_24h:.0f}",
            ])

    def check_alerts(self, snapshots: List[FundingSnapshot]) -> List[str]:
        """Check for alert conditions"""
        alerts = []

        for snap in snapshots:
            # Funding rate rising (opportunity weakening)
            if snap.funding_rate > FUNDING_ALERT_THRESHOLD:
                alerts.append(f"FUNDING RISING on {snap.exchange.upper()}: {snap.funding_rate:.4f}%")

            # Funding flipped positive (opportunity closed)
            if snap.funding_rate >= FUNDING_CLOSE_THRESHOLD:
                alerts.append(f"FUNDING POSITIVE on {snap.exchange.upper()}: {snap.funding_rate:.4f}% - OPPORTUNITY CLOSED!")
                self.opportunity_open = False

            # Spread narrowing
            if 0 < snap.spread_pct < SPREAD_ALERT_THRESHOLD:
                alerts.append(f"SPREAD NARROWING on {snap.exchange.upper()}: {snap.spread_pct:.2f}%")

        return alerts

    def fetch_funding_history(self, exchange_id: str = 'okx', limit: int = 24) -> List[dict]:
        """Fetch historical funding rates"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                return []

            perp_sym = PERP_EXCHANGES.get(exchange_id)
            if not perp_sym:
                return []

            history = exchange.fetch_funding_rate_history(perp_sym, limit=limit)
            return history
        except:
            return []

    def display_dashboard(self, snapshots: List[FundingSnapshot], alerts: List[str]):
        """Display the monitoring dashboard"""
        print("\n" + "=" * 80)
        print(colored("MERL FUNDING RATE TRACKER", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Current status
        print("\n" + colored(" CURRENT FUNDING RATES", "yellow", attrs=['bold']))
        print("-" * 80)
        print(f"{'Exchange':<12} {'Funding':>12} {'Annualized':>14} {'Spot':>12} {'Perp':>12} {'Spread':>10}")
        print("-" * 80)

        for snap in snapshots:
            # Color coding
            if snap.funding_rate <= -1.0:
                fund_color = "green"
                fund_str = colored(f"{snap.funding_rate:+.4f}%", fund_color)
            elif snap.funding_rate < 0:
                fund_color = "cyan"
                fund_str = colored(f"{snap.funding_rate:+.4f}%", fund_color)
            else:
                fund_color = "red"
                fund_str = colored(f"{snap.funding_rate:+.4f}%", fund_color)

            spread_color = "green" if snap.spread_pct > 5 else "yellow" if snap.spread_pct > 2 else "red"
            spread_str = colored(f"{snap.spread_pct:+.2f}%", spread_color)

            print(f"{snap.exchange.upper():<12} {fund_str:>20} {snap.funding_rate_annualized:>+13.0f}% ${snap.spot_price:>11.5f} ${snap.perp_price:>11.5f} {spread_str:>18}")

        # Open Interest
        print("\n" + colored(" OPEN INTEREST", "yellow", attrs=['bold']))
        print("-" * 80)
        for snap in snapshots:
            if snap.open_interest > 0:
                print(f"  {snap.exchange.upper()}: ${snap.open_interest:,.0f}")

        # Funding history (last 8 hours)
        print("\n" + colored(" FUNDING HISTORY (OKX - Last 8 Hours)", "yellow", attrs=['bold']))
        print("-" * 80)
        history = self.fetch_funding_history('okx', 10)
        if history:
            total = 0
            for h in history[-8:]:
                ts = datetime.fromtimestamp(h['timestamp']/1000).strftime('%H:%M')
                rate = h.get('fundingRate', 0) * 100
                total += rate
                bar = colored("█" * int(abs(rate) * 10), "green" if rate < 0 else "red")
                print(f"  {ts}: {rate:+.4f}% {bar}")
            print(f"  ────────────────────")
            print(f"  8h Total: {colored(f'{total:+.4f}%', 'green' if total < 0 else 'red')}")

        # Opportunity status
        print("\n" + colored(" OPPORTUNITY STATUS", "yellow", attrs=['bold']))
        print("-" * 80)
        if self.opportunity_open:
            if snapshots:
                best = max(snapshots, key=lambda x: abs(x.funding_rate))
                if best.funding_rate <= -1.0:
                    status = colored("ACTIVE - STRONG", "green", attrs=['bold'])
                    print(f"  Status: {status}")
                    print(f"  Best funding: {best.funding_rate:.4f}%/hour on {best.exchange.upper()}")
                    print(f"  Estimated daily yield: {best.funding_rate * 24:.2f}%")
                elif best.funding_rate < 0:
                    status = colored("ACTIVE - MODERATE", "cyan")
                    print(f"  Status: {status}")
                    print(f"  Best funding: {best.funding_rate:.4f}%/hour on {best.exchange.upper()}")
                else:
                    status = colored("WEAKENING", "yellow")
                    print(f"  Status: {status}")
        else:
            print(f"  Status: {colored('CLOSED', 'red', attrs=['bold'])}")
            print(f"  Funding has flipped positive - shorts are no longer paying longs")

        # Alerts
        if alerts:
            print("\n" + colored(" ALERTS", "red", attrs=['bold']))
            print("-" * 80)
            for alert in alerts:
                print(colored(f"  !! {alert}", "red"))

        # Session stats
        print("\n" + colored(" SESSION STATS", "yellow", attrs=['bold']))
        print("-" * 80)
        runtime = datetime.now() - self.start_time
        self.checks_performed += 1
        print(f"  Runtime: {runtime}")
        print(f"  Checks: {self.checks_performed}")
        print(f"  Log file: {self.log_file.name}")

        print("\n" + "=" * 80)

    def run(self):
        """Main monitoring loop"""
        cprint("\nStarting MERL Funding Tracker...\n", "green")
        cprint(f"Check interval: {self.check_interval}s", "white")
        cprint(f"Alert threshold: funding > {FUNDING_ALERT_THRESHOLD}% or spread < {SPREAD_ALERT_THRESHOLD}%", "white")
        cprint("\nPress Ctrl+C to stop\n", "yellow")

        while True:
            try:
                # Fetch data from all exchanges
                snapshots = []
                for ex_id in PERP_EXCHANGES:
                    snap = self.fetch_funding_data(ex_id)
                    if snap:
                        snapshots.append(snap)
                        self.log_snapshot(snap)

                if not snapshots:
                    cprint("No data available, waiting...", "yellow")
                    time.sleep(self.check_interval)
                    continue

                # Check alerts
                alerts = self.check_alerts(snapshots)

                # Display dashboard
                self.display_dashboard(snapshots, alerts)

                # Store latest snapshots
                self.snapshots = snapshots

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                cprint("\n\nShutting down MERL Funding Tracker...", "yellow")
                self._print_summary()
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                import traceback
                traceback.print_exc()
                time.sleep(self.check_interval)

    def _print_summary(self):
        """Print session summary"""
        print("\n" + "=" * 60)
        print(colored("SESSION SUMMARY", "cyan", attrs=['bold']))
        print("=" * 60)
        print(f"  Runtime: {datetime.now() - self.start_time}")
        print(f"  Checks performed: {self.checks_performed}")
        print(f"  Log file: {self.log_file}")

        if self.snapshots:
            print(f"\n  Final readings:")
            for snap in self.snapshots:
                print(f"    {snap.exchange.upper()}: funding={snap.funding_rate:.4f}%, spread={snap.spread_pct:.2f}%")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Moon Dev's MERL Funding Rate Tracker")
    parser.add_argument(
        "--interval",
        type=int,
        default=CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {CHECK_INTERVAL})"
    )

    args = parser.parse_args()

    tracker = MERLFundingTracker(check_interval=args.interval)
    tracker.run()


if __name__ == "__main__":
    main()
