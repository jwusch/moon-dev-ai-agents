"""
Moon Dev's DASH Position Tracker
Tracks your DASH long position and funding income over time

Usage:
    python src/agents/dash_position_tracker.py
    python src/agents/dash_position_tracker.py --entry 10 --entry-price 67.50
"""

import ccxt
import time
import csv
import argparse
from datetime import datetime
from termcolor import colored, cprint
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "dash_tracker"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Your position (edit these or pass as args)
ENTRY_SIZE_USD = 10.00
ENTRY_PRICE = 67.50  # Approximate - adjust if you know exact entry
ENTRY_TIME = datetime(2025, 11, 27, 2, 0)  # Approximate entry time

CHECK_INTERVAL = 300  # 5 minutes


class DashPositionTracker:
    def __init__(self, entry_size: float, entry_price: float = None):
        self.entry_size = entry_size
        self.entry_price = entry_price
        self.entry_time = datetime.now()

        # Initialize exchange
        self.kf = ccxt.krakenfutures({'enableRateLimit': True})
        self.kf.load_markets()

        # Get current price if no entry price specified
        if self.entry_price is None:
            ticker = self.kf.fetch_ticker('DASH/USD:USD')
            self.entry_price = ticker['last']

        # Calculate position size in DASH
        self.position_dash = self.entry_size / self.entry_price

        # Tracking
        self.funding_collected = 0.0
        self.snapshots = []

        # Log file
        self.log_file = DATA_DIR / f"position_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        self._init_log()

        cprint(f"\n{'='*60}", "cyan")
        cprint("DASH POSITION TRACKER", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan")
        print(f"  Entry Size:    ${self.entry_size:.2f}")
        print(f"  Entry Price:   ${self.entry_price:.2f}")
        print(f"  Position:      {self.position_dash:.4f} DASH")
        print(f"  Log File:      {self.log_file.name}")

    def _init_log(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'price', 'position_value', 'pnl_price',
                'funding_rate', 'funding_earned', 'total_funding',
                'total_pnl'
            ])

    def get_funding_since_entry(self) -> float:
        """Calculate total funding earned since entry"""
        try:
            # Get funding history
            history = self.kf.fetch_funding_rate_history('DASH/USD:USD', limit=100)

            total_funding_pct = 0
            for h in history:
                ts = datetime.fromisoformat(h['datetime'].replace('Z', '+00:00')).replace(tzinfo=None)
                if ts >= self.entry_time:
                    rate = h.get('fundingRate', 0) * 100
                    total_funding_pct += abs(rate) if rate < 0 else 0  # Only count negative (we get paid)

            # Convert to USD
            funding_usd = self.entry_size * total_funding_pct / 100
            return funding_usd, total_funding_pct

        except Exception as e:
            return 0, 0

    def check_position(self):
        """Check current position status"""
        try:
            # Current price
            ticker = self.kf.fetch_ticker('DASH/USD:USD')
            current_price = ticker['last']

            # Position value from price
            position_value = self.position_dash * current_price
            pnl_price = position_value - self.entry_size
            pnl_price_pct = (pnl_price / self.entry_size) * 100

            # Current funding rate
            history = self.kf.fetch_funding_rate_history('DASH/USD:USD', limit=1)
            current_rate = history[0].get('fundingRate', 0) * 100 if history else 0

            # Total funding earned
            funding_usd, funding_pct = self.get_funding_since_entry()
            self.funding_collected = funding_usd

            # Total P&L
            total_pnl = pnl_price + funding_usd
            total_pnl_pct = (total_pnl / self.entry_size) * 100

            # Log
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    f"{current_price:.2f}",
                    f"{position_value:.4f}",
                    f"{pnl_price:.4f}",
                    f"{current_rate:.4f}",
                    f"{funding_usd:.4f}",
                    f"{funding_usd:.4f}",
                    f"{total_pnl:.4f}"
                ])

            return {
                'price': current_price,
                'position_value': position_value,
                'pnl_price': pnl_price,
                'pnl_price_pct': pnl_price_pct,
                'current_rate': current_rate,
                'funding_earned': funding_usd,
                'funding_pct': funding_pct,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct
            }

        except Exception as e:
            cprint(f"Error: {e}", "red")
            return None

    def display(self, data):
        """Display position status"""
        if not data:
            return

        print("\n" + "=" * 60)
        print(colored("YOUR DASH POSITION", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Price section
        print(f"\n" + colored(" PRICE:", "yellow", attrs=['bold']))
        print(f"  Entry:    ${self.entry_price:.2f}")
        print(f"  Current:  ${data['price']:.2f}")
        price_change = ((data['price'] - self.entry_price) / self.entry_price) * 100
        color = "green" if price_change >= 0 else "red"
        print(f"  Change:   {colored(f'{price_change:+.2f}%', color)}")

        # Position value
        print(f"\n" + colored(" POSITION VALUE:", "yellow", attrs=['bold']))
        print(f"  Entry:    ${self.entry_size:.2f}")
        print(f"  Current:  ${data['position_value']:.2f}")
        color = "green" if data['pnl_price'] >= 0 else "red"
        print(f"  P&L:      {colored(f'${data['pnl_price']:+.2f} ({data['pnl_price_pct']:+.1f}%)', color)}")

        # Funding
        print(f"\n" + colored(" FUNDING INCOME:", "yellow", attrs=['bold']))
        print(f"  Current Rate:  {data['current_rate']:.4f}%/hr", end="")
        if data['current_rate'] < 0:
            print(colored(" (you get paid)", "green"))
        else:
            print(colored(" (you pay)", "red"))
        print(f"  Earned:        {colored(f'${data['funding_earned']:.4f}', 'green')} ({data['funding_pct']:.4f}%)")

        # Total
        print(f"\n" + colored(" TOTAL P&L:", "cyan", attrs=['bold']))
        print("-" * 40)
        print(f"  Price P&L:     ${data['pnl_price']:+.4f}")
        print(f"  Funding:       ${data['funding_earned']:+.4f}")
        print(f"  " + "-" * 30)
        color = "green" if data['total_pnl'] >= 0 else "red"
        print(f"  TOTAL:         {colored(f'${data['total_pnl']:+.4f} ({data['total_pnl_pct']:+.2f}%)', color, attrs=['bold'])}")

        # Projection
        if data['funding_earned'] > 0:
            hours_held = max(1, (datetime.now() - self.entry_time).total_seconds() / 3600)
            hourly_funding = data['funding_earned'] / hours_held
            print(f"\n" + colored(" PROJECTION:", "yellow", attrs=['bold']))
            print(f"  Funding/hour:  ${hourly_funding:.4f}")
            print(f"  Funding/day:   ${hourly_funding * 24:.4f}")
            print(f"  Funding/month: ${hourly_funding * 24 * 30:.2f}")

        print("\n" + "=" * 60)

    def run(self):
        """Run continuous tracking"""
        cprint(f"\nTracking started. Updates every {CHECK_INTERVAL}s", "green")
        cprint("Press Ctrl+C to stop\n", "yellow")

        while True:
            try:
                data = self.check_position()
                self.display(data)
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                cprint("\n\nTracking stopped.", "yellow")
                print(f"Log saved to: {self.log_file}")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Track DASH position")
    parser.add_argument("--entry", type=float, default=10.0, help="Entry size in USD")
    parser.add_argument("--entry-price", type=float, default=None, help="Entry price")

    args = parser.parse_args()

    tracker = DashPositionTracker(
        entry_size=args.entry,
        entry_price=args.entry_price
    )
    tracker.run()


if __name__ == "__main__":
    main()
