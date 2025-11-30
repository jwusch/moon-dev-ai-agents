"""
Moon Dev's $9.48 Zero-to-Hero Challenge
Starting small, growing big with AI-powered strategies

Usage:
    python src/agents/zero_to_hero.py                    # Show dashboard
    python src/agents/zero_to_hero.py --log              # Log current position
    python src/agents/zero_to_hero.py --strategies       # Show strategy ideas
"""

import ccxt
import csv
from datetime import datetime
from termcolor import colored, cprint
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "zero_to_hero"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DATA_DIR / "challenge_log.csv"

# ============================================================================
# CHALLENGE CONFIG
# ============================================================================

STARTING_BALANCE = 10.00
CURRENT_BALANCE = 9.48
START_DATE = datetime(2025, 11, 27)

# Goals
MILESTONES = [
    (10, "Back to even"),
    (15, "50% gain"),
    (20, "Double up"),
    (50, "5x"),
    (100, "10x - Hero status!"),
    (1000, "100x - Legend!"),
]


class ZeroToHero:
    def __init__(self):
        self.kf = ccxt.krakenfutures({'enableRateLimit': True})
        self.kf.load_markets()

    def get_current_position(self):
        """Get current DASH position value"""
        try:
            ticker = self.kf.fetch_ticker('DASH/USD:USD')
            return ticker['last']
        except:
            return None

    def load_log(self):
        """Load trade log"""
        trades = []
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
        return trades

    def log_trade(self, action, coin, side, size, price, balance_before, balance_after, strategy, notes):
        """Log a trade"""
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%H:%M'),
                action, coin, side, f"{size:.2f}", f"{price:.2f}",
                f"{balance_before:.2f}", f"{balance_after:.2f}",
                strategy, notes
            ])

    def display_dashboard(self, current_balance: float):
        """Display the challenge dashboard"""
        print("\n" + "=" * 70)
        print(colored("  $9.48 ZERO-TO-HERO CHALLENGE", "cyan", attrs=['bold']))
        print(colored("  Growing small capital with AI strategies", "cyan"))
        print("=" * 70)

        # Current status
        days_in = (datetime.now() - START_DATE).days + 1
        pnl = current_balance - STARTING_BALANCE
        pnl_pct = (pnl / STARTING_BALANCE) * 100

        print(f"\n" + colored(" STATUS:", "yellow", attrs=['bold']))
        print(f"  Started:     ${STARTING_BALANCE:.2f}")
        print(f"  Current:     ${current_balance:.2f}")

        color = "green" if pnl >= 0 else "red"
        print(f"  P&L:         {colored(f'${pnl:+.2f} ({pnl_pct:+.1f}%)', color)}")
        print(f"  Days:        {days_in}")

        # Milestones
        print(f"\n" + colored(" MILESTONES:", "yellow", attrs=['bold']))
        print("-" * 50)

        for target, name in MILESTONES:
            if current_balance >= target:
                status = colored("[DONE]", "green")
            else:
                needed = target - current_balance
                pct_to_go = (needed / current_balance) * 100
                status = f"${needed:.2f} to go (+{pct_to_go:.0f}%)"

            marker = ">" if current_balance < target and (len([m for m in MILESTONES if current_balance >= m[0]]) == MILESTONES.index((target, name))) else " "
            print(f"  {marker} ${target:>6} - {name:<25} {status}")

        # Trade log summary
        trades = self.load_log()
        print(f"\n" + colored(" TRADE LOG:", "yellow", attrs=['bold']))
        print("-" * 50)

        if trades:
            for trade in trades[-5:]:  # Last 5 trades
                print(f"  {trade['date']} {trade['action']:<8} {trade['coin']:<6} {trade['side']:<5} ${float(trade.get('size_usd', 0)):.2f}")
        else:
            print("  No trades logged yet")

        # Strategy ideas
        print(f"\n" + colored(" ACTIVE STRATEGIES:", "yellow", attrs=['bold']))
        print("-" * 50)
        print("  1. Funding Rate Collection (DASH long)")
        print("     Status: ACTIVE - earning ~$0.08/day")

        print("\n" + "=" * 70)

    def show_strategies(self):
        """Show available strategy ideas"""
        print("\n" + "=" * 70)
        print(colored(" STRATEGY IDEAS FOR ZERO-TO-HERO", "cyan", attrs=['bold']))
        print("=" * 70)

        strategies = [
            {
                "name": "1. Funding Rate Farming (Current)",
                "description": "Collect funding from shorts on DASH",
                "expected": "~$0.08/day on $10 (0.8%/day)",
                "risk": "LOW (if hedged) / MEDIUM (unhedged)",
                "status": "ACTIVE"
            },
            {
                "name": "2. Momentum Scalping",
                "description": "Quick trades on strong moves, 1-2% targets",
                "expected": "Variable, 1-5%/trade",
                "risk": "MEDIUM-HIGH",
                "status": "Needs bot"
            },
            {
                "name": "3. Mean Reversion",
                "description": "Buy dips, sell rips on ranging coins",
                "expected": "2-5% per swing",
                "risk": "MEDIUM",
                "status": "Needs bot"
            },
            {
                "name": "4. Breakout Trading",
                "description": "Enter on confirmed breakouts with momentum",
                "expected": "5-20% on good setups",
                "risk": "MEDIUM-HIGH",
                "status": "Needs signals"
            },
            {
                "name": "5. Grid Trading",
                "description": "Place orders at intervals, profit from chop",
                "expected": "0.5-2%/day in sideways market",
                "risk": "LOW-MEDIUM",
                "status": "Needs bot"
            },
            {
                "name": "6. News/Catalyst Trading",
                "description": "Trade around announcements and events",
                "expected": "Variable, can be large",
                "risk": "HIGH",
                "status": "Needs monitoring"
            },
            {
                "name": "7. Cross-Exchange Arb",
                "description": "Exploit price differences between exchanges",
                "expected": "0.1-1% per arb",
                "risk": "LOW",
                "status": "Need access to multiple exchanges"
            },
            {
                "name": "8. Compound & Reinvest",
                "description": "Take profits and add to positions",
                "expected": "Accelerates all other strategies",
                "risk": "N/A",
                "status": "Always active"
            },
        ]

        for s in strategies:
            status_color = "green" if s["status"] == "ACTIVE" else "yellow"
            print(f"\n{colored(s['name'], 'cyan', attrs=['bold'])}")
            print(f"  {s['description']}")
            print(f"  Expected: {s['expected']}")
            print(f"  Risk: {s['risk']}")
            print(f"  Status: {colored(s['status'], status_color)}")

        print("\n" + "-" * 70)
        print(colored(" RECOMMENDED NEXT STEPS:", "green", attrs=['bold']))
        print("  1. Keep funding position running (passive income)")
        print("  2. Build a momentum scalper for quick gains")
        print("  3. Set up alerts for breakout opportunities")
        print("  4. Compound any profits back into positions")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Zero-to-Hero Challenge")
    parser.add_argument("--log", action="store_true", help="Log current position")
    parser.add_argument("--strategies", action="store_true", help="Show strategy ideas")
    parser.add_argument("--balance", type=float, default=9.48, help="Current balance")

    args = parser.parse_args()

    hero = ZeroToHero()

    if args.strategies:
        hero.show_strategies()
    else:
        hero.display_dashboard(args.balance)


if __name__ == "__main__":
    main()
