"""
Moon Dev's DASH Whale Tracker
Monitors large positions and whale activity on DASH

Features:
1. Large trade detection on Kraken Futures
2. Open Interest tracking (whale accumulation/distribution)
3. Position delta monitoring
4. Funding rate spikes (indicates positioning changes)

Usage:
    python src/agents/dash_whale_tracker.py                # Run continuous monitoring
    python src/agents/dash_whale_tracker.py --snapshot     # Single snapshot
"""

import ccxt
import time
import csv
from datetime import datetime, timedelta
from termcolor import colored, cprint
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "dash_whales"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# What counts as a "whale" trade
WHALE_TRADE_USD = 10000      # $10k+ = whale trade
LARGE_TRADE_USD = 5000       # $5k+ = large trade

# Open Interest thresholds
OI_CHANGE_ALERT_PCT = 5      # Alert if OI changes 5%+ in an hour
OI_SPIKE_PCT = 10            # Major spike alert

# Monitoring settings
CHECK_INTERVAL = 60          # Check every minute
TRADE_HISTORY_HOURS = 1      # Look back 1 hour for large trades

# Alert settings
ALERT_COOLDOWN = 300         # Don't repeat alerts within 5 minutes


@dataclass
class WhaleAlert:
    """Whale activity alert"""
    timestamp: datetime
    alert_type: str          # TRADE, OI_CHANGE, FUNDING_SPIKE
    direction: str           # BUY, SELL, NEUTRAL
    size_usd: float
    details: str
    significance: str        # LOW, MEDIUM, HIGH


class DashWhaleTracker:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint("DASH WHALE TRACKER", "cyan", attrs=['bold'])
        cprint("Monitoring large positions and whale activity", "cyan")
        cprint(f"{'='*70}", "cyan")

        # Initialize exchange
        self.kf = ccxt.krakenfutures({'enableRateLimit': True})
        self.kf.load_markets()
        cprint("  Kraken Futures: Connected", "green")

        # Symbol
        self.symbol = 'DASH/USD:USD'

        # State tracking
        self.last_oi = None
        self.last_funding = None
        self.last_price = None
        self.alerts: List[WhaleAlert] = []
        self.recent_alert_types: Dict[str, datetime] = {}

        # Price/OI history
        self.oi_history: List[dict] = []
        self.price_history: List[dict] = []

        # Log file
        self.log_file = DATA_DIR / f"whale_alerts_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_log()

        # Get initial state
        self._initialize_state()

    def _init_log(self):
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'alert_type', 'direction', 'size_usd',
                    'details', 'significance'
                ])

    def _initialize_state(self):
        """Get initial market state"""
        try:
            ticker = self.kf.fetch_ticker(self.symbol)
            self.last_price = ticker.get('last', 0)

            # Get funding rate
            history = self.kf.fetch_funding_rate_history(self.symbol, limit=1)
            if history:
                self.last_funding = history[0].get('fundingRate', 0) * 100

            # Get approximate OI from ticker info
            self.last_oi = ticker.get('info', {}).get('openInterest', 0)
            if isinstance(self.last_oi, str):
                self.last_oi = float(self.last_oi) if self.last_oi else 0

            cprint(f"\n  Initial State:", "yellow")
            cprint(f"    Price: ${self.last_price:.2f}", "white")
            cprint(f"    Funding: {self.last_funding:.4f}%/hr", "white")
            cprint(f"    Open Interest: {self.last_oi}", "white")

        except Exception as e:
            cprint(f"Error initializing: {e}", "red")

    def log_alert(self, alert: WhaleAlert):
        """Log a whale alert"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.direction,
                f"{alert.size_usd:.2f}",
                alert.details,
                alert.significance
            ])
        self.alerts.append(alert)

    def _should_alert(self, alert_type: str) -> bool:
        """Check if we should send this alert (cooldown)"""
        if alert_type in self.recent_alert_types:
            last_time = self.recent_alert_types[alert_type]
            if (datetime.now() - last_time).total_seconds() < ALERT_COOLDOWN:
                return False
        self.recent_alert_types[alert_type] = datetime.now()
        return True

    def get_recent_trades(self) -> List[dict]:
        """Get recent large trades"""
        try:
            # Fetch recent trades
            trades = self.kf.fetch_trades(self.symbol, limit=100)

            large_trades = []
            for t in trades:
                trade_usd = t['amount'] * t['price']
                if trade_usd >= LARGE_TRADE_USD:
                    large_trades.append({
                        'timestamp': t['datetime'],
                        'side': t['side'],
                        'price': t['price'],
                        'amount': t['amount'],
                        'usd_value': trade_usd,
                        'is_whale': trade_usd >= WHALE_TRADE_USD
                    })

            return large_trades

        except Exception as e:
            return []

    def check_oi_change(self) -> Optional[WhaleAlert]:
        """Check for significant Open Interest changes"""
        try:
            ticker = self.kf.fetch_ticker(self.symbol)
            current_oi = ticker.get('info', {}).get('openInterest', 0)
            if isinstance(current_oi, str):
                current_oi = float(current_oi) if current_oi else 0

            if self.last_oi and self.last_oi > 0 and current_oi > 0:
                oi_change_pct = ((current_oi - self.last_oi) / self.last_oi) * 100

                # Store history
                self.oi_history.append({
                    'timestamp': datetime.now(),
                    'oi': current_oi,
                    'change_pct': oi_change_pct
                })

                # Keep last 60 entries (1 hour if checking every minute)
                if len(self.oi_history) > 60:
                    self.oi_history = self.oi_history[-60:]

                # Check for significant change
                if abs(oi_change_pct) >= OI_CHANGE_ALERT_PCT:
                    direction = "BUY" if oi_change_pct > 0 else "SELL"
                    significance = "HIGH" if abs(oi_change_pct) >= OI_SPIKE_PCT else "MEDIUM"

                    self.last_oi = current_oi

                    if self._should_alert(f"OI_{direction}"):
                        return WhaleAlert(
                            timestamp=datetime.now(),
                            alert_type="OI_CHANGE",
                            direction=direction,
                            size_usd=current_oi * self.last_price if self.last_price else 0,
                            details=f"OI {'increased' if oi_change_pct > 0 else 'decreased'} {abs(oi_change_pct):.1f}% - Whales {'accumulating' if oi_change_pct > 0 else 'exiting'}",
                            significance=significance
                        )

            self.last_oi = current_oi
            return None

        except Exception as e:
            return None

    def check_funding_spike(self) -> Optional[WhaleAlert]:
        """Check for funding rate spikes"""
        try:
            history = self.kf.fetch_funding_rate_history(self.symbol, limit=24)
            if not history:
                return None

            current_rate = history[0].get('fundingRate', 0) * 100
            rates = [h.get('fundingRate', 0) * 100 for h in history]
            avg_rate = sum(rates) / len(rates) if rates else 0

            # Check for significant deviation from average
            if len(rates) >= 8:  # Need at least 8 hours of history
                std_dev = (sum((r - avg_rate) ** 2 for r in rates) / len(rates)) ** 0.5
                z_score = (current_rate - avg_rate) / std_dev if std_dev > 0 else 0

                if abs(z_score) > 2:  # More than 2 standard deviations
                    direction = "SHORT" if current_rate > avg_rate else "LONG"
                    details = f"Funding rate {'spiked to' if z_score > 0 else 'dropped to'} {current_rate:.4f}%/hr (avg: {avg_rate:.4f}%)"

                    if self._should_alert(f"FUNDING_{direction}"):
                        return WhaleAlert(
                            timestamp=datetime.now(),
                            alert_type="FUNDING_SPIKE",
                            direction=direction,
                            size_usd=0,
                            details=details + f" - Heavy {'short' if current_rate > 0 else 'long'} positioning",
                            significance="MEDIUM" if abs(z_score) < 3 else "HIGH"
                        )

            self.last_funding = current_rate
            return None

        except Exception as e:
            return None

    def check_large_trades(self) -> List[WhaleAlert]:
        """Check for whale trades"""
        alerts = []
        trades = self.get_recent_trades()

        for t in trades:
            if t['is_whale'] and self._should_alert(f"TRADE_{t['side']}_{int(t['usd_value'])}"):
                alerts.append(WhaleAlert(
                    timestamp=datetime.now(),
                    alert_type="WHALE_TRADE",
                    direction=t['side'].upper(),
                    size_usd=t['usd_value'],
                    details=f"${t['usd_value']:,.0f} {t['side']} at ${t['price']:.2f}",
                    significance="HIGH" if t['usd_value'] >= WHALE_TRADE_USD * 2 else "MEDIUM"
                ))

        return alerts

    def display_alert(self, alert: WhaleAlert):
        """Display an alert"""
        sig_colors = {"LOW": "white", "MEDIUM": "yellow", "HIGH": "red"}
        dir_colors = {"BUY": "green", "SELL": "red", "LONG": "green", "SHORT": "red", "NEUTRAL": "white"}

        print("\n" + "!" * 60)
        cprint(f"  WHALE ALERT - {alert.alert_type}", sig_colors.get(alert.significance, "white"), attrs=['bold'])
        print("!" * 60)
        print(f"  Time:        {alert.timestamp.strftime('%H:%M:%S')}")
        print(f"  Direction:   {colored(alert.direction, dir_colors.get(alert.direction, 'white'))}")
        if alert.size_usd > 0:
            print(f"  Size:        ${alert.size_usd:,.0f}")
        print(f"  Details:     {alert.details}")
        print(f"  Significance: {colored(alert.significance, sig_colors.get(alert.significance, 'white'))}")
        print("!" * 60)

    def get_market_snapshot(self) -> dict:
        """Get current market snapshot"""
        try:
            ticker = self.kf.fetch_ticker(self.symbol)
            current_price = ticker.get('last', 0)
            volume_24h = ticker.get('quoteVolume', 0)
            high_24h = ticker.get('high', 0)
            low_24h = ticker.get('low', 0)

            # Get OI
            current_oi = ticker.get('info', {}).get('openInterest', 0)
            if isinstance(current_oi, str):
                current_oi = float(current_oi) if current_oi else 0

            # Get funding
            history = self.kf.fetch_funding_rate_history(self.symbol, limit=24)
            rates = [h.get('fundingRate', 0) * 100 for h in history]
            current_rate = rates[0] if rates else 0
            avg_rate_24h = sum(rates) / len(rates) if rates else 0

            return {
                'price': current_price,
                'volume_24h': volume_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'open_interest': current_oi,
                'funding_rate': current_rate,
                'avg_funding_24h': avg_rate_24h,
                'price_range_24h': ((high_24h - low_24h) / low_24h * 100) if low_24h else 0
            }

        except Exception as e:
            return {}

    def display_dashboard(self, snapshot: dict):
        """Display whale tracker dashboard"""
        print("\n" + "=" * 70)
        print(colored("  DASH WHALE TRACKER", "cyan", attrs=['bold']))
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Market Data
        print(f"\n" + colored(" MARKET DATA:", "yellow", attrs=['bold']))
        price = snapshot.get('price') or 0
        print(f"  Price:        ${price:.2f}")

        # 24h range
        high = snapshot.get('high_24h') or 0
        low = snapshot.get('low_24h') or 0
        if high and low:
            range_pct = snapshot.get('price_range_24h') or 0
            print(f"  24h Range:    ${low:.2f} - ${high:.2f} ({range_pct:.1f}%)")
        volume = snapshot.get('volume_24h') or 0
        print(f"  24h Volume:   ${volume:,.0f}")

        # Whale Indicators
        print(f"\n" + colored(" WHALE INDICATORS:", "yellow", attrs=['bold']))
        oi = snapshot.get('open_interest') or 0
        print(f"  Open Interest: {oi:,.0f} contracts")

        funding = snapshot.get('funding_rate') or 0
        funding_color = "green" if funding < 0 else "red"
        print(f"  Funding Rate:  {colored(f'{funding:.4f}%/hr', funding_color)}", end="")
        if funding < 0:
            print(colored(" (longs get paid)", "green"))
        else:
            print(colored(" (shorts get paid)", "red"))

        avg_funding = snapshot.get('avg_funding_24h') or 0
        print(f"  Avg Funding:   {avg_funding:.4f}%/hr (24h)")

        # Interpretation
        print(f"\n" + colored(" WHALE INTERPRETATION:", "yellow", attrs=['bold']))

        if funding < -0.03:
            print(colored("    Strong short positioning - whales betting on downside", "red"))
            print("    You earn funding by going LONG")
        elif funding > 0.03:
            print(colored("    Strong long positioning - whales betting on upside", "green"))
            print("    You earn funding by going SHORT")
        else:
            print("    Neutral positioning - no strong whale bias")

        # Recent alerts
        recent = [a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]
        if recent:
            print(f"\n" + colored(" RECENT ALERTS (1hr):", "yellow", attrs=['bold']))
            for a in recent[-5:]:  # Last 5
                color = "green" if a.direction in ["BUY", "LONG"] else "red"
                print(f"    {a.timestamp.strftime('%H:%M')} [{a.alert_type}] {colored(a.direction, color)} - {a.details[:50]}")

        print("\n" + "=" * 70)

    def scan(self):
        """Run a single scan"""
        snapshot = self.get_market_snapshot()
        self.display_dashboard(snapshot)

        # Check for alerts
        alerts = []

        oi_alert = self.check_oi_change()
        if oi_alert:
            alerts.append(oi_alert)

        funding_alert = self.check_funding_spike()
        if funding_alert:
            alerts.append(funding_alert)

        trade_alerts = self.check_large_trades()
        alerts.extend(trade_alerts)

        for alert in alerts:
            self.display_alert(alert)
            self.log_alert(alert)

        return alerts

    def run(self):
        """Run continuous monitoring"""
        cprint(f"\nStarting whale tracker...", "green")
        cprint(f"  Check interval: {CHECK_INTERVAL}s", "white")
        cprint("Press Ctrl+C to stop\n", "yellow")

        while True:
            try:
                alerts = self.scan()

                if alerts:
                    cprint(f"\n[WHALE ACTIVITY] {len(alerts)} alert(s)!", "red", attrs=['bold'])

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n\nStopping whale tracker...", "yellow")
                print(f"Alerts logged to: {self.log_file}")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="DASH Whale Tracker")
    parser.add_argument("--snapshot", action="store_true", help="Single snapshot")

    args = parser.parse_args()

    tracker = DashWhaleTracker()

    if args.snapshot:
        tracker.scan()
    else:
        tracker.run()


if __name__ == "__main__":
    main()
