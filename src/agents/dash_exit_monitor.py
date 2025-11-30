"""
Moon Dev's DASH Exit Monitor
Monitors your position and alerts when to EXIT WITH PROFIT

This runs continuously and will alert you when:
1. You're in profit (price recovery + funding)
2. Danger signals (funding flips, whale dumps)
3. Good exit opportunities
"""

import ccxt
import time
from datetime import datetime, timedelta
from termcolor import colored, cprint

# ============================================================================
# YOUR POSITION CONFIG
# ============================================================================

ENTRY_SIZE = 10.00
ENTRY_PRICE = 67.75
ENTRY_TIME = datetime(2025, 11, 27, 2, 0)

# Exit targets
MIN_PROFIT_USD = 0.10        # Alert at $0.10+ profit
GOOD_PROFIT_USD = 0.50       # Strong exit signal at $0.50+
DANGER_LOSS_USD = -2.00      # Danger alert at -$2.00

# Check interval
CHECK_INTERVAL = 60  # Every minute

# ============================================================================


class DashExitMonitor:
    def __init__(self):
        self.kf = ccxt.krakenfutures({'enableRateLimit': True})
        self.kf.load_markets()

        self.position_dash = ENTRY_SIZE / ENTRY_PRICE
        self.best_pnl = -999
        self.last_alert = None

        cprint(f"\n{'='*60}", "cyan")
        cprint("DASH EXIT MONITOR - WATCHING YOUR POSITION", "cyan", attrs=['bold'])
        cprint(f"{'='*60}", "cyan")
        print(f"  Entry: ${ENTRY_SIZE:.2f} at ${ENTRY_PRICE:.2f}")
        print(f"  Position: {self.position_dash:.4f} DASH")
        print(f"  Monitoring for exit signals...")
        cprint("  Press Ctrl+C to stop\n", "yellow")

    def get_status(self):
        """Get current position status"""
        ticker = self.kf.fetch_ticker('DASH/USD:USD')
        current_price = ticker['last']

        # Position value from price
        position_value = self.position_dash * current_price
        price_pnl = position_value - ENTRY_SIZE

        # Funding earned
        history = self.kf.fetch_funding_rate_history('DASH/USD:USD', limit=24)
        rates = [h.get('fundingRate', 0) * 100 for h in history]
        avg_rate = sum(rates) / len(rates) if rates else 0
        current_rate = rates[0] if rates else 0

        hours_held = (datetime.now() - ENTRY_TIME).total_seconds() / 3600
        funding_earned = ENTRY_SIZE * abs(avg_rate) * hours_held / 100

        # Total P&L
        total_pnl = price_pnl + funding_earned

        return {
            'price': current_price,
            'price_pnl': price_pnl,
            'funding_earned': funding_earned,
            'total_pnl': total_pnl,
            'hours_held': hours_held,
            'current_rate': current_rate,
            'avg_rate': avg_rate,
            'funding_positive': current_rate > 0  # Bad for longs
        }

    def check_exit_signals(self, status):
        """Check if we should exit"""
        signals = []

        # PROFIT SIGNALS
        if status['total_pnl'] >= GOOD_PROFIT_USD:
            signals.append({
                'type': 'TAKE_PROFIT',
                'urgency': 'HIGH',
                'message': f"GOOD PROFIT! +${status['total_pnl']:.2f} - Consider taking profits!"
            })
        elif status['total_pnl'] >= MIN_PROFIT_USD:
            signals.append({
                'type': 'IN_PROFIT',
                'urgency': 'MEDIUM',
                'message': f"You're in profit! +${status['total_pnl']:.2f}"
            })

        # DANGER SIGNALS
        if status['total_pnl'] <= DANGER_LOSS_USD:
            signals.append({
                'type': 'STOP_LOSS',
                'urgency': 'HIGH',
                'message': f"DANGER! Loss at ${status['total_pnl']:.2f} - Consider cutting losses"
            })

        # Funding flipped (bad for longs)
        if status['funding_positive']:
            signals.append({
                'type': 'FUNDING_WARNING',
                'urgency': 'MEDIUM',
                'message': f"Funding turned positive ({status['current_rate']:.4f}%) - You now PAY funding!"
            })

        return signals

    def display_status(self, status, signals):
        """Display current status"""
        now = datetime.now().strftime('%H:%M:%S')

        # Color based on P&L
        if status['total_pnl'] >= GOOD_PROFIT_USD:
            pnl_color = 'green'
            status_icon = '💰'
        elif status['total_pnl'] >= 0:
            pnl_color = 'green'
            status_icon = '✓'
        elif status['total_pnl'] >= DANGER_LOSS_USD:
            pnl_color = 'yellow'
            status_icon = '⚠'
        else:
            pnl_color = 'red'
            status_icon = '🚨'

        # Status line
        pnl_str = colored(f"${status['total_pnl']:+.2f}", pnl_color)
        print(f"[{now}] {status_icon} DASH ${status['price']:.2f} | P&L: {pnl_str} (price: ${status['price_pnl']:+.2f}, funding: ${status['funding_earned']:+.4f})")

        # Alert on signals
        for sig in signals:
            if sig['urgency'] == 'HIGH':
                print("\n" + "!" * 60)
                cprint(f"  🚨 {sig['type']}: {sig['message']}", "red", attrs=['bold'])
                print("!" * 60 + "\n")
            elif sig['urgency'] == 'MEDIUM':
                cprint(f"  ⚠ {sig['type']}: {sig['message']}", "yellow")

        # Track best P&L
        if status['total_pnl'] > self.best_pnl:
            self.best_pnl = status['total_pnl']
            if self.best_pnl > 0:
                cprint(f"  📈 New best P&L: ${self.best_pnl:.2f}!", "green")

    def run(self):
        """Run continuous monitoring"""
        print(f"\n{'─'*60}")
        print("Starting continuous monitoring...")
        print(f"{'─'*60}\n")

        check_count = 0

        while True:
            try:
                status = self.get_status()
                signals = self.check_exit_signals(status)
                self.display_status(status, signals)

                check_count += 1

                # Every 10 checks, show summary
                if check_count % 10 == 0:
                    hours = status['hours_held']
                    daily_funding = abs(status['avg_rate']) * 24
                    print(f"\n  ── {hours:.1f}h held | Funding: {daily_funding:.2f}%/day | Best P&L: ${self.best_pnl:.2f} ──\n")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")
                print(f"Final Status:")
                print(f"  Hours held: {status['hours_held']:.1f}")
                print(f"  Best P&L: ${self.best_pnl:.2f}")
                print(f"  Final P&L: ${status['total_pnl']:.2f}")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(30)


if __name__ == "__main__":
    monitor = DashExitMonitor()
    monitor.run()
