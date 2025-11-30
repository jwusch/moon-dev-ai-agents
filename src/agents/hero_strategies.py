"""
Moon Dev's Zero-to-Hero Multi-Strategy Bot
4 strategies to grow $9.48 to $1000+

Strategies:
1. Funding Farming - Collect funding from shorts (PASSIVE)
2. Momentum Scalper - Quick trades on strong moves (ACTIVE)
3. Breakout Alerts - Notify on key level breaks (ALERTS)
4. Grid Bot - Profit from sideways chop (SEMI-PASSIVE)

Usage:
    python src/agents/hero_strategies.py --all           # Run all strategies
    python src/agents/hero_strategies.py --scalper      # Run scalper only
    python src/agents/hero_strategies.py --breakout     # Run breakout alerts
    python src/agents/hero_strategies.py --grid         # Run grid bot
    python src/agents/hero_strategies.py --scan         # Scan for opportunities
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
DATA_DIR = PROJECT_ROOT / "src" / "data" / "zero_to_hero"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Account
BALANCE = 9.48
RISK_PER_TRADE = 0.02  # 2% of balance per trade

# Scalper settings
SCALP_TARGET_PCT = 1.5      # Take profit at 1.5%
SCALP_STOP_PCT = 1.0        # Stop loss at 1%
SCALP_LOOKBACK = 5          # Minutes to detect momentum

# Breakout settings
BREAKOUT_LOOKBACK = 24      # Hours for high/low
BREAKOUT_THRESHOLD = 0.5    # % above high to trigger

# Grid settings
GRID_LEVELS = 5             # Number of grid levels
GRID_SPACING_PCT = 1.0      # 1% between levels
GRID_SIZE_PER_LEVEL = 2.0   # $2 per grid level

# Coins to trade
TRADING_COINS = ['DASH', 'ETH', 'BTC', 'SOL', 'LTC']

CHECK_INTERVAL = 30  # Seconds


@dataclass
class Signal:
    """Trading signal"""
    timestamp: datetime
    coin: str
    strategy: str
    direction: str  # LONG or SHORT
    entry_price: float
    target_price: float
    stop_price: float
    size_usd: float
    confidence: float
    reason: str


class HeroStrategies:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint("ZERO-TO-HERO MULTI-STRATEGY BOT", "cyan", attrs=['bold'])
        cprint(f"Starting Balance: ${BALANCE:.2f}", "cyan")
        cprint(f"{'='*70}", "cyan")

        # Initialize exchange
        self.kf = ccxt.krakenfutures({'enableRateLimit': True})
        self.kf.load_markets()
        cprint("  Kraken Futures: Connected", "green")

        # Signal log
        self.signals: List[Signal] = []
        self.signal_log = DATA_DIR / f"signals_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_signal_log()

        # Price history cache
        self.price_history: Dict[str, List[dict]] = {}

    def _init_signal_log(self):
        if not self.signal_log.exists():
            with open(self.signal_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'coin', 'strategy', 'direction',
                    'entry', 'target', 'stop', 'size', 'confidence', 'reason'
                ])

    def log_signal(self, signal: Signal):
        """Log a signal"""
        with open(self.signal_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                signal.timestamp.isoformat(),
                signal.coin, signal.strategy, signal.direction,
                f"{signal.entry_price:.4f}", f"{signal.target_price:.4f}",
                f"{signal.stop_price:.4f}", f"{signal.size_usd:.2f}",
                f"{signal.confidence:.2f}", signal.reason
            ])
        self.signals.append(signal)

    # =========================================================================
    # STRATEGY 1: FUNDING FARMING (Already implemented separately)
    # =========================================================================

    def check_funding(self, coin: str) -> Optional[dict]:
        """Check funding rate for a coin"""
        try:
            sym = f"{coin}/USD:USD"
            history = self.kf.fetch_funding_rate_history(sym, limit=24)
            rates = [h.get('fundingRate', 0) * 100 for h in history]
            avg_rate = sum(rates) / len(rates) if rates else 0

            return {
                'coin': coin,
                'avg_rate': avg_rate,
                'direction': 'LONG' if avg_rate < 0 else 'SHORT',
                'daily_yield': abs(avg_rate) * 24
            }
        except:
            return None

    # =========================================================================
    # STRATEGY 2: MOMENTUM SCALPER
    # =========================================================================

    def get_recent_candles(self, coin: str, minutes: int = 60) -> List[dict]:
        """Get recent 1-minute candles"""
        try:
            sym = f"{coin}/USD:USD"
            since = int((datetime.now() - timedelta(minutes=minutes)).timestamp() * 1000)
            ohlcv = self.kf.fetch_ohlcv(sym, '1m', since=since, limit=minutes)

            candles = []
            for c in ohlcv:
                candles.append({
                    'timestamp': c[0],
                    'open': c[1],
                    'high': c[2],
                    'low': c[3],
                    'close': c[4],
                    'volume': c[5]
                })
            return candles
        except:
            return []

    def check_momentum(self, coin: str) -> Optional[Signal]:
        """
        Check for momentum scalp opportunity

        Signal: Strong move in last 5 minutes with volume
        """
        candles = self.get_recent_candles(coin, 30)
        if len(candles) < 10:
            return None

        # Recent candles
        recent = candles[-SCALP_LOOKBACK:]
        older = candles[-15:-SCALP_LOOKBACK]

        # Calculate momentum
        recent_move = (recent[-1]['close'] - recent[0]['open']) / recent[0]['open'] * 100
        avg_volume_recent = sum(c['volume'] for c in recent) / len(recent)
        avg_volume_older = sum(c['volume'] for c in older) / len(older) if older else 1

        volume_spike = avg_volume_recent / avg_volume_older if avg_volume_older > 0 else 1

        current_price = recent[-1]['close']

        # Strong bullish momentum
        if recent_move > 0.5 and volume_spike > 1.5:
            target = current_price * (1 + SCALP_TARGET_PCT / 100)
            stop = current_price * (1 - SCALP_STOP_PCT / 100)
            confidence = min(0.9, 0.5 + recent_move * 0.1 + volume_spike * 0.1)

            return Signal(
                timestamp=datetime.now(),
                coin=coin,
                strategy="SCALPER",
                direction="LONG",
                entry_price=current_price,
                target_price=target,
                stop_price=stop,
                size_usd=BALANCE * RISK_PER_TRADE,
                confidence=confidence,
                reason=f"Bullish momentum +{recent_move:.1f}% with {volume_spike:.1f}x volume"
            )

        # Strong bearish momentum
        elif recent_move < -0.5 and volume_spike > 1.5:
            target = current_price * (1 - SCALP_TARGET_PCT / 100)
            stop = current_price * (1 + SCALP_STOP_PCT / 100)
            confidence = min(0.9, 0.5 + abs(recent_move) * 0.1 + volume_spike * 0.1)

            return Signal(
                timestamp=datetime.now(),
                coin=coin,
                strategy="SCALPER",
                direction="SHORT",
                entry_price=current_price,
                target_price=target,
                stop_price=stop,
                size_usd=BALANCE * RISK_PER_TRADE,
                confidence=confidence,
                reason=f"Bearish momentum {recent_move:.1f}% with {volume_spike:.1f}x volume"
            )

        return None

    # =========================================================================
    # STRATEGY 3: BREAKOUT ALERTS
    # =========================================================================

    def get_daily_range(self, coin: str) -> Optional[dict]:
        """Get 24h high/low"""
        try:
            sym = f"{coin}/USD:USD"
            ticker = self.kf.fetch_ticker(sym)

            return {
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'current': ticker.get('last', 0),
                'volume': ticker.get('quoteVolume', 0)
            }
        except:
            return None

    def check_breakout(self, coin: str) -> Optional[Signal]:
        """
        Check for breakout opportunity

        Signal: Price breaking above 24h high or below 24h low
        """
        data = self.get_daily_range(coin)
        if not data:
            return None

        current = data['current']
        high = data['high']
        low = data['low']

        # Check for None values
        if not current or not high or not low or high == 0 or low == 0:
            return None

        # Distance from extremes
        pct_from_high = (current - high) / high * 100
        pct_from_low = (current - low) / low * 100

        # Breakout above high
        if pct_from_high > BREAKOUT_THRESHOLD:
            target = current * 1.03  # 3% target on breakout
            stop = high * 0.995  # Stop just below the breakout level

            return Signal(
                timestamp=datetime.now(),
                coin=coin,
                strategy="BREAKOUT",
                direction="LONG",
                entry_price=current,
                target_price=target,
                stop_price=stop,
                size_usd=BALANCE * RISK_PER_TRADE * 1.5,  # Larger size on breakouts
                confidence=0.7,
                reason=f"Breaking 24h high! +{pct_from_high:.2f}% above ${high:.2f}"
            )

        # Breakdown below low
        elif pct_from_low < -BREAKOUT_THRESHOLD:
            target = current * 0.97  # 3% target on breakdown
            stop = low * 1.005  # Stop just above the breakdown level

            return Signal(
                timestamp=datetime.now(),
                coin=coin,
                strategy="BREAKOUT",
                direction="SHORT",
                entry_price=current,
                target_price=target,
                stop_price=stop,
                size_usd=BALANCE * RISK_PER_TRADE * 1.5,
                confidence=0.7,
                reason=f"Breaking 24h low! {pct_from_low:.2f}% below ${low:.2f}"
            )

        return None

    # =========================================================================
    # STRATEGY 4: GRID BOT
    # =========================================================================

    def calculate_grid_levels(self, coin: str) -> Optional[dict]:
        """
        Calculate grid levels for a coin

        Grid: Place buy orders below current price, sell orders above
        """
        data = self.get_daily_range(coin)
        if not data:
            return None

        current = data['current']
        high = data['high']
        low = data['low']

        # Check for None values
        if not current or not high or not low:
            return None

        # Use the daily range to set grid
        range_pct = (high - low) / low * 100 if low > 0 else 10

        # Grid spacing based on volatility
        spacing = max(GRID_SPACING_PCT, range_pct / (GRID_LEVELS * 2))

        buy_levels = []
        sell_levels = []

        for i in range(1, GRID_LEVELS + 1):
            buy_price = current * (1 - spacing * i / 100)
            sell_price = current * (1 + spacing * i / 100)
            buy_levels.append(buy_price)
            sell_levels.append(sell_price)

        return {
            'coin': coin,
            'current': current,
            'buy_levels': buy_levels,
            'sell_levels': sell_levels,
            'spacing_pct': spacing,
            'size_per_level': GRID_SIZE_PER_LEVEL
        }

    # =========================================================================
    # SCANNER - Check all strategies
    # =========================================================================

    def scan_all(self):
        """Scan all coins for all strategies"""
        print("\n" + "=" * 70)
        print(colored("SCANNING ALL STRATEGIES", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        all_signals = []

        for coin in TRADING_COINS:
            cprint(f"\n  Checking {coin}...", "white")

            # 1. Funding
            funding = self.check_funding(coin)
            if funding and abs(funding['avg_rate']) > 0.02:
                cprint(f"    [FUNDING] {funding['direction']} - {funding['daily_yield']:.2f}%/day", "green")

            # 2. Momentum
            momentum = self.check_momentum(coin)
            if momentum:
                color = "green" if momentum.direction == "LONG" else "red"
                cprint(f"    [SCALPER] {momentum.direction} - {momentum.reason}", color)
                all_signals.append(momentum)
                self.log_signal(momentum)

            # 3. Breakout
            breakout = self.check_breakout(coin)
            if breakout:
                color = "green" if breakout.direction == "LONG" else "red"
                cprint(f"    [BREAKOUT] {breakout.direction} - {breakout.reason}", color, attrs=['bold'])
                all_signals.append(breakout)
                self.log_signal(breakout)

            # 4. Grid levels
            grid = self.calculate_grid_levels(coin)
            if grid:
                cprint(f"    [GRID] {GRID_LEVELS} levels, {grid['spacing_pct']:.1f}% spacing", "yellow")

        # Summary
        print("\n" + "=" * 70)
        print(colored("SIGNAL SUMMARY", "yellow", attrs=['bold']))
        print("-" * 70)

        if all_signals:
            print(f"\n{'Coin':<8} {'Strategy':<12} {'Direction':<8} {'Entry':>10} {'Target':>10} {'Conf':>6}")
            print("-" * 60)
            for s in all_signals:
                color = "green" if s.direction == "LONG" else "red"
                print(f"{s.coin:<8} {s.strategy:<12} {colored(s.direction, color):<16} ${s.entry_price:>9.2f} ${s.target_price:>9.2f} {s.confidence:>5.0%}")
        else:
            print("  No active signals right now")

        # Best opportunity
        if all_signals:
            best = max(all_signals, key=lambda x: x.confidence)
            print(f"\n" + colored("BEST OPPORTUNITY:", "green", attrs=['bold']))
            print(f"  {best.coin} - {best.strategy} {best.direction}")
            print(f"  Entry: ${best.entry_price:.2f} -> Target: ${best.target_price:.2f}")
            print(f"  {best.reason}")

        print("=" * 70)

        return all_signals

    # =========================================================================
    # DISPLAY GRID
    # =========================================================================

    def display_grid(self, coin: str = 'DASH'):
        """Display grid levels for a coin"""
        grid = self.calculate_grid_levels(coin)
        if not grid:
            print("Could not calculate grid")
            return

        print("\n" + "=" * 60)
        print(colored(f"GRID BOT LEVELS - {coin}", "cyan", attrs=['bold']))
        print("=" * 60)

        print(f"\n  Current Price: ${grid['current']:.2f}")
        print(f"  Grid Spacing: {grid['spacing_pct']:.1f}%")
        print(f"  Size per Level: ${grid['size_per_level']:.2f}")

        print(f"\n" + colored("  SELL ORDERS (above current):", "red"))
        for i, price in enumerate(grid['sell_levels']):
            pct = (price - grid['current']) / grid['current'] * 100
            print(f"    Level {i+1}: ${price:.2f} (+{pct:.1f}%)")

        print(f"\n  --- Current: ${grid['current']:.2f} ---")

        print(f"\n" + colored("  BUY ORDERS (below current):", "green"))
        for i, price in enumerate(grid['buy_levels']):
            pct = (price - grid['current']) / grid['current'] * 100
            print(f"    Level {i+1}: ${price:.2f} ({pct:.1f}%)")

        print("\n" + "=" * 60)

    # =========================================================================
    # CONTINUOUS MONITORING
    # =========================================================================

    def run(self, strategies: List[str] = None):
        """Run continuous monitoring"""
        if strategies is None:
            strategies = ['all']

        cprint(f"\nStarting strategy monitor...", "green")
        cprint(f"  Strategies: {', '.join(strategies)}", "white")
        cprint(f"  Check interval: {CHECK_INTERVAL}s", "white")
        cprint("Press Ctrl+C to stop\n", "yellow")

        while True:
            try:
                signals = self.scan_all()

                if signals:
                    cprint(f"\n[ALERT] {len(signals)} signal(s) found!", "yellow", attrs=['bold'])

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\n\nStopping monitor...", "yellow")
                print(f"Signals logged to: {self.signal_log}")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Zero-to-Hero Multi-Strategy Bot")
    parser.add_argument("--all", action="store_true", help="Run all strategies")
    parser.add_argument("--scan", action="store_true", help="Single scan")
    parser.add_argument("--scalper", action="store_true", help="Run scalper")
    parser.add_argument("--breakout", action="store_true", help="Run breakout alerts")
    parser.add_argument("--grid", type=str, default=None, help="Show grid for coin")

    args = parser.parse_args()

    bot = HeroStrategies()

    if args.grid:
        bot.display_grid(args.grid.upper())
    elif args.scan:
        bot.scan_all()
    else:
        bot.run()


if __name__ == "__main__":
    main()
