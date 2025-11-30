"""
Moon Dev's MERL Arbitrage Bot
Monitors MERL prices across multiple exchanges and identifies arbitrage opportunities

Features:
- Multi-exchange price monitoring (spot & perp)
- Real-time arbitrage opportunity detection
- Fee-adjusted profit calculations
- Cross-exchange spread analysis
- PAPER TRADING MODE with simulated P&L
- Optional live trade execution (with safety mode)

Usage:
    python src/agents/merl_arb_bot.py                    # Monitor only
    python src/agents/merl_arb_bot.py --paper            # Paper trading mode
    python src/agents/merl_arb_bot.py --paper --size 50  # Paper trade $50 per trade
    python src/agents/merl_arb_bot.py --execute          # Live execution (DANGEROUS)
"""

import ccxt
import time
import argparse
import csv
from datetime import datetime
from termcolor import colored, cprint
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import os
load_dotenv()

# Data directory for paper trading logs
DATA_DIR = PROJECT_ROOT / "src" / "data" / "merl_arb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Check interval in seconds
CHECK_INTERVAL = 5

# Minimum profit threshold (in %) to flag as opportunity
MIN_PROFIT_THRESHOLD = 0.3  # 0.3% after fees

# Trading fees per exchange (maker/taker in %)
EXCHANGE_FEES = {
    "okx": {"maker": 0.08, "taker": 0.10},
    "kraken": {"maker": 0.16, "taker": 0.26},
    "kucoin": {"maker": 0.10, "taker": 0.10},
    "gate": {"maker": 0.20, "taker": 0.20},
    "bybit": {"maker": 0.10, "taker": 0.10},
    "binance": {"maker": 0.10, "taker": 0.10},
    "htx": {"maker": 0.20, "taker": 0.20},
    "mexc": {"maker": 0.00, "taker": 0.10},
}

# Exchange configurations for MERL
SPOT_EXCHANGES = {
    "okx": {"symbol": "MERL/USDT", "name": "OKX"},
    "kucoin": {"symbol": "MERL/USDT", "name": "KuCoin"},
    "gate": {"symbol": "MERL/USDT", "name": "Gate.io"},
    "bybit": {"symbol": "MERL/USDT", "name": "Bybit"},
    "mexc": {"symbol": "MERL/USDT", "name": "MEXC"},
    "htx": {"symbol": "MERL/USDT", "name": "HTX"},
}

# Perp exchange (for spot-perp arb)
PERP_EXCHANGE = "okx"
PERP_SYMBOL = "MERL/USDT:USDT"

# Position sizing for execution
DEFAULT_TRADE_SIZE_USD = 100  # USD amount per arb trade
MAX_SLIPPAGE_PERCENT = 0.5  # Maximum acceptable slippage


@dataclass
class ExchangePrice:
    """Price data from an exchange"""
    exchange: str
    symbol: str
    bid: float  # Best bid (what you can sell at)
    ask: float  # Best ask (what you can buy at)
    bid_volume: float
    ask_volume: float
    last: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """Represents an arbitrage opportunity"""
    buy_exchange: str
    sell_exchange: str
    buy_price: float  # Ask price on buy exchange
    sell_price: float  # Bid price on sell exchange
    gross_spread_pct: float
    buy_fee_pct: float
    sell_fee_pct: float
    net_profit_pct: float
    available_volume: float  # Min of bid/ask volumes
    potential_profit_usd: float
    timestamp: datetime


@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    id: str
    open_time: datetime
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    amount: float  # MERL amount
    size_usd: float
    expected_profit_pct: float
    status: str = "open"  # open, closed
    close_time: Optional[datetime] = None
    close_buy_price: Optional[float] = None
    close_sell_price: Optional[float] = None
    realized_pnl: Optional[float] = None


@dataclass
class PaperTradingState:
    """Paper trading account state"""
    starting_balance: float = 1000.0
    current_balance: float = 1000.0
    positions: List[PaperPosition] = field(default_factory=list)
    closed_positions: List[PaperPosition] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees_paid: float = 0.0


class MERLArbBot:
    def __init__(
        self,
        execute_mode: bool = False,
        paper_mode: bool = False,
        min_profit_threshold: float = None,
        check_interval: int = None,
        trade_size_usd: float = None,
        starting_balance: float = 1000.0,
        auto_close_threshold: float = 2.0,  # Close position when spread drops below this %
    ):
        """Initialize the arbitrage bot"""
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's MERL Arbitrage Bot", "cyan", attrs=['bold'])
        cprint(f"Cross-Exchange Arbitrage Detection & Execution", "cyan")
        cprint(f"{'='*70}", "cyan")

        self.execute_mode = execute_mode
        self.paper_mode = paper_mode
        self.min_profit_threshold = min_profit_threshold if min_profit_threshold is not None else MIN_PROFIT_THRESHOLD
        self.check_interval = check_interval if check_interval is not None else CHECK_INTERVAL
        self.trade_size_usd = trade_size_usd if trade_size_usd is not None else DEFAULT_TRADE_SIZE_USD
        self.auto_close_threshold = auto_close_threshold

        if execute_mode:
            cprint("\n  *** EXECUTION MODE ENABLED - LIVE TRADES WILL BE PLACED ***", "red", attrs=['bold'])
            cprint("  Press Ctrl+C within 5 seconds to abort...", "yellow")
            time.sleep(5)

        if paper_mode:
            cprint("\n  PAPER TRADING MODE - Simulated trades will be tracked", "green", attrs=['bold'])
            self.paper_state = PaperTradingState(
                starting_balance=starting_balance,
                current_balance=starting_balance
            )
            self.trade_log_file = DATA_DIR / f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self._init_trade_log()
        else:
            self.paper_state = None
            self.trade_log_file = None

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.symbols: Dict[str, str] = {}
        self.prices: Dict[str, ExchangePrice] = {}
        self.opportunities: List[ArbOpportunity] = []
        self.total_opportunities = 0
        self.total_theoretical_profit = 0.0

        # Initialize exchanges
        self._init_exchanges()

        # Stats tracking
        self.start_time = datetime.now()
        self.checks_performed = 0
        self.position_counter = 0

    def _init_trade_log(self):
        """Initialize the CSV trade log file"""
        with open(self.trade_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'action', 'position_id', 'buy_exchange', 'sell_exchange',
                'buy_price', 'sell_price', 'amount', 'size_usd', 'spread_pct',
                'fees_paid', 'pnl', 'balance'
            ])
        cprint(f"  Trade log: {self.trade_log_file}", "white")

    def _init_exchanges(self):
        """Initialize connections to all exchanges"""
        cprint("\nInitializing exchanges...", "white")

        for ex_id, config in SPOT_EXCHANGES.items():
            try:
                # Create exchange instance with API keys if available
                exchange_config = {'enableRateLimit': True}

                # Add API keys if available for execution
                if self.execute_mode:
                    api_key = os.getenv(f"{ex_id.upper()}_API_KEY")
                    api_secret = os.getenv(f"{ex_id.upper()}_API_SECRET")
                    if api_key and api_secret:
                        exchange_config['apiKey'] = api_key
                        exchange_config['secret'] = api_secret

                exchange = getattr(ccxt, ex_id)(exchange_config)
                exchange.load_markets()
                symbol = config["symbol"]

                if symbol in exchange.markets:
                    self.exchanges[ex_id] = exchange
                    self.symbols[ex_id] = symbol
                    cprint(f"  {config['name']}: {symbol}", "green")
                else:
                    cprint(f"  {config['name']}: MERL not found", "yellow")
            except Exception as e:
                cprint(f"  {config['name']}: Failed - {str(e)[:50]}", "red")

        # Initialize perp exchange
        try:
            perp_ex = getattr(ccxt, PERP_EXCHANGE)({'enableRateLimit': True})
            perp_ex.load_markets()
            if PERP_SYMBOL in perp_ex.markets:
                self.exchanges[f"{PERP_EXCHANGE}_perp"] = perp_ex
                self.symbols[f"{PERP_EXCHANGE}_perp"] = PERP_SYMBOL
                cprint(f"  {PERP_EXCHANGE.upper()} Perp: {PERP_SYMBOL}", "green")
        except Exception as e:
            cprint(f"  Perp exchange failed: {str(e)[:50]}", "yellow")

        cprint(f"\n  Active exchanges: {len(self.exchanges)}", "white")

    def fetch_orderbook(self, exchange_id: str) -> Optional[ExchangePrice]:
        """Fetch orderbook data from an exchange"""
        try:
            exchange = self.exchanges[exchange_id]
            symbol = self.symbols[exchange_id]
            orderbook = exchange.fetch_order_book(symbol, limit=5)

            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return None

            # Get ticker for last price
            ticker = exchange.fetch_ticker(symbol)

            return ExchangePrice(
                exchange=exchange_id,
                symbol=symbol,
                bid=bids[0][0],
                ask=asks[0][0],
                bid_volume=sum(b[1] for b in bids[:3]),  # Top 3 levels
                ask_volume=sum(a[1] for a in asks[:3]),
                last=ticker['last'],
                timestamp=datetime.now()
            )
        except Exception as e:
            return None

    def fetch_all_prices(self) -> Dict[str, ExchangePrice]:
        """Fetch prices from all exchanges"""
        prices = {}
        for ex_id in self.exchanges:
            price = self.fetch_orderbook(ex_id)
            if price:
                prices[ex_id] = price
        return prices

    def calculate_arb_opportunity(
        self,
        buy_ex: str,
        sell_ex: str,
        buy_price: ExchangePrice,
        sell_price: ExchangePrice
    ) -> Optional[ArbOpportunity]:
        """Calculate arbitrage opportunity between two exchanges"""
        # Buy at ask price on buy_ex, sell at bid price on sell_ex
        buy_at = buy_price.ask
        sell_at = sell_price.bid

        if buy_at <= 0 or sell_at <= 0:
            return None

        # Gross spread (before fees)
        gross_spread_pct = ((sell_at - buy_at) / buy_at) * 100

        # Get fees
        buy_ex_base = buy_ex.replace("_perp", "")
        sell_ex_base = sell_ex.replace("_perp", "")

        buy_fee = EXCHANGE_FEES.get(buy_ex_base, {"taker": 0.20})["taker"]
        sell_fee = EXCHANGE_FEES.get(sell_ex_base, {"taker": 0.20})["taker"]

        # Net profit after fees
        net_profit_pct = gross_spread_pct - buy_fee - sell_fee

        # Available volume (minimum of what we can buy/sell)
        available_volume = min(buy_price.ask_volume, sell_price.bid_volume)

        # Potential profit in USD
        avg_price = (buy_at + sell_at) / 2
        potential_profit_usd = (net_profit_pct / 100) * available_volume * avg_price

        return ArbOpportunity(
            buy_exchange=buy_ex,
            sell_exchange=sell_ex,
            buy_price=buy_at,
            sell_price=sell_at,
            gross_spread_pct=gross_spread_pct,
            buy_fee_pct=buy_fee,
            sell_fee_pct=sell_fee,
            net_profit_pct=net_profit_pct,
            available_volume=available_volume,
            potential_profit_usd=potential_profit_usd,
            timestamp=datetime.now()
        )

    def find_opportunities(self, prices: Dict[str, ExchangePrice]) -> List[ArbOpportunity]:
        """Find all arbitrage opportunities across exchanges"""
        opportunities = []
        exchanges = list(prices.keys())

        # Compare all exchange pairs
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges[i+1:]:
                # Try buying on buy_ex, selling on sell_ex
                opp1 = self.calculate_arb_opportunity(
                    buy_ex, sell_ex, prices[buy_ex], prices[sell_ex]
                )
                if opp1 and opp1.net_profit_pct > 0:
                    opportunities.append(opp1)

                # Try the reverse
                opp2 = self.calculate_arb_opportunity(
                    sell_ex, buy_ex, prices[sell_ex], prices[buy_ex]
                )
                if opp2 and opp2.net_profit_pct > 0:
                    opportunities.append(opp2)

        # Sort by profit
        opportunities.sort(key=lambda x: x.net_profit_pct, reverse=True)
        return opportunities

    def display_dashboard(self, prices: Dict[str, ExchangePrice], opportunities: List[ArbOpportunity]):
        """Display the arbitrage dashboard"""
        print("\n" + "=" * 80)
        print(colored("MERL ARBITRAGE BOT", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.execute_mode:
            print(colored("  [EXECUTION MODE ACTIVE]", "red", attrs=['bold']))
        print("=" * 80)

        # Price comparison table
        print("\n" + colored(" EXCHANGE PRICES", "yellow", attrs=['bold']))
        print("-" * 80)
        print(f"{'Exchange':<15} {'Bid':>12} {'Ask':>12} {'Spread':>10} {'Bid Vol':>12} {'Ask Vol':>12}")
        print("-" * 80)

        for ex_id, price in sorted(prices.items(), key=lambda x: x[1].ask):
            spread = (price.ask - price.bid) / price.bid * 100
            ex_name = SPOT_EXCHANGES.get(ex_id.replace("_perp", ""), {}).get("name", ex_id)
            if "_perp" in ex_id:
                ex_name += " (Perp)"
            print(f"{ex_name:<15} ${price.bid:>11.5f} ${price.ask:>11.5f} {spread:>9.3f}% {price.bid_volume:>11.1f} {price.ask_volume:>11.1f}")

        # Price range
        if prices:
            min_ask = min(p.ask for p in prices.values())
            max_bid = max(p.bid for p in prices.values())
            max_spread = (max_bid - min_ask) / min_ask * 100
            print("-" * 80)
            print(f"  Best Buy: ${min_ask:.5f}  |  Best Sell: ${max_bid:.5f}  |  Max Spread: {max_spread:.3f}%")

        # Arbitrage opportunities
        print("\n" + colored(" ARBITRAGE OPPORTUNITIES", "yellow", attrs=['bold']))
        print("-" * 80)

        profitable_opps = [o for o in opportunities if o.net_profit_pct >= self.min_profit_threshold]

        if not profitable_opps:
            print(colored(f"  No opportunities above {self.min_profit_threshold}% threshold", "white"))
            # Show best available even if below threshold
            if opportunities:
                best = opportunities[0]
                if best.net_profit_pct > 0:
                    print(f"  Best available: {best.net_profit_pct:.3f}% (below threshold)")
        else:
            print(f"{'Route':<30} {'Gross':>10} {'Fees':>10} {'Net':>10} {'Vol':>10} {'Profit':>12}")
            print("-" * 80)

            for opp in profitable_opps[:10]:  # Show top 10
                buy_name = SPOT_EXCHANGES.get(opp.buy_exchange.replace("_perp", ""), {}).get("name", opp.buy_exchange)
                sell_name = SPOT_EXCHANGES.get(opp.sell_exchange.replace("_perp", ""), {}).get("name", opp.sell_exchange)
                route = f"{buy_name} -> {sell_name}"
                fees = opp.buy_fee_pct + opp.sell_fee_pct

                # Color based on profitability
                if opp.net_profit_pct >= 1.0:
                    color = "green"
                    emoji = "***"
                elif opp.net_profit_pct >= 0.5:
                    color = "cyan"
                    emoji = "**"
                else:
                    color = "white"
                    emoji = "*"

                print(colored(
                    f"{emoji} {route:<27} {opp.gross_spread_pct:>9.3f}% {fees:>9.2f}% {opp.net_profit_pct:>9.3f}% {opp.available_volume:>9.0f} ${opp.potential_profit_usd:>10.2f}",
                    color
                ))

            # Summary
            self.total_opportunities += len(profitable_opps)
            session_profit = sum(o.potential_profit_usd for o in profitable_opps)
            self.total_theoretical_profit += session_profit

        # Session stats
        print("\n" + colored(" SESSION STATS", "yellow", attrs=['bold']))
        print("-" * 80)
        runtime = datetime.now() - self.start_time
        self.checks_performed += 1
        print(f"  Runtime: {runtime}  |  Checks: {self.checks_performed}")
        print(f"  Total Opportunities: {self.total_opportunities}  |  Theoretical Profit: ${self.total_theoretical_profit:.2f}")

        # Paper trading status
        if self.paper_mode and self.paper_state:
            print("\n" + colored(" PAPER TRADING STATUS", "green", attrs=['bold']))
            print("-" * 80)
            ps = self.paper_state
            pnl_color = "green" if ps.total_pnl >= 0 else "red"
            pnl_pct = (ps.total_pnl / ps.starting_balance) * 100
            print(f"  Balance: ${ps.current_balance:.2f} (started: ${ps.starting_balance:.2f})")
            print(f"  Total P&L: {colored(f'${ps.total_pnl:+.2f} ({pnl_pct:+.2f}%)', pnl_color)}")
            print(f"  Trades: {ps.total_trades} (W: {ps.winning_trades} / L: {ps.losing_trades})")
            print(f"  Open Positions: {len(ps.positions)}")
            if ps.positions:
                for pos in ps.positions:
                    print(f"    - {pos.id}: {pos.buy_exchange}->{pos.sell_exchange} | {pos.amount:.1f} MERL @ {pos.expected_profit_pct:.2f}% spread")
            print(f"  Fees Paid: ${ps.total_fees_paid:.2f}")

        print("\n" + "=" * 80)

    # =========================================================================
    # PAPER TRADING METHODS
    # =========================================================================

    def paper_open_position(self, opportunity: ArbOpportunity, prices: Dict[str, ExchangePrice]) -> Optional[PaperPosition]:
        """Open a paper trading position"""
        if not self.paper_mode or not self.paper_state:
            return None

        # Check if we have enough balance
        if self.paper_state.current_balance < self.trade_size_usd:
            cprint(f"  Insufficient paper balance: ${self.paper_state.current_balance:.2f}", "yellow")
            return None

        # Check if we already have a position in this pair
        for pos in self.paper_state.positions:
            if pos.buy_exchange == opportunity.buy_exchange and pos.sell_exchange == opportunity.sell_exchange:
                return None  # Already have this position

        # Calculate position size
        amount = self.trade_size_usd / opportunity.buy_price

        # Calculate fees
        buy_fee = (self.trade_size_usd * opportunity.buy_fee_pct / 100)
        sell_fee = (self.trade_size_usd * opportunity.sell_fee_pct / 100)
        total_fees = buy_fee + sell_fee

        # Create position
        self.position_counter += 1
        position = PaperPosition(
            id=f"P{self.position_counter:04d}",
            open_time=datetime.now(),
            buy_exchange=opportunity.buy_exchange,
            sell_exchange=opportunity.sell_exchange,
            buy_price=opportunity.buy_price,
            sell_price=opportunity.sell_price,
            amount=amount,
            size_usd=self.trade_size_usd,
            expected_profit_pct=opportunity.net_profit_pct,
        )

        # Deduct from balance (we're buying the asset)
        self.paper_state.current_balance -= self.trade_size_usd
        self.paper_state.total_fees_paid += total_fees
        self.paper_state.positions.append(position)

        # Log the trade
        self._log_trade('OPEN', position, total_fees, 0)

        cprint(f"\n  PAPER TRADE OPENED: {position.id}", "green", attrs=['bold'])
        cprint(f"  Buy {amount:.2f} MERL on {opportunity.buy_exchange} @ ${opportunity.buy_price:.5f}", "green")
        cprint(f"  Sell on {opportunity.sell_exchange} @ ${opportunity.sell_price:.5f}", "red")
        cprint(f"  Expected profit: {opportunity.net_profit_pct:.3f}% (${opportunity.net_profit_pct/100 * self.trade_size_usd:.2f})", "cyan")
        cprint(f"  Fees: ${total_fees:.2f}", "yellow")

        return position

    def paper_close_position(self, position: PaperPosition, prices: Dict[str, ExchangePrice], reason: str = "manual") -> float:
        """Close a paper trading position and calculate P&L"""
        if not self.paper_mode or not self.paper_state:
            return 0.0

        # Get current prices
        buy_price_data = prices.get(position.buy_exchange)
        sell_price_data = prices.get(position.sell_exchange)

        if not buy_price_data or not sell_price_data:
            cprint(f"  Cannot close {position.id}: missing price data", "yellow")
            return 0.0

        # To close: sell on buy_exchange (where we bought), buy on sell_exchange (where we sold short)
        # This reverses our position
        close_buy_price = sell_price_data.ask  # Buy back on sell exchange
        close_sell_price = buy_price_data.bid  # Sell on buy exchange

        # Calculate P&L
        # Original: bought at buy_price, sold at sell_price
        # Close: sell at close_sell_price, buy at close_buy_price
        # Profit from original spread: (sell_price - buy_price) * amount
        # Cost of closing: (close_buy_price - close_sell_price) * amount (usually negative if spread narrowed)

        original_profit = (position.sell_price - position.buy_price) * position.amount

        # For simplicity in paper trading, we assume we capture the original spread
        # minus any spread change and fees
        current_spread = (sell_price_data.bid - buy_price_data.ask)
        original_spread = (position.sell_price - position.buy_price)
        spread_change = current_spread - original_spread

        # Calculate fees for closing
        buy_ex_base = position.sell_exchange.replace("_perp", "")
        sell_ex_base = position.buy_exchange.replace("_perp", "")
        close_buy_fee = EXCHANGE_FEES.get(buy_ex_base, {"taker": 0.20})["taker"]
        close_sell_fee = EXCHANGE_FEES.get(sell_ex_base, {"taker": 0.20})["taker"]
        close_fees = (close_buy_fee + close_sell_fee) / 100 * position.size_usd

        # Net P&L = original spread profit + spread change adjustment - close fees
        realized_pnl = original_profit + (spread_change * position.amount) - close_fees

        # Update position
        position.status = "closed"
        position.close_time = datetime.now()
        position.close_buy_price = close_buy_price
        position.close_sell_price = close_sell_price
        position.realized_pnl = realized_pnl

        # Update state
        self.paper_state.current_balance += position.size_usd + realized_pnl
        self.paper_state.total_pnl += realized_pnl
        self.paper_state.total_fees_paid += close_fees
        self.paper_state.total_trades += 1

        if realized_pnl >= 0:
            self.paper_state.winning_trades += 1
        else:
            self.paper_state.losing_trades += 1

        # Move to closed positions
        self.paper_state.positions.remove(position)
        self.paper_state.closed_positions.append(position)

        # Log the trade
        self._log_trade('CLOSE', position, close_fees, realized_pnl)

        pnl_color = "green" if realized_pnl >= 0 else "red"
        cprint(f"\n  PAPER TRADE CLOSED: {position.id} ({reason})", pnl_color, attrs=['bold'])
        cprint(f"  P&L: {colored(f'${realized_pnl:+.2f}', pnl_color)}", "white")
        cprint(f"  Duration: {position.close_time - position.open_time}", "white")

        return realized_pnl

    def paper_check_close_conditions(self, prices: Dict[str, ExchangePrice]) -> List[Tuple[PaperPosition, str]]:
        """Check which positions should be closed"""
        to_close = []

        if not self.paper_mode or not self.paper_state:
            return to_close

        for position in self.paper_state.positions:
            buy_price_data = prices.get(position.buy_exchange)
            sell_price_data = prices.get(position.sell_exchange)

            if not buy_price_data or not sell_price_data:
                continue

            # Calculate current spread
            current_spread_pct = ((sell_price_data.bid - buy_price_data.ask) / buy_price_data.ask) * 100

            # Close if spread has narrowed below threshold (we've captured the arb)
            if current_spread_pct < self.auto_close_threshold:
                to_close.append((position, f"spread_narrowed ({current_spread_pct:.2f}%)"))

            # Close if position is old (> 1 hour) to prevent holding too long
            age = datetime.now() - position.open_time
            if age.total_seconds() > 3600:  # 1 hour
                to_close.append((position, f"timeout ({age})"))

        return to_close

    def _log_trade(self, action: str, position: PaperPosition, fees: float, pnl: float):
        """Log a trade to CSV file"""
        if not self.trade_log_file:
            return

        spread_pct = ((position.sell_price - position.buy_price) / position.buy_price) * 100

        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                action,
                position.id,
                position.buy_exchange,
                position.sell_exchange,
                f"{position.buy_price:.6f}",
                f"{position.sell_price:.6f}",
                f"{position.amount:.4f}",
                f"{position.size_usd:.2f}",
                f"{spread_pct:.4f}",
                f"{fees:.4f}",
                f"{pnl:.4f}",
                f"{self.paper_state.current_balance:.2f}" if self.paper_state else "0"
            ])

    def execute_arb(self, opportunity: ArbOpportunity, size_usd: float = DEFAULT_TRADE_SIZE_USD) -> bool:
        """Execute an arbitrage trade (DANGEROUS - only in execute mode)"""
        if not self.execute_mode:
            cprint("  Execution mode not enabled!", "red")
            return False

        buy_ex = opportunity.buy_exchange
        sell_ex = opportunity.sell_exchange

        # Check we have both exchanges
        if buy_ex not in self.exchanges or sell_ex not in self.exchanges:
            cprint(f"  Missing exchange connection!", "red")
            return False

        # Calculate amount to trade
        amount = size_usd / opportunity.buy_price

        # Check against available volume
        if amount > opportunity.available_volume * 0.5:  # Don't take more than 50% of available
            amount = opportunity.available_volume * 0.5
            cprint(f"  Reduced amount to {amount:.2f} MERL (volume limit)", "yellow")

        try:
            cprint(f"\n  Executing arbitrage: {buy_ex} -> {sell_ex}", "cyan")
            cprint(f"  Amount: {amount:.2f} MERL (${size_usd:.2f})", "white")
            cprint(f"  Expected profit: {opportunity.net_profit_pct:.3f}%", "white")

            # Place buy order
            buy_exchange = self.exchanges[buy_ex]
            buy_symbol = self.symbols[buy_ex]
            cprint(f"  Placing BUY on {buy_ex}...", "green")
            # buy_order = buy_exchange.create_market_buy_order(buy_symbol, amount)

            # Place sell order
            sell_exchange = self.exchanges[sell_ex]
            sell_symbol = self.symbols[sell_ex]
            cprint(f"  Placing SELL on {sell_ex}...", "red")
            # sell_order = sell_exchange.create_market_sell_order(sell_symbol, amount)

            cprint(f"  SIMULATED - Orders would be placed here", "yellow")
            return True

        except Exception as e:
            cprint(f"  Execution failed: {e}", "red")
            return False

    def run(self):
        """Main bot loop"""
        cprint("\nStarting MERL Arbitrage Bot...\n", "green")
        cprint(f"Monitoring {len(self.exchanges)} exchanges", "white")
        cprint(f"Min profit threshold: {self.min_profit_threshold}%", "white")
        cprint(f"Check interval: {self.check_interval}s", "white")
        if self.paper_mode:
            cprint(f"Paper trading: ENABLED (${self.trade_size_usd} per trade)", "green")
            cprint(f"Auto-close threshold: {self.auto_close_threshold}%", "white")
        cprint("\nPress Ctrl+C to stop\n", "yellow")

        while True:
            try:
                # Fetch all prices
                prices = self.fetch_all_prices()

                if len(prices) < 2:
                    cprint("Not enough exchanges responding, waiting...", "yellow")
                    time.sleep(self.check_interval)
                    continue

                # Find opportunities
                opportunities = self.find_opportunities(prices)

                # Paper trading: check if we should close any positions
                if self.paper_mode:
                    positions_to_close = self.paper_check_close_conditions(prices)
                    for position, reason in positions_to_close:
                        self.paper_close_position(position, prices, reason)

                # Display dashboard
                self.display_dashboard(prices, opportunities)

                # Paper trading: open new positions for profitable opportunities
                if self.paper_mode:
                    profitable = [o for o in opportunities if o.net_profit_pct >= self.min_profit_threshold]
                    if profitable and self.paper_state and len(self.paper_state.positions) < 3:  # Max 3 concurrent positions
                        best = profitable[0]
                        # Only open if spread is significant enough
                        if best.net_profit_pct >= 5.0:  # At least 5% spread for paper trading
                            self.paper_open_position(best, prices)

                # Execute if profitable and in execute mode
                elif self.execute_mode:
                    profitable = [o for o in opportunities if o.net_profit_pct >= self.min_profit_threshold]
                    if profitable:
                        best = profitable[0]
                        cprint(f"\n  PROFITABLE OPPORTUNITY DETECTED: {best.net_profit_pct:.3f}%", "green", attrs=['bold'])
                        # Uncomment to enable execution:
                        # self.execute_arb(best)

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                cprint("\n\nShutting down MERL Arbitrage Bot...", "yellow")
                self._print_final_stats()
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                import traceback
                traceback.print_exc()
                time.sleep(self.check_interval)

    def _print_final_stats(self):
        """Print final session statistics"""
        runtime = datetime.now() - self.start_time
        print("\n" + "=" * 60)
        print(colored("FINAL SESSION STATISTICS", "cyan", attrs=['bold']))
        print("=" * 60)
        print(f"  Total Runtime: {runtime}")
        print(f"  Total Checks: {self.checks_performed}")
        print(f"  Opportunities Found: {self.total_opportunities}")
        print(f"  Theoretical Profit: ${self.total_theoretical_profit:.2f}")
        if self.checks_performed > 0:
            avg_per_check = self.total_opportunities / self.checks_performed
            print(f"  Avg Opportunities/Check: {avg_per_check:.2f}")

        # Paper trading final stats
        if self.paper_mode and self.paper_state:
            print("\n" + "-" * 60)
            print(colored("PAPER TRADING RESULTS", "green", attrs=['bold']))
            print("-" * 60)
            ps = self.paper_state
            pnl_color = "green" if ps.total_pnl >= 0 else "red"
            pnl_pct = (ps.total_pnl / ps.starting_balance) * 100

            print(f"  Starting Balance: ${ps.starting_balance:.2f}")
            print(f"  Final Balance: ${ps.current_balance:.2f}")
            print(f"  Total P&L: {colored(f'${ps.total_pnl:+.2f} ({pnl_pct:+.2f}%)', pnl_color)}")
            print(f"  Total Trades: {ps.total_trades}")
            print(f"  Win Rate: {ps.winning_trades}/{ps.total_trades} ({ps.winning_trades/max(ps.total_trades,1)*100:.1f}%)")
            print(f"  Total Fees: ${ps.total_fees_paid:.2f}")

            if ps.positions:
                print(f"\n  Open Positions (unrealized):")
                for pos in ps.positions:
                    print(f"    - {pos.id}: {pos.amount:.2f} MERL ({pos.buy_exchange} -> {pos.sell_exchange})")

            if self.trade_log_file:
                print(f"\n  Trade log saved to: {self.trade_log_file}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Moon Dev's MERL Arbitrage Bot")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable live execution mode (DANGEROUS)"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode (simulated trades)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MIN_PROFIT_THRESHOLD,
        help=f"Minimum profit threshold in percent (default: {MIN_PROFIT_THRESHOLD})"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {CHECK_INTERVAL})"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=DEFAULT_TRADE_SIZE_USD,
        help=f"Trade size in USD for paper/live trading (default: {DEFAULT_TRADE_SIZE_USD})"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Starting balance for paper trading (default: 1000)"
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=2.0,
        help="Auto-close positions when spread drops below this percent (default: 2.0)"
    )

    args = parser.parse_args()

    # Create bot with custom settings
    bot = MERLArbBot(
        execute_mode=args.execute,
        paper_mode=args.paper,
        min_profit_threshold=args.threshold,
        check_interval=args.interval,
        trade_size_usd=args.size,
        starting_balance=args.balance,
        auto_close_threshold=args.close_threshold,
    )
    bot.run()


if __name__ == "__main__":
    main()
