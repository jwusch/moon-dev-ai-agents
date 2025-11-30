"""
Moon Dev's MERL Swarm Agent
Orchestrates multiple agents to monitor MERL and uses AI to analyze combined data

Features:
- Trade watcher data
- Liquidation/derivatives data
- Multi-exchange orderbook data
- AI analysis combining all signals

Usage:
    python src/agents/merl_swarm_agent.py
"""

import ccxt
import time
from datetime import datetime
from termcolor import colored, cprint
import sys
import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Try to import model factory
try:
    from src.models.model_factory import ModelFactory
    HAS_MODEL_FACTORY = True
except:
    HAS_MODEL_FACTORY = False

# Configuration
CHECK_INTERVAL = 30  # Main loop interval
WHALE_THRESHOLD_USD = 5000
LEVERAGE_LEVELS = [3, 5, 10, 20, 25]

# Exchanges for MERL
SPOT_EXCHANGES = {
    "okx": "MERL/USDT",
    "kraken": "MERL/USD",
    "kucoin": "MERL/USDT",
    "gate": "MERL/USDT",
}

PERP_EXCHANGE = "okx"
PERP_SYMBOL = "MERL/USDT:USDT"

class MERLSwarmAgent:
    def __init__(self):
        cprint(f"\n{'='*70}", "cyan")
        cprint(f"Moon Dev's MERL Swarm Agent", "cyan", attrs=['bold'])
        cprint(f"Multi-Agent Orchestration with AI Analysis", "cyan")
        cprint(f"{'='*70}", "cyan")

        # Initialize exchanges
        self.exchanges = {}
        self.spot_symbols = {}

        cprint("\nInitializing exchanges...", "white")
        for ex_id, symbol in SPOT_EXCHANGES.items():
            try:
                exchange = getattr(ccxt, ex_id)({'enableRateLimit': True})
                exchange.load_markets()
                if symbol in exchange.markets:
                    self.exchanges[ex_id] = exchange
                    self.spot_symbols[ex_id] = symbol
                    cprint(f"  {ex_id.upper()}: {symbol}", "green")
            except Exception as e:
                cprint(f"  {ex_id.upper()}: Failed", "red")

        # Initialize perp exchange
        try:
            self.perp_exchange = getattr(ccxt, PERP_EXCHANGE)({'enableRateLimit': True})
            self.perp_exchange.load_markets()
            if PERP_SYMBOL in self.perp_exchange.markets:
                cprint(f"  {PERP_EXCHANGE.upper()} Perp: {PERP_SYMBOL}", "green")
                self.has_perp = True
            else:
                self.has_perp = False
        except:
            self.has_perp = False

        # Initialize AI model
        self.model = None
        if HAS_MODEL_FACTORY:
            try:
                factory = ModelFactory()
                # Try different models in order of preference
                for model_type in ['claude', 'deepseek', 'openai', 'groq', 'xai']:
                    try:
                        model = factory.get_model(model_type)
                        if model:
                            self.model = model
                            cprint(f"\n  AI Model: {model_type.upper()} ({model.model_name})", "green")
                            break
                    except:
                        continue
            except Exception as e:
                cprint(f"\n  AI Model init error: {e}", "yellow")

        if not self.model:
            cprint("\n  AI Model: Not available (will show data only)", "yellow")

        # Data storage
        self.trade_history = []
        self.last_analysis = None
        self.signal_history = []

        cprint(f"\nSwarm initialized with {len(self.exchanges)} exchanges", "green")

    def fetch_spot_prices(self):
        """Fetch spot prices from all exchanges"""
        prices = {}
        for ex_id, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(self.spot_symbols[ex_id])
                prices[ex_id] = {
                    'price': ticker['last'],
                    'change_24h': ticker.get('percentage', 0) or 0,
                    'volume': ticker.get('quoteVolume') or ticker.get('baseVolume') or 0,
                    'high': ticker.get('high') or 0,
                    'low': ticker.get('low') or 0,
                }
            except:
                pass
        return prices

    def fetch_perp_data(self):
        """Fetch perpetual futures data"""
        if not self.has_perp:
            return None

        try:
            ticker = self.perp_exchange.fetch_ticker(PERP_SYMBOL)

            # Funding rate
            funding = None
            try:
                if hasattr(self.perp_exchange, 'fetch_funding_rate'):
                    funding = self.perp_exchange.fetch_funding_rate(PERP_SYMBOL)
            except:
                pass

            # Open interest
            oi = None
            try:
                if hasattr(self.perp_exchange, 'fetch_open_interest'):
                    oi = self.perp_exchange.fetch_open_interest(PERP_SYMBOL)
            except:
                pass

            return {
                'price': ticker['last'],
                'funding_rate': funding.get('fundingRate', 0) * 100 if funding else 0,
                'open_interest': oi.get('openInterestValue', 0) if oi else 0,
                'volume': ticker.get('quoteVolume') or 0,
            }
        except:
            return None

    def fetch_orderbooks(self):
        """Fetch orderbook data from all exchanges"""
        orderbooks = {}
        for ex_id, exchange in self.exchanges.items():
            try:
                ob = exchange.fetch_order_book(self.spot_symbols[ex_id], limit=20)

                bids = ob.get('bids', [])
                asks = ob.get('asks', [])

                bid_value = sum(b[0] * b[1] for b in bids)
                ask_value = sum(a[0] * a[1] for a in asks)

                whale_bids = [b for b in bids if b[0] * b[1] >= WHALE_THRESHOLD_USD]
                whale_asks = [a for a in asks if a[0] * a[1] >= WHALE_THRESHOLD_USD]

                orderbooks[ex_id] = {
                    'bid_value': bid_value,
                    'ask_value': ask_value,
                    'spread': (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
                    'whale_bids': len(whale_bids),
                    'whale_asks': len(whale_asks),
                    'imbalance': (bid_value - ask_value) / (bid_value + ask_value) * 100 if (bid_value + ask_value) > 0 else 0,
                }
            except:
                pass
        return orderbooks

    def fetch_recent_trades(self):
        """Fetch recent trades from primary exchange"""
        try:
            if 'okx' in self.exchanges:
                trades = self.exchanges['okx'].fetch_trades(self.spot_symbols['okx'], limit=100)

                buy_volume = sum(t['amount'] for t in trades if t['side'] == 'buy')
                sell_volume = sum(t['amount'] for t in trades if t['side'] == 'sell')
                buy_value = sum(t['cost'] for t in trades if t['side'] == 'buy')
                sell_value = sum(t['cost'] for t in trades if t['side'] == 'sell')

                return {
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_value': buy_value,
                    'sell_value': sell_value,
                    'trade_count': len(trades),
                    'buy_ratio': buy_value / (buy_value + sell_value) * 100 if (buy_value + sell_value) > 0 else 50,
                }
        except:
            pass
        return None

    def calculate_liquidation_levels(self, price):
        """Calculate liquidation levels"""
        levels = {'long': {}, 'short': {}}
        for lev in LEVERAGE_LEVELS:
            maint = 0.005
            levels['long'][lev] = price * (1 - (1/lev) + maint)
            levels['short'][lev] = price * (1 + (1/lev) - maint)
        return levels

    def generate_ai_analysis(self, market_data):
        """Generate AI analysis of combined market data"""
        if not self.model:
            return None

        prompt = f"""Analyze this MERL (Merlin Chain) market data and provide a trading signal.

SPOT PRICES:
{self._format_prices(market_data['spot_prices'])}

DERIVATIVES:
{self._format_perp(market_data['perp_data'])}

ORDERBOOK ANALYSIS:
{self._format_orderbooks(market_data['orderbooks'])}

RECENT TRADES:
{self._format_trades(market_data['trades'])}

Based on this data, provide:
1. SIGNAL: BULLISH, BEARISH, or NEUTRAL
2. CONFIDENCE: 1-10
3. KEY_FACTORS: 3 bullet points
4. RISK_LEVEL: LOW, MEDIUM, HIGH
5. RECOMMENDATION: One sentence

Keep response concise and actionable."""

        try:
            response = self.model.generate_response(
                system_prompt="You are a crypto trading analyst. Provide concise, actionable analysis.",
                user_content=prompt,
                temperature=0.3,
                max_tokens=500
            )
            # Extract text content from response
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            return f"AI analysis error: {e}"

    def _format_prices(self, prices):
        if not prices:
            return "No data"
        lines = []
        for ex, data in prices.items():
            lines.append(f"- {ex.upper()}: ${data['price']:.4f} ({data['change_24h']:+.1f}% 24h)")
        return "\n".join(lines)

    def _format_perp(self, perp):
        if not perp:
            return "No perpetual data"
        return f"""- Price: ${perp['price']:.4f}
- Funding Rate: {perp['funding_rate']:.4f}%
- Open Interest: ${perp['open_interest']:,.0f}"""

    def _format_orderbooks(self, obs):
        if not obs:
            return "No data"
        total_bid = sum(o['bid_value'] for o in obs.values())
        total_ask = sum(o['ask_value'] for o in obs.values())
        avg_imbalance = sum(o['imbalance'] for o in obs.values()) / len(obs) if obs else 0
        return f"""- Total Bid Value: ${total_bid:,.0f}
- Total Ask Value: ${total_ask:,.0f}
- Avg Imbalance: {avg_imbalance:+.1f}%
- Whale Orders: {sum(o['whale_bids'] for o in obs.values())} bids, {sum(o['whale_asks'] for o in obs.values())} asks"""

    def _format_trades(self, trades):
        if not trades:
            return "No data"
        return f"""- Buy Ratio: {trades['buy_ratio']:.1f}%
- Buy Value: ${trades['buy_value']:,.0f}
- Sell Value: ${trades['sell_value']:,.0f}"""

    def display_dashboard(self, market_data, ai_analysis=None):
        """Display the combined dashboard"""
        print("\n" + "=" * 80)
        print(colored("MERL SWARM AGENT - COMBINED ANALYSIS", "cyan", attrs=['bold']))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Spot prices
        print("\n" + colored("SPOT PRICES", "yellow", attrs=['bold']))
        print("-" * 40)
        if market_data['spot_prices']:
            for ex, data in market_data['spot_prices'].items():
                change_24h = data['change_24h']
                change_color = "green" if change_24h >= 0 else "red"
                change_str = colored(f"{change_24h:+.1f}%", change_color)
                print(f"  {ex.upper():<10} ${data['price']:<10.4f} {change_str}")

        # Derivatives
        if market_data['perp_data']:
            print("\n" + colored("DERIVATIVES (OKX Perp)", "yellow", attrs=['bold']))
            print("-" * 40)
            perp = market_data['perp_data']
            funding_rate = perp['funding_rate']
            fund_color = "green" if funding_rate < 0 else "red"
            fund_str = colored(f"{funding_rate:.4f}%", fund_color)
            print(f"  Funding Rate: {fund_str}")
            print(f"  Open Interest: ${perp['open_interest']:,.0f}")

            if funding_rate < -0.5:
                print(colored("  >> SHORTS PAYING LONGS - Bullish signal!", "green"))
            elif funding_rate > 0.5:
                print(colored("  >> LONGS PAYING SHORTS - Bearish signal!", "red"))

        # Orderbook summary
        print("\n" + colored("ORDERBOOK PRESSURE", "yellow", attrs=['bold']))
        print("-" * 40)
        if market_data['orderbooks']:
            total_bid = sum(o['bid_value'] for o in market_data['orderbooks'].values())
            total_ask = sum(o['ask_value'] for o in market_data['orderbooks'].values())
            pressure = total_bid / (total_bid + total_ask) * 100 if (total_bid + total_ask) > 0 else 50

            bar_len = 30
            buy_bars = int(pressure / 100 * bar_len)
            print(f"  {colored('█' * buy_bars, 'green')}{colored('█' * (bar_len - buy_bars), 'red')}")
            print(f"  {pressure:.1f}% BUY | ${total_bid:,.0f} vs ${total_ask:,.0f} | {100-pressure:.1f}% SELL")

        # Trade flow
        if market_data['trades']:
            print("\n" + colored("RECENT TRADE FLOW", "yellow", attrs=['bold']))
            print("-" * 40)
            trades = market_data['trades']
            buy_ratio = trades['buy_ratio']
            buy_color = "green" if buy_ratio > 50 else "red"
            buy_ratio_str = colored(f"{buy_ratio:.1f}%", buy_color)
            print(f"  Buy Ratio: {buy_ratio_str}")
            print(f"  Volume: ${trades['buy_value'] + trades['sell_value']:,.0f}")

        # Liquidation levels
        if market_data['spot_prices']:
            avg_price = sum(p['price'] for p in market_data['spot_prices'].values()) / len(market_data['spot_prices'])
            levels = self.calculate_liquidation_levels(avg_price)

            print("\n" + colored("LIQUIDATION LEVELS (from avg price)", "yellow", attrs=['bold']))
            print("-" * 40)
            print(colored("  LONG liquidations (price drops to):", "green"))
            for lev in [5, 10, 20]:
                dist = ((avg_price - levels['long'][lev]) / avg_price) * 100
                print(f"    {lev}x: ${levels['long'][lev]:.4f} (-{dist:.1f}%)")
            print(colored("  SHORT liquidations (price rises to):", "red"))
            for lev in [5, 10, 20]:
                dist = ((levels['short'][lev] - avg_price) / avg_price) * 100
                print(f"    {lev}x: ${levels['short'][lev]:.4f} (+{dist:.1f}%)")

        # AI Analysis
        print("\n" + colored("AI ANALYSIS", "cyan", attrs=['bold']))
        print("-" * 40)
        if ai_analysis:
            print(ai_analysis)
        else:
            print("  AI model not available")

        print("\n" + "=" * 80)

    def run(self):
        """Main orchestration loop"""
        cprint("\nStarting MERL Swarm Agent...\n", "green")

        iteration = 0
        while True:
            try:
                # Gather data from all agents
                cprint("Gathering market data...", "cyan")

                market_data = {
                    'spot_prices': self.fetch_spot_prices(),
                    'perp_data': self.fetch_perp_data(),
                    'orderbooks': self.fetch_orderbooks(),
                    'trades': self.fetch_recent_trades(),
                    'timestamp': datetime.now(),
                }

                # Generate AI analysis every 3rd iteration (to save API calls)
                ai_analysis = None
                if self.model and iteration % 3 == 0:
                    cprint("Generating AI analysis...", "cyan")
                    ai_analysis = self.generate_ai_analysis(market_data)
                    self.last_analysis = ai_analysis
                else:
                    ai_analysis = self.last_analysis

                # Display combined dashboard
                self.display_dashboard(market_data, ai_analysis)

                iteration += 1
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                cprint("\nMERL Swarm Agent shutting down...", "yellow")
                break
            except Exception as e:
                cprint(f"Error: {e}", "red")
                time.sleep(5)

def main():
    agent = MERLSwarmAgent()
    agent.run()

if __name__ == "__main__":
    main()
