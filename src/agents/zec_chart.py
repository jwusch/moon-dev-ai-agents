"""
🌙 Moon Dev's ZEC Chart
Terminal-based candlestick chart for Zcash (ZEC)
Similar to TradingView style

Usage:
    python src/agents/zec_chart.py
"""

import ccxt
import time
from datetime import datetime
import plotext as plt
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SYMBOL = "ZEC/USDT:USDT"  # ZEC perpetual
EXCHANGE = "okx"
TIMEFRAME = "5m"  # 5 minute candles
CANDLE_COUNT = 50  # Number of candles to show
REFRESH_INTERVAL = 30  # Seconds between refreshes

class ZECChart:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })

        print(f"🌙 Moon Dev's ZEC Chart")
        print(f"📊 Loading {exchange_id.upper()} markets...")

        self.exchange.load_markets()

        # Find ZEC perpetual symbol
        self.symbol = self._find_zec_perp()
        if not self.symbol:
            print(f"❌ No ZEC perpetual found on {exchange_id}")
            return

        print(f"💎 Symbol: {self.symbol}")
        print(f"⏱️  Timeframe: {TIMEFRAME}")
        print(f"📈 Candles: {CANDLE_COUNT}")

    def _find_zec_perp(self):
        """Find the ZEC perpetual futures symbol"""
        for symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            if 'ZEC' in symbol and market.get('swap', False):
                return symbol
        candidates = ["ZEC/USDT:USDT", "ZEC-USDT-SWAP", "ZECUSDT"]
        for c in candidates:
            if c in self.exchange.markets:
                return c
        return None

    def fetch_ohlcv(self):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=CANDLE_COUNT)
            return ohlcv
        except Exception as e:
            print(f"❌ Error fetching OHLCV: {e}")
            return None

    def display_chart(self):
        """Display the candlestick chart"""
        ohlcv = self.fetch_ohlcv()
        if not ohlcv:
            return

        # Extract data
        timestamps = [candle[0] for candle in ohlcv]
        opens = [candle[1] for candle in ohlcv]
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        closes = [candle[4] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]

        # Current price info
        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else closes[-1]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100 if prev_close > 0 else 0

        high_24h = max(highs)
        low_24h = min(lows)

        # Clear terminal
        plt.clear_terminal()

        # Create candlestick chart
        plt.clear_figure()
        plt.theme('dark')
        plt.title(f"🌙 ZEC/USDT - ${current_price:,.2f} ({pct_change:+.2f}%)")

        # Plot candlesticks
        plt.candlestick(timestamps, {"Open": opens, "Close": closes, "High": highs, "Low": lows})

        # Configure axes
        plt.xlabel("Time")
        plt.ylabel("Price ($)")

        # Set date labels
        plt.date_form('H:M')

        # Plot size
        plt.plotsize(100, 25)

        # Show the chart
        plt.show()

        # Print stats below chart
        print("─" * 80)
        print(f"💰 Current: ${current_price:,.2f}  |  📈 High: ${high_24h:,.2f}  |  📉 Low: ${low_24h:,.2f}")
        print(f"📊 Timeframe: {TIMEFRAME}  |  🕐 Updated: {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 80)

    def display_simple_chart(self):
        """Display a simple line chart with volume"""
        ohlcv = self.fetch_ohlcv()
        if not ohlcv:
            return

        # Extract data
        closes = [candle[4] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]
        times = list(range(len(closes)))

        # Current price info
        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else closes[-1]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100 if prev_close > 0 else 0

        high_price = max(closes)
        low_price = min(closes)

        # Clear terminal
        plt.clear_terminal()

        # Create subplot layout
        plt.clear_figure()
        plt.theme('dark')

        # Price chart
        plt.subplots(2, 1)

        plt.subplot(1, 1)
        plt.title(f"🌙 ZEC/USDT - ${current_price:,.2f} ({pct_change:+.2f}%)")

        # Color based on trend
        if pct_change >= 0:
            plt.plot(times, closes, marker='braille', color='green')
        else:
            plt.plot(times, closes, marker='braille', color='red')

        # Add horizontal lines for high/low
        plt.hline(high_price, color='cyan')
        plt.hline(low_price, color='magenta')

        plt.ylabel("Price ($)")
        plt.plotsize(100, 18)

        # Volume chart
        plt.subplot(2, 1)
        plt.bar(times, volumes, color='blue')
        plt.ylabel("Volume")
        plt.xlabel(f"Last {CANDLE_COUNT} candles ({TIMEFRAME})")
        plt.plotsize(100, 6)

        plt.show()

        # Print stats
        print("─" * 80)
        print(f"💰 Price: ${current_price:,.2f}  |  📈 High: ${high_price:,.2f}  |  📉 Low: ${low_price:,.2f}")
        print(f"📊 {TIMEFRAME} candles  |  🕐 {datetime.now().strftime('%H:%M:%S')}  |  Press Ctrl+C to exit")
        print("─" * 80)

    def run(self, chart_type='candle'):
        """Run the chart display"""
        print(f"\n✅ Starting ZEC chart (type: {chart_type})...\n")
        time.sleep(1)

        while True:
            try:
                if chart_type == 'candle':
                    self.display_chart()
                else:
                    self.display_simple_chart()

                time.sleep(REFRESH_INTERVAL)

            except KeyboardInterrupt:
                print("\n👋 Chart closed.")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(5)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ZEC Chart')
    parser.add_argument('--type', '-t', choices=['candle', 'line'], default='candle',
                        help='Chart type: candle or line')
    parser.add_argument('--timeframe', '-tf', default='5m',
                        help='Timeframe: 1m, 5m, 15m, 1h, 4h, 1d')
    args = parser.parse_args()

    global TIMEFRAME
    TIMEFRAME = args.timeframe

    chart = ZECChart()
    if chart.symbol:
        chart.run(chart_type=args.type)

if __name__ == "__main__":
    main()
