"""
Moon Dev's MERL Chart
Terminal-based chart for Merlin Chain (MERL)

Usage:
    python src/agents/merl_chart.py
    python src/agents/merl_chart.py --type line
"""

import ccxt
import time
from datetime import datetime
import plotext as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "MERL/USDT"
EXCHANGE = "okx"
TIMEFRAME = "5m"
CANDLE_COUNT = 50
REFRESH_INTERVAL = 30

class MERLChart:
    def __init__(self, exchange_id=EXCHANGE):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})

        print(f"Moon Dev's MERL Chart")
        print(f"Loading {exchange_id.upper()} markets...")

        self.exchange.load_markets()
        self.symbol = SYMBOL if SYMBOL in self.exchange.markets else None

        if not self.symbol:
            print(f"MERL not found on {exchange_id}")
            return

        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {TIMEFRAME}")

    def fetch_ohlcv(self):
        try:
            return self.exchange.fetch_ohlcv(self.symbol, TIMEFRAME, limit=CANDLE_COUNT)
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return None

    def display_chart(self):
        ohlcv = self.fetch_ohlcv()
        if not ohlcv:
            return

        timestamps = [candle[0] for candle in ohlcv]
        opens = [candle[1] for candle in ohlcv]
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        closes = [candle[4] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]

        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else closes[-1]
        pct_change = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

        plt.clear_terminal()
        plt.clear_figure()
        plt.theme('dark')
        plt.title(f"MERL/USDT - ${current_price:.4f} ({pct_change:+.2f}%)")
        plt.candlestick(timestamps, {"Open": opens, "Close": closes, "High": highs, "Low": lows})
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.date_form('H:M')
        plt.plotsize(100, 25)
        plt.show()

        print("-" * 80)
        print(f"Current: ${current_price:.4f}  |  High: ${max(highs):.4f}  |  Low: ${min(lows):.4f}")
        print(f"Timeframe: {TIMEFRAME}  |  Updated: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80)

    def display_simple_chart(self):
        ohlcv = self.fetch_ohlcv()
        if not ohlcv:
            return

        closes = [candle[4] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]
        times = list(range(len(closes)))

        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else closes[-1]
        pct_change = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0

        plt.clear_terminal()
        plt.clear_figure()
        plt.theme('dark')
        plt.subplots(2, 1)

        plt.subplot(1, 1)
        plt.title(f"MERL/USDT - ${current_price:.4f} ({pct_change:+.2f}%)")
        color = 'green' if pct_change >= 0 else 'red'
        plt.plot(times, closes, marker='braille', color=color)
        plt.hline(max(closes), color='cyan')
        plt.hline(min(closes), color='magenta')
        plt.ylabel("Price ($)")
        plt.plotsize(100, 18)

        plt.subplot(2, 1)
        plt.bar(times, volumes, color='blue')
        plt.ylabel("Volume")
        plt.xlabel(f"Last {CANDLE_COUNT} candles ({TIMEFRAME})")
        plt.plotsize(100, 6)

        plt.show()

        print("-" * 80)
        print(f"Price: ${current_price:.4f}  |  High: ${max(closes):.4f}  |  Low: ${min(closes):.4f}")
        print(f"{TIMEFRAME} candles  |  {datetime.now().strftime('%H:%M:%S')}  |  Ctrl+C to exit")
        print("-" * 80)

    def run(self, chart_type='candle'):
        print(f"\nStarting MERL chart (type: {chart_type})...\n")
        time.sleep(1)

        while True:
            try:
                if chart_type == 'candle':
                    self.display_chart()
                else:
                    self.display_simple_chart()
                time.sleep(REFRESH_INTERVAL)
            except KeyboardInterrupt:
                print("\nChart closed.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MERL Chart')
    parser.add_argument('--type', '-t', choices=['candle', 'line'], default='candle')
    parser.add_argument('--timeframe', '-tf', default='5m')
    args = parser.parse_args()

    global TIMEFRAME
    TIMEFRAME = args.timeframe

    chart = MERLChart()
    if chart.symbol:
        chart.run(chart_type=args.type)

if __name__ == "__main__":
    main()
