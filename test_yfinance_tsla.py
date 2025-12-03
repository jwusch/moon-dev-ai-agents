"""Test YFinance TSLA data fetch"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

from agents.yfinance_adapter import YFinanceAdapter
import yfinance as yf

# Test direct yfinance
print("Testing direct yfinance:")
tsla = yf.Ticker("TSLA")
hist = tsla.history(period="500d")
print(f"Direct yfinance: {len(hist)} days")

# Test our adapter
print("\nTesting YFinance adapter:")
adapter = YFinanceAdapter()
data = adapter.get_ohlcv_data('TSLA', '1d', 500)
print(f"Adapter data: {len(data)} days")

if not data.empty:
    print(f"First row: {data.iloc[0]}")
    print(f"Last row: {data.iloc[-1]}")