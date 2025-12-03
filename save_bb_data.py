"""
Save current BB (BlackBerry) data for analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# Download current BB data
print("Downloading BB (BlackBerry) data...")
bb = yf.Ticker('BB')
df = bb.history(period='5d', interval='1h')

# Save to CSV
filename = f'BB_current_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(filename)

print(f'✅ Saved BB data to: {filename}')
print(f'   Latest price: ${df["Close"].iloc[-1]:.2f}')
print(f'   Data points: {len(df)} hours')
print(f'   Date range: {df.index[0]} to {df.index[-1]}')

# Show last few data points
print("\nLast 5 data points:")
print(df.tail())