import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Quick test of bbands
ticker = yf.Ticker("VXX")
df = ticker.history(period="5d", interval="15m")

print("Testing pandas_ta bbands...")
bb_result = ta.bbands(df['Close'], length=20, std=2)
print(f"Type: {type(bb_result)}")
print(f"Columns: {bb_result.columns.tolist() if hasattr(bb_result, 'columns') else 'N/A'}")
print(f"Shape: {bb_result.shape if hasattr(bb_result, 'shape') else 'N/A'}")
print(bb_result.head())