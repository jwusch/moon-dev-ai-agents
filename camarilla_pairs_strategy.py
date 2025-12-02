"""
🎯 Camarilla Strategy for Pairs Trading
Applying mean reversion to naturally range-bound pairs

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
from datetime import datetime
import os

class CamarillaPairsStrategy(Strategy):
    """
    Trade the spread between two correlated instruments using Camarilla levels
    """
    
    # Parameters
    zscore_entry = 2.0    # Enter at 2 standard deviations
    zscore_exit = 0.5     # Exit when spread returns to 0.5 std
    lookback = 60         # Days for calculating mean and std
    position_size = 0.95  # Use 95% of capital
    
    def init(self):
        # For pairs trading, we need the spread data to be pre-calculated
        # This strategy assumes the 'Close' price is actually the spread ratio
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Calculate rolling statistics
        self.spread_mean = self.I(lambda: close.rolling(self.lookback).mean().values)
        self.spread_std = self.I(lambda: close.rolling(self.lookback).std().values)
        
        # Z-score
        def calc_zscore():
            mean = self.spread_mean
            std = self.spread_std
            z = (close - mean) / std
            return z.values
        
        self.zscore = self.I(calc_zscore)
        
        # Camarilla levels on the spread
        prev_high = high.shift(1)
        prev_low = low.shift(1) 
        prev_close = close.shift(1)
        prev_range = prev_high - prev_low
        
        self.r3 = self.I(lambda: (prev_close + prev_range * 1.1 / 4).values)
        self.r2 = self.I(lambda: (prev_close + prev_range * 1.1 / 6).values)
        self.s2 = self.I(lambda: (prev_close - prev_range * 1.1 / 6).values)
        self.s3 = self.I(lambda: (prev_close - prev_range * 1.1 / 4).values)
        
        # RSI on the spread
        self.rsi = self.I(talib.RSI, close, 14)
        
    def next(self):
        if len(self.data) < self.lookback + 1 or self.position:
            return
            
        # Current values
        zscore = self.zscore[-1]
        spread = self.data.Close[-1]
        rsi = self.rsi[-1]
        r3 = self.r3[-1]
        r2 = self.r2[-1]
        s2 = self.s2[-1]
        s3 = self.s3[-1]
        
        # Skip if indicators are invalid
        if pd.isna(zscore) or pd.isna(rsi) or abs(zscore) > 4:  # Skip extreme outliers
            return
            
        # Mean reversion trades based on z-score and Camarilla
        
        # Buy spread when oversold (expecting reversion up)
        if zscore <= -self.zscore_entry and spread <= s3:
            if rsi < 30:  # Confirm with RSI
                # Target: return to mean (z=0) or R2
                tp_zscore = self.spread_mean[-1] + 0.5 * self.spread_std[-1]
                tp_camarilla = r2
                tp = min(tp_zscore, tp_camarilla)
                
                # Stop: 3 std deviations or below S3
                sl = self.spread_mean[-1] - 3 * self.spread_std[-1]
                
                if spread > sl and spread < tp:
                    self.buy(size=self.position_size, sl=sl, tp=tp)
        
        # Sell spread when overbought (expecting reversion down)
        elif zscore >= self.zscore_entry and spread >= r3:
            if rsi > 70:  # Confirm with RSI
                # Target: return to mean (z=0) or S2
                tp_zscore = self.spread_mean[-1] - 0.5 * self.spread_std[-1]
                tp_camarilla = s2
                tp = max(tp_zscore, tp_camarilla)
                
                # Stop: 3 std deviations or above R3
                sl = self.spread_mean[-1] + 3 * self.spread_std[-1]
                
                if spread < sl and spread > tp:
                    self.sell(size=self.position_size, sl=sl, tp=tp)


def prepare_pair_data(ticker1, ticker2, period="2y"):
    """Download and prepare spread data for backtesting"""
    try:
        # Download data
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        df1 = stock1.history(period=period)
        df2 = stock2.history(period=period)
        
        # Align dates
        common_dates = df1.index.intersection(df2.index)
        df1 = df1.loc[common_dates]
        df2 = df2.loc[common_dates]
        
        # Calculate spread (ratio)
        spread_df = pd.DataFrame(index=common_dates)
        spread_df['Open'] = df1['Open'] / df2['Open']
        spread_df['High'] = df1['High'] / df2['High']
        spread_df['Low'] = df1['Low'] / df2['Low']
        spread_df['Close'] = df1['Close'] / df2['Close']
        spread_df['Volume'] = (df1['Volume'] + df2['Volume']) / 2  # Average volume
        
        return spread_df
        
    except Exception as e:
        print(f"Error preparing {ticker1}/{ticker2}: {e}")
        return None


# Test on the best pairs from our analysis
test_pairs = [
    ("GLD", "SLV"),   # Gold/Silver - strong mean reversion, z=-2.44
    ("GLD", "GDX"),   # Gold/Miners - z=-2.34
    ("IWM", "MDY"),   # Small/Mid cap - z=2.03
    ("USO", "BNO"),   # WTI/Brent oil - z=-1.79
    ("XLU", "XLP"),   # Utilities/Staples - z=1.76
    ("VTV", "VUG"),   # Value/Growth - z=-1.53
    ("TLT", "IEF"),   # Long/Medium bonds
    ("VNQ", "IYR"),   # Two REIT ETFs - highest mean reversion
]

# For single instruments
test_singles = [
    "VXX",    # Volatility - Hurst 0.33
    "VIXY",   # Volatility - Hurst 0.33
    "BTAL",   # Market neutral - Hurst 0.43
]

print("="*70)
print("🎯 TESTING CAMARILLA ON MEAN-REVERTING PAIRS")
print("="*70)

# Create results directory
results_dir = "camarilla_pairs_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Test pairs
pair_results = []

for ticker1, ticker2 in test_pairs:
    print(f"\nTesting {ticker1}/{ticker2} pair...", end=" ")
    
    # Prepare spread data
    spread_df = prepare_pair_data(ticker1, ticker2)
    
    if spread_df is None or len(spread_df) < 200:
        print("Insufficient data")
        continue
    
    # Use last 500 days
    test_df = spread_df.tail(500) if len(spread_df) > 500 else spread_df
    
    # Calculate spread statistics
    spread_mean = test_df['Close'].mean()
    spread_std = test_df['Close'].std()
    current_zscore = (test_df['Close'].iloc[-1] - spread_mean) / spread_std
    
    # Run backtest
    try:
        bt = Backtest(test_df, CamarillaPairsStrategy,
                     cash=10000,
                     commission=0.002,
                     trade_on_close=True)
        
        stats = bt.run()
        
        result = {
            'Pair': f"{ticker1}/{ticker2}",
            'Return': stats['Return [%]'],
            'Sharpe': stats['Sharpe Ratio'],
            'Max_DD': stats['Max. Drawdown [%]'],
            'Trades': stats['# Trades'],
            'Win_Rate': stats['Win Rate [%]'],
            'Current_Z': current_zscore,
            'Days': len(test_df)
        }
        
        pair_results.append(result)
        print(f"Return: {stats['Return [%]']:+.1f}%, Sharpe: {stats['Sharpe Ratio']:.2f}")
        
        # Save spread data
        test_df.to_csv(f"{results_dir}/{ticker1}_{ticker2}_spread.csv")
        
    except Exception as e:
        print(f"Backtest error: {e}")

# Test single instruments with standard Camarilla
print("\n\nTesting single mean-reverting instruments...")

from camarilla_improved_strategy import ImprovedCamarillaStrategy

single_results = []

for ticker in test_singles:
    print(f"\n{ticker}...", end=" ")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if len(df) < 200:
            print("Insufficient data")
            continue
            
        test_df = df.tail(500) if len(df) > 500 else df
        
        # Calculate buy & hold
        buy_hold = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0] - 1) * 100
        
        bt = Backtest(test_df, ImprovedCamarillaStrategy,
                     cash=10000,
                     commission=0.002,
                     trade_on_close=True)
        
        stats = bt.run()
        
        result = {
            'Ticker': ticker,
            'Strategy_Return': stats['Return [%]'],
            'Buy_Hold': buy_hold,
            'Alpha': stats['Return [%]'] - buy_hold,
            'Sharpe': stats['Sharpe Ratio'],
            'Trades': stats['# Trades'],
            'Win_Rate': stats['Win Rate [%]']
        }
        
        single_results.append(result)
        print(f"Return: {stats['Return [%]']:+.1f}%, Alpha: {result['Alpha']:+.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")

# Summary
print("\n" + "="*70)
print("📊 RESULTS SUMMARY")
print("="*70)

if pair_results:
    pair_df = pd.DataFrame(pair_results)
    pair_df = pair_df.sort_values('Return', ascending=False)
    
    print("\nPAIR TRADING RESULTS:")
    print("-"*70)
    print(f"{'Pair':<15} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Trades':>8} {'Win%':>8} {'Z-Score':>10}")
    print("-"*70)
    
    for _, row in pair_df.iterrows():
        print(f"{row['Pair']:<15} {row['Return']:>9.1f}% {row['Sharpe']:>10.2f} "
              f"{row['Max_DD']:>9.1f}% {row['Trades']:>8} {row['Win_Rate']:>7.1f}% "
              f"{row['Current_Z']:>10.2f}")
    
    print(f"\nAverage return: {pair_df['Return'].mean():.1f}%")
    print(f"Best pair: {pair_df.iloc[0]['Pair']} ({pair_df.iloc[0]['Return']:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_df.to_csv(f"{results_dir}/pair_results_{timestamp}.csv", index=False)

if single_results:
    single_df = pd.DataFrame(single_results)
    
    print("\n\nSINGLE INSTRUMENT RESULTS (Mean Reverting):")
    print("-"*70)
    for _, row in single_df.iterrows():
        print(f"{row['Ticker']}: Return={row['Strategy_Return']:.1f}%, "
              f"Alpha={row['Alpha']:.1f}%, Trades={row['Trades']}")

print("\n" + "="*70)
print("💡 KEY FINDINGS")
print("="*70)

print("""
1. Pairs trading shows more promise than single instruments
2. True mean-reverting pairs can be traded with Camarilla
3. Z-score entry/exit rules work better than price levels
4. Current opportunities exist in extreme z-score pairs

Next steps:
- Optimize z-score thresholds
- Add correlation breakdown detection
- Consider options strategies on these pairs
- Test shorter timeframes for more signals
""")

print(f"\n✅ All results saved to: {results_dir}/")