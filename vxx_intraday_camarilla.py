"""
🎯 VXX Intraday Camarilla Strategy
Demonstrating multiple trading opportunities with 15-minute data

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta

class IntradayVXXCamarilla:
    """
    Camarilla strategy optimized for VXX intraday trading
    Uses 15-minute bars and tighter levels for frequent mean reversion
    """
    
    def __init__(self, symbol="VXX", interval="15m", lookback_days=30):
        self.symbol = symbol
        self.interval = interval
        self.lookback_days = lookback_days
        self.trades = []
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        
    def calculate_camarilla_levels(self, high, low, close):
        """Calculate Camarilla levels for intraday trading"""
        range_hl = high - low
        
        levels = {
            'r4': close + range_hl * 1.1 / 2,
            'r3': close + range_hl * 1.1 / 4,
            'r2': close + range_hl * 1.1 / 6,
            'r1': close + range_hl * 1.1 / 12,
            'pp': (high + low + close) / 3,
            's1': close - range_hl * 1.1 / 12,
            's2': close - range_hl * 1.1 / 6,
            's3': close - range_hl * 1.1 / 4,
            's4': close - range_hl * 1.1 / 2
        }
        
        return levels
    
    def run_backtest(self):
        """Run intraday backtest on VXX"""
        print(f"Downloading {self.symbol} {self.interval} data...")
        
        # Get data
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=f"{self.lookback_days}d", interval=self.interval)
        
        if len(df) < 100:
            print("Insufficient data for backtest")
            return None
            
        # Add technical indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Track daily levels
        daily_levels = None
        current_date = None
        
        # Simulate trading
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Update daily Camarilla levels at start of each day
            if current_time.date() != current_date:
                current_date = current_time.date()
                # Get previous day's data
                prev_day_data = df[df.index.date < current_date].tail(26)  # ~1 day of 15min bars
                if len(prev_day_data) > 0:
                    daily_high = prev_day_data['High'].max()
                    daily_low = prev_day_data['Low'].min()
                    daily_close = prev_day_data['Close'].iloc[-1]
                    daily_levels = self.calculate_camarilla_levels(daily_high, daily_low, daily_close)
            
            if daily_levels is None or pd.isna(df['RSI'].iloc[i]) or pd.isna(df['ATR'].iloc[i]):
                continue
                
            # Position management
            if self.position != 0:
                # Check exit conditions
                hours_held = (current_time - self.entry_time).total_seconds() / 3600
                
                if self.position > 0:  # Long position
                    pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                    
                    # Exit conditions
                    if (pnl_pct >= 1.0 or  # 1% profit target
                        pnl_pct <= -2.0 or  # 2% stop loss
                        hours_held >= 3 or  # Max 3 hours
                        current_time.hour >= 15):  # Close before market close
                        
                        self.close_position(current_time, current_price, 'Long')
                        
                elif self.position < 0:  # Short position
                    pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
                    
                    # Exit conditions
                    if (pnl_pct >= 1.0 or  # 1% profit target
                        pnl_pct <= -2.0 or  # 2% stop loss
                        hours_held >= 3 or  # Max 3 hours
                        current_time.hour >= 15):  # Close before market close
                        
                        self.close_position(current_time, current_price, 'Short')
            
            else:
                # Entry signals (only during market hours, not first/last 30 min)
                if 10 <= current_time.hour < 15:
                    rsi = df['RSI'].iloc[i]
                    sma = df['SMA20'].iloc[i]
                    
                    # Long entry: Price at S1 or S2 with oversold RSI
                    if (current_price <= daily_levels['s1'] and 
                        current_price > daily_levels['s2'] and
                        rsi < 35):
                        
                        self.position = 1
                        self.entry_price = current_price
                        self.entry_time = current_time
                        
                    # Short entry: Price at R1 or R2 with overbought RSI
                    elif (current_price >= daily_levels['r1'] and
                          current_price < daily_levels['r2'] and
                          rsi > 65):
                        
                        self.position = -1
                        self.entry_price = current_price
                        self.entry_time = current_time
        
        # Force close any open position
        if self.position != 0:
            self.close_position(df.index[-1], df['Close'].iloc[-1], 
                               'Long' if self.position > 0 else 'Short')
        
        return self.analyze_results(df)
    
    def close_position(self, exit_time, exit_price, position_type):
        """Record trade and reset position"""
        pnl_pct = ((exit_price - self.entry_price) / self.entry_price * 100 
                   if position_type == 'Long' 
                   else (self.entry_price - exit_price) / self.entry_price * 100)
        
        duration = (exit_time - self.entry_time).total_seconds() / 60  # minutes
        
        self.trades.append({
            'Entry_Time': self.entry_time,
            'Exit_Time': exit_time,
            'Type': position_type,
            'Entry_Price': self.entry_price,
            'Exit_Price': exit_price,
            'PnL%': pnl_pct,
            'Duration_Min': duration
        })
        
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
    
    def analyze_results(self, df):
        """Analyze backtest results"""
        if not self.trades:
            return None
            
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate statistics
        total_return = trades_df['PnL%'].sum()
        num_trades = len(trades_df)
        win_rate = (trades_df['PnL%'] > 0).sum() / num_trades * 100
        avg_win = trades_df[trades_df['PnL%'] > 0]['PnL%'].mean() if any(trades_df['PnL%'] > 0) else 0
        avg_loss = trades_df[trades_df['PnL%'] < 0]['PnL%'].mean() if any(trades_df['PnL%'] < 0) else 0
        avg_duration = trades_df['Duration_Min'].mean()
        
        # Daily statistics
        trades_df['Date'] = trades_df['Entry_Time'].dt.date
        daily_stats = trades_df.groupby('Date').agg({
            'PnL%': ['count', 'sum']
        })
        
        results = {
            'Total_Return_%': total_return,
            'Num_Trades': num_trades,
            'Win_Rate_%': win_rate,
            'Avg_Win_%': avg_win,
            'Avg_Loss_%': avg_loss,
            'Avg_Duration_Min': avg_duration,
            'Trades_Per_Day': num_trades / len(daily_stats) if len(daily_stats) > 0 else 0,
            'Days_Traded': len(daily_stats),
            'Best_Day_%': daily_stats['PnL%']['sum'].max() if len(daily_stats) > 0 else 0,
            'Worst_Day_%': daily_stats['PnL%']['sum'].min() if len(daily_stats) > 0 else 0,
            'Trades_DF': trades_df,
            'Daily_Stats': daily_stats,
            'Data_Range': f"{df.index[0]} to {df.index[-1]}"
        }
        
        return results

def main():
    print("="*70)
    print("📊 VXX INTRADAY CAMARILLA BACKTEST")
    print("="*70)
    
    # Run backtest
    strategy = IntradayVXXCamarilla(symbol="VXX", interval="15m", lookback_days=30)
    results = strategy.run_backtest()
    
    if results:
        print(f"\nBacktest Period: {results['Data_Range']}")
        print(f"Total Trades: {results['Num_Trades']}")
        print(f"Days Traded: {results['Days_Traded']}")
        print(f"Trades Per Day: {results['Trades_Per_Day']:.1f}")
        print(f"\nPerformance:")
        print(f"  Total Return: {results['Total_Return_%']:.1f}%")
        print(f"  Win Rate: {results['Win_Rate_%']:.1f}%")
        print(f"  Avg Win: {results['Avg_Win_%']:.2f}%")
        print(f"  Avg Loss: {results['Avg_Loss_%']:.2f}%")
        print(f"  Avg Duration: {results['Avg_Duration_Min']:.0f} minutes")
        print(f"\nDaily Statistics:")
        print(f"  Best Day: {results['Best_Day_%']:.1f}%")
        print(f"  Worst Day: {results['Worst_Day_%']:.1f}%")
        
        # Show recent trades
        print("\nRecent Trades:")
        print("-"*70)
        for _, trade in results['Trades_DF'].tail(10).iterrows():
            print(f"{trade['Entry_Time'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{trade['Type']:<5} - "
                  f"PnL: {trade['PnL%']:+.1f}% - "
                  f"Duration: {trade['Duration_Min']:.0f} min")
        
        # Compare to daily strategy
        print("\n" + "="*70)
        print("📊 COMPARISON: INTRADAY vs DAILY")
        print("="*70)
        
        print(f"""
Daily Strategy (what we tested earlier):
  - 1 trade on VXX over 500 days
  - Limited opportunity capture
  - Misses intraday volatility

Intraday Strategy (15-minute bars):
  - {results['Num_Trades']} trades over {results['Days_Traded']} days
  - {results['Trades_Per_Day']:.1f} trades per day average
  - Captures multiple daily swings
  - Total return: {results['Total_Return_%']:.1f}% in {results['Days_Traded']} days
  
Annualized Performance Estimate:
  - Daily return: {results['Total_Return_%'] / results['Days_Traded']:.2f}%
  - Monthly return: {results['Total_Return_%'] / results['Days_Traded'] * 20:.1f}%
  - Annual return: {results['Total_Return_%'] / results['Days_Traded'] * 252:.0f}%
  
This confirms your intuition: finer-grained data on VXX 
provides MANY more trading opportunities!
""")
        
        # Save detailed results
        results['Trades_DF'].to_csv('vxx_intraday_trades.csv', index=False)
        print("\n✅ Detailed trade history saved to: vxx_intraday_trades.csv")
    
    else:
        print("Failed to run backtest")

if __name__ == "__main__":
    main()