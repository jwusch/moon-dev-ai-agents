"""
🎯 VXX Intraday Camarilla - Optimized Version
Using proper volatility-adjusted levels and multiple timeframe confirmation

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from yfinance_cache_demo import YFinanceCache

class OptimizedVXXStrategy:
    """
    Enhanced intraday strategy for VXX with:
    - Dynamic Camarilla levels based on ATR
    - Multiple timeframe analysis
    - Better entry/exit rules
    - Session-based trading
    """
    
    def __init__(self, symbol="VXX"):
        self.symbol = symbol
        self.cache = YFinanceCache()
        self.trades = []
        self.daily_stats = {}
        
    def calculate_dynamic_camarilla(self, high, low, close, atr):
        """Calculate Camarilla levels adjusted for volatility"""
        range_hl = high - low
        
        # Adjust multiplier based on ATR
        # Higher volatility = wider levels
        volatility_factor = min(2.0, max(0.5, atr / close))
        
        levels = {
            'r4': close + range_hl * 1.1 / 2 * volatility_factor,
            'r3': close + range_hl * 1.1 / 4 * volatility_factor,
            'r2': close + range_hl * 1.1 / 6 * volatility_factor,
            'r1': close + range_hl * 1.1 / 12 * volatility_factor,
            'pp': (high + low + close) / 3,
            's1': close - range_hl * 1.1 / 12 * volatility_factor,
            's2': close - range_hl * 1.1 / 6 * volatility_factor,
            's3': close - range_hl * 1.1 / 4 * volatility_factor,
            's4': close - range_hl * 1.1 / 2 * volatility_factor
        }
        
        return levels
    
    def identify_session_volatility(self, df_15m):
        """Identify high volatility sessions for trading"""
        sessions = {
            'pre_market': (9, 30, 10, 0),    # 9:30-10:00 AM
            'morning': (10, 0, 11, 30),      # 10:00-11:30 AM  
            'lunch': (11, 30, 13, 30),       # 11:30-1:30 PM
            'afternoon': (13, 30, 15, 30),   # 1:30-3:30 PM
            'close': (15, 30, 16, 0)         # 3:30-4:00 PM
        }
        
        volatility_by_session = {}
        
        for session_name, (start_h, start_m, end_h, end_m) in sessions.items():
            session_data = df_15m[
                (df_15m.index.hour >= start_h) & 
                (df_15m.index.hour < end_h) |
                ((df_15m.index.hour == start_h) & (df_15m.index.minute >= start_m)) |
                ((df_15m.index.hour == end_h) & (df_15m.index.minute < end_m))
            ]
            
            if len(session_data) > 0:
                avg_range = ((session_data['High'] - session_data['Low']) / 
                           session_data['Low'] * 100).mean()
                volatility_by_session[session_name] = avg_range
        
        return volatility_by_session
    
    def run_comprehensive_backtest(self):
        """Run backtest with multiple timeframes and optimizations"""
        print(f"📊 Loading data for {self.symbol}...")
        
        # Get multiple timeframes
        df_daily = self.cache.get_data(self.symbol, period="6mo", interval="1d")
        df_60m = self.cache.get_data(self.symbol, period="3mo", interval="60m")
        df_15m = self.cache.get_data(self.symbol, period="60d", interval="15m")
        
        if len(df_15m) < 100:
            print("Insufficient data")
            return None
        
        # Calculate indicators on different timeframes
        # Daily for trend
        df_daily['SMA20'] = df_daily['Close'].rolling(20).mean()
        df_daily['Trend'] = np.where(df_daily['Close'] > df_daily['SMA20'], 1, -1)
        
        # 60min for medium-term levels
        df_60m['VWAP'] = ta.vwap(df_60m['High'], df_60m['Low'], df_60m['Close'], df_60m['Volume'])
        df_60m['RSI'] = ta.rsi(df_60m['Close'], length=14)
        
        # 15min for entries
        df_15m['ATR'] = ta.atr(df_15m['High'], df_15m['Low'], df_15m['Close'], length=14)
        df_15m['RSI'] = ta.rsi(df_15m['Close'], length=14)
        bb_result = ta.bbands(df_15m['Close'], length=20, std=2)
        df_15m['BB_lower'] = bb_result.iloc[:, 0]  # First column is lower band
        df_15m['BB_middle'] = bb_result.iloc[:, 1]  # Second column is middle band
        df_15m['BB_upper'] = bb_result.iloc[:, 2]  # Third column is upper band
        
        # Identify best trading sessions
        session_vol = self.identify_session_volatility(df_15m)
        print("\n📈 Session Volatility Analysis:")
        for session, vol in sorted(session_vol.items(), key=lambda x: x[1], reverse=True):
            print(f"  {session}: {vol:.2f}% average range")
        
        # Trading simulation
        position = 0
        entry_price = 0
        entry_time = None
        daily_trades = 0
        current_date = None
        
        for i in range(100, len(df_15m)):
            current_time = df_15m.index[i]
            current_price = df_15m['Close'].iloc[i]
            
            # New day setup
            if current_time.date() != current_date:
                current_date = current_time.date()
                daily_trades = 0
                
                # Get previous day's data for Camarilla
                prev_date = current_date - timedelta(days=1)
                prev_day_data = df_15m[df_15m.index.date == prev_date]
                
                if len(prev_day_data) > 0:
                    daily_high = prev_day_data['High'].max()
                    daily_low = prev_day_data['Low'].min()
                    daily_close = prev_day_data['Close'].iloc[-1]
                    daily_atr = prev_day_data['ATR'].iloc[-1] if not pd.isna(prev_day_data['ATR'].iloc[-1]) else (daily_high - daily_low)
                    
                    camarilla = self.calculate_dynamic_camarilla(
                        daily_high, daily_low, daily_close, daily_atr
                    )
                else:
                    camarilla = None
            
            if camarilla is None or pd.isna(df_15m['RSI'].iloc[i]) or pd.isna(df_15m['ATR'].iloc[i]):
                continue
            
            # Position management
            if position != 0:
                hours_held = (current_time - entry_time).total_seconds() / 3600
                pnl_pct = ((current_price - entry_price) / entry_price * 100 
                          if position > 0 
                          else (entry_price - current_price) / entry_price * 100)
                
                # Dynamic exit based on volatility
                atr_pct = df_15m['ATR'].iloc[i] / current_price * 100
                profit_target = min(2.0, max(0.5, atr_pct * 0.5))  # 0.5-2% based on ATR
                stop_loss = min(3.0, max(1.0, atr_pct * 1.0))      # 1-3% based on ATR
                
                # Exit conditions
                exit_signal = False
                exit_reason = ""
                
                if pnl_pct >= profit_target:
                    exit_signal = True
                    exit_reason = "Profit target"
                elif pnl_pct <= -stop_loss:
                    exit_signal = True
                    exit_reason = "Stop loss"
                elif hours_held >= 2:
                    exit_signal = True
                    exit_reason = "Time limit"
                elif current_time.hour >= 15 and current_time.minute >= 45:
                    exit_signal = True
                    exit_reason = "End of day"
                elif position > 0 and current_price >= camarilla['r2']:
                    exit_signal = True
                    exit_reason = "Resistance hit"
                elif position < 0 and current_price <= camarilla['s2']:
                    exit_signal = True
                    exit_reason = "Support hit"
                
                if exit_signal:
                    self.record_trade(entry_time, current_time, 
                                    'Long' if position > 0 else 'Short',
                                    entry_price, current_price, pnl_pct, exit_reason)
                    position = 0
                    daily_trades += 1
            
            else:
                # Entry signals - limit trades per day
                if daily_trades < 3 and 9 <= current_time.hour < 15:
                    rsi = df_15m['RSI'].iloc[i]
                    bb_pct = (current_price - df_15m['BB_lower'].iloc[i]) / (
                        df_15m['BB_upper'].iloc[i] - df_15m['BB_lower'].iloc[i]
                    )
                    
                    # Get hourly trend
                    hourly_idx = df_60m.index.get_indexer([current_time], method='ffill')[0]
                    if hourly_idx >= 0 and hourly_idx < len(df_60m):
                        hourly_rsi = df_60m['RSI'].iloc[hourly_idx]
                    else:
                        hourly_rsi = 50
                    
                    # Long setup: Multiple confirmations
                    if (current_price <= camarilla['s1'] and 
                        current_price > camarilla['s2'] and
                        rsi < 40 and
                        bb_pct < 0.2 and
                        hourly_rsi < 45):
                        
                        position = 1
                        entry_price = current_price
                        entry_time = current_time
                    
                    # Short setup: Multiple confirmations
                    elif (current_price >= camarilla['r1'] and
                          current_price < camarilla['r2'] and
                          rsi > 60 and
                          bb_pct > 0.8 and
                          hourly_rsi > 55):
                        
                        position = -1
                        entry_price = current_price
                        entry_time = current_time
        
        # Close any open position
        if position != 0:
            self.record_trade(entry_time, df_15m.index[-1],
                            'Long' if position > 0 else 'Short',
                            entry_price, df_15m['Close'].iloc[-1],
                            pnl_pct, "End of data")
        
        return self.analyze_results()
    
    def record_trade(self, entry_time, exit_time, trade_type, entry_price, 
                    exit_price, pnl_pct, exit_reason):
        """Record completed trade"""
        self.trades.append({
            'Entry_Time': entry_time,
            'Exit_Time': exit_time,
            'Type': trade_type,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL%': pnl_pct,
            'Exit_Reason': exit_reason,
            'Duration_Min': (exit_time - entry_time).total_seconds() / 60
        })
        
        # Update daily stats
        date = entry_time.date()
        if date not in self.daily_stats:
            self.daily_stats[date] = {'trades': 0, 'pnl': 0}
        self.daily_stats[date]['trades'] += 1
        self.daily_stats[date]['pnl'] += pnl_pct
    
    def analyze_results(self):
        """Comprehensive results analysis"""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Overall statistics
        total_return = trades_df['PnL%'].sum()
        num_trades = len(trades_df)
        win_rate = (trades_df['PnL%'] > 0).sum() / num_trades * 100
        
        # By exit reason
        exit_stats = trades_df.groupby('Exit_Reason').agg({
            'PnL%': ['count', 'mean', 'sum']
        }).round(2)
        
        # By trade type
        type_stats = trades_df.groupby('Type').agg({
            'PnL%': ['count', 'mean', 'sum'],
            'Duration_Min': 'mean'
        }).round(2)
        
        # Time of day analysis
        trades_df['Hour'] = trades_df['Entry_Time'].dt.hour
        hour_stats = trades_df.groupby('Hour').agg({
            'PnL%': ['count', 'mean', 'sum']
        }).round(2)
        
        return {
            'trades_df': trades_df,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': trades_df[trades_df['PnL%'] > 0]['PnL%'].mean() if any(trades_df['PnL%'] > 0) else 0,
            'avg_loss': trades_df[trades_df['PnL%'] < 0]['PnL%'].mean() if any(trades_df['PnL%'] < 0) else 0,
            'exit_stats': exit_stats,
            'type_stats': type_stats,
            'hour_stats': hour_stats,
            'daily_stats': self.daily_stats
        }


def main():
    print("="*70)
    print("🚀 VXX INTRADAY STRATEGY - OPTIMIZED VERSION")
    print("="*70)
    
    strategy = OptimizedVXXStrategy("VXX")
    results = strategy.run_comprehensive_backtest()
    
    if results:
        print(f"\n📊 BACKTEST RESULTS")
        print("-"*50)
        print(f"Total Trades: {results['num_trades']}")
        print(f"Total Return: {results['total_return']:.1f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Average Win: {results['avg_win']:.2f}%")
        print(f"Average Loss: {results['avg_loss']:.2f}%")
        print(f"Profit Factor: {abs(results['avg_win']/results['avg_loss']) if results['avg_loss'] != 0 else 0:.2f}")
        
        print(f"\n📊 EXIT REASON ANALYSIS")
        print("-"*50)
        print(results['exit_stats'])
        
        print(f"\n📊 TRADE TYPE ANALYSIS")
        print("-"*50)
        print(results['type_stats'])
        
        print(f"\n📊 BEST TRADING HOURS")
        print("-"*50)
        best_hours = results['hour_stats'].sort_values(('PnL%', 'sum'), ascending=False).head(5)
        for hour, stats in best_hours.iterrows():
            print(f"{hour}:00 - Trades: {stats[('PnL%', 'count')]}, "
                  f"Avg: {stats[('PnL%', 'mean')]:.2f}%, "
                  f"Total: {stats[('PnL%', 'sum')]:.1f}%")
        
        # Daily breakdown
        print(f"\n📊 DAILY PERFORMANCE (Last 10 days)")
        print("-"*50)
        daily_df = pd.DataFrame.from_dict(results['daily_stats'], orient='index')
        for date, stats in daily_df.tail(10).iterrows():
            print(f"{date}: {stats['trades']} trades, {stats['pnl']:.1f}% PnL")
        
        # Save results
        results['trades_df'].to_csv('vxx_optimized_trades.csv', index=False)
        print(f"\n✅ Detailed results saved to: vxx_optimized_trades.csv")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS")
        print("-"*50)
        print(f"1. Best exit reason: {results['exit_stats'].idxmax()[('PnL%', 'mean')]}")
        print(f"2. Best trading hour: {results['hour_stats'].idxmax()[('PnL%', 'mean')]}:00")
        print(f"3. Trades per day: {results['num_trades'] / len(results['daily_stats']):.1f}")
        print(f"4. Daily expectation: {results['total_return'] / len(results['daily_stats']):.2f}%")
        
        # Comparison to daily strategy
        print(f"\n📊 INTRADAY vs DAILY COMPARISON")
        print("-"*50)
        print(f"Daily Strategy: 1 trade over 500 days = 0.002 trades/day")
        print(f"Intraday Strategy: {results['num_trades']} trades over {len(results['daily_stats'])} days = {results['num_trades']/len(results['daily_stats']):.1f} trades/day")
        print(f"\nThis confirms: VXX with intraday data provides {results['num_trades']/0.002:.0f}x more opportunities!")

if __name__ == "__main__":
    main()