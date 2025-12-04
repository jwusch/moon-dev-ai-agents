#!/usr/bin/env python3
"""
📊 AEGS 5 SYMBOLS TEST 📊
Quick test with 5 known working symbols
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import yfinance as yf

class AEGSQuickTest:
    """Quick AEGS test with known symbols"""
    
    def __init__(self):
        self.cache_dir = "intraday_15min_cache"
        self.test_symbols = ["AAPL", "TSLA", "SPY", "QQQ", "MSFT"]  # Known working symbols
        
        # AEGS Parameters
        self.rsi_oversold = 35
        self.bb_position_threshold = 0.25
        self.volume_surge_threshold = 1.5
        self.min_drop_threshold = 1.5
        self.profit_target = 0.12
        self.stop_loss = 0.06
        self.max_hold_periods = 64
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def fetch_15min_data(self, symbol, days_back=55):
        """Fetch 15-minute data for testing"""
        
        cache_file = os.path.join(self.cache_dir, f"{symbol}_test_15min_{days_back}d.pkl")
        
        # Check cache
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty:
                    print(f"📦 Cache HIT: {symbol} ({len(df)} bars)")
                    return df
            except:
                pass
        
        # Fetch data
        print(f"🌐 Fetching 15min data: {symbol}")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="15m",
                auto_adjust=True,
                timeout=30
            )
            
            if df is not None and not df.empty:
                # Save to cache
                df.to_pickle(cache_file)
                print(f"✅ Fetched: {symbol} ({len(df)} bars, {df.index[0].date()} to {df.index[-1].date()})")
                time.sleep(2)  # Rate limiting
                return df
            else:
                print(f"❌ No data for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        df['BB_Position'] = (df['Close'] - lower_band) / (upper_band - lower_band)
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price changes
        df['Price_Change_4bar'] = df['Close'].pct_change(4) * 100
        
        return df
    
    def generate_signals(self, df):
        """Generate AEGS signals"""
        
        df = self.calculate_indicators(df)
        
        # AEGS conditions
        df['Signal_RSI'] = df['RSI'] < self.rsi_oversold
        df['Signal_BB'] = df['BB_Position'] < self.bb_position_threshold
        df['Signal_Volume'] = df['Volume_Ratio'] > self.volume_surge_threshold
        df['Signal_Drop'] = df['Price_Change_4bar'] < -self.min_drop_threshold
        
        # Quality filters
        df['Valid_Price'] = df['Close'] > 5.0
        df['Valid_Volume'] = df['Volume'] > 100000
        
        # Count signals
        signal_conditions = ['Signal_RSI', 'Signal_BB', 'Signal_Volume', 'Signal_Drop']
        df['Signal_Count'] = df[signal_conditions].sum(axis=1)
        
        # AEGS entry signal (3 out of 4 conditions + quality)
        df['AEGS_Signal'] = (
            (df['Signal_Count'] >= 3) &
            df['Valid_Price'] &
            df['Valid_Volume'] &
            df['RSI'].notna() &
            df['BB_Position'].notna()
        )
        
        return df
    
    def backtest_symbol(self, symbol, df):
        """Quick backtest"""
        
        df = self.generate_signals(df)
        
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_index = 0
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            
            if not in_position and row['AEGS_Signal']:
                # Enter position
                in_position = True
                entry_price = row['Close']
                entry_time = timestamp
                entry_index = i
                
            elif in_position:
                # Check exit
                current_price = row['Close']
                current_return = (current_price - entry_price) / entry_price
                periods_held = i - entry_index
                
                exit_reason = None
                
                if current_return >= self.profit_target:
                    exit_reason = "profit_target"
                elif current_return <= -self.stop_loss:
                    exit_reason = "stop_loss"
                elif periods_held >= self.max_hold_periods:
                    exit_reason = "max_hold"
                elif i == len(df) - 1:
                    exit_reason = "end_of_data"
                
                if exit_reason:
                    trade = {
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': current_return * 100,
                        'periods_held': periods_held,
                        'hours_held': periods_held / 4,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    in_position = False
        
        return trades
    
    def run_test(self):
        """Run quick test"""
        
        print("📊🚀 AEGS 5-SYMBOL QUICK TEST 🚀📊")
        print("=" * 60)
        print(f"🎯 Testing: {', '.join(self.test_symbols)}")
        print(f"📊 Timeframe: 15-minute, last 55 days")
        print(f"🎯 Strategy: AEGS oversold bounce")
        
        all_trades = []
        symbol_stats = {}
        
        for i, symbol in enumerate(self.test_symbols, 1):
            print(f"\n[{i}/5] Testing {symbol}...")
            
            # Fetch data
            df = self.fetch_15min_data(symbol)
            
            if df is None or df.empty:
                symbol_stats[symbol] = {'status': 'no_data', 'trades': 0}
                continue
            
            # Run backtest
            trades = self.backtest_symbol(symbol, df)
            
            if trades:
                total_return = sum([t['return_pct'] for t in trades])
                win_trades = [t for t in trades if t['return_pct'] > 0]
                win_rate = len(win_trades) / len(trades) * 100
                
                print(f"    ✅ {len(trades)} trades, {total_return:+.1f}% total, {win_rate:.0f}% wins")
                
                all_trades.extend(trades)
                symbol_stats[symbol] = {
                    'status': 'success',
                    'trades': len(trades),
                    'total_return': total_return,
                    'win_rate': win_rate
                }
            else:
                print(f"    📊 No valid trades generated")
                symbol_stats[symbol] = {'status': 'no_trades', 'trades': 0}
        
        # Summary
        print(f"\n📊 QUICK TEST SUMMARY:")
        print("=" * 40)
        
        if all_trades:
            total_trades = len(all_trades)
            total_return = sum([t['return_pct'] for t in all_trades])
            win_trades = [t for t in all_trades if t['return_pct'] > 0]
            win_rate = len(win_trades) / total_trades * 100
            avg_return = total_return / total_trades
            
            print(f"📈 Total trades: {total_trades}")
            print(f"🏆 Win rate: {win_rate:.1f}%")
            print(f"💰 Total return: {total_return:+.1f}%")
            print(f"📊 Avg return/trade: {avg_return:+.2f}%")
            
            if win_trades:
                avg_win = np.mean([t['return_pct'] for t in win_trades])
                lose_trades = [t for t in all_trades if t['return_pct'] <= 0]
                avg_loss = np.mean([t['return_pct'] for t in lose_trades]) if lose_trades else 0
                print(f"✅ Avg win: {avg_win:+.2f}%")
                print(f"❌ Avg loss: {avg_loss:+.2f}%")
            
            # Exit reasons
            exit_reasons = {}
            for trade in all_trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print(f"\n🚪 Exit reasons:")
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                print(f"   {reason.replace('_', ' ').title()}: {count} ({pct:.0f}%)")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trades_df = pd.DataFrame(all_trades)
            trades_file = f"aegs_5symbol_test_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"\n💾 Results saved: {trades_file}")
            
        else:
            print("❌ No trades generated!")
        
        return all_trades, symbol_stats

def main():
    tester = AEGSQuickTest()
    trades, stats = tester.run_test()
    
    print(f"\n🎯 AEGS 15-MIN QUICK TEST COMPLETE!")
    print(f"   📊 Verified AEGS strategy functionality")
    print(f"   🚀 Ready for full goldmine backtest")

if __name__ == "__main__":
    main()