#!/usr/bin/env python3
"""
📊 AEGS CACHED SYMBOLS BACKTEST 📊
Run AEGS backtest only on symbols with cached 15-minute data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import glob

class AEGSCachedBacktester:
    """AEGS backtest focused on available cached data"""
    
    def __init__(self):
        self.cache_dir = "intraday_15min_cache"
        self.results_dir = "aegs_15min_backtest_results"
        
        # AEGS Strategy Parameters optimized for 15-min
        self.rsi_oversold = 35
        self.bb_position_threshold = 0.25
        self.volume_surge_threshold = 1.5  # Lower for 15-min volatility
        self.min_drop_threshold = 1.5  # 1.5% drop (more sensitive)
        
        # Exit rules (aggressive for intraday)
        self.profit_target = 0.12  # 12% profit target
        self.stop_loss = 0.06  # 6% stop loss
        self.max_hold_periods = 64  # 16 hours max
        
        print(f"🎯 AEGS 15-MIN STRATEGY (OPTIMIZED):")
        print(f"   RSI Oversold: <{self.rsi_oversold}")
        print(f"   BB Position: <{self.bb_position_threshold}")
        print(f"   Volume Surge: >{self.volume_surge_threshold}x")
        print(f"   Min Drop: >{self.min_drop_threshold}%")
        print(f"   Profit Target: {self.profit_target*100}%")
        print(f"   Stop Loss: {self.stop_loss*100}%")
        print(f"   Max Hold: {self.max_hold_periods} bars ({self.max_hold_periods/4:.0f} hours)")
    
    def get_cached_symbols(self):
        """Get list of symbols with cached data"""
        
        cache_files = glob.glob(os.path.join(self.cache_dir, "*_15min_*.pkl"))
        symbols = []
        
        for cache_file in cache_files:
            filename = os.path.basename(cache_file)
            symbol = filename.split('_15min_')[0]
            symbols.append(symbol)
        
        symbols = list(set(symbols))  # Remove duplicates
        symbols.sort()
        
        print(f"📦 CACHED SYMBOLS FOUND: {len(symbols)} symbols")
        for i, symbol in enumerate(symbols):
            print(f"   {i+1:2}. {symbol}")
        
        return symbols
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        
        # RSI calculation
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            return bb_position, upper_band, lower_band
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['BB_Position'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Price change indicators
        df['Price_Change_1bar'] = df['Close'].pct_change() * 100
        df['Price_Change_4bar'] = df['Close'].pct_change(4) * 100
        df['Price_Change_8bar'] = df['Close'].pct_change(8) * 100
        
        # Volatility
        df['High_Low_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
        
        return df
    
    def generate_aegs_signals(self, df):
        """Generate AEGS signals with multiple strength levels"""
        
        if df is None or df.empty or len(df) < 50:
            return df
        
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        
        # AEGS conditions
        df['Condition_RSI'] = df['RSI'] < self.rsi_oversold
        df['Condition_BB'] = df['BB_Position'] < self.bb_position_threshold
        df['Condition_Volume'] = df['Volume_Ratio'] > self.volume_surge_threshold
        df['Condition_Drop'] = df['Price_Change_4bar'] < -self.min_drop_threshold
        df['Condition_Recent_Drop'] = df['Price_Change_1bar'] < -0.5  # Recent 15-min drop
        
        # Additional quality filters
        df['Condition_Price'] = df['Close'] > 1.0  # Minimum price
        df['Condition_Volume_Min'] = df['Volume'] > 10000  # Minimum volume
        df['Condition_Volatility'] = df['High_Low_Pct'] > 1.0  # Minimum volatility
        
        # Signal strength levels
        core_conditions = ['Condition_RSI', 'Condition_BB', 'Condition_Volume', 'Condition_Drop']
        df['Core_Signals'] = df[core_conditions].sum(axis=1)
        
        quality_conditions = ['Condition_Price', 'Condition_Volume_Min', 'Condition_Volatility']
        df['Quality_Signals'] = df[quality_conditions].sum(axis=1)
        
        # Different signal strengths
        df['AEGS_Strong'] = (df['Core_Signals'] >= 4) & (df['Quality_Signals'] >= 3)  # All conditions
        df['AEGS_Medium'] = (df['Core_Signals'] >= 3) & (df['Quality_Signals'] >= 2)  # 3/4 core + quality
        df['AEGS_Weak'] = (df['Core_Signals'] >= 2) & (df['Quality_Signals'] >= 2)    # 2/4 core + quality
        
        # Final signal (start with medium strength)
        df['AEGS_Signal'] = df['AEGS_Medium']
        
        return df
    
    def backtest_single_symbol(self, symbol, df, signal_strength='AEGS_Medium'):
        """Backtest AEGS on single symbol with specified signal strength"""
        
        if df is None or df.empty:
            return []
        
        # Generate signals
        df = self.generate_aegs_signals(df)
        
        if signal_strength not in df.columns:
            return []
        
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_index = 0
        entry_signal_count = 0
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            
            if not in_position and row[signal_strength]:
                # Enter position
                in_position = True
                entry_price = row['Close']
                entry_time = timestamp
                entry_index = i
                entry_signal_count = row['Core_Signals']
                
            elif in_position:
                # Check exit conditions
                current_price = row['Close']
                current_return = (current_price - entry_price) / entry_price
                periods_held = i - entry_index
                
                exit_reason = None
                
                # Profit target
                if current_return >= self.profit_target:
                    exit_reason = "profit_target"
                
                # Stop loss
                elif current_return <= -self.stop_loss:
                    exit_reason = "stop_loss"
                
                # Max hold period
                elif periods_held >= self.max_hold_periods:
                    exit_reason = "max_hold"
                
                # End of data
                elif i == len(df) - 1:
                    exit_reason = "end_of_data"
                
                if exit_reason:
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': current_return * 100,
                        'periods_held': periods_held,
                        'hours_held': periods_held / 4,  # 15-min bars to hours
                        'exit_reason': exit_reason,
                        'entry_signal_count': entry_signal_count,
                        'entry_rsi': df.iloc[entry_index]['RSI'] if 'RSI' in df.columns else 0,
                        'entry_bb_position': df.iloc[entry_index]['BB_Position'] if 'BB_Position' in df.columns else 0,
                        'entry_volume_ratio': df.iloc[entry_index]['Volume_Ratio'] if 'Volume_Ratio' in df.columns else 0
                    }
                    trades.append(trade)
                    
                    in_position = False
                    entry_price = 0
                    entry_time = None
                    entry_index = 0
        
        return trades
    
    def load_cached_data(self, symbol):
        """Load the most recent cached data for symbol"""
        
        cache_pattern = os.path.join(self.cache_dir, f"{symbol}_15min_*.pkl")
        cache_files = glob.glob(cache_pattern)
        
        if not cache_files:
            return None
        
        # Use most recent file
        latest_cache = max(cache_files, key=os.path.getctime)
        
        try:
            df = pd.read_pickle(latest_cache)
            return df
        except Exception as e:
            print(f"❌ Cache read error {symbol}: {e}")
            return None
    
    def run_comprehensive_backtest(self, symbols):
        """Run backtest with multiple signal strengths"""
        
        print(f"\n🚀 COMPREHENSIVE AEGS CACHED BACKTEST")
        print("=" * 60)
        print(f"📊 Testing {len(symbols)} cached symbols")
        print("🎯 Multiple signal strength levels")
        
        all_results = {
            'AEGS_Strong': {'trades': [], 'symbol_stats': {}},
            'AEGS_Medium': {'trades': [], 'symbol_stats': {}},
            'AEGS_Weak': {'trades': [], 'symbol_stats': {}}
        }
        
        start_time = time.time()
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i:2}/{len(symbols)}] Backtesting {symbol}...")
            
            # Load cached data
            df = self.load_cached_data(symbol)
            
            if df is None or df.empty:
                print(f"    ❌ No valid cache data")
                continue
            
            print(f"    📊 Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            
            # Test all signal strengths
            symbol_summary = {}
            
            for strength in ['AEGS_Strong', 'AEGS_Medium', 'AEGS_Weak']:
                trades = self.backtest_single_symbol(symbol, df.copy(), strength)
                
                if trades:
                    total_return = sum([t['return_pct'] for t in trades])
                    win_trades = [t for t in trades if t['return_pct'] > 0]
                    win_rate = len(win_trades) / len(trades) * 100
                    
                    all_results[strength]['trades'].extend(trades)
                    all_results[strength]['symbol_stats'][symbol] = {
                        'trades': len(trades),
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'avg_return': total_return / len(trades)
                    }
                    
                    symbol_summary[strength] = f"{len(trades)} trades, {total_return:+.1f}%, {win_rate:.0f}% wins"
                else:
                    symbol_summary[strength] = "No trades"
            
            # Show symbol summary
            print(f"    🎯 Strong: {symbol_summary.get('AEGS_Strong', 'No trades')}")
            print(f"    📈 Medium: {symbol_summary.get('AEGS_Medium', 'No trades')}")
            print(f"    📊 Weak: {symbol_summary.get('AEGS_Weak', 'No trades')}")
        
        backtest_time = time.time() - start_time
        return all_results, backtest_time
    
    def analyze_comprehensive_results(self, all_results, backtest_time):
        """Analyze results across all signal strengths"""
        
        print(f"\n📊 COMPREHENSIVE AEGS BACKTEST COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Backtest time: {backtest_time:.1f} seconds")
        
        for strength in ['AEGS_Strong', 'AEGS_Medium', 'AEGS_Weak']:
            trades = all_results[strength]['trades']
            
            print(f"\n🎯 {strength.replace('_', ' ').upper()} RESULTS:")
            
            if not trades:
                print("   ❌ No trades generated")
                continue
            
            # Calculate statistics
            total_trades = len(trades)
            win_trades = [t for t in trades if t['return_pct'] > 0]
            total_return = sum([t['return_pct'] for t in trades])
            win_rate = len(win_trades) / total_trades * 100
            avg_return = total_return / total_trades
            
            avg_win = np.mean([t['return_pct'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t['return_pct'] for t in trades if t['return_pct'] < 0])
            avg_hold_hours = np.mean([t['hours_held'] for t in trades])
            
            print(f"   📈 Total trades: {total_trades}")
            print(f"   🏆 Win rate: {win_rate:.1f}% ({len(win_trades)}/{total_trades})")
            print(f"   💰 Total return: {total_return:+.1f}%")
            print(f"   📊 Avg return/trade: {avg_return:+.2f}%")
            print(f"   ✅ Avg win: {avg_win:+.2f}%")
            print(f"   ❌ Avg loss: {avg_loss:+.2f}%")
            print(f"   ⏱️  Avg hold time: {avg_hold_hours:.1f} hours")
            
            # Exit reasons
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print(f"   🚪 Exit reasons:")
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                print(f"      {reason.replace('_', ' ').title()}: {count} ({pct:.0f}%)")
            
            # Top symbols
            symbol_stats = all_results[strength]['symbol_stats']
            profitable_symbols = {k: v for k, v in symbol_stats.items() if v['total_return'] > 0}
            top_symbols = sorted(profitable_symbols.items(), key=lambda x: x[1]['total_return'], reverse=True)
            
            if top_symbols:
                print(f"   🏆 Top performers:")
                for sym, stats in top_symbols[:5]:
                    print(f"      {sym}: {stats['total_return']:+.1f}% ({stats['trades']} trades)")
        
        return all_results

def main():
    """Execute comprehensive cached backtest"""
    
    print("📊💎 AEGS CACHED SYMBOLS COMPREHENSIVE BACKTEST 💎📊")
    print("=" * 80)
    print("🎯 Testing multiple AEGS signal strengths on all cached data")
    print("⚡ Optimized parameters for 15-minute intraday trading")
    
    backtester = AEGSCachedBacktester()
    
    # Get symbols with cached data
    cached_symbols = backtester.get_cached_symbols()
    
    if not cached_symbols:
        print("❌ No cached symbols found!")
        return
    
    # Run comprehensive backtest
    all_results, backtest_time = backtester.run_comprehensive_backtest(cached_symbols)
    
    # Analyze results
    final_results = backtester.analyze_comprehensive_results(all_results, backtest_time)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save all trades for each strength level
    for strength in ['AEGS_Strong', 'AEGS_Medium', 'AEGS_Weak']:
        trades = all_results[strength]['trades']
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_file = f"aegs_cached_{strength.lower()}_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Saved {strength}: {trades_file} ({len(trades)} trades)")
    
    print(f"\n🎯 COMPREHENSIVE BACKTEST COMPLETE!")
    print(f"   📊 Tested {len(cached_symbols)} symbols with cached 15-min data")
    print(f"   ⚡ Optimized AEGS parameters for intraday performance")
    print(f"   💾 Results saved with timestamp {timestamp}")

if __name__ == "__main__":
    main()