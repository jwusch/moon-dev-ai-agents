#!/usr/bin/env python3
"""
📊 AEGS WORKING SYMBOLS COMPREHENSIVE BACKTEST 📊
Run comprehensive AEGS backtest on verified working symbols with 1m and 15m data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import yfinance as yf
from yfinance_proxy_wrapper import YFinanceProxyWrapper

class AEGSWorkingSymbolsBacktester:
    """AEGS backtest focused on working symbols with 1m and 15m data"""
    
    def __init__(self):
        self.cache_dir = "working_symbols_cache"
        self.results_dir = "aegs_working_symbols_results"
        self.create_directories()
        
        # Initialize PIA proxy wrapper
        self.proxy_wrapper = YFinanceProxyWrapper()
        print(f"🌐 PIA Proxy: {'Enabled' if self.proxy_wrapper.proxy_session else 'Disabled'}")
        
        # Known working symbols from previous tests
        self.working_symbols = [
            # Major stocks (from 5-symbol test)
            "AAPL", "TSLA", "SPY", "QQQ", "MSFT",
            # Additional major symbols likely to work
            "NVDA", "GOOGL", "AMZN", "META", "NFLX",
            "AMD", "CRM", "ADBE", "INTC", "PYPL",
            # ETFs and indices
            "IWM", "DIA", "VTI", "XLF", "XLK",
            # Volatile stocks good for AEGS
            "GME", "AMC", "PLTR", "RIVN", "LCID",
            # Cached symbols we know work
            "TLRY", "BKKT", "ACHV", "ALDX", "ATRA"
        ]
        
        # AEGS parameters for different timeframes
        self.strategies = {
            "1m_aggressive": {
                "rsi_oversold": 30,
                "bb_position": 0.15,
                "volume_surge": 2.0,
                "min_drop": 1.0,
                "profit_target": 0.06,
                "stop_loss": 0.03,
                "max_hold": 240  # 4 hours in 1-min bars
            },
            "1m_conservative": {
                "rsi_oversold": 25,
                "bb_position": 0.10,
                "volume_surge": 2.5,
                "min_drop": 1.5,
                "profit_target": 0.08,
                "stop_loss": 0.04,
                "max_hold": 180  # 3 hours
            },
            "15m_standard": {
                "rsi_oversold": 35,
                "bb_position": 0.25,
                "volume_surge": 1.5,
                "min_drop": 1.5,
                "profit_target": 0.12,
                "stop_loss": 0.06,
                "max_hold": 64  # 16 hours in 15-min bars
            },
            "15m_aggressive": {
                "rsi_oversold": 40,
                "bb_position": 0.30,
                "volume_surge": 1.8,
                "min_drop": 2.0,
                "profit_target": 0.15,
                "stop_loss": 0.08,
                "max_hold": 48  # 12 hours
            },
            "1h_standard": {
                "rsi_oversold": 40,
                "bb_position": 0.35,
                "volume_surge": 1.3,
                "min_drop": 2.5,
                "profit_target": 0.20,
                "stop_loss": 0.10,
                "max_hold": 72  # 3 days in 1-hour bars
            },
            "1h_aggressive": {
                "rsi_oversold": 45,
                "bb_position": 0.40,
                "volume_surge": 1.5,
                "min_drop": 3.0,
                "profit_target": 0.25,
                "stop_loss": 0.12,
                "max_hold": 48  # 2 days in 1-hour bars
            }
        }
        
        print(f"🎯 WORKING SYMBOLS: {len(self.working_symbols)} symbols")
        print(f"📊 STRATEGIES: {len(self.strategies)} variations")
        print(f"   {', '.join(self.strategies.keys())}")
    
    def create_directories(self):
        """Create cache and results directories"""
        
        for directory in [self.cache_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
            
            # Create subdirectories for timeframes
            for timeframe in ["1m", "15m"]:
                subdir = os.path.join(directory, timeframe)
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
    
    def fetch_timeframe_data(self, symbol, timeframe):
        """Fetch data for specific timeframe with appropriate lookback"""
        
        # Timeframe limits (yfinance constraints)
        timeframe_limits = {
            "1m": 7,      # 1-minute: 7 days max
            "15m": 55,    # 15-minute: 55 days max (stay under 60-day limit) 
            "1h": 365     # 1-hour: 1 year lookback
        }
        max_days = timeframe_limits.get(timeframe, 55)
        
        # Cache file
        cache_file = os.path.join(self.cache_dir, timeframe, f"{symbol}_{timeframe}_{max_days}d.pkl")
        
        # Check cache (valid for 6 hours)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getctime(cache_file)
            file_age_hours = file_age / 3600
            if file_age < (6 * 3600):  # 6 hours
                try:
                    df = pd.read_pickle(cache_file)
                    if df is not None and not df.empty:
                        print(f"    📦 {timeframe} Cache HIT: {len(df)} bars (cached {file_age_hours:.1f}h ago)")
                        return df
                except Exception as e:
                    print(f"    ⚠️  {timeframe} Cache corrupted: {e}")
            else:
                print(f"    📦 {timeframe} Cache EXPIRED: {file_age_hours:.1f}h old (>6h limit)")
        
        # Fetch fresh data
        print(f"    🌐 Fetching {timeframe} data ({max_days}d)...")
        
        try:
            # Use proxy wrapper for data fetching
            df = self.proxy_wrapper.fetch_with_proxy(
                symbol,
                period=f"{max_days}d",
                interval=timeframe,
                auto_adjust=True,
                prepost=True,
                timeout=30
            )
            
            if df is not None and not df.empty:
                # Save to cache
                try:
                    df.to_pickle(cache_file)
                    print(f"    ✅ {timeframe}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
                except Exception as e:
                    print(f"    ⚠️  Cache save error: {e}")
                
                # Rate limiting
                time.sleep(1.5)
                return df
            else:
                print(f"    ❌ {timeframe}: No data")
                return None
                
        except Exception as e:
            print(f"    ❌ {timeframe} error: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        
        if df is None or df.empty or len(df) < 50:
            return df
        
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
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Price changes (different lookbacks for different timeframes)
        df['Price_Change_1bar'] = df['Close'].pct_change() * 100
        df['Price_Change_4bar'] = df['Close'].pct_change(4) * 100
        df['Price_Change_8bar'] = df['Close'].pct_change(8) * 100
        
        return df
    
    def generate_aegs_signals(self, df, strategy_params):
        """Generate AEGS signals with strategy parameters"""
        
        df = self.calculate_indicators(df)
        
        if df is None or df.empty:
            return df
        
        # AEGS conditions
        df['Signal_RSI'] = df['RSI'] < strategy_params['rsi_oversold']
        df['Signal_BB'] = df['BB_Position'] < strategy_params['bb_position']
        df['Signal_Volume'] = df['Volume_Ratio'] > strategy_params['volume_surge']
        df['Signal_Drop'] = df['Price_Change_4bar'] < -strategy_params['min_drop']
        
        # Quality filters
        df['Quality_Price'] = df['Close'] > 5.0
        df['Quality_Volume'] = df['Volume'] > 50000
        df['Quality_Indicators'] = df['RSI'].notna() & df['BB_Position'].notna()
        
        # Signal count
        signal_cols = ['Signal_RSI', 'Signal_BB', 'Signal_Volume', 'Signal_Drop']
        quality_cols = ['Quality_Price', 'Quality_Volume', 'Quality_Indicators']
        
        df['Signal_Count'] = df[signal_cols].sum(axis=1)
        df['Quality_Count'] = df[quality_cols].sum(axis=1)
        
        # AEGS entry signal (3+ signals + all quality filters)
        df['AEGS_Signal'] = (df['Signal_Count'] >= 3) & (df['Quality_Count'] >= 3)
        
        return df
    
    def backtest_strategy(self, symbol, df, strategy_name, strategy_params):
        """Backtest single strategy on symbol"""
        
        if df is None or df.empty:
            return []
        
        # Generate signals
        df = self.generate_aegs_signals(df, strategy_params)
        
        if df is None or df.empty or 'AEGS_Signal' not in df.columns:
            return []
        
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
                # Check exit conditions
                current_price = row['Close']
                current_return = (current_price - entry_price) / entry_price
                periods_held = i - entry_index
                
                exit_reason = None
                
                if current_return >= strategy_params['profit_target']:
                    exit_reason = "profit_target"
                elif current_return <= -strategy_params['stop_loss']:
                    exit_reason = "stop_loss"
                elif periods_held >= strategy_params['max_hold']:
                    exit_reason = "max_hold"
                elif i == len(df) - 1:
                    exit_reason = "end_of_data"
                
                if exit_reason:
                    trade = {
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'timeframe': strategy_name.split('_')[0],
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': current_return * 100,
                        'periods_held': periods_held,
                        'exit_reason': exit_reason,
                        'entry_signal_count': df.iloc[entry_index]['Signal_Count'] if 'Signal_Count' in df.columns else 0,
                        'entry_rsi': df.iloc[entry_index]['RSI'] if 'RSI' in df.columns else 0
                    }
                    trades.append(trade)
                    
                    in_position = False
        
        return trades
    
    def run_comprehensive_backtest(self):
        """Run comprehensive backtest on all working symbols and strategies"""
        
        print(f"\n🚀 COMPREHENSIVE AEGS WORKING SYMBOLS BACKTEST")
        print("=" * 70)
        print(f"📊 Symbols: {len(self.working_symbols)}")
        print(f"🎯 Strategies: {len(self.strategies)}")
        print(f"📈 Timeframes: 1-minute (7 days), 15-minute (55 days), 1-hour (365 days)")
        
        all_trades = []
        symbol_results = {}
        
        start_time = time.time()
        
        for i, symbol in enumerate(self.working_symbols, 1):
            print(f"\n[{i:2}/{len(self.working_symbols)}] Testing {symbol}...")
            
            symbol_results[symbol] = {}
            
            # Fetch data for all timeframes
            data_1m = self.fetch_timeframe_data(symbol, "1m")
            data_15m = self.fetch_timeframe_data(symbol, "15m")
            data_1h = self.fetch_timeframe_data(symbol, "1h")
            
            # Test all strategies
            for strategy_name, params in self.strategies.items():
                timeframe = strategy_name.split('_')[0]
                
                # Select appropriate data
                if timeframe == "1m":
                    df = data_1m
                elif timeframe == "15m":
                    df = data_15m
                elif timeframe == "1h":
                    df = data_1h
                else:
                    continue
                
                if df is None or df.empty:
                    symbol_results[symbol][strategy_name] = {'status': 'no_data', 'trades': 0}
                    continue
                
                # Run backtest
                trades = self.backtest_strategy(symbol, df.copy(), strategy_name, params)
                
                if trades:
                    total_return = sum([t['return_pct'] for t in trades])
                    win_trades = [t for t in trades if t['return_pct'] > 0]
                    win_rate = len(win_trades) / len(trades) * 100
                    
                    all_trades.extend(trades)
                    symbol_results[symbol][strategy_name] = {
                        'status': 'success',
                        'trades': len(trades),
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'avg_return': total_return / len(trades)
                    }
                    
                    print(f"    🎯 {strategy_name}: {len(trades)} trades, {total_return:+.1f}%, {win_rate:.0f}% wins")
                else:
                    symbol_results[symbol][strategy_name] = {'status': 'no_trades', 'trades': 0}
            
            # Rate limiting between symbols
            if i < len(self.working_symbols):
                time.sleep(2)
        
        backtest_time = time.time() - start_time
        return all_trades, symbol_results, backtest_time
    
    def analyze_comprehensive_results(self, all_trades, symbol_results, backtest_time):
        """Analyze comprehensive backtest results"""
        
        print(f"\n📊 COMPREHENSIVE AEGS BACKTEST COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Backtest time: {backtest_time/60:.1f} minutes")
        print(f"📊 Symbols tested: {len(symbol_results)}")
        print(f"🎯 Total trades: {len(all_trades)}")
        
        if not all_trades:
            print("❌ No trades generated!")
            return
        
        # Analyze by strategy
        print(f"\n📈 STRATEGY PERFORMANCE:")
        
        for strategy_name in self.strategies.keys():
            strategy_trades = [t for t in all_trades if t['strategy'] == strategy_name]
            
            if not strategy_trades:
                print(f"   {strategy_name}: No trades")
                continue
            
            total_return = sum([t['return_pct'] for t in strategy_trades])
            win_trades = [t for t in strategy_trades if t['return_pct'] > 0]
            win_rate = len(win_trades) / len(strategy_trades) * 100
            avg_return = total_return / len(strategy_trades)
            
            avg_win = np.mean([t['return_pct'] for t in win_trades]) if win_trades else 0
            lose_trades = [t for t in strategy_trades if t['return_pct'] <= 0]
            avg_loss = np.mean([t['return_pct'] for t in lose_trades]) if lose_trades else 0
            
            print(f"   {strategy_name}:")
            print(f"      Trades: {len(strategy_trades)}, Win Rate: {win_rate:.1f}%")
            print(f"      Total Return: {total_return:+.1f}%, Avg: {avg_return:+.2f}%")
            print(f"      Avg Win: {avg_win:+.2f}%, Avg Loss: {avg_loss:+.2f}%")
        
        # Analyze by timeframe
        print(f"\n📊 TIMEFRAME PERFORMANCE:")
        
        for timeframe in ["1m", "15m", "1h"]:
            tf_trades = [t for t in all_trades if t['timeframe'] == timeframe]
            
            if tf_trades:
                total_return = sum([t['return_pct'] for t in tf_trades])
                win_rate = len([t for t in tf_trades if t['return_pct'] > 0]) / len(tf_trades) * 100
                avg_return = total_return / len(tf_trades)
                
                print(f"   {timeframe}: {len(tf_trades)} trades, {win_rate:.1f}% wins, {total_return:+.1f}% total, {avg_return:+.2f}% avg")
        
        # Top performing symbols
        symbol_performance = {}
        for symbol in self.working_symbols:
            symbol_trades = [t for t in all_trades if t['symbol'] == symbol]
            if symbol_trades:
                total_return = sum([t['return_pct'] for t in symbol_trades])
                symbol_performance[symbol] = total_return
        
        top_symbols = sorted(symbol_performance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP PERFORMING SYMBOLS:")
        for i, (symbol, return_pct) in enumerate(top_symbols[:10], 1):
            symbol_trade_count = len([t for t in all_trades if t['symbol'] == symbol])
            print(f"   {i:2}. {symbol}: {return_pct:+6.1f}% ({symbol_trade_count} trades)")
        
        return {
            'total_trades': len(all_trades),
            'total_return': sum([t['return_pct'] for t in all_trades]),
            'win_rate': len([t for t in all_trades if t['return_pct'] > 0]) / len(all_trades) * 100,
            'strategies_tested': len(self.strategies),
            'symbols_tested': len(self.working_symbols)
        }
    
    def save_results(self, all_trades, symbol_results, summary_stats):
        """Save comprehensive results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_file = os.path.join(self.results_dir, f"aegs_comprehensive_trades_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Trades saved: {trades_file}")
        
        # Save summary
        summary = {
            'backtest_date': datetime.now().isoformat(),
            'working_symbols': self.working_symbols,
            'strategies': self.strategies,
            'summary_statistics': summary_stats,
            'symbol_results': symbol_results
        }
        
        summary_file = os.path.join(self.results_dir, f"aegs_comprehensive_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Summary saved: {summary_file}")
        
        return trades_file if all_trades else None, summary_file

def main():
    """Execute comprehensive working symbols backtest"""
    
    print("📊🚀 AEGS WORKING SYMBOLS COMPREHENSIVE BACKTEST 🚀📊")
    print("=" * 80)
    print("🎯 Testing multiple AEGS strategies on verified working symbols")
    print("⚡ 1-minute, 15-minute, and 1-hour timeframes with optimized parameters")
    print("💎 Known working symbols from previous successful tests")
    
    # Initialize backtester
    backtester = AEGSWorkingSymbolsBacktester()
    
    # Run comprehensive backtest
    all_trades, symbol_results, backtest_time = backtester.run_comprehensive_backtest()
    
    # Analyze results
    summary_stats = backtester.analyze_comprehensive_results(all_trades, symbol_results, backtest_time)
    
    # Save results
    if summary_stats:
        trades_file, summary_file = backtester.save_results(all_trades, symbol_results, summary_stats)
        
        print(f"\n🎯 COMPREHENSIVE AEGS BACKTEST COMPLETE!")
        print(f"   📊 {summary_stats['total_trades']} trades across {summary_stats['strategies_tested']} strategies")
        print(f"   💰 Total return: {summary_stats['total_return']:+.1f}%")
        print(f"   🏆 Win rate: {summary_stats['win_rate']:.1f}%")
        print(f"   📁 Files: {summary_file}")

if __name__ == "__main__":
    main()