#!/usr/bin/env python3
"""
📊 AEGS 15-MIN GOLDMINE BACKTESTER 📊
Backtest AEGS strategy on all goldmine symbols using cached 15-minute data
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from pathlib import Path

class AEGS15MinBacktester:
    """AEGS strategy backtester for 15-minute intraday data"""
    
    def __init__(self):
        self.cache_dir = "intraday_15min_cache"
        self.results_dir = "aegs_15min_backtest_results"
        self.create_results_directory()
        
        # AEGS Strategy Parameters for 15-min timeframe
        self.rsi_oversold = 35  # Slightly higher for 15-min noise
        self.bb_position_threshold = 0.25  # Bottom 25% of BB range
        self.volume_surge_threshold = 1.8  # 80% above average volume
        self.min_drop_threshold = 2.0  # 2% drop in recent periods
        
        # Exit rules
        self.profit_target = 0.15  # 15% profit target (more conservative for 15-min)
        self.stop_loss = 0.08  # 8% stop loss (tighter for 15-min)
        self.max_hold_periods = 96  # 24 hours (96 x 15-min bars)
        
        print(f"🎯 AEGS 15-MIN STRATEGY PARAMETERS:")
        print(f"   RSI Oversold: <{self.rsi_oversold}")
        print(f"   BB Position: <{self.bb_position_threshold}")
        print(f"   Volume Surge: >{self.volume_surge_threshold}x")
        print(f"   Min Drop: >{self.min_drop_threshold}%")
        print(f"   Profit Target: {self.profit_target*100}%")
        print(f"   Stop Loss: {self.stop_loss*100}%")
        print(f"   Max Hold: {self.max_hold_periods} bars (24 hours)")
    
    def create_results_directory(self):
        """Create results directory"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"📁 Created results directory: {self.results_dir}")
        else:
            print(f"📁 Using existing results directory: {self.results_dir}")
    
    def load_goldmine_symbols(self):
        """Load all AEGS goldmine symbols"""
        
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            all_symbols = []
            for category in registry['goldmine_symbols']:
                symbols = list(registry['goldmine_symbols'][category].keys())
                # Filter out invalid symbols
                valid_symbols = [s for s in symbols if len(s) <= 5 and s.replace('.', '').replace('-', '').isalpha()]
                all_symbols.extend(valid_symbols)
            
            # Remove duplicates
            unique_symbols = list(set(all_symbols))
            unique_symbols.sort()
            
            print(f"💎 GOLDMINE SYMBOLS LOADED: {len(unique_symbols)} symbols")
            return unique_symbols
            
        except Exception as e:
            print(f"❌ Error loading goldmine symbols: {e}")
            return []
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # BB position: 0 = at lower band, 1 = at upper band
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return bb_position, upper_band, lower_band
    
    def generate_aegs_signals(self, df):
        """Generate AEGS entry signals for 15-minute data"""
        
        if df is None or df.empty or len(df) < 50:
            return pd.DataFrame()
        
        # Calculate indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['BB_Position'], df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Recent price drop analysis (last 4 bars = 1 hour)
        df['Price_Change_4bar'] = df['Close'].pct_change(4) * 100
        
        # AEGS entry conditions
        conditions = {
            'rsi_oversold': df['RSI'] < self.rsi_oversold,
            'bb_oversold': df['BB_Position'] < self.bb_position_threshold,
            'volume_surge': df['Volume_Ratio'] > self.volume_surge_threshold,
            'price_drop': df['Price_Change_4bar'] < -self.min_drop_threshold
        }
        
        # Signal when at least 3 out of 4 conditions are met
        df['Signal_Count'] = sum(conditions.values())
        df['AEGS_Signal'] = df['Signal_Count'] >= 3
        
        # Additional filters
        df['Valid_Signal'] = (
            df['AEGS_Signal'] & 
            (df['Volume'] > 1000) &  # Minimum volume
            (df['Close'] > 1.0) &    # Minimum price
            df['RSI'].notna() & 
            df['BB_Position'].notna()
        )
        
        return df
    
    def backtest_symbol(self, symbol, df):
        """Backtest AEGS strategy on a single symbol"""
        
        if df is None or df.empty:
            return None
        
        # Generate signals
        df_signals = self.generate_aegs_signals(df)
        
        if df_signals.empty or 'Valid_Signal' not in df_signals.columns:
            return None
        
        trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        entry_index = 0
        
        for i, (timestamp, row) in enumerate(df_signals.iterrows()):
            
            if not in_position and row['Valid_Signal']:
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
                elif i == len(df_signals) - 1:
                    exit_reason = "end_of_data"
                
                if exit_reason:
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': current_return * 100,
                        'periods_held': periods_held,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    in_position = False
                    entry_price = 0
                    entry_time = None
                    entry_index = 0
        
        return trades
    
    def load_or_fetch_data(self, symbol, days_back=55):
        """Load cached data or fetch if not available - returns sufficient data for backtesting (55 days max for 15-min data)"""
        
        import yfinance as yf
        
        # Check for cached backtest data (longer timeframe)
        backtest_cache_file = os.path.join(self.cache_dir, f"{symbol}_backtest_15min_{days_back}d.pkl")
        
        if os.path.exists(backtest_cache_file):
            try:
                df = pd.read_pickle(backtest_cache_file)
                if df is not None and not df.empty and len(df) > 100:
                    print(f"📦 Cache HIT: {symbol} ({len(df)} bars, {days_back}d)")
                    return df
            except Exception as e:
                print(f"⚠️  Cache read error: {e}")
        
        # Cache miss - fetch data for backtesting
        print(f"🌐 Cache MISS - Fetching {days_back}d of 15min data: {symbol}")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="15m",
                auto_adjust=True,
                prepost=True,
                timeout=30
            )
            
            if df is not None and not df.empty:
                # Save to backtest cache
                try:
                    df.to_pickle(backtest_cache_file)
                    print(f"💾 Cached backtest data: {symbol} ({len(df)} bars)")
                except Exception as e:
                    print(f"⚠️  Cache save error: {e}")
                
                print(f"✅ Fetched: {symbol} ({len(df)} bars, {df.index[0].date()} to {df.index[-1].date()})")
                
                # Rate limiting after successful fetch
                print(f"    ⏳ Rate limit delay: 3.0s")
                time.sleep(3.0)
                
                return df
            else:
                print(f"❌ No data returned for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Fetch error {symbol}: {e}")
            # Still add delay on error to be safe
            time.sleep(2.0)
            return None
    
    def run_goldmine_backtest(self, symbols, max_symbols=None):
        """Run AEGS backtest on all goldmine symbols"""
        
        print(f"\n🚀 AEGS 15-MIN GOLDMINE BACKTEST STARTING")
        print("=" * 70)
        
        if max_symbols:
            symbols = symbols[:max_symbols]
            print(f"🎯 Testing first {max_symbols} symbols for speed")
        
        print(f"📊 Total symbols: {len(symbols)}")
        print(f"📁 Cache directory: {self.cache_dir}")
        
        all_trades = []
        symbol_results = {}
        
        start_time = time.time()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\n[{i:3}/{len(symbols)}] Backtesting {symbol}...")
                
                # Load cached data or fetch if needed
                df = self.load_or_fetch_data(symbol, days_back=55)
                
                if df is None:
                    print(f"    ❌ No data available (cache miss + fetch failed)")
                    symbol_results[symbol] = {'status': 'no_data', 'trades': 0, 'total_return': 0}
                    continue
                
                # Run backtest
                trades = self.backtest_symbol(symbol, df)
                
                if trades is None or len(trades) == 0:
                    print(f"    📊 No valid trades generated")
                    symbol_results[symbol] = {'status': 'no_trades', 'trades': 0, 'total_return': 0}
                    continue
                
                # Analyze trades
                total_return = sum([trade['return_pct'] for trade in trades])
                win_trades = [t for t in trades if t['return_pct'] > 0]
                win_rate = len(win_trades) / len(trades) * 100 if trades else 0
                
                print(f"    ✅ {len(trades)} trades, {total_return:+.1f}% total, {win_rate:.1f}% wins")
                
                # Store results
                all_trades.extend(trades)
                symbol_results[symbol] = {
                    'status': 'success',
                    'trades': len(trades),
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'avg_return': total_return / len(trades) if trades else 0
                }
                
                # Progress update every 25 symbols
                if i % 25 == 0:
                    elapsed = time.time() - start_time
                    processed = i
                    successful = len([s for s in symbol_results.values() if s['status'] == 'success'])
                    total_trades = len(all_trades)
                    
                    print(f"\n📊 BACKTEST PROGRESS:")
                    print(f"   Processed: {processed}/{len(symbols)} ({processed/len(symbols)*100:.1f}%)")
                    print(f"   Successful: {successful}")
                    print(f"   Total trades: {total_trades}")
                    print(f"   Elapsed: {elapsed/60:.1f} minutes")
                    
                    if processed > 0:
                        rate = processed / elapsed
                        eta = (len(symbols) - processed) / rate
                        print(f"   ETA: {eta/60:.1f} minutes")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                symbol_results[symbol] = {'status': 'error', 'trades': 0, 'total_return': 0}
        
        backtest_time = time.time() - start_time
        
        return all_trades, symbol_results, backtest_time
    
    def analyze_backtest_results(self, all_trades, symbol_results, backtest_time):
        """Analyze complete backtest results"""
        
        print(f"\n📊 AEGS 15-MIN GOLDMINE BACKTEST COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Backtest time: {backtest_time/60:.1f} minutes")
        print(f"📊 Symbols processed: {len(symbol_results)}")
        
        # Symbol-level analysis
        successful_symbols = {k: v for k, v in symbol_results.items() if v['status'] == 'success'}
        no_data_count = len([v for v in symbol_results.values() if v['status'] == 'no_data'])
        no_trades_count = len([v for v in symbol_results.values() if v['status'] == 'no_trades'])
        
        print(f"✅ Successful backtests: {len(successful_symbols)}")
        print(f"❌ No cache data: {no_data_count}")
        print(f"📊 No valid trades: {no_trades_count}")
        
        if not all_trades:
            print("❌ No trades generated across all symbols!")
            return
        
        # Trade-level analysis
        total_trades = len(all_trades)
        win_trades = [t for t in all_trades if t['return_pct'] > 0]
        lose_trades = [t for t in all_trades if t['return_pct'] < 0]
        
        total_return = sum([t['return_pct'] for t in all_trades])
        avg_return = total_return / total_trades
        win_rate = len(win_trades) / total_trades * 100
        
        avg_win = np.mean([t['return_pct'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in lose_trades]) if lose_trades else 0
        
        print(f"\n📈 TRADE ANALYSIS:")
        print(f"   Total trades: {total_trades}")
        print(f"   Win trades: {len(win_trades)} ({win_rate:.1f}%)")
        print(f"   Lose trades: {len(lose_trades)} ({100-win_rate:.1f}%)")
        print(f"   Total return: {total_return:+.1f}%")
        print(f"   Average return per trade: {avg_return:+.2f}%")
        print(f"   Average win: {avg_win:+.2f}%")
        print(f"   Average loss: {avg_loss:+.2f}%")
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in all_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        print(f"\n🚪 EXIT REASONS:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_trades * 100
            print(f"   {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # Top performing symbols
        profitable_symbols = {k: v for k, v in successful_symbols.items() if v['total_return'] > 0}
        top_symbols = sorted(profitable_symbols.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        print(f"\n🏆 TOP 10 PERFORMING SYMBOLS:")
        for i, (symbol, data) in enumerate(top_symbols[:10], 1):
            print(f"   {i:2}. {symbol:<6}: {data['total_return']:+6.1f}% ({data['trades']} trades, {data['win_rate']:.0f}% wins)")
        
        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'successful_symbols': len(successful_symbols),
            'profitable_symbols': len(profitable_symbols)
        }
    
    def save_backtest_results(self, all_trades, symbol_results, backtest_time, summary_stats):
        """Save detailed backtest results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades CSV
        trades_file = os.path.join(self.results_dir, f"aegs_15min_trades_{timestamp}.csv")
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Trades saved: {trades_file}")
        
        # Save summary JSON
        summary_file = os.path.join(self.results_dir, f"aegs_15min_summary_{timestamp}.json")
        summary_data = {
            'backtest_date': datetime.now().isoformat(),
            'strategy': 'AEGS_15min',
            'backtest_duration_minutes': backtest_time / 60,
            'total_symbols_tested': len(symbol_results),
            'summary_statistics': summary_stats,
            'symbol_results': symbol_results,
            'strategy_parameters': {
                'rsi_oversold': self.rsi_oversold,
                'bb_position_threshold': self.bb_position_threshold,
                'volume_surge_threshold': self.volume_surge_threshold,
                'min_drop_threshold': self.min_drop_threshold,
                'profit_target': self.profit_target,
                'stop_loss': self.stop_loss,
                'max_hold_periods': self.max_hold_periods
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Summary saved: {summary_file}")
        
        return trades_file, summary_file

def main():
    """Execute AEGS 15-minute goldmine backtest"""
    
    print("📊🚀 AEGS 15-MIN GOLDMINE BACKTESTER 🚀📊")
    print("=" * 80)
    print("🎯 Running AEGS strategy backtest on cached 15-minute data")
    print("💎 Testing all goldmine symbols for optimal intraday performance")
    
    # Initialize backtester
    backtester = AEGS15MinBacktester()
    
    # Load goldmine symbols
    symbols = backtester.load_goldmine_symbols()
    
    if not symbols:
        print("❌ No goldmine symbols found!")
        return
    
    # Run backtest on first 25 symbols (with rate limiting and auto-fetch)
    all_trades, symbol_results, backtest_time = backtester.run_goldmine_backtest(
        symbols, 
        max_symbols=25  # Test 25 symbols with auto-fetch capability
    )
    
    # Analyze results
    summary_stats = backtester.analyze_backtest_results(all_trades, symbol_results, backtest_time)
    
    # Save results
    if summary_stats:
        trades_file, summary_file = backtester.save_backtest_results(
            all_trades, symbol_results, backtest_time, summary_stats
        )
        
        print(f"\n🎯 AEGS 15-MIN BACKTEST COMPLETE!")
        print(f"   📊 Results: {len(all_trades)} trades across {summary_stats['successful_symbols']} symbols")
        print(f"   💰 Total return: {summary_stats['total_return']:+.1f}%")
        print(f"   🎯 Win rate: {summary_stats['win_rate']:.1f}%")
        print(f"   📁 Files: {trades_file}, {summary_file}")

if __name__ == "__main__":
    main()