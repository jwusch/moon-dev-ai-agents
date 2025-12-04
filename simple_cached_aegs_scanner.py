#!/usr/bin/env python3
"""
🔥💎 SIMPLE CACHED AEGS SCANNER 💎🔥
Fresh cache system to avoid rate limiting
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import time
import json
import os
import pickle
import hashlib
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

class SimpleCacheAEGS:
    """Simple caching system for AEGS scanning"""
    
    def __init__(self, cache_dir="simple_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cache_key(self, symbol, period="730d", interval="1d"):
        """Generate cache key"""
        key_string = f"{symbol}_{period}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_data(self, symbol, period="730d", interval="1d"):
        """Get data from cache or download"""
        cache_key = self.get_cache_key(symbol, period, interval)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                # Check if cache is recent (less than 1 day old)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 86400:  # 24 hours
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    
                    print(f"📦 Cache HIT: {symbol} ({len(df)} bars)")
                    self.cache_stats['hits'] += 1
                    return df
            except Exception as e:
                print(f"Cache read error for {symbol}: {e}")
        
        # Download fresh data
        print(f"🌐 Cache MISS - Downloading: {symbol}")
        self.cache_stats['misses'] += 1
        
        try:
            # Add small delay to be nice to yfinance
            time.sleep(0.1)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if len(df) > 0:
                # Cache the data
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"💾 Cached {symbol}: {len(df)} bars")
            
            return df
            
        except Exception as e:
            print(f"❌ Download failed for {symbol}: {e}")
            return None
    
    def backtest_aegs_strategy(self, df):
        """Backtest AEGS strategy"""
        if df is None or len(df) < 100:
            return None
            
        # Calculate indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bbands = ta.bbands(df['Close'], length=20)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0'] 
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        
        trades = []
        position = None
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Entry signal: AEGS criteria
            if position is None:
                # Calculate BB position
                bb_range = current['BB_Upper'] - current['BB_Lower']
                if bb_range > 0:
                    bb_position = (current['Close'] - current['BB_Lower']) / bb_range
                else:
                    bb_position = 0.5
                
                # Volume surge
                volume_ratio = current['Volume'] / current['Volume_SMA'] if current['Volume_SMA'] > 0 else 0
                
                # Daily drop
                if prev['Close'] > 0:
                    daily_change = (current['Close'] - prev['Close']) / prev['Close']
                else:
                    daily_change = 0
                
                # AEGS entry criteria
                if (current['RSI'] < 30 and 
                    bb_position < 0.2 and
                    volume_ratio > 2.0 and 
                    daily_change < -0.05):
                    
                    position = {
                        'entry_date': current.name,
                        'entry_price': current['Close'],
                        'entry_index': i
                    }
            
            # Exit logic
            elif position is not None:
                days_held = i - position['entry_index']
                current_price = current['Close']
                entry_price = position['entry_price']
                return_pct = (current_price - entry_price) / entry_price
                
                exit_reason = None
                
                # 30% profit target
                if return_pct >= 0.30:
                    exit_reason = "Profit Target 30%"
                # -20% stop loss
                elif return_pct <= -0.20:
                    exit_reason = "Stop Loss 20%"
                # 30-day profitable exit
                elif days_held >= 30 and return_pct > 0:
                    exit_reason = "Time Exit (Profitable)"
                # 60-day force exit
                elif days_held >= 60:
                    exit_reason = "Force Exit"
                
                if exit_reason:
                    trades.append({
                        'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': current.name.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct * 100,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    position = None
        
        if not trades:
            return None
        
        # Calculate strategy performance
        returns = [t['return_pct'] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        total_return = 1.0
        for ret in returns:
            total_return *= (1 + ret/100)
        
        strategy_return = (total_return - 1) * 100
        
        # Buy and hold return
        buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
        
        excess_return = strategy_return - buy_hold_return
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': excess_return,
            'trades': trades
        }
    
    def scan_symbol(self, symbol):
        """Scan single symbol"""
        try:
            # Get cached data
            df = self.get_cached_data(symbol)
            
            if df is None or len(df) < 100:
                return None
            
            # Price filter
            current_price = df.iloc[-1]['Close']
            if current_price < 1.0:
                return None
            
            # Backtest
            result = self.backtest_aegs_strategy(df.copy())
            
            if result and result['strategy_return'] > 0:
                result['symbol'] = symbol
                result['current_price'] = current_price
                return result
            
            return None
            
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            return None

def main():
    """Test D-Z cached scanning"""
    print("🔥💎 TESTING CACHED D-Z AEGS SCAN 💎🔥")
    print("=" * 50)
    
    # Initialize cache
    cache = SimpleCacheAEGS()
    
    # Sample D-Z symbols to test
    test_symbols = [
        'DASH', 'DDOG', 'DOCU', 'DKNG', 'DLTR',  # D symbols
        'EBAY', 'EQIX', 'ETSY', 'EXPE', 'ENPH',  # E symbols  
        'FTNT', 'FISV', 'FANG', 'FAST', 'FSLR',  # F symbols
        'GILD', 'GOLD', 'GOOGL', 'GOOG', 'GRUB', # G symbols
        'HUBS', 'HZNP', 'HELE', 'HMSY', 'HTHT',  # H symbols
        'INTC', 'IDXX', 'ILMN', 'INCY', 'ISRG',  # I symbols
    ]
    
    print(f"🎯 Testing {len(test_symbols)} D-I symbols")
    print("👀 Watch for cache hits vs downloads")
    
    profitable_symbols = []
    start_time = time.time()
    
    for i, symbol in enumerate(test_symbols, 1):
        print(f"\n[{i:2}/{len(test_symbols)}] Processing {symbol}...")
        
        result = cache.scan_symbol(symbol)
        
        if result:
            profitable_symbols.append(result)
            ret = result['strategy_return']
            trades = result['total_trades']
            win_rate = result['win_rate']
            print(f"    ✅ PROFITABLE: +{ret:.1f}% ({trades} trades, {win_rate:.0f}% wins)")
        else:
            print(f"    ❌ Not profitable")
    
    # Results
    scan_time = time.time() - start_time
    
    print(f"\n🎉 SCAN COMPLETE!")
    print(f"⏱️  Total time: {scan_time:.1f}s")
    print(f"📊 Cache stats: {cache.cache_stats['hits']} hits, {cache.cache_stats['misses']} misses")
    print(f"💰 Profitable symbols: {len(profitable_symbols)}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cached_test_scan_{timestamp}.json"
    
    output = {
        'scan_date': datetime.now().isoformat(),
        'symbols_tested': test_symbols,
        'cache_stats': cache.cache_stats,
        'profitable_count': len(profitable_symbols),
        'scan_time_seconds': scan_time,
        'profitable_symbols': profitable_symbols
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")
    
    # Show top performers
    if profitable_symbols:
        profitable_symbols.sort(key=lambda x: x['strategy_return'], reverse=True)
        print(f"\n🏆 Top performers:")
        for i, result in enumerate(profitable_symbols[:5], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            print(f"   {i}. {symbol:<6} +{ret:6.1f}% ({trades} trades)")
    
    # Test second run (should be all cache hits)
    print(f"\n🔄 TESTING SECOND RUN (should be all cache hits)...")
    
    cache2 = SimpleCacheAEGS()
    for symbol in test_symbols[:5]:  # Just test first 5
        print(f"Testing {symbol}...")
        cache2.get_cached_data(symbol)
    
    print(f"📊 Second run cache stats: {cache2.cache_stats['hits']} hits, {cache2.cache_stats['misses']} misses")
    
    if cache2.cache_stats['hits'] == 5 and cache2.cache_stats['misses'] == 0:
        print("✅ CACHING WORKING PERFECTLY!")
    else:
        print("⚠️  Caching issue detected")

if __name__ == "__main__":
    main()