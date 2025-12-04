#!/usr/bin/env python3
"""
🔥💎 WORKING CACHED AEGS SCANNER 💎🔥
Fixed version with proper BB bands calculation
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

class WorkingCacheAEGS:
    """Working caching system for AEGS scanning with rate limiting protection"""
    
    def __init__(self, cache_dir="working_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_stats = {'hits': 0, 'misses': 0, 'rate_limited': 0, 'retries': 0}
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 500ms between requests
    
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
        
        # Download fresh data with retry backoff
        print(f"🌐 Cache MISS - Downloading: {symbol}")
        self.cache_stats['misses'] += 1
        
        # Retry with exponential backoff
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Global rate limiting - ensure minimum time between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    sleep_time = self.min_request_interval - time_since_last
                    time.sleep(sleep_time)
                
                # Progressive delay: 1s, 2s, 4s
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"⏳ Retry {attempt}/{max_retries-1} for {symbol} - waiting {delay}s...")
                    self.cache_stats['retries'] += 1
                    time.sleep(delay)
                
                # Update last request time
                self.last_request_time = time.time()
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if len(df) > 0:
                    # Cache the data
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    print(f"💾 Cached {symbol}: {len(df)} bars")
                    return df
                elif attempt == max_retries - 1:
                    print(f"📊 No data for {symbol} (possibly delisted)")
                    return None
                else:
                    print(f"⚠️  Empty data for {symbol}, retrying...")
                    continue
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limiting indicators
                if any(indicator in error_msg for indicator in ['rate limit', '429', 'too many requests', 'quota exceeded']):
                    self.cache_stats['rate_limited'] += 1
                    if attempt < max_retries - 1:
                        # Aggressive backoff for rate limits: 5s, 15s, 45s
                        backoff_delay = 5 * (3 ** attempt)
                        print(f"🚨 RATE LIMITED for {symbol} - aggressive backoff {backoff_delay}s...")
                        print(f"   📊 Rate limit count: {self.cache_stats['rate_limited']}")
                        time.sleep(backoff_delay)
                        # Increase minimum interval for future requests
                        self.min_request_interval = min(2.0, self.min_request_interval * 1.5)
                        continue
                    else:
                        print(f"❌ Rate limited - max retries exceeded for {symbol}")
                        return None
                        
                # Check for network errors
                elif any(indicator in error_msg for indicator in ['timeout', 'connection', 'network']):
                    if attempt < max_retries - 1:
                        print(f"🌐 Network error for {symbol}, retrying...")
                        continue
                    else:
                        print(f"❌ Network error - max retries exceeded for {symbol}")
                        return None
                        
                # Other errors
                else:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Error for {symbol}: {e}, retrying...")
                        continue
                    else:
                        print(f"❌ Download failed for {symbol} after {max_retries} attempts: {e}")
                        return None
        
        return None
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands manually"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    def backtest_aegs_strategy(self, df):
        """Backtest AEGS strategy with manual BB calculation"""
        if df is None or len(df) < 100:
            return None
        
        # Reset index to ensure we have a proper datetime index
        df = df.copy()
        df.reset_index(inplace=True)
        if 'Date' not in df.columns:
            df['Date'] = df.index
        
        try:
            # Calculate indicators manually to avoid pandas_ta issues
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # Manual Bollinger Bands calculation
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
            
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return None
        
        trades = []
        position = None
        
        for i in range(50, len(df)):
            if i >= len(df) or (i-1) >= len(df):
                break
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Skip if we have NaN values
            if pd.isna(current['RSI']) or pd.isna(current['BB_Upper']) or pd.isna(current['BB_Lower']):
                continue
            
            # Entry signal: AEGS criteria
            if position is None:
                # Calculate BB position
                bb_range = current['BB_Upper'] - current['BB_Lower']
                if bb_range > 0:
                    bb_position = (current['Close'] - current['BB_Lower']) / bb_range
                else:
                    bb_position = 0.5
                
                # Volume surge
                if pd.notna(current['Volume_SMA']) and current['Volume_SMA'] > 0:
                    volume_ratio = current['Volume'] / current['Volume_SMA']
                else:
                    volume_ratio = 0
                
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
                        'entry_date': current['Date'] if 'Date' in df.columns else i,
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
                    entry_date = position['entry_date']
                    exit_date = current['Date'] if 'Date' in df.columns else i
                    
                    # Format dates properly
                    if hasattr(entry_date, 'strftime'):
                        entry_date_str = entry_date.strftime('%Y-%m-%d')
                    else:
                        entry_date_str = str(entry_date)
                        
                    if hasattr(exit_date, 'strftime'):
                        exit_date_str = exit_date.strftime('%Y-%m-%d')
                    else:
                        exit_date_str = str(exit_date)
                    
                    trades.append({
                        'entry_date': entry_date_str,
                        'exit_date': exit_date_str,
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

def get_dz_symbols():
    """Get a comprehensive list of D-Z NASDAQ symbols"""
    return [
        # D symbols
        'DASH', 'DDOG', 'DOCU', 'DKNG', 'DLTR', 'DNA', 'DNLI', 'DRNA', 'DXCM', 'DLO',
        # E symbols  
        'EBAY', 'EQIX', 'ETSY', 'EXPE', 'ENPH', 'ERIC', 'EVBG', 'EA', 'EPAM', 'ENTG',
        # F symbols
        'FTNT', 'FISV', 'FANG', 'FAST', 'FSLR', 'FLEX', 'FFIV', 'FULT', 'FOLD', 'FATE',
        # G symbols
        'GILD', 'GOLD', 'GOOGL', 'GOOG', 'GNTX', 'GSIT', 'GPRO', 'GRWG', 'GSKY', 'GWRE',
        # H symbols
        'HUBS', 'HELE', 'HTHT', 'HLIT', 'HAIN', 'HUBG', 'HMHC', 'HALO', 'HGEN', 'HEAR',
        # I symbols
        'INTC', 'IDXX', 'ILMN', 'INCY', 'ISRG', 'INFO', 'INTU', 'IEX', 'IOVA', 'IART',
        # J symbols
        'JBHT', 'JCOM', 'JD', 'JOBS', 'JBLU', 'JAKK', 'JAZZ', 'JNPR', 'JOE', 'JYNT',
        # K symbols  
        'KHC', 'KLAC', 'KERX', 'KRNT', 'KTOS', 'KURA', 'KDNY', 'KYMR', 'KPTI', 'KIRK',
        # L symbols
        'LRCX', 'LULU', 'LYFT', 'LBTYK', 'LBTYA', 'LPLA', 'LOGI', 'LOPE', 'LIVN', 'LMAT',
        # M symbols
        'MSFT', 'MU', 'MRVL', 'MAR', 'MELI', 'MTCH', 'MCHP', 'MDLZ', 'MNST', 'MOMO',
        # N symbols
        'NFLX', 'NVDA', 'NTES', 'NTAP', 'NDAQ', 'NKTR', 'NLOK', 'NTNX', 'NUAN', 'NICE',
        # O symbols
        'OKTA', 'ORLY', 'ORCL', 'OSTK', 'OLED', 'OMCL', 'OMER', 'OSUR', 'OFIX', 'OPRA',
        # P symbols
        'PYPL', 'PCAR', 'PCTY', 'PTC', 'PNFP', 'POOL', 'PSTG', 'PTCT', 'PDCO', 'PINS',
        # Q symbols
        'QCOM', 'QLYS', 'QRVO', 'QTT', 'QTNT', 'QADA', 'QNST', 'QURE', 'QUOT', 'QFIN',
        # R symbols
        'REGN', 'ROKU', 'ROST', 'RMBS', 'RCKT', 'RGEN', 'RPAY', 'RPRX', 'RRGB', 'REAL',
        # S symbols
        'SBUX', 'SIRI', 'SPLK', 'SWKS', 'SGEN', 'SAGE', 'SAIL', 'SAVE', 'SCKT', 'SEDG',
        # T symbols
        'TSLA', 'TTWO', 'TXN', 'TMUS', 'TLRY', 'TEAM', 'TECH', 'TGTX', 'TNDM', 'TRIP',
        # U symbols
        'UBER', 'ULTA', 'URBN', 'UMBF', 'UMPQ', 'UEPS', 'UNIT', 'UNFI', 'UONE', 'UPWK',
        # V symbols
        'VRTX', 'VRSK', 'VRSN', 'VSAT', 'VTRS', 'VCYT', 'VCRA', 'VEON', 'VIAV', 'VIAC',
        # W symbols
        'WDC', 'WDAY', 'WING', 'WIX', 'WKHS', 'WMGI', 'WOOF', 'WRAP', 'WYNN', 'WIFI',
        # X symbols
        'XLNX', 'XRAY', 'XPER', 'XPEL', 'XENE', 'XOMA', 'XBIT', 'XCUR', 'XNCR', 'XMTR',
        # Y symbols
        'YY', 'YELP', 'YEXT', 'YALA', 'YORW', 'YTEN', 'YTRA', 'YRCW', 'YTPG', 'YELL',
        # Z symbols
        'ZM', 'ZS', 'ZNGA', 'ZLAB', 'ZION', 'ZGNX', 'ZBRA', 'ZUMZ', 'ZUORA', 'ZOOM'
    ]

def main():
    """Run comprehensive D-Z cached scan"""
    print("🔥💎 COMPREHENSIVE CACHED D-Z AEGS SCAN 💎🔥")
    print("=" * 60)
    
    # Initialize cache
    cache = WorkingCacheAEGS()
    
    # Get D-Z symbols
    symbols = get_dz_symbols()
    print(f"🎯 Scanning {len(symbols)} D-Z symbols")
    print("📦 Cache will save all data for future scans")
    
    profitable_symbols = []
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i:3}/{len(symbols)}] Processing {symbol}...")
        
        result = cache.scan_symbol(symbol)
        
        if result:
            profitable_symbols.append(result)
            ret = result['strategy_return']
            trades = result['total_trades']
            win_rate = result['win_rate']
            print(f"    ✅ PROFITABLE: +{ret:.1f}% ({trades} trades, {win_rate:.0f}% wins)")
        else:
            print(f"    ❌ Not profitable")
        
        # Progress update every 20 symbols
        if i % 20 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (len(symbols) - i) / rate / 60
            print(f"\n📊 Progress: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%) - {eta:.1f}min remaining")
            print(f"   Cache: {cache.cache_stats['hits']} hits, {cache.cache_stats['misses']} misses")
            print(f"   Rate limits: {cache.cache_stats['rate_limited']}, Retries: {cache.cache_stats['retries']}")
            print(f"   Current interval: {cache.min_request_interval:.1f}s between requests")
            print(f"   Found: {len(profitable_symbols)} profitable symbols so far")
    
    # Final results
    scan_time = time.time() - start_time
    
    print(f"\n🎉 COMPREHENSIVE D-Z SCAN COMPLETE!")
    print(f"⏱️  Total time: {scan_time/60:.1f} minutes")
    print(f"📊 Cache stats: {cache.cache_stats['hits']} hits, {cache.cache_stats['misses']} misses")
    print(f"🚨 Rate limiting: {cache.cache_stats['rate_limited']} incidents, {cache.cache_stats['retries']} retries")
    print(f"⚡ Final request interval: {cache.min_request_interval:.1f}s")
    print(f"💰 Profitable symbols found: {len(profitable_symbols)}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cached_dz_aegs_scan_{timestamp}.json"
    
    output = {
        'scan_date': datetime.now().isoformat(),
        'method': 'COMPREHENSIVE CACHED D-Z AEGS SCAN',
        'symbols_scanned': len(symbols),
        'cache_stats': cache.cache_stats,
        'profitable_count': len(profitable_symbols),
        'scan_time_minutes': scan_time / 60,
        'profitable_symbols': profitable_symbols
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")
    
    # Show top performers
    if profitable_symbols:
        profitable_symbols.sort(key=lambda x: x['strategy_return'], reverse=True)
        print(f"\n🏆 Top 20 D-Z Performers:")
        for i, result in enumerate(profitable_symbols[:20], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            print(f"   {i:2}. {symbol:<6} +{ret:6.1f}% ({trades} trades)")
    
    # Test cache efficiency
    print(f"\n🔄 Testing cache efficiency...")
    print(f"Cache hit rate: {cache.cache_stats['hits']}/{cache.cache_stats['hits'] + cache.cache_stats['misses']} = {cache.cache_stats['hits']/(cache.cache_stats['hits'] + cache.cache_stats['misses'])*100:.1f}%")
    
    if cache.cache_stats['misses'] > 0:
        print(f"✅ Successfully cached {cache.cache_stats['misses']} new symbols")
        print(f"🚀 Future scans will be much faster with cache hits!")

if __name__ == "__main__":
    main()