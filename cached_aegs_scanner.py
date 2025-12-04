#!/usr/bin/env python3
"""
🔥💎 CACHED AEGS SCANNER 💎🔥
Uses permanent cache to avoid rate limiting
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from aegs_permanent_cache import AEGSPermanentCache

class CachedAEGSScanner:
    """AEGS scanner that uses permanent cache"""
    
    def __init__(self):
        self.cache = AEGSPermanentCache()
        self.results = []
        
    def calculate_aegs_signal_strength(self, df):
        """Calculate AEGS signal strength"""
        if len(df) < 50:
            return 0
        
        # Calculate indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bbands = ta.bbands(df['Close'], length=20)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # AEGS criteria scoring
        score = 0
        
        # RSI < 30 (oversold)
        if latest['RSI'] < 30:
            score += 25
        elif latest['RSI'] < 35:
            score += 15
        
        # Below lower Bollinger Band
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        if bb_position < 0.2:
            score += 25
        elif bb_position < 0.3:
            score += 15
        
        # Volume surge (>2x average)
        volume_ratio = latest['Volume'] / latest['Volume_SMA']
        if volume_ratio > 2.0:
            score += 25
        elif volume_ratio > 1.5:
            score += 15
        
        # Daily drop >5%
        daily_change = (latest['Close'] - prev['Close']) / prev['Close']
        if daily_change < -0.05:
            score += 25
        elif daily_change < -0.03:
            score += 15
        
        return score
    
    def backtest_aegs_strategy(self, df):
        """Backtest AEGS strategy on symbol data"""
        if len(df) < 100:
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
                bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
                
                # Volume surge
                volume_ratio = current['Volume'] / current['Volume_SMA'] if current['Volume_SMA'] > 0 else 0
                
                # Daily drop
                daily_change = (current['Close'] - prev['Close']) / prev['Close']
                
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
    
    def process_symbol_batch(self, symbols_batch, batch_id, total_symbols):
        """Process a batch of symbols with caching"""
        batch_results = []
        batch_start = time.time()
        
        for i, symbol in enumerate(symbols_batch):
            try:
                # Progress
                overall_progress = ((batch_id * len(symbols_batch)) + i + 1) / total_symbols * 100
                print(f"🔍 [{batch_id+1}] {symbol:<6} ({i+1}/{len(symbols_batch)}) - {overall_progress:.1f}% overall")
                
                # Get cached data (avoids rate limiting!)
                df = self.cache.get_data(symbol, period="730d", interval="1d")
                
                if df is None or len(df) < 100:
                    continue
                
                # Current price filter
                current_price = df.iloc[-1]['Close']
                if current_price < 1.0:
                    continue
                
                # Backtest AEGS strategy  
                result = self.backtest_aegs_strategy(df.copy())
                
                if result and result['strategy_return'] > 0:  # Only profitable
                    result['symbol'] = symbol
                    result['current_price'] = current_price
                    batch_results.append(result)
                    
                    print(f"   ✅ {symbol}: +{result['strategy_return']:.1f}% ({result['total_trades']} trades)")
                
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
                continue
        
        batch_time = time.time() - batch_start
        print(f"📊 Batch {batch_id+1} complete: {len(batch_results)} profitable symbols in {batch_time:.1f}s")
        
        return batch_results
    
    def scan_symbols(self, symbols, num_threads=4):
        """Scan symbols using cached data"""
        print(f"🔥 Starting CACHED AEGS scan of {len(symbols)} symbols")
        print(f"⚡ Using {num_threads} threads with permanent cache")
        
        # Split into batches
        batch_size = max(1, len(symbols) // num_threads)
        symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        all_results = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.process_symbol_batch, batch, i, len(symbols))
                for i, batch in enumerate(symbol_batches)
            ]
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Batch error: {e}")
        
        # Sort by strategy return
        all_results.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        return all_results

def get_nasdaq_symbols(letter_filter=None):
    """Get NASDAQ symbols from our cache or use a predefined list"""
    # Quick list of common NASDAQ symbols by letter range
    nasdaq_symbols = {
        'D': ['DASH', 'DDOG', 'DOCU', 'DOCN', 'DKNG', 'DLTR', 'DLO', 'DNA', 'DNLI', 'DRNA'],
        'E': ['EBAY', 'EQIX', 'ERIC', 'ETSY', 'EXPE', 'ENPH', 'EVBG', 'EVRG', 'EA', 'EPAM'],
        'F': ['FB', 'FTNT', 'FISV', 'FLEX', 'FANG', 'FAST', 'FFIV', 'FSLR', 'FULT', 'FOLD'],
        'G': ['GOOG', 'GOOGL', 'GILD', 'GLUU', 'GOLD', 'GH', 'GRUB', 'GNTX', 'GSIT', 'GPRO'],
        'H': ['HUBS', 'HZNP', 'HAS', 'HLIT', 'HELE', 'HMSY', 'HALO', 'HTHT', 'HUBG', 'HAIN'],
        'I': ['INTC', 'IDXX', 'ILMN', 'INCY', 'INFO', 'INTU', 'ISRG', 'IEX', 'IOVA', 'IART'],
        'J': ['JBHT', 'JCOM', 'JD', 'JOBS', 'JBLU', 'JAKK', 'JAZZ', 'JNPR', 'JOE', 'JYNT'],
        'K': ['KHC', 'KLAC', 'KOD', 'KDMN', 'KERX', 'KRNT', 'KTOS', 'KURA', 'KDNY', 'KYMR'],
        'L': ['LRCX', 'LULU', 'LYFT', 'LBTYK', 'LBTYA', 'LBRDK', 'LBRDA', 'LPLA', 'LOGI', 'LOPE'],
        'M': ['MSFT', 'MU', 'MRVL', 'MAR', 'MXIM', 'MELI', 'MTCH', 'MCHP', 'MDLZ', 'MNST'],
        'N': ['NFLX', 'NVDA', 'NTES', 'NTAP', 'NDAQ', 'NKTR', 'NLOK', 'NTNX', 'NUAN', 'NICE'],
        'O': ['OKTA', 'ORLY', 'ORCL', 'OSTK', 'OLED', 'OMCL', 'OMER', 'OSUR', 'OFIX', 'OPRA'],
        'P': ['PYPL', 'PEP', 'PCAR', 'PCTY', 'PTC', 'PNFP', 'POOL', 'PSTG', 'PTCT', 'PDCO'],
        'Q': ['QCOM', 'QLYS', 'QRVO', 'QTT', 'QTNT', 'QADA', 'QNST', 'QURE', 'QUOT', 'QFIN'],
        'R': ['REGN', 'ROKU', 'ROST', 'RYAAY', 'RMBS', 'RCKT', 'RGEN', 'RPAY', 'RPRX', 'RRGB'],
        'S': ['SBUX', 'SIRI', 'SPLK', 'SWKS', 'SGEN', 'SAGE', 'SAIL', 'SAVE', 'SCKT', 'SEDG'],
        'T': ['TSLA', 'TTWO', 'TXN', 'TMUS', 'TLRY', 'TEAM', 'TECH', 'TGTX', 'TNDM', 'TWTR'],
        'U': ['UBER', 'ULTA', 'URBN', 'UHAL', 'UMBF', 'UMPQ', 'UEPS', 'UNIT', 'UNFI', 'UONE'],
        'V': ['VIAB', 'VIAC', 'VRTX', 'VRSK', 'VRSN', 'VSAT', 'VTRS', 'VCYT', 'VCRA', 'VEON'],
        'W': ['WBA', 'WDC', 'WDAY', 'WING', 'WIX', 'WKHS', 'WLTW', 'WMGI', 'WOOF', 'WRAP'],
        'X': ['XLNX', 'XEL', 'XRAY', 'XPER', 'XPEL', 'XENE', 'XOMA', 'XBIT', 'XCUR', 'XNCR'],
        'Y': ['YY', 'YELP', 'YEXT', 'YALA', 'YOGA', 'YORW', 'YTEN', 'YTRA', 'YRCW', 'YTPG'],
        'Z': ['ZM', 'ZS', 'ZNGA', 'ZLAB', 'ZIXI', 'ZION', 'ZGNX', 'ZEAL', 'ZBRA', 'ZUMZ']
    }
    
    if letter_filter:
        return nasdaq_symbols.get(letter_filter, [])
    
    # Return all symbols
    all_symbols = []
    for symbols in nasdaq_symbols.values():
        all_symbols.extend(symbols)
    
    return all_symbols

def main():
    """Main scanning function"""
    print("🔥💎 CACHED AEGS SCANNER - No Rate Limits! 💎🔥")
    print("=" * 60)
    
    # Initialize scanner
    scanner = CachedAEGSScanner()
    
    # Show cache status
    inventory = scanner.cache.get_cache_inventory()
    print(f"📦 Cache Status: {inventory['total_files']} cached symbols ({inventory['total_size_mb']:.1f} MB)")
    
    # Get symbols to scan
    print(f"\n🎯 Choose scan range:")
    print("1. Single letter (D-Z)")
    print("2. Letter range (D-Z)")  
    print("3. All cached symbols")
    print("4. Custom symbol list")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    symbols = []
    if choice == "1":
        letter = input("Enter letter (D-Z): ").upper().strip()
        symbols = get_nasdaq_symbols(letter)
        scan_name = f"Letter {letter}"
    elif choice == "2":
        # Get D-Z symbols
        letters = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        for letter in letters:
            symbols.extend(get_nasdaq_symbols(letter))
        scan_name = "Letters D-Z"
    elif choice == "3":
        symbols = list(inventory['by_symbol'].keys())
        scan_name = "All Cached"
    else:
        symbol_input = input("Enter symbols (comma separated): ")
        symbols = [s.strip().upper() for s in symbol_input.split(',')]
        scan_name = "Custom List"
    
    if not symbols:
        print("❌ No symbols to scan!")
        return
    
    print(f"\n🚀 Starting scan of {len(symbols)} symbols ({scan_name})")
    
    # Run the scan
    start_time = time.time()
    results = scanner.scan_symbols(symbols, num_threads=6)
    scan_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cached_aegs_scan_{timestamp}.json"
    
    output = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'CACHED AEGS SCAN',
        'scan_range': scan_name,
        'total_analyzed': len(symbols),
        'profitable_count': len(results),
        'scan_time_seconds': scan_time,
        'cache_hits': inventory['total_files'],
        'profitable_symbols': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Show summary
    print(f"\n🎉 SCAN COMPLETE!")
    print(f"⏱️  Scan time: {scan_time:.1f}s")
    print(f"📊 Total analyzed: {len(symbols)}")
    print(f"💰 Profitable symbols found: {len(results)}")
    print(f"📁 Results saved to: {filename}")
    
    if results:
        print(f"\n🏆 Top 10 Performers:")
        for i, result in enumerate(results[:10], 1):
            symbol = result['symbol']
            ret = result['strategy_return']
            trades = result['total_trades']
            win_rate = result['win_rate']
            print(f"   {i:2}. {symbol:<6} +{ret:6.1f}% ({trades} trades, {win_rate:.0f}% wins)")
    
    return results

if __name__ == "__main__":
    main()