#!/usr/bin/env python3
"""
🔥💎 NASDAQ 'A' SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥
No filtering - pure brute force backtest on all NASDAQ A symbols
Compare to AI-filtered results
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
import time
import concurrent.futures
from typing import List, Dict, Optional
warnings.filterwarnings('ignore')

class BruteForceAEGSBacktester:
    """Brute force AEGS backtest - no filtering, just raw backtesting power"""
    
    def __init__(self):
        self.nasdaq_a_symbols = []
        self.brute_force_results = []
        
    def get_all_nasdaq_a_symbols(self) -> List[str]:
        """Get comprehensive list of NASDAQ A symbols - no filtering"""
        
        print(colored("🔥 BRUTE FORCE MODE: All NASDAQ 'A' symbols", 'red', attrs=['bold']))
        print("=" * 60)
        
        # Comprehensive NASDAQ A-symbol list (same as before but NO filtering)
        nasdaq_a_symbols = [
            # Major Tech
            'AAPL', 'AMZN', 'AMD', 'ADBE', 'AVGO', 'ABNB', 'ALGN', 'ADSK', 'ADP', 'ANSS',
            
            # All Biotech/Pharma (no quality filter)
            'AMGN', 'ALNY', 'ACAD', 'AKBA', 'ADMA', 'AEHR', 'AGEN', 'AIMD', 'ATNM', 'ATOS',
            'APLS', 'ARDX', 'ARCT', 'AGIO', 'ACRS', 'ACIU', 'ADAP', 'ADIL', 'ADVM', 'AERI',
            'AFIB', 'AGEN', 'AIRG', 'AKTS', 'ALDX', 'ALEC', 'ALGM', 'ALIM', 'ALPN', 'ALVR',
            'AMPH', 'AMRN', 'AMRS', 'ANAB', 'ANEB', 'ANGO', 'ANIC', 'ANTE', 'APCX', 'APDN',
            'APHA', 'APLT', 'APVO', 'ARAY', 'ARCB', 'ARCT', 'AREC', 'ARGO', 'ARIB', 'ARNA',
            'ARTL', 'ARVN', 'ASND', 'ASTC', 'ASXC', 'ATAK', 'ATEC', 'ATHA', 'ATHE', 'ATLC',
            'ATNI', 'ATOM', 'ATOS', 'ATRA', 'ATRC', 'ATRI', 'ATRN', 'ATRS', 'ATSG', 'ATUS',
            'ATVI', 'ATXS', 'AUDC', 'AUPH', 'AVAV', 'AVCO', 'AVDL', 'AVEO', 'AVGO', 'AVGR',
            'AVID', 'AVNW', 'AVRO', 'AVTX', 'AVXL', 'AWRE', 'AXDX', 'AXGN', 'AXNX', 'AXON',
            'AXSM', 'AXTG', 'AYTU', 'AZPN', 'AZTA',
            
            # All Financial/Fintech (no quality filter)  
            'AFRM', 'ALLY', 'AMAT', 'APA', 'ACIW', 'ACMR', 'ACOR', 'ACRE', 'ACRX', 'ACTG',
            
            # All Energy
            'APA', 'AROC', 'ARTW', 'ASIX', 'ATMU', 'ATUS', 'AUMN', 'AVAV', 'AWRE',
            
            # All Consumer
            'AMCX', 'ANGI', 'APRN', 'ARCO', 'ARVL', 'ATSG', 'AUTO', 'AVYA',
            
            # All Real Estate  
            'AMH', 'APLE', 'ARE', 'ARCP', 'ASGN', 'ATCO', 'ATKR',
            
            # All Industrial (kitchen sink approach)
            'AMED', 'AMKR', 'AMWD', 'ANCN', 'ANDE', 'ANDV', 'ANET', 'ANGI', 'ANGN', 'ANIK',
            'ANIP', 'ANSS', 'ANTE', 'ANTM', 'AORT', 'AOUT', 'APAM', 'APCX', 'APEI', 'APEN',
            'APLS', 'APOG', 'APPS', 'APPN', 'APRE', 'APTV', 'APVO', 'AQUA', 'ARAV', 'ARCB',
            'ARCE', 'ARCH', 'ARCO', 'ARCW', 'ARDS', 'AREC', 'ARES', 'ARGO', 'ARGT', 'ARGS',
            'ARIA', 'ARIS', 'ARKR', 'ARMK', 'ARMP', 'ARNC', 'AROC', 'ARQT', 'ARRY', 'ARTL',
            'ARTW', 'ARVL', 'ARWR', 'ASIX', 'ASMB', 'ASML', 'ASPN', 'ASPS', 'ASRT', 'ASRV',
            'ASTC', 'ASTE', 'ASUR', 'ASYS', 'ATAI', 'ATCX', 'ATEC', 'ATEN', 'ATER', 'ATHX',
            'ATIF', 'ATLC', 'ATLO', 'ATNF', 'ATNI', 'ATNM', 'ATOM', 'ATOS', 'ATRA', 'ATRC',
            'ATRI', 'ATRO', 'ATSG', 'ATUS', 'ATVI', 'ATXG', 'ATXI', 'ATXS', 'AUBN', 'AUDC',
            'AUPH', 'AUTL', 'AUTO', 'AUVI', 'AUXL', 'AVAV', 'AVCO', 'AVCT', 'AVDL', 'AVEO',
            'AVGO', 'AVGR', 'AVID', 'AVIR', 'AVNW', 'AVRO', 'AVTE', 'AVTR', 'AVTX', 'AVXL',
            'AVYA', 'AWRE', 'AXAS', 'AXDX', 'AXGN', 'AXLA', 'AXNX', 'AXON', 'AXSM', 'AXTG',
            'AYRO', 'AYTU', 'AZEK', 'AZPN', 'AZTA',
            
            # Add more A symbols that might exist
            'AADI', 'AAIC', 'AAME', 'AAON', 'AAPL', 'AAWW', 'ABCB', 'ABCL', 'ABCM', 'ABEO',
            'ABEV', 'ABIO', 'ABMD', 'ABNB', 'ABOS', 'ABSI', 'ABST', 'ABUS', 'ABVC', 'ACAD',
            'ACAH', 'ACAQ', 'ACAT', 'ACBA', 'ACCD', 'ACCO', 'ACEL', 'ACER', 'ACES', 'ACET',
            'ACGL', 'ACGN', 'ACHC', 'ACHL', 'ACIU', 'ACIW', 'ACLS', 'ACMR', 'ACNB', 'ACOR',
            'ACRS', 'ACRX', 'ACST', 'ACTG', 'ADBE', 'ADCT', 'ADER', 'ADES', 'ADGI', 'ADIL',
            'ADMA', 'ADMP', 'ADMS', 'ADNT', 'ADOM', 'ADORX', 'ADPT', 'ADRE', 'ADSK', 'ADTN',
            'ADUS', 'ADVM', 'ADVS', 'ADXN', 'AEHL', 'AEHR', 'AEI', 'AEIS', 'AEMD', 'AENT',
            'AEP', 'AERI', 'AESE', 'AFBI', 'AFIB', 'AFIN', 'AFMD', 'AFRI', 'AFRM', 'AFYA',
            'AGAE', 'AGEN', 'AGER', 'AGES', 'AGGR', 'AGIL', 'AGIO', 'AGMH', 'AGNC', 'AGRI',
            'AGRX', 'AGTI', 'AGYS', 'AHCO', 'AHHX', 'AHPI', 'AIHS', 'AIMD', 'AIMC', 'AINV',
            'AIR', 'AIRG', 'AIRI', 'AIRT', 'AISP', 'AIT', 'AIV', 'AIXI', 'AKAM', 'AKAN',
            'AKBA', 'AKRO', 'AKTS', 'AKUS', 'ALAC', 'ALBO', 'ALCC', 'ALCO', 'ALDX', 'ALEC',
            'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALHC', 'ALIM', 'ALJJ', 'ALKS', 'ALLK', 'ALLO',
            'ALLR', 'ALLT', 'ALLY', 'ALNY', 'ALOT', 'ALPN', 'ALPP', 'ALRM', 'ALRS', 'ALSK',
            'ALTI', 'ALTM', 'ALTR', 'ALTU', 'ALVR', 'ALVY', 'ALXO', 'ALZN', 'AMAT', 'AMBA',
            'AMBC', 'AMBP', 'AMCI', 'AMCR', 'AMCX', 'AMD', 'AMED', 'AMEH', 'AMG', 'AMGN',
            'AMH', 'AMKR', 'AMLI', 'AMLX', 'AMN', 'AMNB', 'AMOT', 'AMOV', 'AMPH', 'AMPL',
            'AMRB', 'AMRK', 'AMRN', 'AMRS', 'AMRX', 'AMSC', 'AMSF', 'AMST', 'AMSW', 'AMT',
            'AMTB', 'AMTD', 'AMTI', 'AMTX', 'AMWD', 'AMZN', 'ANAB', 'ANDE', 'ANEB', 'ANET',
            'ANEW', 'ANF', 'ANGI', 'ANGN', 'ANGO', 'ANIK', 'ANIP', 'ANIX', 'ANSS', 'ANTE',
            'ANY', 'AOSL', 'AOUT', 'APA', 'APAM', 'APD', 'APDN', 'APEI', 'APEN', 'APG',
            'APHA', 'API', 'APLD', 'APLE', 'APLM', 'APLS', 'APLT', 'APM', 'APOG', 'APOP',
            'APPF', 'APPN', 'APPS', 'APRE', 'APRN', 'APSG', 'APTI', 'APTM', 'APTS', 'APTV',
            'APVO', 'APWC', 'APYX', 'AQB', 'AQMS', 'AQN', 'AQST', 'AQUA', 'AR', 'ARAV',
            'ARAY', 'ARCB', 'ARCC', 'ARCH', 'ARCO', 'ARCT', 'ARDS', 'AREC', 'AREN', 'ARES',
            'ARGX', 'ARHS', 'ARI', 'ARIB', 'ARKR', 'ARL', 'ARLO', 'ARMK', 'ARMP', 'ARNA',
            'ARNC', 'AROC', 'AROW', 'ARQT', 'ARR', 'ARRY', 'ARTL', 'ARTNA', 'ARTW', 'ARVL',
            'ARVN', 'ARWR', 'ASA', 'ASAI', 'ASAQ', 'ASGN', 'ASIX', 'ASLE', 'ASMB', 'ASML',
            'ASND', 'ASNS', 'ASO', 'ASPN', 'ASPS', 'ASRT', 'ASRV', 'ASTC', 'ASTE', 'ASTI',
            'ASTL', 'ASTR', 'ASUR', 'ASX', 'ASXC', 'ASYS', 'ATA', 'ATAI', 'ATAK', 'ATAQ',
            'ATCX', 'ATE', 'ATER', 'ATEX', 'ATHA', 'ATHE', 'ATHM', 'ATHX', 'ATI', 'ATIF',
            'ATLC', 'ATLO', 'ATNF', 'ATNI', 'ATNM', 'ATO', 'ATOM', 'ATOS', 'ATR', 'ATRA',
            'ATRC', 'ATRI', 'ATRO', 'ATRN', 'ATRS', 'ATSG', 'ATUS', 'ATV', 'ATVI', 'ATXG',
            'ATXI', 'ATXS', 'AU', 'AUBN', 'AUDC', 'AUGX', 'AUID', 'AUMN', 'AUPH', 'AUR',
            'AURA', 'AUST', 'AUTL', 'AUTO', 'AUUD', 'AUVI', 'AVAV', 'AVCO', 'AVCT', 'AVDL',
            'AVDX', 'AVEO', 'AVGO', 'AVGR', 'AVHI', 'AVID', 'AVIR', 'AVK', 'AVNS', 'AVNW',
            'AVPT', 'AVRO', 'AVT', 'AVTE', 'AVTR', 'AVTX', 'AVXL', 'AVY', 'AVYA', 'AWAY',
            'AWH', 'AWI', 'AWIN', 'AWK', 'AWP', 'AWR', 'AWRE', 'AWX', 'AX', 'AXAS', 'AXDX',
            'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXON', 'AXP', 'AXR', 'AXS', 'AXSM', 'AXTA',
            'AXTG', 'AXU', 'AY', 'AYI', 'AYRO', 'AYTU', 'AYX', 'AZ', 'AZEK', 'AZN',
            'AZO', 'AZPN', 'AZTA', 'AZUL', 'AZZ'
        ]
        
        # Remove duplicates, filter valid symbols, and sort
        nasdaq_a_symbols = sorted(list(set([s for s in nasdaq_a_symbols if len(s) <= 5])))
        
        print(f"🔥 BRUTE FORCE TARGET: {len(nasdaq_a_symbols)} symbols")
        print("📊 NO FILTERING - Testing everything that moves!")
        
        # Show sample
        print("\nSample symbols:")
        for i in range(0, min(25, len(nasdaq_a_symbols)), 5):
            sample = nasdaq_a_symbols[i:i+5]
            print("   " + ", ".join(sample))
        
        if len(nasdaq_a_symbols) > 25:
            print(f"   ... and {len(nasdaq_a_symbols) - 25} more symbols")
        
        self.nasdaq_a_symbols = nasdaq_a_symbols
        return nasdaq_a_symbols
    
    def brute_force_backtest_all(self, max_workers=15):
        """Brute force backtest ALL symbols - no filtering"""
        
        symbols = self.nasdaq_a_symbols
        
        print(colored(f"\n🔥💎 BRUTE FORCE AEGS BACKTEST - ALL {len(symbols)} SYMBOLS 💎🔥", 'red', attrs=['bold']))
        print("=" * 80)
        print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO MERCY")
        print("=" * 80)
        
        results = []
        
        # Use ThreadPoolExecutor for maximum parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._brute_force_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            completed = 0
            total = len(symbols)
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Progress with key metrics
                        trades = result['total_trades']
                        win_rate = result['win_rate']
                        strategy_return = result['strategy_return']
                        excess_return = result['excess_return']
                        
                        status = "🔥" if excess_return > 100 else "✅" if excess_return > 0 else "❌"
                        
                        print(f"[{completed:3d}/{total}] {status} {symbol:6s}: "
                              f"{trades:2d} trades, {win_rate:4.0f}% win, "
                              f"Strategy: {strategy_return:+6.1f}%, "
                              f"Excess: {excess_return:+6.1f}%")
                    else:
                        print(f"[{completed:3d}/{total}] 💀 {symbol:6s}: NO DATA")
                        
                except Exception as e:
                    print(f"[{completed:3d}/{total}] 💥 {symbol:6s}: ERROR")
        
        self.brute_force_results = results
        print(f"\n🎯 BRUTE FORCE COMPLETE: {len(results)}/{total} symbols backtested successfully")
        return results
    
    def _brute_force_single_symbol(self, symbol):
        """Brute force backtest single symbol"""
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')
            
            if len(df) < 50:
                return None
            
            # Simple price filter - exclude penny stocks
            current_price = df['Close'].iloc[-1]
            if current_price < 1.0:
                return None
            
            # Calculate AEGS indicators (same as before)
            df = self._calculate_aegs_indicators(df)
            df = self._apply_aegs_strategy(df)
            
            # Run backtest
            return self._run_aegs_backtest(df, symbol)
            
        except Exception as e:
            return None
    
    def _calculate_aegs_indicators(self, df):
        """Calculate AEGS technical indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Daily change
        df['Daily_Change'] = df['Close'].pct_change()
        
        return df
    
    def _apply_aegs_strategy(self, df):
        """Apply AEGS strategy signals - SAME AS FILTERED VERSION"""
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            signal_strength = 0
            
            # RSI oversold
            if pd.notna(row['RSI']):
                if row['RSI'] < 30:
                    signal_strength += 35
                elif row['RSI'] < 35:
                    signal_strength += 20
            
            # Bollinger Band position
            if pd.notna(row['BB_Position']):
                if row['BB_Position'] < 0:
                    signal_strength += 35
                elif row['BB_Position'] < 0.2:
                    signal_strength += 20
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
                if row['Volume_Ratio'] > 2.0 and row['Daily_Change'] < -0.05:
                    signal_strength += 30
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
            
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            
            # Signal threshold
            if signal_strength >= 70:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        return df
    
    def _run_aegs_backtest(self, df, symbol):
        """Run AEGS backtest - SAME AS FILTERED VERSION"""
        
        df['Position'] = 0
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = 0
        
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        
        for i in range(len(df)):
            current_date = df.index[i]
            
            if df.iloc[i]['Signal'] == 1 and position == 0:
                # Enter position
                position = 1
                entry_price = df.iloc[i]['Close']
                entry_date = current_date
                df.iloc[i, df.columns.get_loc('Position')] = 1
                
            elif position == 1:
                df.iloc[i, df.columns.get_loc('Position')] = 1
                
                # Exit conditions
                current_price = df.iloc[i]['Close']
                returns = (current_price - entry_price) / entry_price
                days_held = (current_date - entry_date).days
                
                exit_position = False
                exit_reason = ""
                
                # AEGS exit rules
                if returns >= 0.3:  # 30% profit target
                    exit_position = True
                    exit_reason = "Profit Target 30%"
                elif returns <= -0.2:  # 20% stop loss
                    exit_position = True
                    exit_reason = "Stop Loss 20%"
                elif days_held >= 30 and returns > 0:
                    exit_position = True
                    exit_reason = "Time Exit (Profitable)"
                elif days_held >= 60:
                    exit_position = True
                    exit_reason = "Force Exit"
                
                if exit_position:
                    position = 0
                    
                    trades.append({
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'exit_date': current_date.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': returns * 100,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    df.iloc[i, df.columns.get_loc('Position')] = 0
        
        # Calculate strategy returns
        for i in range(1, len(df)):
            if df.iloc[i]['Position'] == 1:
                df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
        
        # Performance metrics
        total_return = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        winning_trades = [t for t in trades if t['return_pct'] > 0]
        losing_trades = [t for t in trades if t['return_pct'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Current price for context
        current_price = df['Close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'strategy_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': (total_return - buy_hold_return) * 100,
            'trades': trades[-3:] if trades else []
        }
    
    def analyze_brute_force_results(self):
        """Analyze brute force results and compare to filtered approach"""
        
        if not self.brute_force_results:
            print("❌ No brute force results to analyze")
            return
        
        print(colored(f"\n🔥💎 BRUTE FORCE AEGS RESULTS ANALYSIS 💎🔥", 'red', attrs=['bold']))
        print("=" * 80)
        
        # Sort by excess return
        sorted_results = sorted(self.brute_force_results, key=lambda x: x['excess_return'], reverse=True)
        
        # Top performers
        print(colored(f"\n🏆 TOP 20 BRUTE FORCE WINNERS (Ranked by Excess Return):", 'green', attrs=['bold']))
        print("=" * 100)
        print(f"{'#':<3} {'Symbol':<8} {'Price':<10} {'Trades':<7} {'Win%':<6} {'Strategy%':<11} {'Excess%':<9} {'Avg Win':<9}")
        print("=" * 100)
        
        for i, result in enumerate(sorted_results[:20], 1):
            symbol = result['symbol']
            price = result['current_price']
            trades = result['total_trades']
            win_rate = result['win_rate']
            strategy_return = result['strategy_return']
            excess_return = result['excess_return']
            avg_win = result['avg_win']
            
            # Color coding for performance
            if excess_return > 100:
                color = 'red'
                attrs = ['bold']
            elif excess_return > 50:
                color = 'yellow'
                attrs = ['bold']
            elif excess_return > 0:
                color = 'green'
                attrs = []
            else:
                color = 'white'
                attrs = []
            
            print(colored(f"{i:<3} {symbol:<8} ${price:<9.2f} {trades:<7} {win_rate:<6.0f} "
                         f"{strategy_return:<11.1f} {excess_return:<9.1f} {avg_win:<9.1f}", color, attrs=attrs))
        
        # Summary statistics
        total_analyzed = len(self.brute_force_results)
        with_trades = [r for r in self.brute_force_results if r['total_trades'] > 0]
        positive_excess = [r for r in self.brute_force_results if r['excess_return'] > 0]
        goldmines = [r for r in self.brute_force_results if r['excess_return'] > 100]
        hidden_gems = [r for r in self.brute_force_results if r['excess_return'] > 50 and r['total_trades'] >= 3]
        
        if with_trades:
            avg_win_rate = np.mean([r['win_rate'] for r in with_trades])
            avg_excess = np.mean([r['excess_return'] for r in self.brute_force_results])
            
            print(colored(f"\n📊 BRUTE FORCE SUMMARY STATISTICS:", 'cyan', attrs=['bold']))
            print("=" * 50)
            print(f"   Total Symbols Tested: {total_analyzed}")
            print(f"   Symbols with Trades: {len(with_trades)}")
            print(f"   Average Win Rate: {avg_win_rate:.1f}%")
            print(f"   Average Excess Return: {avg_excess:.1f}%")
            print(f"   Positive Excess Returns: {len(positive_excess)}/{total_analyzed} ({len(positive_excess)/total_analyzed*100:.0f}%)")
            print(colored(f"   🔥 GOLDMINES (>100% excess): {len(goldmines)}", 'red', attrs=['bold']))
            print(colored(f"   💎 HIDDEN GEMS (>50% excess): {len(hidden_gems)}", 'yellow', attrs=['bold']))
        
        # Compare to filtered results
        self._compare_to_filtered_results(sorted_results)
        
        # Save brute force results
        self._save_brute_force_results(sorted_results)
        
        return sorted_results
    
    def _compare_to_filtered_results(self, brute_force_sorted):
        """Compare brute force results to AI-filtered results"""
        
        print(colored(f"\n🔍 BRUTE FORCE vs AI-FILTERED COMPARISON", 'magenta', attrs=['bold']))
        print("=" * 80)
        
        # AI-filtered top performers (from previous run)
        ai_filtered_top = ['ATRA', 'ADVM', 'APDN', 'AVXL', 'AFRM', 'APPN', 'AVTR', 'AGIO', 'ASPN', 'AIMD']
        
        # Brute force top performers
        brute_force_top = [r['symbol'] for r in brute_force_sorted[:20]]
        
        print(f"🤖 AI-Filtered Top 10: {', '.join(ai_filtered_top[:10])}")
        print(f"🔥 Brute Force Top 10: {', '.join(brute_force_top[:10])}")
        
        # Find hidden gems that AI filtering missed
        ai_filtered_symbols = set(ai_filtered_top)
        brute_force_gems = [r for r in brute_force_sorted[:30] if r['symbol'] not in ai_filtered_symbols and r['excess_return'] > 20]
        
        if brute_force_gems:
            print(colored(f"\n💎 HIDDEN GEMS MISSED BY AI FILTERING:", 'yellow', attrs=['bold']))
            print("=" * 70)
            for i, gem in enumerate(brute_force_gems[:10], 1):
                symbol = gem['symbol']
                excess = gem['excess_return']
                trades = gem['total_trades']
                win_rate = gem['win_rate']
                price = gem['current_price']
                
                print(f"{i:2d}. {symbol:6s} (${price:6.2f}): {excess:+6.1f}% excess, "
                      f"{trades:2d} trades, {win_rate:4.0f}% win rate")
        
        # Find overlap
        overlap = set(ai_filtered_top[:10]) & set(brute_force_top[:10])
        
        print(f"\n🎯 OVERLAP: {len(overlap)}/10 symbols found by both methods")
        if overlap:
            print(f"   Common winners: {', '.join(sorted(overlap))}")
        
        # Performance comparison
        brute_force_top_10_excess = np.mean([r['excess_return'] for r in brute_force_sorted[:10]])
        
        print(colored(f"\n📈 PERFORMANCE METRICS:", 'cyan', attrs=['bold']))
        print("=" * 40)
        print(f"   Brute Force Top 10 Avg Excess: {brute_force_top_10_excess:.1f}%")
        print(f"   Total Symbols Analyzed: {len(self.brute_force_results)} vs ~50 (filtered)")
        print(f"   Hidden Gems Discovered: {len(brute_force_gems)}")
    
    def _save_brute_force_results(self, sorted_results):
        """Save brute force results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nasdaq_a_brute_force_aegs_results_{timestamp}.json'
        
        # Top performers
        top_performers = sorted_results[:25]
        
        export_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'BRUTE FORCE - No filtering',
            'criteria': 'All NASDAQ A symbols, $1+ price filter only',
            'total_analyzed': len(self.brute_force_results),
            'top_performers': top_performers,
            'goldmines': [r for r in sorted_results if r['excess_return'] > 100],
            'hidden_gems': [r for r in sorted_results if r['excess_return'] > 50 and r['total_trades'] >= 3],
            'summary': {
                'total_analyzed': len(self.brute_force_results),
                'positive_excess_count': len([r for r in self.brute_force_results if r['excess_return'] > 0]),
                'avg_excess_return': np.mean([r['excess_return'] for r in self.brute_force_results]),
                'goldmine_count': len([r for r in sorted_results if r['excess_return'] > 100]),
                'hidden_gem_count': len([r for r in sorted_results if r['excess_return'] > 50 and r['total_trades'] >= 3])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n💾 Brute force results saved to: {filename}")

def main():
    """Run brute force AEGS backtest on NASDAQ A symbols"""
    
    print(colored("🔥💎 NASDAQ 'A' SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥", 'red', attrs=['bold']))
    print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO PRESELECTION")
    print("Pure raw computational power applied to every symbol")
    print("=" * 70)
    
    backtester = BruteForceAEGSBacktester()
    
    # Step 1: Get ALL NASDAQ A symbols
    symbols = backtester.get_all_nasdaq_a_symbols()
    
    if not symbols:
        print("❌ No symbols found")
        return
    
    # Step 2: Brute force backtest everything
    print(f"\n🔥 INITIATING BRUTE FORCE ATTACK ON {len(symbols)} SYMBOLS...")
    results = backtester.brute_force_backtest_all(max_workers=20)
    
    if not results:
        print("❌ No successful backtests")
        return
    
    # Step 3: Analyze and compare results
    backtester.analyze_brute_force_results()
    
    print(colored(f"\n🎯 BRUTE FORCE MISSION COMPLETE!", 'green', attrs=['bold']))
    print(f"Discovered hidden gems that AI filtering might have missed!")

if __name__ == "__main__":
    main()