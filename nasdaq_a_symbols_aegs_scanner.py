#!/usr/bin/env python3
"""
🔥💎 NASDAQ 'A' SYMBOLS AEGS CANDIDATE SCANNER 💎🔥
Comprehensive AEGS analysis on NASDAQ symbols starting with 'A'
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from termcolor import colored
import warnings
import time
import concurrent.futures
from typing import List, Dict, Optional
warnings.filterwarnings('ignore')

class NASDAQASymbolsAEGSScanner:
    """AEGS candidate scanner for NASDAQ symbols starting with 'A'"""
    
    def __init__(self):
        self.nasdaq_a_symbols = []
        self.aegs_candidates = []
        self.backtest_results = {}
        
    def get_nasdaq_a_symbols(self) -> List[str]:
        """Get NASDAQ symbols starting with 'A'"""
        
        print(colored("🔍 Fetching NASDAQ symbols starting with 'A'...", 'yellow', attrs=['bold']))
        print("=" * 60)
        
        # Comprehensive list of NASDAQ A-symbols (major ones)
        nasdaq_a_symbols = [
            # Major Tech
            'AAPL', 'AMZN', 'AMD', 'ADBE', 'AVGO', 'ABNB', 'ALGN', 'ADSK', 'ADP', 'ANSS',
            
            # Biotech/Pharma
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
            
            # Financial/Fintech
            'AFRM', 'ALLY', 'AMAT', 'APA', 'ACIW', 'ACMR', 'ACOR', 'ACRE', 'ACRX', 'ACTG',
            
            # Energy/Commodities
            'APA', 'AROC', 'ARTW', 'ASIX', 'ATMU', 'ATUS', 'AUMN', 'AVAV', 'AWRE',
            
            # Consumer/Retail
            'AMCX', 'ANGI', 'APRN', 'ARCO', 'ARVL', 'ATSG', 'AUTO', 'AVYA',
            
            # Real Estate/REITs
            'AMH', 'APLE', 'ARE', 'ARCP', 'ASGN', 'ATCO', 'ATKR',
            
            # Industrial/Manufacturing
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
            
            # Communication/Media
            'AKAM', 'AMCX', 'ANGI', 'APPS', 'APPN', 'ARQT', 'ASUR', 'AUVI', 'AVYA',
            
            # Crypto/Blockchain
            'ARBK', 'ARBE', 'ARDX',
            
            # Cannabis/CBD
            'APHA', 'AXIM',
            
            # Gaming/Entertainment
            'ATVI', 'APPS', 'AUVI',
            
            # Space/Defense
            'ASTR', 'AVAV',
            
            # EV/Clean Energy
            'ARVL', 'AYRO', 'ATMU',
            
            # Cloud/SaaS
            'APPN', 'ASUR', 'AVYA',
            
            # AI/Machine Learning
            'ASML', 'ANET', 'NVDA'  # Some overlap but important
        ]
        
        # Remove duplicates and sort
        nasdaq_a_symbols = sorted(list(set(nasdaq_a_symbols)))
        
        print(f"📊 Found {len(nasdaq_a_symbols)} NASDAQ symbols starting with 'A'")
        
        # Show sample
        print("\nSample symbols:")
        for i in range(0, min(20, len(nasdaq_a_symbols)), 5):
            sample = nasdaq_a_symbols[i:i+5]
            print("   " + ", ".join(sample))
        
        if len(nasdaq_a_symbols) > 20:
            print(f"   ... and {len(nasdaq_a_symbols) - 20} more symbols")
        
        self.nasdaq_a_symbols = nasdaq_a_symbols
        return nasdaq_a_symbols
    
    def screen_for_aegs_criteria(self, symbols: List[str], min_price=2.0) -> List[Dict]:
        """Screen symbols for AEGS candidate criteria"""
        
        print(colored(f"\n🔍 Screening {len(symbols)} symbols for AEGS criteria...", 'cyan', attrs=['bold']))
        print("=" * 70)
        
        candidates = []
        processed = 0
        
        # Process in batches
        batch_size = 25
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_candidates = self._process_screening_batch(batch, min_price)
            candidates.extend(batch_candidates)
            
            processed += len(batch)
            print(f"   Processed {processed}/{len(symbols)} symbols...")
        
        if not candidates:
            print("❌ No AEGS candidates found")
            return []
        
        # Create DataFrame and rank candidates
        df = pd.DataFrame(candidates)
        
        # AEGS scoring system
        df['aegs_score'] = (
            df['volatility_score'] * 0.25 +           # Volatility (25%)
            abs(df['daily_change_pct']) * 0.2 +       # Today's move (20%)
            df['volume_surge'] * 0.2 +                # Volume surge (20%)
            df['oversold_score'] * 0.15 +             # Oversold conditions (15%)
            df['mean_reversion_score'] * 0.1 +        # Mean reversion potential (10%)
            df['liquidity_score'] * 0.1               # Liquidity (10%)
        )
        
        # Sort by AEGS score
        top_candidates = df.nlargest(50, 'aegs_score')
        
        print(colored(f"\n🔥 TOP 50 AEGS CANDIDATES (NASDAQ 'A' symbols):", 'red', attrs=['bold']))
        print("=" * 100)
        print(f"{'#':<3} {'Symbol':<8} {'Price':<10} {'Change%':<10} {'Vol':<8} {'RSI':<8} {'AEGS Score':<12}")
        print("=" * 100)
        
        for i, (_, row) in enumerate(top_candidates.head(25).iterrows(), 1):
            symbol = row['symbol']
            price = row['current_price']
            change = row['daily_change_pct']
            vol = row['volatility_score']
            rsi = row.get('rsi', 0)
            aegs_score = row['aegs_score']
            
            change_color = 'green' if change > 0 else 'red'
            change_symbol = '+' if change > 0 else ''
            
            print(f"{i:<3} {symbol:<8} ${price:<9.2f} "
                  f"{colored(f'{change_symbol}{change:.1f}%', change_color):<15} "
                  f"{vol:<7.1f} {rsi:<7.1f} {aegs_score:<11.2f}")
        
        if len(top_candidates) > 25:
            print(f"... and {len(top_candidates) - 25} more candidates")
        
        self.aegs_candidates = top_candidates.to_dict('records')
        return self.aegs_candidates
    
    def _process_screening_batch(self, symbols: List[str], min_price: float) -> List[Dict]:
        """Process screening batch for AEGS criteria"""
        
        results = []
        
        for symbol in symbols:
            try:
                # Skip if symbol is too long (likely invalid)
                if len(symbol) > 5:
                    continue
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='60d', interval='1d')
                
                if len(hist) < 30:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Price filter
                if current_price < min_price:
                    continue
                
                # Basic calculations
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                daily_change = (current_price - prev_close) / prev_close if prev_close > 0 else 0
                
                # RSI calculation
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50
                
                # Volatility
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
                
                # Volume analysis
                avg_volume = hist['Volume'].tail(20).mean()
                recent_volume = hist['Volume'].tail(3).mean()
                volume_surge = recent_volume / avg_volume if avg_volume > 0 else 0
                
                # Bollinger Bands position
                sma20 = hist['Close'].rolling(20).mean()
                bb_std = hist['Close'].rolling(20).std()
                bb_lower = sma20 - (bb_std * 2)
                bb_upper = sma20 + (bb_std * 2)
                bb_position = ((current_price - bb_lower.iloc[-1]) / 
                              (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if not bb_lower.empty else 0.5
                
                # AEGS-specific scoring
                oversold_score = max(0, 40 - current_rsi) if not np.isnan(current_rsi) else 0  # Higher score for more oversold
                volatility_score = min(100, volatility)  # Cap at 100%
                mean_reversion_score = 100 - abs(bb_position - 0.5) * 200  # Higher when near BB bands
                liquidity_score = min(100, avg_volume / 100000)  # Volume in 100k shares
                
                results.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'daily_change_pct': daily_change * 100,
                    'rsi': current_rsi,
                    'bb_position': bb_position,
                    'volatility_score': volatility_score,
                    'volume_surge': volume_surge,
                    'oversold_score': oversold_score,
                    'mean_reversion_score': mean_reversion_score,
                    'liquidity_score': liquidity_score,
                    'avg_volume': avg_volume
                })
                
            except Exception as e:
                continue
        
        return results
    
    def backtest_top_candidates(self, top_n=20):
        """Backtest top AEGS candidates"""
        
        if not self.aegs_candidates:
            print("❌ No candidates to backtest")
            return
        
        top_candidates = self.aegs_candidates[:top_n]
        
        print(colored(f"\n🔥💎 BACKTESTING TOP {top_n} AEGS CANDIDATES 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._backtest_single_candidate, candidate): candidate['symbol']
                for candidate in top_candidates
            }
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol), 1):
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Progress update
                        trades = result['total_trades']
                        win_rate = result['win_rate']
                        strategy_return = result['strategy_return']
                        excess_return = result['excess_return']
                        
                        print(f"[{i:2d}/{top_n}] {symbol:6s}: "
                              f"{trades:2d} trades, {win_rate:4.0f}% win, "
                              f"Return: {strategy_return:+6.1f}%, "
                              f"Excess: {excess_return:+6.1f}%")
                        
                except Exception as e:
                    print(f"[{i:2d}/{top_n}] {symbol:6s}: ERROR - {str(e)[:40]}")
        
        self.backtest_results = results
        return results
    
    def _backtest_single_candidate(self, candidate):
        """Backtest single AEGS candidate"""
        
        symbol = candidate['symbol']
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')
            
            if len(df) < 50:
                return None
            
            # Calculate AEGS indicators
            df = self._calculate_aegs_indicators(df)
            df = self._apply_aegs_strategy(df)
            
            # Run backtest
            return self._run_backtest(df, symbol)
            
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
        """Apply AEGS strategy signals"""
        
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
    
    def _run_backtest(self, df, symbol):
        """Run backtest on AEGS signals"""
        
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
                
                # Dynamic exits based on volatility
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
        
        return {
            'symbol': symbol,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'strategy_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': (total_return - buy_hold_return) * 100,
            'trades': trades[-3:] if trades else []
        }
    
    def analyze_results(self):
        """Analyze and report results"""
        
        if not self.backtest_results:
            print("❌ No backtest results to analyze")
            return
        
        print(colored(f"\n📊 NASDAQ 'A' SYMBOLS AEGS ANALYSIS RESULTS", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        # Sort by excess return
        sorted_results = sorted(self.backtest_results, key=lambda x: x['excess_return'], reverse=True)
        
        # Top performers
        print(colored(f"\n🏆 TOP 10 PERFORMERS (Ranked by Excess Return):", 'green', attrs=['bold']))
        print("=" * 90)
        print(f"{'#':<3} {'Symbol':<8} {'Trades':<7} {'Win%':<6} {'Strategy%':<11} {'Excess%':<9} {'Best Win':<10}")
        print("=" * 90)
        
        for i, result in enumerate(sorted_results[:10], 1):
            trades = result['total_trades']
            win_rate = result['win_rate']
            strategy_return = result['strategy_return']
            excess_return = result['excess_return']
            best_win = result['avg_win']
            
            print(f"{i:<3} {result['symbol']:<8} {trades:<7} {win_rate:<6.0f} "
                  f"{strategy_return:<11.1f} {excess_return:<9.1f} {best_win:<10.1f}")
        
        # Summary stats
        total_analyzed = len(self.backtest_results)
        with_trades = [r for r in self.backtest_results if r['total_trades'] > 0]
        positive_excess = [r for r in self.backtest_results if r['excess_return'] > 0]
        
        if with_trades:
            avg_win_rate = np.mean([r['win_rate'] for r in with_trades])
            avg_excess = np.mean([r['excess_return'] for r in self.backtest_results])
            
            print(colored(f"\n📈 SUMMARY STATISTICS:", 'yellow', attrs=['bold']))
            print("=" * 40)
            print(f"   Symbols Analyzed: {total_analyzed}")
            print(f"   Symbols with Trades: {len(with_trades)}")
            print(f"   Average Win Rate: {avg_win_rate:.1f}%")
            print(f"   Average Excess Return: {avg_excess:.1f}%")
            print(f"   Outperforming B&H: {len(positive_excess)}/{total_analyzed} ({len(positive_excess)/total_analyzed*100:.0f}%)")
        
        # Save results
        self._save_results(sorted_results[:10])
    
    def _save_results(self, top_performers):
        """Save top performers to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nasdaq_a_symbols_aegs_results_{timestamp}.json'
        
        export_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'criteria': 'NASDAQ symbols starting with A',
            'top_performers': top_performers,
            'summary': {
                'total_analyzed': len(self.backtest_results),
                'top_count': len(top_performers)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show candidates for watchlist
        print(colored(f"\n🎯 RECOMMENDED FOR AEGS WATCHLIST:", 'magenta', attrs=['bold']))
        print("=" * 50)
        
        for i, result in enumerate(top_performers[:5], 1):
            if result['excess_return'] > 50 and result['total_trades'] >= 2:
                symbol = result['symbol']
                excess = result['excess_return']
                trades = result['total_trades']
                win_rate = result['win_rate']
                
                print(f"{i}. {symbol}: {excess:+.1f}% excess return, "
                      f"{trades} trades, {win_rate:.0f}% win rate")

def main():
    """Run NASDAQ 'A' symbols AEGS scanner"""
    
    print(colored("🔥💎 NASDAQ 'A' SYMBOLS AEGS CANDIDATE SCANNER 💎🔥", 'red', attrs=['bold']))
    print("=" * 70)
    
    scanner = NASDAQASymbolsAEGSScanner()
    
    # Step 1: Get NASDAQ A symbols
    symbols = scanner.get_nasdaq_a_symbols()
    
    if not symbols:
        print("❌ No NASDAQ A symbols found")
        return
    
    # Step 2: Screen for AEGS criteria
    candidates = scanner.screen_for_aegs_criteria(symbols, min_price=2.0)
    
    if not candidates:
        print("❌ No AEGS candidates found")
        return
    
    print(f"\n✅ Found {len(candidates)} AEGS candidates for backtesting")
    
    # Step 3: Backtest top candidates
    scanner.backtest_top_candidates(top_n=25)
    
    # Step 4: Analyze results
    scanner.analyze_results()
    
    print(colored(f"\n🎯 NASDAQ 'A' SYMBOLS AEGS SCAN COMPLETE!", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()