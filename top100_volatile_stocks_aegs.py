#!/usr/bin/env python3
"""
🔥💎 TOP 100 VOLATILE STOCKS ($5+) AEGS BACKTEST SUITE 💎🔥
Comprehensive AEGS analysis on top 100 volatile stocks with $5+ closing prices
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

class Top100VolatileStocksAEGS:
    """AEGS backtester for top 100 volatile stocks with $5+ price filter"""
    
    def __init__(self):
        self.top_volatile_stocks = []
        self.backtest_results = {}
        
        # Expanded stock universe for better coverage
        self.stock_universe = self._build_comprehensive_universe()
        
    def _build_comprehensive_universe(self) -> List[str]:
        """Build comprehensive stock universe for volatility screening"""
        
        universe = []
        
        # Large cap tech with high beta
        tech_stocks = [
            'TSLA', 'NVDA', 'AMD', 'NFLX', 'META', 'GOOGL', 'AMZN', 'MSFT',
            'CRM', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'SHOP',
            'SQ', 'PYPL', 'ROKU', 'ZOOM', 'DOCU', 'PTON', 'ZM', 'UBER',
            'LYFT', 'ABNB', 'COIN', 'HOOD', 'SOFI', 'PLTR', 'RBLX'
        ]
        
        # Biotech/Pharma (historically volatile)
        biotech_stocks = [
            'MRNA', 'BNTX', 'NVAX', 'GILD', 'BIIB', 'AMGN', 'REGN', 'VRTX',
            'ILMN', 'BMRN', 'ALNY', 'TECH', 'SAGE', 'BLUE', 'EDIT', 'CRSP',
            'NTLA', 'BEAM', 'VCYT', 'PACB', 'CDNA', 'FATE', 'FOLD', 'ARCT',
            'SRPT', 'IONS', 'EXAS', 'VEEV', 'IRTC', 'ICPT'
        ]
        
        # Energy/Oil (volatile sector)
        energy_stocks = [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'OXY',
            'MPC', 'VLO', 'PSX', 'HES', 'DVN', 'FANG', 'EQT', 'AR',
            'CLR', 'MRO', 'APA', 'OVV', 'SM', 'NOV', 'FTI', 'RIG'
        ]
        
        # Financial/Banking
        financial_stocks = [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC',
            'TFC', 'COF', 'AXP', 'SCHW', 'BLK', 'SPGI', 'ICE',
            'CME', 'NDAQ', 'MCO', 'V', 'MA', 'PYPL', 'FIS', 'FISV'
        ]
        
        # Consumer Discretionary
        consumer_stocks = [
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW',
            'BKNG', 'ORLY', 'YUM', 'CMG', 'MAR', 'HLT', 'MGM', 'WYNN',
            'LVS', 'DIS', 'NFLX', 'WBD', 'PARA', 'FOXA', 'MTCH', 'ETSY'
        ]
        
        # Healthcare
        healthcare_stocks = [
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'ABBV',
            'MRK', 'LLY', 'MDT', 'ISRG', 'SYK', 'BSX', 'EW', 'ZBH',
            'DXCM', 'HOLX', 'VAR', 'TDOC', 'VEEV', 'IQV', 'CRL'
        ]
        
        # High-beta ETFs and leveraged products
        etf_stocks = [
            'QQQ', 'SPY', 'IWM', 'EFA', 'EEM', 'VTI', 'VEA', 'VWO',
            'TQQQ', 'SQQQ', 'SPXL', 'SPXU', 'TNA', 'TZA', 'UPRO', 'UVXY',
            'VXX', 'SVXY', 'LABU', 'LABD', 'SOXL', 'SOXS', 'ERX', 'ERY',
            'GUSH', 'DRIP', 'CURE', 'RXL', 'TECL', 'TECS', 'FAS', 'FAZ'
        ]
        
        # Crypto-related
        crypto_stocks = [
            'COIN', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN', 'HIVE',
            'EBON', 'SOS', 'EBANG', 'GBTC', 'ETHE', 'BITO', 'BITI'
        ]
        
        # Cannabis
        cannabis_stocks = [
            'TLRY', 'CGC', 'ACB', 'CRON', 'HEXO', 'OGI', 'SNDL', 'GRWG', 'SMG'
        ]
        
        # Gaming/Entertainment
        gaming_stocks = [
            'EA', 'ATVI', 'TTWO', 'RBLX', 'U', 'DKNG', 'PENN', 'MGM',
            'WYNN', 'LVS', 'CZR', 'BYD', 'FLUT', 'RSI'
        ]
        
        # Airlines/Travel
        travel_stocks = [
            'AAL', 'DAL', 'UAL', 'LUV', 'ALK', 'JBLU', 'SAVE', 'HA',
            'CCL', 'RCL', 'NCLH', 'MAR', 'HLT', 'H', 'EXPE', 'BKNG'
        ]
        
        # REITs (can be volatile)
        reit_stocks = [
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR',
            'DLR', 'WELL', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'CPT'
        ]
        
        # Combine all categories
        all_categories = [
            tech_stocks, biotech_stocks, energy_stocks, financial_stocks,
            consumer_stocks, healthcare_stocks, etf_stocks, crypto_stocks,
            cannabis_stocks, gaming_stocks, travel_stocks, reit_stocks
        ]
        
        for category in all_categories:
            universe.extend(category)
        
        # Remove duplicates and return
        return list(set(universe))
    
    def get_volatile_stocks_with_price_filter(self, limit=100, min_price=5.0):
        """Get most volatile stocks with price filter"""
        print(colored(f"🔍 Screening {len(self.stock_universe)} stocks for volatility (Price >${min_price})...", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        volatility_data = []
        processed = 0
        
        # Process in batches for better performance
        batch_size = 50
        for i in range(0, len(self.stock_universe), batch_size):
            batch = self.stock_universe[i:i+batch_size]
            batch_results = self._process_batch(batch, min_price)
            volatility_data.extend(batch_results)
            
            processed += len(batch)
            print(f"   Processed {processed}/{len(self.stock_universe)} stocks...")
        
        if not volatility_data:
            print("❌ No stocks found meeting criteria")
            return []
        
        # Create DataFrame and calculate composite volatility score
        df = pd.DataFrame(volatility_data)
        
        # Enhanced volatility scoring
        df['volatility_score'] = (
            df['volatility_annualized'] * 0.3 +           # Annual volatility (30%)
            abs(df['daily_change_pct']) * 0.25 +          # Today's move (25%)
            df['intraday_range_pct'] * 0.2 +              # Intraday range (20%)
            np.log(df['volume_ratio'] + 1) * 0.15 +       # Volume surge (15%)
            df['avg_true_range_pct'] * 0.1                # ATR (10%)
        )
        
        # Sort and get top stocks
        top_volatile = df.nlargest(limit, 'volatility_score')
        
        print(colored(f"\n🔥 TOP {limit} MOST VOLATILE STOCKS (>${min_price}):", 'red', attrs=['bold']))
        print("=" * 100)
        print(f"{'#':<3} {'Symbol':<8} {'Price':<10} {'Change%':<10} {'Vol%':<8} {'ATR%':<8} {'Score':<8}")
        print("=" * 100)
        
        for i, (_, row) in enumerate(top_volatile.iterrows(), 1):
            symbol = row['symbol']
            price = row['current_price']
            change = row['daily_change_pct']
            vol = row['volatility_annualized']
            atr = row['avg_true_range_pct']
            score = row['volatility_score']
            
            change_color = 'green' if change > 0 else 'red'
            change_symbol = '+' if change > 0 else ''
            
            if i <= 20:  # Show top 20 in detail
                print(f"{i:<3} {symbol:<8} ${price:<9.2f} "
                      f"{colored(f'{change_symbol}{change:.1f}%', change_color):<15} "
                      f"{vol:<7.1f} {atr:<7.1f} {score:<7.1f}")
        
        if len(top_volatile) > 20:
            print(f"... and {len(top_volatile) - 20} more stocks")
        
        self.top_volatile_stocks = top_volatile['symbol'].tolist()
        return self.top_volatile_stocks
    
    def _process_batch(self, symbols: List[str], min_price: float) -> List[Dict]:
        """Process a batch of symbols for volatility analysis"""
        results = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='30d', interval='1d')
                
                if len(hist) < 20:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                
                # Price filter
                if current_price < min_price:
                    continue
                
                # Calculate metrics
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                daily_change = (current_price - prev_close) / prev_close if prev_close > 0 else 0
                
                # Volatility calculations
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
                
                # Average True Range
                hist['TR'] = np.maximum(
                    hist['High'] - hist['Low'],
                    np.maximum(
                        abs(hist['High'] - hist['Close'].shift(1)),
                        abs(hist['Low'] - hist['Close'].shift(1))
                    )
                )
                atr = hist['TR'].rolling(14).mean().iloc[-1] if len(hist) >= 14 else 0
                atr_pct = (atr / current_price * 100) if current_price > 0 else 0
                
                # Volume analysis
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].tail(3).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                
                # Intraday range
                high_today = hist['High'].iloc[-1]
                low_today = hist['Low'].iloc[-1]
                intraday_range = (high_today - low_today) / low_today * 100 if low_today > 0 else 0
                
                results.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'daily_change_pct': daily_change * 100,
                    'volatility_annualized': volatility,
                    'avg_true_range_pct': atr_pct,
                    'volume_ratio': volume_ratio,
                    'intraday_range_pct': intraday_range,
                    'volume': recent_volume
                })
                
            except Exception as e:
                continue
                
        return results
    
    def calculate_aegs_indicators(self, df):
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
        
        # Additional volatility indicators for better AEGS performance
        df['ATR'] = self._calculate_atr(df)
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        return df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        return df['TR'].rolling(period).mean()
    
    def enhanced_aegs_strategy(self, df):
        """Enhanced AEGS strategy with volatility adjustments"""
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Volatility_Adjustment'] = 0
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            signal_strength = 0
            vol_adjustment = 0
            
            # Base AEGS signals
            # RSI oversold
            if pd.notna(row['RSI']):
                if row['RSI'] < 25:  # More stringent for volatile stocks
                    signal_strength += 40
                elif row['RSI'] < 30:
                    signal_strength += 25
                elif row['RSI'] < 35:
                    signal_strength += 15
            
            # Bollinger Band position
            if pd.notna(row['BB_Position']):
                if row['BB_Position'] < -0.1:  # Well below lower band
                    signal_strength += 40
                elif row['BB_Position'] < 0:
                    signal_strength += 25
                elif row['BB_Position'] < 0.15:
                    signal_strength += 15
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
                if row['Volume_Ratio'] > 3.0 and row['Daily_Change'] < -0.03:
                    signal_strength += 35
                elif row['Volume_Ratio'] > 2.0 and row['Daily_Change'] < -0.02:
                    signal_strength += 25
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
            
            # Price drop magnitude (adjusted for volatility)
            if pd.notna(row['Daily_Change']):
                daily_change_pct = row['Daily_Change'] * 100
                if daily_change_pct < -15:  # Larger drops for volatile stocks
                    signal_strength += 40
                elif daily_change_pct < -10:
                    signal_strength += 25
                elif daily_change_pct < -5:
                    signal_strength += 15
            
            # Volatility adjustment (reduce signals in extreme volatility)
            if pd.notna(row.get('ATR_Ratio')):
                atr_ratio = row['ATR_Ratio']
                if atr_ratio > 0.1:  # Very high volatility - reduce signal
                    vol_adjustment = -20
                elif atr_ratio > 0.05:  # High volatility - slight reduction
                    vol_adjustment = -10
                elif atr_ratio < 0.02:  # Low volatility - boost signal
                    vol_adjustment = 10
            
            # Apply volatility adjustment
            adjusted_strength = signal_strength + vol_adjustment
            
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            df.iloc[i, df.columns.get_loc('Volatility_Adjustment')] = vol_adjustment
            
            # Higher threshold for volatile stocks
            if adjusted_strength >= 75:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        return df
    
    def backtest_single_stock(self, symbol, period='1y'):
        """Backtest enhanced AEGS strategy on single stock"""
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if len(df) < 50:
                return None
            
            # Calculate indicators and signals
            df = self.calculate_aegs_indicators(df)
            df = self.enhanced_aegs_strategy(df)
            
            # Backtest with improved exit logic
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
                    
                    # Dynamic exit conditions based on volatility
                    current_price = df.iloc[i]['Close']
                    returns = (current_price - entry_price) / entry_price
                    days_held = (current_date - entry_date).days
                    
                    # ATR-based stops
                    atr_ratio = df.iloc[i].get('ATR_Ratio', 0.05)
                    
                    # Dynamic profit target (higher for more volatile stocks)
                    profit_target = 0.3 if atr_ratio < 0.03 else 0.5 if atr_ratio < 0.08 else 0.7
                    
                    # Dynamic stop loss (wider for more volatile stocks)
                    stop_loss = -0.15 if atr_ratio < 0.03 else -0.25 if atr_ratio < 0.08 else -0.35
                    
                    # Exit conditions
                    exit_position = False
                    exit_reason = ""
                    
                    if returns >= profit_target:
                        exit_position = True
                        exit_reason = f"Profit Target {profit_target*100:.0f}%"
                    elif returns <= stop_loss:
                        exit_position = True
                        exit_reason = f"Stop Loss {stop_loss*100:.0f}%"
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
            avg_days_held = np.mean([t['days_held'] for t in trades]) if trades else 0
            
            # Risk metrics
            recent_vol = df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
            current_price = df['Close'].iloc[-1]
            
            # Sharpe ratio approximation
            if len(df['Strategy_Returns']) > 0:
                strategy_vol = df['Strategy_Returns'].std() * np.sqrt(252)
                sharpe = (total_return / strategy_vol) if strategy_vol > 0 else 0
            else:
                sharpe = 0
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_days_held': avg_days_held,
                'strategy_return': total_return * 100,
                'buy_hold_return': buy_hold_return * 100,
                'excess_return': (total_return - buy_hold_return) * 100,
                'volatility': recent_vol,
                'sharpe_ratio': sharpe,
                'trades': trades[-3:] if trades else []
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error backtesting {symbol}: {e}")
            return None
    
    def run_parallel_backtests(self, max_workers=10):
        """Run backtests in parallel for faster processing"""
        
        print(colored(f"\n🔥💎 RUNNING AEGS BACKTESTS ON TOP {len(self.top_volatile_stocks)} STOCKS 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.backtest_single_stock, symbol, '1y'): symbol 
                for symbol in self.top_volatile_stocks
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
                        
                        print(f"[{i:3d}/{len(self.top_volatile_stocks)}] {symbol:6s}: "
                              f"{trades:2d} trades, {win_rate:4.0f}% win, "
                              f"Return: {strategy_return:+6.1f}%, "
                              f"Excess: {excess_return:+6.1f}%")
                        
                except Exception as e:
                    print(f"[{i:3d}/{len(self.top_volatile_stocks)}] {symbol:6s}: ERROR - {str(e)[:50]}")
        
        self.backtest_results = results
        return results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance analysis"""
        
        if not self.backtest_results:
            print("❌ No backtest results to report")
            return
        
        print(colored(f"\n📊 TOP 100 VOLATILE STOCKS AEGS COMPREHENSIVE REPORT", 'cyan', attrs=['bold']))
        print("=" * 100)
        
        # Sort by excess return
        sorted_results = sorted(self.backtest_results, key=lambda x: x['excess_return'], reverse=True)
        
        # Top performers
        print(colored(f"\n🏆 TOP 20 PERFORMERS (Ranked by Excess Return vs Buy & Hold):", 'green', attrs=['bold']))
        print("=" * 110)
        print(f"{'#':<3} {'Symbol':<8} {'Price':<8} {'Trades':<7} {'Win%':<6} {'Avg Hold':<9} {'Strategy%':<11} {'Excess%':<9} {'Sharpe':<7}")
        print("=" * 110)
        
        for i, result in enumerate(sorted_results[:20], 1):
            trades = result['total_trades']
            win_rate = result['win_rate']
            avg_hold = result['avg_days_held']
            strategy_return = result['strategy_return']
            excess_return = result['excess_return']
            sharpe = result['sharpe_ratio']
            price = result['current_price']
            
            print(f"{i:<3} {result['symbol']:<8} ${price:<7.2f} {trades:<7} {win_rate:<6.0f} "
                  f"{avg_hold:<9.1f} {strategy_return:<11.1f} {excess_return:<9.1f} {sharpe:<7.2f}")
        
        # Summary statistics
        total_stocks = len(self.backtest_results)
        total_trades = sum(r['total_trades'] for r in self.backtest_results)
        stocks_with_trades = [r for r in self.backtest_results if r['total_trades'] > 0]
        
        if stocks_with_trades:
            avg_win_rate = np.mean([r['win_rate'] for r in stocks_with_trades])
            avg_strategy_return = np.mean([r['strategy_return'] for r in self.backtest_results])
            avg_excess_return = np.mean([r['excess_return'] for r in self.backtest_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in stocks_with_trades])
            
            positive_excess = len([r for r in self.backtest_results if r['excess_return'] > 0])
            positive_return = len([r for r in self.backtest_results if r['strategy_return'] > 0])
            
            print(colored(f"\n📈 PORTFOLIO SUMMARY STATISTICS:", 'yellow', attrs=['bold']))
            print("=" * 50)
            print(f"   Stocks Analyzed: {total_stocks}")
            print(f"   Stocks with Trades: {len(stocks_with_trades)}")
            print(f"   Total AEGS Trades: {total_trades}")
            print(f"   Average Win Rate: {avg_win_rate:.1f}%")
            print(f"   Average Strategy Return: {avg_strategy_return:.1f}%")
            print(f"   Average Excess Return: {avg_excess_return:.1f}%")
            print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
            print(f"   Positive Returns: {positive_return}/{total_stocks} ({positive_return/total_stocks*100:.0f}%)")
            print(f"   Outperforming B&H: {positive_excess}/{total_stocks} ({positive_excess/total_stocks*100:.0f}%)")
            
            # Volatility buckets analysis
            print(colored(f"\n📊 PERFORMANCE BY VOLATILITY BUCKETS:", 'cyan', attrs=['bold']))
            print("=" * 70)
            
            # Create volatility buckets
            volatilities = [r['volatility'] for r in self.backtest_results]
            low_vol = [r for r in self.backtest_results if r['volatility'] < 40]
            med_vol = [r for r in self.backtest_results if 40 <= r['volatility'] < 80]
            high_vol = [r for r in self.backtest_results if r['volatility'] >= 80]
            
            buckets = [
                ("Low Volatility (<40%)", low_vol),
                ("Medium Volatility (40-80%)", med_vol),
                ("High Volatility (>80%)", high_vol)
            ]
            
            for bucket_name, bucket_data in buckets:
                if bucket_data:
                    avg_excess = np.mean([r['excess_return'] for r in bucket_data])
                    count = len(bucket_data)
                    outperforming = len([r for r in bucket_data if r['excess_return'] > 0])
                    print(f"   {bucket_name:<25}: {count:3d} stocks, {avg_excess:+6.1f}% avg excess, "
                          f"{outperforming}/{count} outperforming")
            
            # Best and worst performers
            print(colored(f"\n🎯 NOTABLE RESULTS:", 'magenta', attrs=['bold']))
            print("=" * 50)
            
            best_performer = max(self.backtest_results, key=lambda x: x['excess_return'])
            worst_performer = min(self.backtest_results, key=lambda x: x['excess_return'])
            most_trades = max(self.backtest_results, key=lambda x: x['total_trades'])
            best_win_rate = max(stocks_with_trades, key=lambda x: x['win_rate'])
            
            print(f"   Best Excess Return: {best_performer['symbol']} ({best_performer['excess_return']:+.1f}%)")
            print(f"   Worst Excess Return: {worst_performer['symbol']} ({worst_performer['excess_return']:+.1f}%)")
            print(f"   Most Active: {most_trades['symbol']} ({most_trades['total_trades']} trades)")
            print(f"   Best Win Rate: {best_win_rate['symbol']} ({best_win_rate['win_rate']:.1f}%)")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'top100_volatile_aegs_results_{timestamp}.json'
        
        export_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'criteria': {
                'min_price': 5.0,
                'top_count': len(self.top_volatile_stocks),
                'period': '1 year'
            },
            'top_volatile_stocks': self.top_volatile_stocks,
            'results': self.backtest_results,
            'summary': {
                'total_stocks': total_stocks,
                'stocks_with_trades': len(stocks_with_trades),
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate if stocks_with_trades else 0,
                'avg_strategy_return': avg_strategy_return,
                'avg_excess_return': avg_excess_return,
                'avg_sharpe_ratio': avg_sharpe if stocks_with_trades else 0,
                'positive_return_count': positive_return if stocks_with_trades else 0,
                'outperforming_count': positive_excess
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")
        print(colored(f"\n🎯 ANALYSIS COMPLETE: {total_stocks} stocks analyzed, {total_trades} total trades", 'green', attrs=['bold']))

def main():
    """Run top 100 volatile stocks AEGS analysis"""
    
    print(colored("🔥💎 TOP 100 VOLATILE STOCKS ($5+) AEGS ANALYSIS 💎🔥", 'red', attrs=['bold']))
    print("=" * 80)
    
    backtester = Top100VolatileStocksAEGS()
    
    # Step 1: Screen for volatile stocks with price filter
    volatile_stocks = backtester.get_volatile_stocks_with_price_filter(limit=100, min_price=5.0)
    
    if not volatile_stocks:
        print("❌ No volatile stocks found meeting criteria")
        return
    
    print(f"\n✅ Found {len(volatile_stocks)} volatile stocks for AEGS analysis")
    
    # Step 2: Run parallel backtests
    results = backtester.run_parallel_backtests(max_workers=15)
    
    print(f"\n✅ Completed backtests on {len(results)} stocks")
    
    # Step 3: Generate comprehensive analysis
    backtester.generate_comprehensive_report()

if __name__ == "__main__":
    main()