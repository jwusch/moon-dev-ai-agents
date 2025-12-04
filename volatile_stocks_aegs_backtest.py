#!/usr/bin/env python3
"""
🔥💎 VOLATILE STOCKS AEGS BACKTEST SUITE 💎🔥
Find today's most volatile stocks and run AEGS backtests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class VolatileStocksAEGS:
    """AEGS backtester for volatile stocks"""
    
    def __init__(self):
        self.top_volatile_stocks = []
        self.backtest_results = {}
        
    def get_volatile_stocks_today(self, limit=10):
        """Get most volatile stocks today"""
        print(colored("🔍 Identifying Today's Most Volatile Stocks...", 'yellow', attrs=['bold']))
        print("=" * 60)
        
        # High volatility stock universe (known volatile names)
        volatile_universe = [
            # Meme/High Beta Stocks
            'GME', 'AMC', 'BBBY', 'MULN', 'GNUS', 'SNDL', 'TLRY', 'CGC',
            'WISH', 'CLOV', 'SPCE', 'PLTR', 'NIO', 'LCID', 'RIVN',
            
            # Biotech/Pharma (high volatility)
            'MRNA', 'NVAX', 'BNTX', 'GILD', 'BIIB', 'VERU', 'SENS',
            'ATOS', 'PROG', 'OCGN', 'VXRT', 'INO', 'SRNE',
            
            # Tech/Growth (volatile)
            'TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM', 'ROKU', 'SQ',
            'SHOP', 'SNOW', 'CRWD', 'ZM', 'DOCU', 'PTON',
            
            # Crypto-related
            'COIN', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITI', 'BITO',
            
            # Energy/Commodities
            'USO', 'UCO', 'DRIP', 'GUSH', 'ERX', 'XLE', 'HAL',
            
            # Leveraged ETFs
            'SQQQ', 'TQQQ', 'UVXY', 'VXX', 'SVXY', 'SPXU', 'UPRO',
            'TZA', 'TNA', 'SOXS', 'SOXL', 'LABU', 'LABD',
            
            # Recent movers/penny stocks
            'DWAC', 'PHUN', 'BENE', 'MARK', 'EXPR', 'KOSS', 'NAKD',
            'BB', 'NOK', 'FIZZ', 'WKHS', 'RIDE', 'HYMC',
            
            # Cannabis
            'ACB', 'HEXO', 'OGI', 'CRON', 'APHA', 'GRWG', 'SMG'
        ]
        
        print(f"📊 Analyzing {len(volatile_universe)} volatile stocks...")
        
        volatility_data = []
        
        for i, symbol in enumerate(volatile_universe):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get recent data for volatility calculation
                hist = ticker.history(period='5d', interval='1d')
                
                if len(hist) < 3:
                    continue
                    
                # Calculate metrics
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                daily_change = (current_price - prev_close) / prev_close
                
                # Calculate recent volatility (5-day)
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized vol %
                
                # Volume analysis
                avg_volume = hist['Volume'].mean()
                today_volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
                volume_ratio = today_volume / avg_volume if avg_volume > 0 else 0
                
                # Price range today
                high_today = hist['High'].iloc[-1]
                low_today = hist['Low'].iloc[-1]
                intraday_range = (high_today - low_today) / low_today * 100
                
                volatility_data.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'daily_change_pct': daily_change * 100,
                    'volatility_annualized': volatility,
                    'volume_ratio': volume_ratio,
                    'intraday_range_pct': intraday_range,
                    'volume': today_volume
                })
                
                if (i + 1) % 20 == 0:
                    print(f"   Processed {i + 1}/{len(volatile_universe)} stocks...")
                    
            except Exception as e:
                continue
        
        # Create DataFrame and sort by composite volatility score
        df = pd.DataFrame(volatility_data)
        
        if df.empty:
            print("❌ No volatile stocks found")
            return []
        
        # Create composite volatility score
        df['volatility_score'] = (
            df['volatility_annualized'] * 0.4 +           # Annual volatility (40%)
            abs(df['daily_change_pct']) * 0.3 +          # Today's move (30%)
            df['intraday_range_pct'] * 0.2 +             # Intraday range (20%)
            np.log(df['volume_ratio'] + 1) * 0.1         # Volume surge (10%)
        )
        
        # Sort by volatility score and get top stocks
        top_volatile = df.nlargest(limit, 'volatility_score')
        
        print(colored(f"\n🔥 TOP {limit} MOST VOLATILE STOCKS TODAY:", 'red', attrs=['bold']))
        print("=" * 80)
        
        for i, (_, row) in enumerate(top_volatile.iterrows(), 1):
            symbol = row['symbol']
            price = row['current_price']
            change = row['daily_change_pct']
            vol = row['volatility_annualized']
            vol_ratio = row['volume_ratio']
            
            change_color = 'green' if change > 0 else 'red'
            change_symbol = '+' if change > 0 else ''
            
            print(f"{i:2d}. {symbol:6s} ${price:8.2f} "
                  f"{colored(f'{change_symbol}{change:.1f}%', change_color)} "
                  f"Vol: {vol:5.1f}% "
                  f"VolRatio: {vol_ratio:.1f}x")
        
        self.top_volatile_stocks = top_volatile['symbol'].tolist()
        return self.top_volatile_stocks
    
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
        
        return df
    
    def aegs_strategy(self, df):
        """AEGS buy/sell signals"""
        
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
                if row['BB_Position'] < 0:  # Below lower band
                    signal_strength += 35
                elif row['BB_Position'] < 0.2:
                    signal_strength += 20
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
                if row['Volume_Ratio'] > 2.0 and row['Daily_Change'] < -0.02:
                    signal_strength += 30
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
            
            # Price drop magnitude
            if pd.notna(row['Daily_Change']):
                daily_change_pct = row['Daily_Change'] * 100
                if daily_change_pct < -10:
                    signal_strength += 35
                elif daily_change_pct < -5:
                    signal_strength += 20
            
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            
            # Buy signal if strength >= 70
            if signal_strength >= 70:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        return df
    
    def backtest_single_stock(self, symbol, period='1y'):
        """Backtest AEGS strategy on single stock"""
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if len(df) < 50:
                return None
                
            print(f"\n📊 Backtesting {symbol}...")
            
            # Calculate indicators and signals
            df = self.calculate_aegs_indicators(df)
            df = self.aegs_strategy(df)
            
            # Backtest
            df['Position'] = 0
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = 0
            
            position = 0
            entry_price = 0
            trades = []
            
            for i in range(len(df)):
                if df.iloc[i]['Signal'] == 1 and position == 0:
                    # Enter position
                    position = 1
                    entry_price = df.iloc[i]['Close']
                    df.iloc[i, df.columns.get_loc('Position')] = 1
                    
                elif position == 1:
                    df.iloc[i, df.columns.get_loc('Position')] = 1
                    
                    # Exit conditions: 50% profit or -20% stop loss
                    current_price = df.iloc[i]['Close']
                    returns = (current_price - entry_price) / entry_price
                    
                    if returns >= 0.5 or returns <= -0.2:
                        # Exit position
                        position = 0
                        exit_price = current_price
                        
                        trades.append({
                            'entry_date': df.index[i-1].strftime('%Y-%m-%d'),
                            'exit_date': df.index[i].strftime('%Y-%m-%d'),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': returns * 100
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
            
            # Get current metrics for volatility context
            recent_vol = df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
            current_price = df['Close'].iloc[-1]
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'strategy_return': total_return * 100,
                'buy_hold_return': buy_hold_return * 100,
                'excess_return': (total_return - buy_hold_return) * 100,
                'volatility': recent_vol,
                'trades': trades[-3:] if trades else []  # Last 3 trades
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error backtesting {symbol}: {e}")
            return None
    
    def run_volatile_stocks_backtests(self):
        """Run AEGS backtests on all volatile stocks"""
        
        print(colored("\n🔥💎 RUNNING AEGS BACKTESTS ON VOLATILE STOCKS 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        results = []
        
        for i, symbol in enumerate(self.top_volatile_stocks, 1):
            print(f"\n[{i}/{len(self.top_volatile_stocks)}] Processing {symbol}...")
            
            result = self.backtest_single_stock(symbol, period='1y')
            if result:
                results.append(result)
                
                # Quick summary
                trades = result['total_trades']
                win_rate = result['win_rate']
                strategy_return = result['strategy_return']
                excess_return = result['excess_return']
                
                if strategy_return > 0:
                    return_color = 'green'
                    return_symbol = '+'
                else:
                    return_color = 'red'
                    return_symbol = ''
                
                print(f"   📈 {trades} trades, {win_rate:.0f}% win rate")
                print(colored(f"   💰 Strategy: {return_symbol}{strategy_return:.1f}%, Excess: {return_symbol}{excess_return:.1f}%", 
                            return_color))
        
        self.backtest_results = results
        return results
    
    def generate_volatility_report(self):
        """Generate comprehensive report"""
        
        if not self.backtest_results:
            print("❌ No backtest results to report")
            return
        
        print(colored("\n📊 VOLATILE STOCKS AEGS PERFORMANCE REPORT", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        # Sort by excess return
        sorted_results = sorted(self.backtest_results, key=lambda x: x['excess_return'], reverse=True)
        
        print(f"\n🎯 TOP PERFORMERS (Ranked by Excess Return vs Buy & Hold):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Symbol':<8} {'Trades':<6} {'Win%':<5} {'Strategy%':<10} {'Excess%':<8} {'Vol%':<6}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10], 1):
            trades = result['total_trades']
            win_rate = result['win_rate']
            strategy_return = result['strategy_return']
            excess_return = result['excess_return']
            volatility = result['volatility']
            
            print(f"{i:<4} {result['symbol']:<8} {trades:<6} {win_rate:<5.0f} "
                  f"{strategy_return:<10.1f} {excess_return:<8.1f} {volatility:<6.0f}")
        
        # Summary statistics
        total_trades = sum(r['total_trades'] for r in self.backtest_results)
        avg_win_rate = np.mean([r['win_rate'] for r in self.backtest_results if r['total_trades'] > 0])
        avg_strategy_return = np.mean([r['strategy_return'] for r in self.backtest_results])
        avg_excess_return = np.mean([r['excess_return'] for r in self.backtest_results])
        
        positive_excess = len([r for r in self.backtest_results if r['excess_return'] > 0])
        total_stocks = len(self.backtest_results)
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Stocks Analyzed: {total_stocks}")
        print(f"   Total AEGS Trades: {total_trades}")
        print(f"   Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   Average Strategy Return: {avg_strategy_return:.1f}%")
        print(f"   Average Excess Return: {avg_excess_return:.1f}%")
        print(f"   Stocks Outperforming B&H: {positive_excess}/{total_stocks} ({positive_excess/total_stocks*100:.0f}%)")
        
        # Volatility vs Performance analysis
        high_vol_stocks = [r for r in self.backtest_results if r['volatility'] > 50]
        if high_vol_stocks:
            high_vol_excess = np.mean([r['excess_return'] for r in high_vol_stocks])
            print(f"   High Volatility (>50%) Performance: {high_vol_excess:.1f}% avg excess return")
        
        # Best individual trades
        all_trades = []
        for result in self.backtest_results:
            for trade in result['trades']:
                trade['symbol'] = result['symbol']
                all_trades.append(trade)
        
        if all_trades:
            best_trades = sorted(all_trades, key=lambda x: x['return_pct'], reverse=True)[:5]
            print(f"\n🏆 BEST INDIVIDUAL TRADES:")
            for trade in best_trades:
                print(f"   {trade['symbol']}: {trade['return_pct']:+.1f}% on {trade['exit_date']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'volatile_stocks_aegs_backtest_{timestamp}.json'
        
        export_data = {
            'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_volatile_stocks': self.top_volatile_stocks,
            'results': self.backtest_results,
            'summary': {
                'total_stocks': total_stocks,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_strategy_return': avg_strategy_return,
                'avg_excess_return': avg_excess_return,
                'outperforming_count': positive_excess
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run volatile stocks AEGS backtest suite"""
    
    print(colored("🔥💎 VOLATILE STOCKS AEGS BACKTEST SUITE 💎🔥", 'red', attrs=['bold']))
    print("=" * 60)
    
    backtester = VolatileStocksAEGS()
    
    # Step 1: Identify volatile stocks
    volatile_stocks = backtester.get_volatile_stocks_today(limit=10)
    
    if not volatile_stocks:
        print("❌ No volatile stocks identified")
        return
    
    # Step 2: Run backtests
    results = backtester.run_volatile_stocks_backtests()
    
    # Step 3: Generate report
    backtester.generate_volatility_report()
    
    print(colored("\n🎯 AEGS VOLATILE STOCKS ANALYSIS COMPLETE! 🎯", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()