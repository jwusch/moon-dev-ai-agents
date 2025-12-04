#!/usr/bin/env python3
"""
🔥💎 ZM 15-MINUTE INTRADAY AEGS BACKTEST 💎🔥
Compare 15-minute vs daily AEGS performance on ZM
Testing the Fractal Market Efficiency Principle
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class ZM15MinuteAEGS:
    """15-minute intraday AEGS backtest for ZM"""
    
    def __init__(self):
        self.symbol = 'ZM'
        
    def get_15min_data(self, days_back=60):
        """Get 15-minute data for ZM"""
        print(colored(f"📊 Fetching {days_back} days of 15-minute ZM data...", 'yellow'))
        
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=f'{days_back}d', interval='15m')
        
        print(f"   Data Range: {df.index[0]} to {df.index[-1]}")
        print(f"   Total Bars: {len(df)}")
        print(f"   Current Price: ${df['Close'].iloc[-1]:.2f}")
        
        return df
    
    def calculate_intraday_aegs_indicators(self, df):
        """Calculate AEGS indicators optimized for 15-minute bars"""
        
        # RSI (shorter period for intraday)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20 periods = 5 hours of 15m bars)
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume analysis (20 periods = 5 hours)
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price changes (bar-to-bar)
        df['Bar_Change'] = df['Close'].pct_change()
        
        # Intraday momentum (4 bars = 1 hour lookback)
        df['Momentum_1H'] = df['Close'] / df['Close'].shift(4) - 1
        
        # ATR for volatility context (14 periods)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Market session indicator
        df['Hour'] = df.index.hour
        df['Market_Session'] = df['Hour'].apply(self._get_market_session)
        
        return df
    
    def _get_market_session(self, hour):
        """Classify market session"""
        if 9 <= hour < 10:
            return 'Open'
        elif 10 <= hour < 15:
            return 'Mid'
        elif 15 <= hour < 16:
            return 'Close'
        else:
            return 'Extended'
    
    def intraday_aegs_strategy(self, df):
        """Intraday AEGS strategy with 15-minute optimizations"""
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Signal_Details'] = ''
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            signal_strength = 0
            signal_details = []
            
            # RSI oversold (more aggressive thresholds for intraday)
            if pd.notna(row['RSI']):
                if row['RSI'] < 25:
                    signal_strength += 40
                    signal_details.append(f"RSI Oversold ({row['RSI']:.1f})")
                elif row['RSI'] < 30:
                    signal_strength += 25
                    signal_details.append(f"RSI Weak ({row['RSI']:.1f})")
                elif row['RSI'] < 35:
                    signal_strength += 15
                    signal_details.append(f"RSI Mild ({row['RSI']:.1f})")
            
            # Bollinger Band position
            if pd.notna(row['BB_Position']):
                if row['BB_Position'] < -0.1:
                    signal_strength += 40
                    signal_details.append(f"Below BB ({row['BB_Position']:.2f})")
                elif row['BB_Position'] < 0:
                    signal_strength += 25
                    signal_details.append(f"Near BB Lower ({row['BB_Position']:.2f})")
                elif row['BB_Position'] < 0.2:
                    signal_strength += 15
                    signal_details.append(f"Low BB Zone ({row['BB_Position']:.2f})")
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Bar_Change']):
                bar_change_pct = row['Bar_Change'] * 100
                if row['Volume_Ratio'] > 2.5 and bar_change_pct < -1.5:
                    signal_strength += 35
                    signal_details.append(f"Vol Surge Down ({row['Volume_Ratio']:.1f}x, {bar_change_pct:.1f}%)")
                elif row['Volume_Ratio'] > 2.0 and bar_change_pct < -1.0:
                    signal_strength += 25
                    signal_details.append(f"High Vol Drop ({row['Volume_Ratio']:.1f}x, {bar_change_pct:.1f}%)")
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
                    signal_details.append(f"Above Avg Vol ({row['Volume_Ratio']:.1f}x)")
            
            # Intraday momentum drop
            if pd.notna(row['Momentum_1H']):
                momentum_pct = row['Momentum_1H'] * 100
                if momentum_pct < -3:
                    signal_strength += 30
                    signal_details.append(f"1H Drop ({momentum_pct:.1f}%)")
                elif momentum_pct < -2:
                    signal_strength += 20
                    signal_details.append(f"1H Weak ({momentum_pct:.1f}%)")
                elif momentum_pct < -1:
                    signal_strength += 10
                    signal_details.append(f"1H Mild ({momentum_pct:.1f}%)")
            
            # Single bar drop magnitude
            if pd.notna(row['Bar_Change']):
                bar_change_pct = row['Bar_Change'] * 100
                if bar_change_pct < -3:
                    signal_strength += 35
                    signal_details.append(f"Big Bar Drop ({bar_change_pct:.1f}%)")
                elif bar_change_pct < -2:
                    signal_strength += 25
                    signal_details.append(f"Bar Drop ({bar_change_pct:.1f}%)")
                elif bar_change_pct < -1:
                    signal_strength += 15
                    signal_details.append(f"Small Drop ({bar_change_pct:.1f}%)")
            
            # Market session bonus/penalty
            session = row['Market_Session']
            if session == 'Open':
                signal_strength += 10
                signal_details.append("Opening Session")
            elif session == 'Close':
                signal_strength += 5
                signal_details.append("Closing Session")
            elif session == 'Extended':
                signal_strength -= 20  # Avoid extended hours
                signal_details.append("Extended Hours (Avoid)")
            
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            df.iloc[i, df.columns.get_loc('Signal_Details')] = ' | '.join(signal_details)
            
            # Lower threshold for intraday (faster signals)
            if signal_strength >= 70 and session != 'Extended':
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        return df
    
    def run_intraday_backtest(self, df):
        """Run intraday backtest with optimized exits"""
        
        df['Position'] = 0
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = 0
        df['Trade_ID'] = 0
        
        position = 0
        entry_price = 0
        entry_datetime = None
        entry_index = 0
        trades = []
        trade_id = 0
        
        for i in range(len(df)):
            current_datetime = df.index[i]
            current_price = df.iloc[i]['Close']
            
            if df.iloc[i]['Signal'] == 1 and position == 0:
                # Enter position
                position = 1
                entry_price = current_price
                entry_datetime = current_datetime
                entry_index = i
                trade_id += 1
                
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Trade_ID')] = trade_id
                
            elif position == 1:
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Trade_ID')] = trade_id
                
                # Calculate current performance
                returns = (current_price - entry_price) / entry_price
                bars_held = i - entry_index
                hours_held = bars_held * 0.25  # 15-min bars = 0.25 hours each
                
                # Intraday exit conditions
                exit_trade = False
                exit_reason = ""
                
                # Quick profit targets (intraday scalping)
                if returns >= 0.02:  # 2% quick profit
                    exit_trade = True
                    exit_reason = "Quick Profit (2%)"
                elif returns >= 0.015 and hours_held >= 2:  # 1.5% after 2 hours
                    exit_trade = True
                    exit_reason = "Profit Target (1.5%)"
                elif returns >= 0.01 and hours_held >= 4:  # 1% after 4 hours
                    exit_trade = True
                    exit_reason = "Slow Profit (1%)"
                
                # Tight stop loss for intraday
                elif returns <= -0.01:  # 1% stop loss
                    exit_trade = True
                    exit_reason = "Stop Loss (-1%)"
                
                # Time-based exits
                elif hours_held >= 6:  # Max 6 hours (1.5 trading days)
                    exit_trade = True
                    exit_reason = f"Time Exit ({hours_held:.1f}h)"
                
                # End of day exit (avoid overnight risk)
                elif current_datetime.hour >= 15:  # 3 PM ET
                    exit_trade = True
                    exit_reason = "End of Day Exit"
                
                if exit_trade:
                    # Exit position
                    position = 0
                    
                    # Record trade
                    trade_data = {
                        'trade_id': trade_id,
                        'entry_datetime': entry_datetime.strftime('%Y-%m-%d %H:%M'),
                        'exit_datetime': current_datetime.strftime('%Y-%m-%d %H:%M'),
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': returns * 100,
                        'return_dollars': (current_price - entry_price) * 100,
                        'bars_held': bars_held,
                        'hours_held': hours_held,
                        'exit_reason': exit_reason,
                        'entry_signal_strength': df.iloc[entry_index]['Signal_Strength'],
                        'entry_signal_details': df.iloc[entry_index]['Signal_Details']
                    }
                    
                    trades.append(trade_data)
                    df.iloc[i, df.columns.get_loc('Position')] = 0
        
        # Calculate strategy returns
        for i in range(1, len(df)):
            if df.iloc[i]['Position'] == 1:
                df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
        
        return trades, df
    
    def analyze_intraday_results(self, trades, df):
        """Analyze intraday backtest results"""
        
        print(colored(f"\n📊 ZM 15-MINUTE AEGS BACKTEST RESULTS", 'cyan', attrs=['bold']))
        print("=" * 70)
        
        if not trades:
            print("❌ No trades generated in 15-minute backtest")
            return None
        
        # Calculate performance metrics
        winning_trades = [t for t in trades if t['return_pct'] > 0]
        losing_trades = [t for t in trades if t['return_pct'] < 0]
        
        total_return = sum([t['return_pct'] for t in trades])
        total_dollars = sum([t['return_dollars'] for t in trades])
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        avg_hours_held = np.mean([t['hours_held'] for t in trades])
        
        # Display results
        print(f"📈 PERFORMANCE SUMMARY:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate:.1f}% ({len(winning_trades)}W / {len(losing_trades)}L)")
        print(f"   Average Hold: {avg_hours_held:.1f} hours")
        print(f"   ")
        print(colored(f"   Total Strategy Return: {total_return:+.2f}%", 'green' if total_return > 0 else 'red'))
        print(f"   Total P&L (100 shares): ${total_dollars:+.2f}")
        print(f"   Average Win: {avg_win:+.1f}%")
        print(f"   Average Loss: {avg_loss:+.1f}%")
        
        if winning_trades and losing_trades:
            profit_factor = abs(sum([t['return_pct'] for t in winning_trades])) / abs(sum([t['return_pct'] for t in losing_trades]))
            print(f"   Profit Factor: {profit_factor:.2f}")
        
        # Best and worst trades
        if trades:
            best_trade = max(trades, key=lambda x: x['return_pct'])
            worst_trade = min(trades, key=lambda x: x['return_pct'])
            print(f"   Best Trade: {best_trade['return_pct']:+.2f}%")
            print(f"   Worst Trade: {worst_trade['return_pct']:+.2f}%")
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_dollars': total_dollars,
            'avg_hours_held': avg_hours_held,
            'trades': trades
        }
    
    def show_trade_details(self, trades, limit=10):
        """Show detailed trade breakdown"""
        
        print(colored(f"\n🎯 DETAILED TRADE BREAKDOWN (Last {limit})", 'yellow', attrs=['bold']))
        print("=" * 100)
        
        for trade in trades[-limit:]:
            color = 'green' if trade['return_pct'] > 0 else 'red'
            
            print(f"\n📊 Trade #{trade['trade_id']}: {trade['entry_datetime']} → {trade['exit_datetime']}")
            print(f"   Price: ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}")
            print(colored(f"   Return: {trade['return_pct']:+.2f}% (${trade['return_dollars']:+.2f})", color))
            print(f"   Hold: {trade['hours_held']:.1f} hours ({trade['bars_held']} bars)")
            print(f"   Exit: {trade['exit_reason']}")
            print(f"   Signal Strength: {trade['entry_signal_strength']}")
            print(f"   Signals: {trade['entry_signal_details']}")
    
    def compare_to_daily(self, intraday_results):
        """Compare 15-minute results to daily results"""
        
        print(colored(f"\n📊 15-MINUTE vs DAILY AEGS COMPARISON", 'magenta', attrs=['bold']))
        print("=" * 70)
        
        # Daily results (from previous analysis)
        daily_results = {
            'trades': 2,
            'win_rate': 100.0,
            'return': 7.3,
            'hold_period': '30.5 days'
        }
        
        print(f"{'Metric':<20} {'15-Minute':<15} {'Daily':<15} {'Winner'}")
        print("=" * 70)
        
        # Total trades
        intraday_trades = intraday_results['total_trades'] if intraday_results else 0
        print(f"{'Total Trades':<20} {intraday_trades:<15} {daily_results['trades']:<15} {'15-Min' if intraday_trades > daily_results['trades'] else 'Daily'}")
        
        # Win rate
        intraday_wr = intraday_results['win_rate'] if intraday_results else 0
        print(f"{'Win Rate':<20} {intraday_wr:<15.1f} {daily_results['win_rate']:<15.1f} {'Daily' if daily_results['win_rate'] > intraday_wr else '15-Min'}")
        
        # Returns
        intraday_ret = intraday_results['total_return'] if intraday_results else 0
        print(f"{'Return %':<20} {intraday_ret:<15.2f} {daily_results['return']:<15.1f} {'Daily' if daily_results['return'] > intraday_ret else '15-Min'}")
        
        # Hold period
        intraday_hold = f"{intraday_results['avg_hours_held']:.1f}h" if intraday_results else "0h"
        print(f"{'Avg Hold':<20} {intraday_hold:<15} {daily_results['hold_period']:<15} {'15-Min (Speed)' if intraday_results else 'Daily'}")
        
        print(f"\n💡 FRACTAL MARKET EFFICIENCY ANALYSIS:")
        if intraday_results and intraday_results['total_return'] > 0:
            print("   ✅ 15-minute timeframe captured intraday inefficiencies")
        else:
            print("   ⚠️ 15-minute timeframe showed more noise than signal")
        
        if daily_results['return'] > (intraday_ret if intraday_results else 0):
            print("   ✅ Daily timeframe provided cleaner, more profitable signals")
        
        return intraday_results

def main():
    """Run ZM 15-minute AEGS backtest"""
    
    print(colored("🔥💎 ZM 15-MINUTE INTRADAY AEGS ANALYSIS 💎🔥", 'red', attrs=['bold']))
    print("Testing Fractal Market Efficiency: 15-min vs Daily timeframes")
    print("=" * 70)
    
    zm_15min = ZM15MinuteAEGS()
    
    # Get 15-minute data
    df = zm_15min.get_15min_data(days_back=60)
    
    if len(df) < 50:
        print("❌ Insufficient 15-minute data")
        return
    
    # Calculate indicators
    print("\n🔄 Calculating 15-minute AEGS indicators...")
    df = zm_15min.calculate_intraday_aegs_indicators(df)
    df = zm_15min.intraday_aegs_strategy(df)
    
    # Run backtest
    print("🔄 Running 15-minute backtest...")
    trades, df = zm_15min.run_intraday_backtest(df)
    
    # Analyze results
    results = zm_15min.analyze_intraday_results(trades, df)
    
    if results:
        # Show trade details
        zm_15min.show_trade_details(trades, limit=10)
        
        # Compare to daily
        zm_15min.compare_to_daily(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'zm_15min_aegs_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump({
                'symbol': 'ZM',
                'timeframe': '15-minute',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
    
    print(colored("\n🎯 15-MINUTE AEGS ANALYSIS COMPLETE!", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()