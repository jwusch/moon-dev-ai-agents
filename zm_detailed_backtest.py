#!/usr/bin/env python3
"""
🔥💎 ZM (ZOOM) DETAILED AEGS BACKTEST ANALYSIS 💎🔥
Deep dive into ZM's AEGS performance - one of the top performers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

def analyze_zm_detailed():
    """Detailed analysis of ZM AEGS backtest"""
    
    print(colored("🔥💎 ZM (ZOOM) DETAILED AEGS ANALYSIS 💎🔥", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Get ZM data
    symbol = 'ZM'
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='1y')
    
    if len(df) < 50:
        print("❌ Insufficient data for ZM")
        return
    
    # Get current info
    info = ticker.info
    current_price = df['Close'].iloc[-1]
    year_high = df['High'].max()
    year_low = df['Low'].min()
    
    print(f"📊 ZM (Zoom Video Communications) Overview:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   52-Week Range: ${year_low:.2f} - ${year_high:.2f}")
    print(f"   Sector: {info.get('sector', 'Technology')}")
    print(f"   Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "")
    
    # Calculate all indicators
    df = calculate_aegs_indicators(df)
    df = enhanced_aegs_strategy(df)
    
    # Run detailed backtest
    trades, performance = run_detailed_backtest(df)
    
    # Display results
    display_zm_results(trades, performance, df)
    
    # Show trade-by-trade analysis
    analyze_individual_trades(trades, df)
    
    # Technical analysis
    show_technical_context(df)
    
    return trades, performance, df

def calculate_aegs_indicators(df):
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
    
    # Volume analysis
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Daily changes
    df['Daily_Change'] = df['Close'].pct_change()
    
    # ATR for volatility context
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Ratio'] = df['ATR'] / df['Close']
    
    return df

def enhanced_aegs_strategy(df):
    """Enhanced AEGS strategy"""
    
    df['Signal'] = 0
    df['Signal_Strength'] = 0
    df['Signal_Details'] = ''
    
    for i in range(20, len(df)):
        row = df.iloc[i]
        signal_strength = 0
        signal_details = []
        
        # RSI oversold signals
        if pd.notna(row['RSI']):
            if row['RSI'] < 25:
                signal_strength += 40
                signal_details.append(f"RSI Extreme Oversold ({row['RSI']:.1f})")
            elif row['RSI'] < 30:
                signal_strength += 25
                signal_details.append(f"RSI Oversold ({row['RSI']:.1f})")
            elif row['RSI'] < 35:
                signal_strength += 15
                signal_details.append(f"RSI Weak ({row['RSI']:.1f})")
        
        # Bollinger Band signals
        if pd.notna(row['BB_Position']):
            if row['BB_Position'] < -0.1:
                signal_strength += 40
                signal_details.append(f"Well Below BB Lower ({row['BB_Position']:.2f})")
            elif row['BB_Position'] < 0:
                signal_strength += 25
                signal_details.append(f"Below BB Lower ({row['BB_Position']:.2f})")
            elif row['BB_Position'] < 0.15:
                signal_strength += 15
                signal_details.append(f"Near BB Lower ({row['BB_Position']:.2f})")
        
        # Volume surge with price drop
        if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
            daily_change_pct = row['Daily_Change'] * 100
            if row['Volume_Ratio'] > 3.0 and daily_change_pct < -3:
                signal_strength += 35
                signal_details.append(f"High Vol Selloff (Vol:{row['Volume_Ratio']:.1f}x, Price:{daily_change_pct:.1f}%)")
            elif row['Volume_Ratio'] > 2.0 and daily_change_pct < -2:
                signal_strength += 25
                signal_details.append(f"Vol Surge Down (Vol:{row['Volume_Ratio']:.1f}x, Price:{daily_change_pct:.1f}%)")
            elif row['Volume_Ratio'] > 1.5:
                signal_strength += 10
                signal_details.append(f"Above Avg Vol ({row['Volume_Ratio']:.1f}x)")
        
        # Price drop magnitude
        if pd.notna(row['Daily_Change']):
            daily_change_pct = row['Daily_Change'] * 100
            if daily_change_pct < -15:
                signal_strength += 40
                signal_details.append(f"Extreme Drop ({daily_change_pct:.1f}%)")
            elif daily_change_pct < -10:
                signal_strength += 25
                signal_details.append(f"Large Drop ({daily_change_pct:.1f}%)")
            elif daily_change_pct < -5:
                signal_strength += 15
                signal_details.append(f"Moderate Drop ({daily_change_pct:.1f}%)")
        
        df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
        df.iloc[i, df.columns.get_loc('Signal_Details')] = ' | '.join(signal_details)
        
        # Generate signal if strength >= 75
        if signal_strength >= 75:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
    
    return df

def run_detailed_backtest(df):
    """Run detailed backtest with trade tracking"""
    
    df['Position'] = 0
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = 0
    df['Trade_ID'] = 0
    
    position = 0
    entry_price = 0
    entry_date = None
    entry_index = 0
    trades = []
    trade_id = 0
    
    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df.iloc[i]['Close']
        
        if df.iloc[i]['Signal'] == 1 and position == 0:
            # Enter position
            position = 1
            entry_price = current_price
            entry_date = current_date
            entry_index = i
            trade_id += 1
            
            df.iloc[i, df.columns.get_loc('Position')] = 1
            df.iloc[i, df.columns.get_loc('Trade_ID')] = trade_id
            
        elif position == 1:
            df.iloc[i, df.columns.get_loc('Position')] = 1
            df.iloc[i, df.columns.get_loc('Trade_ID')] = trade_id
            
            # Calculate current performance
            returns = (current_price - entry_price) / entry_price
            days_held = (current_date - entry_date).days
            
            # Exit conditions
            exit_trade = False
            exit_reason = ""
            
            # Dynamic exits based on ATR
            atr_ratio = df.iloc[i].get('ATR_Ratio', 0.05)
            profit_target = 0.3 if atr_ratio < 0.05 else 0.5
            stop_loss = -0.25
            
            if returns >= profit_target:
                exit_trade = True
                exit_reason = f"Profit Target ({profit_target*100:.0f}%)"
            elif returns <= stop_loss:
                exit_trade = True
                exit_reason = "Stop Loss (-25%)"
            elif days_held >= 30 and returns > 0:
                exit_trade = True
                exit_reason = "Time Exit (Profitable)"
            elif days_held >= 60:
                exit_trade = True
                exit_reason = "Force Exit (60 days)"
            
            if exit_trade:
                # Exit position
                position = 0
                
                # Record trade details
                trade_data = {
                    'trade_id': trade_id,
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'return_pct': returns * 100,
                    'return_dollars': (current_price - entry_price) * 100,  # Assume 100 shares
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'entry_rsi': df.iloc[entry_index]['RSI'],
                    'entry_bb_position': df.iloc[entry_index]['BB_Position'],
                    'entry_volume_ratio': df.iloc[entry_index]['Volume_Ratio'],
                    'entry_signal_strength': df.iloc[entry_index]['Signal_Strength'],
                    'entry_signal_details': df.iloc[entry_index]['Signal_Details'],
                    'max_gain_during_trade': 0,  # Will calculate
                    'max_loss_during_trade': 0   # Will calculate
                }
                
                # Calculate max gain/loss during trade
                trade_period = df.iloc[entry_index:i+1]
                max_price = trade_period['High'].max()
                min_price = trade_period['Low'].min()
                trade_data['max_gain_during_trade'] = (max_price - entry_price) / entry_price * 100
                trade_data['max_loss_during_trade'] = (min_price - entry_price) / entry_price * 100
                
                trades.append(trade_data)
                df.iloc[i, df.columns.get_loc('Position')] = 0
    
    # Calculate strategy returns
    for i in range(1, len(df)):
        if df.iloc[i]['Position'] == 1:
            df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
    
    # Calculate performance metrics
    total_return = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    
    winning_trades = [t for t in trades if t['return_pct'] > 0]
    losing_trades = [t for t in trades if t['return_pct'] < 0]
    
    performance = {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'avg_win': np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0,
        'avg_loss': np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0,
        'avg_days_held': np.mean([t['days_held'] for t in trades]) if trades else 0,
        'strategy_return': total_return * 100,
        'buy_hold_return': buy_hold_return * 100,
        'excess_return': (total_return - buy_hold_return) * 100,
        'total_dollars_pnl': sum([t['return_dollars'] for t in trades]),
        'best_trade': max([t['return_pct'] for t in trades]) if trades else 0,
        'worst_trade': min([t['return_pct'] for t in trades]) if trades else 0
    }
    
    return trades, performance

def display_zm_results(trades, performance, df):
    """Display comprehensive ZM results"""
    
    print(colored("\n📈 ZM AEGS PERFORMANCE SUMMARY", 'green', attrs=['bold']))
    print("=" * 60)
    
    p = performance
    print(f"   Total Trades: {p['total_trades']}")
    print(f"   Win Rate: {p['win_rate']:.1f}% ({p['winning_trades']}W / {p['losing_trades']}L)")
    print(f"   Average Hold Period: {p['avg_days_held']:.1f} days")
    print(f"   ")
    print(colored(f"   Strategy Return: {p['strategy_return']:+.1f}%", 'green' if p['strategy_return'] > 0 else 'red'))
    print(f"   Buy & Hold Return: {p['buy_hold_return']:+.1f}%")
    print(colored(f"   Excess Return: {p['excess_return']:+.1f}%", 'green' if p['excess_return'] > 0 else 'red'))
    print(f"   ")
    print(f"   Average Win: {p['avg_win']:+.1f}%")
    print(f"   Average Loss: {p['avg_loss']:+.1f}%")
    print(f"   Best Trade: {p['best_trade']:+.1f}%")
    print(f"   Worst Trade: {p['worst_trade']:+.1f}%")
    print(f"   ")
    print(f"   Total P&L (100 shares): ${p['total_dollars_pnl']:+.2f}")

def analyze_individual_trades(trades, df):
    """Analyze each trade in detail"""
    
    print(colored(f"\n🎯 DETAILED TRADE-BY-TRADE ANALYSIS", 'yellow', attrs=['bold']))
    print("=" * 100)
    
    for i, trade in enumerate(trades, 1):
        color = 'green' if trade['return_pct'] > 0 else 'red'
        
        print(colored(f"\n📊 TRADE #{i} ({trade['entry_date']} → {trade['exit_date']})", 'cyan', attrs=['bold']))
        print(f"   Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}")
        print(colored(f"   Return: {trade['return_pct']:+.1f}% (${trade['return_dollars']:+.2f} on 100 shares)", color))
        print(f"   Hold Period: {trade['days_held']} days")
        print(f"   Exit Reason: {trade['exit_reason']}")
        
        print(f"\n   📈 Entry Conditions:")
        print(f"      RSI: {trade['entry_rsi']:.1f}")
        print(f"      BB Position: {trade['entry_bb_position']:.3f}")
        print(f"      Volume Ratio: {trade['entry_volume_ratio']:.1f}x")
        print(f"      Signal Strength: {trade['entry_signal_strength']}/100")
        print(f"      Signals: {trade['entry_signal_details']}")
        
        print(f"\n   📊 Trade Performance:")
        print(f"      Max Gain During Trade: {trade['max_gain_during_trade']:+.1f}%")
        print(f"      Max Loss During Trade: {trade['max_loss_during_trade']:+.1f}%")
        
        # Risk/Reward analysis
        potential_gain = trade['max_gain_during_trade'] - trade['return_pct']
        print(f"      Potential Left on Table: {potential_gain:.1f}%")
        
        # Exit timing analysis
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        trade_period = df[entry_date:exit_date]
        
        if len(trade_period) > 1:
            final_rsi = trade_period['RSI'].iloc[-1]
            final_bb_pos = trade_period['BB_Position'].iloc[-1]
            print(f"      Exit RSI: {final_rsi:.1f}")
            print(f"      Exit BB Position: {final_bb_pos:.3f}")

def show_technical_context(df):
    """Show current technical context"""
    
    print(colored(f"\n📊 CURRENT TECHNICAL CONTEXT", 'magenta', attrs=['bold']))
    print("=" * 50)
    
    latest = df.iloc[-1]
    recent_5d = df.tail(5)
    
    print(f"   Current Price: ${latest['Close']:.2f}")
    print(f"   Current RSI: {latest['RSI']:.1f}")
    print(f"   Current BB Position: {latest['BB_Position']:.3f}")
    print(f"   Current Volume Ratio: {latest['Volume_Ratio']:.1f}x")
    print(f"   Current ATR Ratio: {latest['ATR_Ratio']:.3f} ({latest['ATR_Ratio']*100:.1f}%)")
    print(f"   5-Day Price Change: {(latest['Close'] - recent_5d['Close'].iloc[0]) / recent_5d['Close'].iloc[0] * 100:+.1f}%")
    
    # Signal analysis
    current_signal_strength = latest['Signal_Strength']
    print(f"\n   Current AEGS Signal Strength: {current_signal_strength}/100")
    
    if current_signal_strength >= 75:
        print(colored("   🚨 CURRENT BUY SIGNAL ACTIVE! 🚨", 'green', attrs=['bold']))
    elif current_signal_strength >= 50:
        print(colored("   ⚠️  Moderate oversold conditions", 'yellow'))
    else:
        print("   ℹ️  No current AEGS signal")
    
    if latest['Signal_Details']:
        print(f"   Signal Details: {latest['Signal_Details']}")

def save_zm_analysis(trades, performance, df):
    """Save detailed ZM analysis"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comprehensive report
    report = {
        'symbol': 'ZM',
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_summary': performance,
        'detailed_trades': trades,
        'current_technical': {
            'price': float(df['Close'].iloc[-1]),
            'rsi': float(df['RSI'].iloc[-1]),
            'bb_position': float(df['BB_Position'].iloc[-1]),
            'volume_ratio': float(df['Volume_Ratio'].iloc[-1]),
            'signal_strength': float(df['Signal_Strength'].iloc[-1]),
            'signal_details': df['Signal_Details'].iloc[-1]
        }
    }
    
    filename = f'zm_detailed_analysis_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed ZM analysis saved to: {filename}")
    return filename

if __name__ == "__main__":
    trades, performance, df = analyze_zm_detailed()
    save_zm_analysis(trades, performance, df)