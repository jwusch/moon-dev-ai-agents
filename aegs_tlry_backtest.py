"""
🔥💎 AEGS BACKTEST: TLRY (Tilray) 💎🔥
Enhanced cannabis stock analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

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
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Daily change
    df['Daily_Change'] = df['Close'].pct_change()
    
    return df

def aegs_strategy(df):
    """AEGS buy/sell signals with enhanced exit logic"""
    
    df['Signal'] = 0
    df['Signal_Strength'] = 0
    
    for i in range(20, len(df)):  # Start after indicators are calculated
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

def backtest_aegs_tlry(period='2y'):
    """Backtest AEGS strategy on TLRY"""
    
    symbol = 'TLRY'
    print(f'🔥💎 AEGS BACKTEST: {symbol} (Tilray) 💎🔥')
    print('=' * 60)
    
    # Get data
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if len(df) < 50:
            print(f'❌ Insufficient data for {symbol}')
            return None
            
    except Exception as e:
        print(f'❌ Error fetching {symbol}: {e}')
        return None
    
    print(f'📊 Data Period: {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}')
    print(f'📈 Price Range: ${df["Close"].min():.2f} - ${df["Close"].max():.2f}')
    print(f'📊 Current Price: ${df["Close"].iloc[-1]:.2f}')
    print()
    
    # Calculate indicators and signals
    df = calculate_aegs_indicators(df)
    df = aegs_strategy(df)
    
    # Enhanced backtest with better position management
    df['Position'] = 0
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = 0
    
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    
    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df.iloc[i]['Close']
        
        if df.iloc[i]['Signal'] == 1 and position == 0:
            # Enter long position
            position = 1
            entry_price = current_price
            entry_date = current_date
            df.iloc[i, df.columns.get_loc('Position')] = 1
            
        elif position == 1:
            df.iloc[i, df.columns.get_loc('Position')] = 1
            
            # Calculate current return
            current_return = (current_price - entry_price) / entry_price
            days_held = (current_date - entry_date).days
            
            # Enhanced exit conditions
            exit_trade = False
            exit_reason = ''
            
            # 1. Take profit levels
            if current_return >= 0.5:  # 50% profit
                exit_trade = True
                exit_reason = 'Take Profit 50%'
            elif current_return >= 0.3 and days_held >= 30:  # 30% after 30 days
                exit_trade = True
                exit_reason = 'Take Profit 30% (30+ days)'
                
            # 2. Stop loss
            elif current_return <= -0.2:  # 20% stop loss
                exit_trade = True
                exit_reason = 'Stop Loss -20%'
                
            # 3. Time-based exit
            elif days_held >= 60 and current_return > 0:  # Exit with any profit after 60 days
                exit_trade = True
                exit_reason = 'Time Exit (60+ days, profitable)'
            elif days_held >= 90:  # Force exit after 90 days
                exit_trade = True
                exit_reason = 'Force Exit (90+ days)'
                
            # 4. Technical exit - RSI overbought
            elif pd.notna(df.iloc[i].get('RSI')) and df.iloc[i]['RSI'] > 70 and current_return > 0.1:
                exit_trade = True
                exit_reason = 'RSI Overbought Exit'
            
            if exit_trade:
                # Exit position
                position = 0
                
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return_pct': current_return * 100,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'max_gain': current_return * 100  # Simplified for this example
                })
                
                df.iloc[i, df.columns.get_loc('Position')] = 0
    
    # Calculate strategy returns
    for i in range(1, len(df)):
        if df.iloc[i]['Position'] == 1:
            df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
    
    # Performance metrics
    total_return = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    
    # Trade analysis
    winning_trades = [t for t in trades if t['return_pct'] > 0]
    losing_trades = [t for t in trades if t['return_pct'] < 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
    avg_hold_days = np.mean([t['days_held'] for t in trades]) if trades else 0
    
    # Risk metrics
    if len(df['Strategy_Returns']) > 0:
        strategy_vol = df['Strategy_Returns'].std() * np.sqrt(252) * 100
        max_drawdown = ((df['Strategy_Returns'].cumsum().cummax() - df['Strategy_Returns'].cumsum()).max()) * 100
    else:
        strategy_vol = 0
        max_drawdown = 0
    
    # Display results
    print(f'🎯 AEGS BACKTEST RESULTS:')
    print(f'   Total Trades: {len(trades)}')
    print(f'   Win Rate: {win_rate:.1f}%')
    print(f'   Average Win: {avg_win:.1f}%')
    print(f'   Average Loss: {avg_loss:.1f}%')
    print(f'   Average Hold: {avg_hold_days:.1f} days')
    print(f'   Strategy Return: {total_return*100:.1f}%')
    print(f'   Buy & Hold Return: {buy_hold_return*100:.1f}%')
    print(f'   Excess Return: {(total_return - buy_hold_return)*100:.1f}%')
    print(f'   Strategy Volatility: {strategy_vol:.1f}%')
    print(f'   Max Drawdown: {max_drawdown:.1f}%')
    
    if total_return > 0:
        sharpe = (total_return * 100) / strategy_vol if strategy_vol > 0 else 0
        print(f'   Sharpe Ratio: {sharpe:.2f}')
    
    if trades:
        print(f'\n🎯 RECENT TRADES:')
        for trade in trades[-8:]:  # Last 8 trades
            color = '🟢' if trade['return_pct'] > 0 else '🔴'
            print(f'   {color} {trade["entry_date"]}: {trade["return_pct"]:+.1f}% ({trade["days_held"]}d) - {trade["exit_reason"]}')
            
        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'avg_return': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['avg_return'] += trade['return_pct']
        
        print(f'\n📊 EXIT ANALYSIS:')
        for reason, stats in exit_reasons.items():
            avg_ret = stats['avg_return'] / stats['count']
            print(f'   {reason}: {stats["count"]} trades, {avg_ret:+.1f}% avg')
    
    # Market context
    print(f'\n📈 TLRY CONTEXT:')
    print(f'   Cannabis sector stock - high volatility expected')
    print(f'   Regulatory and sentiment driven')
    print(f'   AEGS works well on oversold bounces in volatile names')
    
    # Save results
    results = {
        'symbol': symbol,
        'sector': 'Cannabis',
        'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold_days': avg_hold_days,
        'strategy_return': total_return * 100,
        'buy_hold_return': buy_hold_return * 100,
        'excess_return': (total_return - buy_hold_return) * 100,
        'strategy_volatility': strategy_vol,
        'max_drawdown': max_drawdown,
        'trades': trades[-15:]  # Last 15 trades
    }
    
    return results

if __name__ == '__main__':
    results = backtest_aegs_tlry(period='2y')
    
    if results:
        # Save to file
        filename = f'aegs_tlry_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n💾 Results saved to {filename}')
        
        # Summary assessment
        print(f'\n💡 AEGS x TLRY ASSESSMENT:')
        if results['excess_return'] > 0:
            print(f'   ✅ Strategy outperformed buy & hold by {results["excess_return"]:.1f}%')
        else:
            print(f'   ⚠️ Strategy underperformed by {abs(results["excess_return"]):.1f}%')
            
        if results['win_rate'] > 60:
            print(f'   ✅ Strong win rate of {results["win_rate"]:.1f}%')
        elif results['win_rate'] > 45:
            print(f'   ⚡ Decent win rate of {results["win_rate"]:.1f}%')
        else:
            print(f'   ⚠️ Low win rate of {results["win_rate"]:.1f}%')