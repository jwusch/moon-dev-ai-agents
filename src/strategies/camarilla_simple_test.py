"""
🌙 Simple Camarilla Strategy Test
A working example of the Camarilla strategy

Author: Moon Dev
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from camarilla_strategy import CamarillaLevels

def create_sample_data():
    """Create simple sample data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Create trending market with volatility
    price = 100
    prices = []
    
    for i in range(len(dates)):
        # Add trend and noise
        trend = 0.001 * i  # Slight uptrend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        price = price * (1 + trend + noise)
        prices.append(price)
    
    # Create OHLCV
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.005, len(dates)))
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + abs(np.random.normal(0, 0.01, len(dates))))
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - abs(np.random.normal(0, 0.01, len(dates))))
    df['Volume'] = np.random.randint(1000, 10000, len(dates))
    
    return df

def test_camarilla_signals():
    """Test Camarilla trading signals"""
    print("🌙 Camarilla Strategy Signal Test")
    print("="*60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created {len(df)} days of sample data")
    print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    
    # Calculate Camarilla levels
    levels_df = CamarillaLevels.calculate_series(df.copy())
    
    # Initialize tracking
    trades = []
    position = None
    
    # Simulate trading
    for i in range(1, len(df)):
        if pd.isna(levels_df['R4'].iloc[i]):
            continue
            
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        
        # Get today's levels
        r4 = levels_df['R4'].iloc[i]
        r3 = levels_df['R3'].iloc[i]
        r2 = levels_df['R2'].iloc[i]
        r1 = levels_df['R1'].iloc[i]
        pivot = levels_df['Pivot'].iloc[i]
        s1 = levels_df['S1'].iloc[i]
        s2 = levels_df['S2'].iloc[i]
        s3 = levels_df['S3'].iloc[i]
        s4 = levels_df['S4'].iloc[i]
        
        # Check for signals (no position)
        if not position:
            # Range-bound: Buy near S3
            if abs(current_price - s3) / s3 < 0.001 and current_price > s4:
                position = {
                    'type': 'range_long',
                    'entry': current_price,
                    'stop': s4 * 0.999,
                    'target': r1,
                    'date': current_date
                }
                print(f"\n🟢 LONG at S3: ${current_price:.2f}")
                print(f"   Stop: ${position['stop']:.2f}, Target: ${position['target']:.2f}")
            
            # Range-bound: Sell near R3
            elif abs(current_price - r3) / r3 < 0.001 and current_price < r4:
                position = {
                    'type': 'range_short',
                    'entry': current_price,
                    'stop': r4 * 1.001,
                    'target': s1,
                    'date': current_date
                }
                print(f"\n🔴 SHORT at R3: ${current_price:.2f}")
                print(f"   Stop: ${position['stop']:.2f}, Target: ${position['target']:.2f}")
            
            # Breakout: Buy above R4
            elif current_price > r4 * 1.001:
                position = {
                    'type': 'breakout_long',
                    'entry': current_price,
                    'stop': r3,
                    'target': current_price + 2 * (current_price - r3),
                    'date': current_date
                }
                print(f"\n🚀 BREAKOUT LONG above R4: ${current_price:.2f}")
                print(f"   Stop: ${position['stop']:.2f}, Target: ${position['target']:.2f}")
            
            # Breakout: Sell below S4
            elif current_price < s4 * 0.999:
                position = {
                    'type': 'breakout_short',
                    'entry': current_price,
                    'stop': s3,
                    'target': current_price - 2 * (s3 - current_price),
                    'date': current_date
                }
                print(f"\n📉 BREAKOUT SHORT below S4: ${current_price:.2f}")
                print(f"   Stop: ${position['stop']:.2f}, Target: ${position['target']:.2f}")
        
        # Check exits
        else:
            exit_price = None
            exit_reason = None
            
            if position['type'] in ['range_long', 'breakout_long']:
                if current_price <= position['stop']:
                    exit_price = current_price
                    exit_reason = 'Stop Loss'
                elif current_price >= position['target']:
                    exit_price = current_price
                    exit_reason = 'Take Profit'
            else:  # Short positions
                if current_price >= position['stop']:
                    exit_price = current_price
                    exit_reason = 'Stop Loss'
                elif current_price <= position['target']:
                    exit_price = current_price
                    exit_reason = 'Take Profit'
            
            if exit_price:
                # Calculate P&L
                if position['type'] in ['range_long', 'breakout_long']:
                    pnl = (exit_price - position['entry']) / position['entry'] * 100
                else:
                    pnl = (position['entry'] - exit_price) / position['entry'] * 100
                
                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'duration': (current_date - position['date']).days
                })
                
                print(f"⚪ CLOSED: {exit_reason} at ${exit_price:.2f}")
                print(f"   P&L: {pnl:+.2f}%")
                
                position = None
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRADING SUMMARY")
    print("="*60)
    
    if trades:
        df_trades = pd.DataFrame(trades)
        
        print(f"\nTotal Trades: {len(trades)}")
        print(f"Win Rate: {(df_trades['pnl'] > 0).mean() * 100:.1f}%")
        print(f"Average P&L: {df_trades['pnl'].mean():+.2f}%")
        print(f"Best Trade: {df_trades['pnl'].max():+.2f}%")
        print(f"Worst Trade: {df_trades['pnl'].min():+.2f}%")
        print(f"Total Return: {df_trades['pnl'].sum():+.2f}%")
        
        print("\nTrade Breakdown:")
        for trade_type in df_trades['type'].unique():
            type_trades = df_trades[df_trades['type'] == trade_type]
            print(f"\n{trade_type}:")
            print(f"  Count: {len(type_trades)}")
            print(f"  Win Rate: {(type_trades['pnl'] > 0).mean() * 100:.1f}%")
            print(f"  Avg P&L: {type_trades['pnl'].mean():+.2f}%")
    else:
        print("No trades executed")
    
    # Show current levels
    print("\n📈 Today's Camarilla Levels:")
    latest = levels_df.iloc[-1]
    print(f"R4 (Breakout): ${latest['R4']:.2f}")
    print(f"R3 (Resistance): ${latest['R3']:.2f}")
    print(f"R2: ${latest['R2']:.2f}")
    print(f"R1: ${latest['R1']:.2f}")
    print(f"Pivot: ${latest['Pivot']:.2f}")
    print(f"S1: ${latest['S1']:.2f}")
    print(f"S2: ${latest['S2']:.2f}")
    print(f"S3 (Support): ${latest['S3']:.2f}")
    print(f"S4 (Breakout): ${latest['S4']:.2f}")
    
    print(f"\nCurrent Price: ${df['Close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    test_camarilla_signals()