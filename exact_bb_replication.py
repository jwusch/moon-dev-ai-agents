"""
🔍 Exact Bollinger Band Alpha Replication
Replicating the EXACT conditions that generated α=5.45 with 100% win rate
Uses identical methodology from alpha_source_mapper.py

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import talib
import yfinance as yf

def replicate_exact_alpha_discovery():
    """Replicate the exact alpha discovery methodology"""
    
    print("🔍 EXACT ALPHA SOURCE REPLICATION")
    print("=" * 60)
    print("Replicating BB_Reversion_1h that achieved α=5.45")
    
    # === STEP 1: IDENTICAL DATA PREPARATION ===
    print("\n📊 Step 1: Data preparation (identical to alpha source)...")
    
    symbol = "VXX"
    timeframe = "1h"
    
    # Use same period as alpha source
    period = "730d"  # 2 years as used in alpha source
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe)
        
        if df.columns.nlevels > 1:
            df.columns = [col[0] for col in df.columns]
            
        print(f"✅ Downloaded {len(df)} hours of VXX data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # === STEP 2: IDENTICAL INDICATOR CALCULATION ===
    print("\n🔧 Step 2: Indicator calculation (matching alpha source)...")
    
    # From alpha_source_mapper.py - timeframe adjustments for 1h
    short, medium, long = 6, 24, 72  # 1h timeframe periods
    
    # Core indicators (exactly as in alpha source)
    df['SMA_Short'] = df['Close'].rolling(short).mean()
    df['SMA_Medium'] = df['Close'].rolling(medium).mean()
    df['SMA_Long'] = df['Close'].rolling(long).mean()
    df['Distance_Short'] = (df['Close'] - df['SMA_Short']) / df['SMA_Short'] * 100
    df['Distance_Medium'] = (df['Close'] - df['SMA_Medium']) / df['SMA_Medium'] * 100
    df['RSI'] = talib.RSI(df['Close'].values, medium)
    
    # Bollinger Bands (exactly as in alpha source)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values, medium, 2, 2)
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    print(f"✅ Calculated indicators for {len(df.dropna())} periods")
    
    # === STEP 3: EXACT STRATEGY LOGIC ===
    print("\n🎯 Step 3: Exact strategy replication...")
    
    df = df.dropna()
    
    # Exact entry/exit from alpha source
    entry_condition = df['BB_Position'] < 0.2
    exit_condition = df['BB_Position'] > 0.5
    
    print(f"Entry signals: {entry_condition.sum()}")
    print(f"Exit signals: {exit_condition.sum()}")
    
    # === STEP 4: EXACT BACKTESTING METHODOLOGY ===
    print("\n💹 Step 4: Exact backtest replication...")
    
    # Replicate backtest_alpha_source function logic
    trades = []
    position = 0
    entry_price = None
    entry_idx = None
    
    initial_capital = 10000
    current_capital = initial_capital
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        
        # Check for entry
        if position == 0 and entry_condition.iloc[i]:
            position = 1
            entry_price = current_price
            entry_idx = i
        
        # Check for exit
        elif position == 1 and exit_condition.iloc[i]:
            if entry_price is not None and entry_idx is not None:
                hold_periods = i - entry_idx
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Position sizing (95% of capital like our other strategies)
                position_size = current_capital * 0.95
                shares = position_size / entry_price
                pnl_dollars = shares * (current_price - entry_price)
                current_capital += pnl_dollars
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'hold_periods': hold_periods,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'win': pnl_pct > 0
                })
            
            position = 0
            entry_price = None
            entry_idx = None
    
    # === STEP 5: ANALYZE RESULTS ===
    print(f"\n📊 Step 5: Results analysis...")
    
    if not trades:
        print("❌ No trades found")
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics exactly like alpha source
    total_trades = len(trades_df)
    win_rate = trades_df['win'].mean() * 100
    total_return_pct = (current_capital / initial_capital - 1) * 100
    
    # Alpha source specific metrics
    avg_hold_periods = trades_df['hold_periods'].mean()
    avg_hold_hours = avg_hold_periods * 1  # 1 hour per period
    
    # Risk metrics
    returns = trades_df['pnl_pct'].values
    wins = trades_df[trades_df['win']]['pnl_pct']
    losses = trades_df[~trades_df['win']]['pnl_pct']
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
    
    # Sharpe approximation
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    # Alpha score (risk-adjusted return per trade - from alpha source)
    alpha_score = total_return_pct / total_trades * (win_rate / 100) if total_trades > 0 else 0
    
    print(f"\n🏆 EXACT REPLICATION RESULTS:")
    print(f"{'='*50}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Return: {total_return_pct:+.1f}%")
    print(f"Alpha Score: {alpha_score:.2f}")
    print(f"Avg Hold Hours: {avg_hold_hours:.1f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Best Trade: {returns.max():+.1f}%")
    print(f"Worst Trade: {returns.min():+.1f}%")
    
    print(f"\n🔬 COMPARISON TO DISCOVERY:")
    print(f"Discovery: α=5.45, 100% win rate, 580-min hold")
    print(f"Replication: α={alpha_score:.2f}, {win_rate:.1f}% win rate, {avg_hold_hours*60:.0f}-min hold")
    
    # Calculate match score
    win_rate_match = min(win_rate, 100) / 100
    alpha_match = min(alpha_score / 5.45, 1) if alpha_score > 0 else 0
    overall_match = (win_rate_match + alpha_match) / 2
    
    print(f"Match Score: {overall_match:.2f} (0=no match, 1=perfect match)")
    
    if overall_match > 0.8:
        print("✅ EXCELLENT REPLICATION - Strategy logic validated!")
    elif overall_match > 0.5:
        print("✅ GOOD REPLICATION - Minor differences")
    elif overall_match > 0.2:
        print("⚠️ PARTIAL REPLICATION - Some differences in methodology")
    else:
        print("❌ POOR REPLICATION - Significant methodology differences")
    
    # Show recent trades
    print(f"\n📋 RECENT TRADE EXAMPLES:")
    recent_trades = trades_df.tail(10)
    for i, trade in recent_trades.iterrows():
        status = "WIN" if trade['win'] else "LOSS"
        print(f"  Trade {i}: {trade['pnl_pct']:+.1f}% in {trade['hold_periods']} hours [{status}]")
    
    return {
        'trades_df': trades_df,
        'df': df,
        'metrics': {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'alpha_score': alpha_score,
            'avg_hold_hours': avg_hold_hours,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe
        }
    }

if __name__ == "__main__":
    result = replicate_exact_alpha_discovery()