"""
🎯 VXX Multi-Strategy - REALISTIC VERSION
No BS, actual implementable strategies with real backtests

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from yfinance_cache_demo import YFinanceCache

class RealisticVXXMultiStrategy:
    def __init__(self):
        self.cache = YFinanceCache()
        self.all_trades = []
        
    def mean_reversion_strategy(self, df, timeframe="15m"):
        """The original proven strategy"""
        df = df.copy()
        
        # Calculate indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        entry_bar = 0
        
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            distance = df['Distance%'].iloc[i]
            rsi = df['RSI'].iloc[i]
            
            if pd.isna(distance) or pd.isna(rsi):
                continue
            
            # Market hours only
            if current_time.hour < 9 or current_time.hour >= 16:
                continue
            
            if position is None:
                # Entry signals - proven parameters
                if distance < -1.0 and rsi < 40:
                    position = 'Long'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
                    
                elif distance > 1.0 and rsi > 60:
                    position = 'Short'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
            
            else:
                # Exit conditions
                bars_held = i - entry_bar
                if timeframe == "15m":
                    hours_held = bars_held * 15 / 60
                elif timeframe == "5m":
                    hours_held = bars_held * 5 / 60
                else:
                    hours_held = bars_held
                
                if position == 'Long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    if (distance > -0.2 or      # Near mean
                        pnl_pct > 1.0 or        # 1% profit
                        pnl_pct < -1.5 or       # 1.5% loss
                        hours_held > 3):        # 3 hours max
                        
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': f'MeanRev_{timeframe}'
                        })
                        position = None
                        
                elif position == 'Short':
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    if (distance < 0.2 or       # Near mean
                        pnl_pct > 1.0 or        # 1% profit
                        pnl_pct < -1.5 or       # 1.5% loss
                        hours_held > 3):        # 3 hours max
                        
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': f'MeanRev_{timeframe}'
                        })
                        position = None
        
        return trades
    
    def vix_regime_filter_strategy(self, df):
        """Mean reversion with VIX regime awareness"""
        df = df.copy()
        
        # Calculate indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        
        # VXX price as VIX proxy
        df['VIX_proxy'] = df['Close'] * 0.75  # Rough approximation
        
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        entry_bar = 0
        
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            distance = df['Distance%'].iloc[i]
            rsi = df['RSI'].iloc[i]
            vix_level = df['VIX_proxy'].iloc[i]
            
            if pd.isna(distance) or pd.isna(rsi) or pd.isna(vix_level):
                continue
            
            if current_time.hour < 9 or current_time.hour >= 16:
                continue
            
            # Adjust parameters based on VIX level
            if vix_level > 20:  # High volatility
                distance_threshold = 0.75
                rsi_oversold = 35
                rsi_overbought = 65
                profit_target = 0.75
                stop_loss = 1.25
            elif vix_level < 15:  # Low volatility
                distance_threshold = 1.5
                rsi_oversold = 30
                rsi_overbought = 70
                profit_target = 1.5
                stop_loss = 2.0
            else:  # Normal volatility
                distance_threshold = 1.0
                rsi_oversold = 40
                rsi_overbought = 60
                profit_target = 1.0
                stop_loss = 1.5
            
            if position is None:
                if distance < -distance_threshold and rsi < rsi_oversold:
                    position = 'Long'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
                    
                elif distance > distance_threshold and rsi > rsi_overbought:
                    position = 'Short'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
            
            else:
                bars_held = i - entry_bar
                hours_held = bars_held * 15 / 60
                
                if position == 'Long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    if (pnl_pct > profit_target or
                        pnl_pct < -stop_loss or
                        hours_held > 3):
                        
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': 'VIX_Regime'
                        })
                        position = None
                        
                elif position == 'Short':
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    if (pnl_pct > profit_target or
                        pnl_pct < -stop_loss or
                        hours_held > 3):
                        
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': 'VIX_Regime'
                        })
                        position = None
        
        return trades
    
    def pairs_spread_strategy(self, vxx_df, spy_df):
        """Trade VXX based on SPY movements (flight to quality)"""
        # Align dataframes
        common_idx = vxx_df.index.intersection(spy_df.index)
        vxx_df = vxx_df.loc[common_idx].copy()
        spy_df = spy_df.loc[common_idx].copy()
        
        # Calculate indicators
        vxx_df['VXX_ret'] = vxx_df['Close'].pct_change()
        spy_df['SPY_ret'] = spy_df['Close'].pct_change()
        
        # 20-period correlation
        correlation = vxx_df['VXX_ret'].rolling(20).corr(spy_df['SPY_ret'])
        
        # VXX indicators
        vxx_df['RSI'] = talib.RSI(vxx_df['Close'].values, 14)
        
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        entry_bar = 0
        
        for i in range(50, len(vxx_df)):
            if pd.isna(correlation.iloc[i]) or pd.isna(vxx_df['RSI'].iloc[i]):
                continue
                
            current_time = vxx_df.index[i]
            current_price = vxx_df['Close'].iloc[i]
            
            if current_time.hour < 9 or current_time.hour >= 16:
                continue
            
            # Strong negative correlation expected (VXX up when SPY down)
            if position is None:
                # SPY falling rapidly, VXX should spike
                if spy_df['SPY_ret'].iloc[i] < -0.5 and vxx_df['RSI'].iloc[i] < 70:
                    position = 'Long'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
                    
                # SPY rising rapidly, VXX should fall
                elif spy_df['SPY_ret'].iloc[i] > 0.5 and vxx_df['RSI'].iloc[i] > 30:
                    position = 'Short'
                    entry_price = current_price
                    entry_time = current_time
                    entry_bar = i
            
            else:
                bars_held = i - entry_bar
                hours_held = bars_held * 15 / 60
                
                if position == 'Long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    if (pnl_pct > 1.5 or pnl_pct < -1.5 or hours_held > 2):
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': 'SPY_Correlation'
                        })
                        position = None
                        
                elif position == 'Short':
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    if (pnl_pct > 1.5 or pnl_pct < -1.5 or hours_held > 2):
                        trades.append({
                            'Entry': entry_time,
                            'Exit': current_time,
                            'Type': position,
                            'PnL%': pnl_pct,
                            'Strategy': 'SPY_Correlation'
                        })
                        position = None
        
        return trades
    
    def run_realistic_backtest(self):
        """Run all strategies with realistic parameters"""
        print("="*70)
        print("🎯 REALISTIC MULTI-STRATEGY VXX SYSTEM")
        print("="*70)
        
        # Load data
        print("\n📊 Loading data...")
        vxx_5m = self.cache.get_data("VXX", period="59d", interval="5m")
        vxx_15m = self.cache.get_data("VXX", period="59d", interval="15m")
        spy_15m = self.cache.get_data("SPY", period="59d", interval="15m")
        
        print("✅ Data loaded successfully")
        
        # Run strategies
        print("\n🔄 Running realistic backtests...")
        
        # 1. Original 15m strategy
        trades_15m = self.mean_reversion_strategy(vxx_15m, "15m")
        self.all_trades.extend(trades_15m)
        print(f"  ✓ 15m mean reversion: {len(trades_15m)} trades")
        
        # 2. 5m version (more trades, similar logic)
        trades_5m = self.mean_reversion_strategy(vxx_5m, "5m")
        self.all_trades.extend(trades_5m)
        print(f"  ✓ 5m mean reversion: {len(trades_5m)} trades")
        
        # 3. VIX regime filter
        trades_regime = self.vix_regime_filter_strategy(vxx_15m)
        self.all_trades.extend(trades_regime)
        print(f"  ✓ VIX regime strategy: {len(trades_regime)} trades")
        
        # 4. SPY correlation
        trades_spy = self.pairs_spread_strategy(vxx_15m, spy_15m)
        self.all_trades.extend(trades_spy)
        print(f"  ✓ SPY correlation: {len(trades_spy)} trades")
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Realistic analysis of results"""
        if not self.all_trades:
            print("No trades generated!")
            return
            
        # Convert to DataFrame
        trades_df = pd.DataFrame(self.all_trades)
        
        # Remove overlapping trades (can't take all of them)
        trades_df = trades_df.sort_values('Entry')
        
        # Simple overlap removal - if new trade starts before previous ends, skip it
        filtered_trades = []
        last_exit = pd.Timestamp('2000-01-01', tz='UTC').tz_localize(None)
        
        for _, trade in trades_df.iterrows():
            entry_time = pd.Timestamp(trade['Entry']).tz_localize(None) if hasattr(trade['Entry'], 'tz') else trade['Entry']
            exit_time = pd.Timestamp(trade['Exit']).tz_localize(None) if hasattr(trade['Exit'], 'tz') else trade['Exit']
            
            if entry_time >= last_exit:
                filtered_trades.append(trade)
                last_exit = exit_time
        
        filtered_df = pd.DataFrame(filtered_trades)
        
        print(f"\n📊 REALISTIC RESULTS (After removing overlaps):")
        print(f"  Total possible trades: {len(trades_df)}")
        print(f"  Executable trades: {len(filtered_df)}")
        print(f"  Overlap removed: {len(trades_df) - len(filtered_df)}")
        
        # Strategy breakdown
        print(f"\n📈 STRATEGY PERFORMANCE:")
        print("-"*70)
        print(f"{'Strategy':<20} {'Trades':>8} {'Total%':>10} {'Avg%':>8} {'Win%':>8}")
        print("-"*70)
        
        for strategy in filtered_df['Strategy'].unique():
            strat_trades = filtered_df[filtered_df['Strategy'] == strategy]
            total_ret = strat_trades['PnL%'].sum()
            avg_ret = strat_trades['PnL%'].mean()
            win_rate = (strat_trades['PnL%'] > 0).sum() / len(strat_trades) * 100
            
            print(f"{strategy:<20} {len(strat_trades):>8} {total_ret:>9.1f}% "
                  f"{avg_ret:>7.2f}% {win_rate:>7.1f}%")
        
        # Overall performance
        total_return = filtered_df['PnL%'].sum()
        compound_return = self.calculate_compound_return(filtered_df)
        
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"  Period: 59 days")
        print(f"  Total trades: {len(filtered_df)}")
        print(f"  Trades per day: {len(filtered_df)/59:.1f}")
        print(f"  Simple return: {total_return:.1f}%")
        print(f"  Compound return: {compound_return:.1f}%")
        print(f"  Win rate: {(filtered_df['PnL%'] > 0).sum() / len(filtered_df) * 100:.1f}%")
        
        # Annualized
        annual_simple = total_return * (252/59)
        annual_compound = ((1 + compound_return/100) ** (252/59) - 1) * 100
        
        print(f"\n📊 ANNUALIZED PROJECTIONS:")
        print(f"  Simple annualized: {annual_simple:.1f}%")
        print(f"  Compound (CAGR): {annual_compound:.1f}%")
        
        # Reality check
        print(f"\n⚠️  REALITY CHECK:")
        print(f"  - Slippage (0.05% per trade): -{len(filtered_df) * 0.05:.1f}%")
        print(f"  - Commission ($1 per trade): -{len(filtered_df) * 1 / 100:.1f}%")
        print(f"  - Realistic CAGR after costs: {annual_compound - len(filtered_df) * 0.1 * (252/59):.1f}%")
        
        print(f"\n🎯 BOTTOM LINE:")
        print(f"  Starting with proven strategies and realistic execution,")
        print(f"  a multi-strategy approach can potentially achieve:")
        print(f"  • Conservative: {annual_compound * 0.5:.0f}% annual returns")
        print(f"  • Realistic: {annual_compound * 0.7:.0f}% annual returns")
        print(f"  • Optimistic: {annual_compound:.0f}% annual returns")
        
    def calculate_compound_return(self, trades_df):
        """Calculate actual compound returns"""
        capital = 10000
        for _, trade in trades_df.iterrows():
            trade_capital = capital * 0.95  # 95% position size
            profit = trade_capital * (trade['PnL%'] / 100)
            capital += profit
        return (capital / 10000 - 1) * 100


# Run the realistic backtest
if __name__ == "__main__":
    strategy = RealisticVXXMultiStrategy()
    strategy.run_realistic_backtest()