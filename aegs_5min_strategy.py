#!/usr/bin/env python3
"""
AEGS 5-Minute Strategy
Optimized version of AEGS for intraday trading on 5-minute bars
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from termcolor import colored
import json
import sqlite3

class AEGS5MinStrategy:
    """
    Automated Enhanced Generation Strategy for 5-minute timeframe
    Optimized for mean reversion and scalping opportunities
    """
    
    def __init__(self):
        self.name = "AEGS_5MIN"
        self.timeframe = "5m"
        self.lookback_days = 30  # Analyze 30 days of 5-min data
        
        # Adjusted parameters for 5-minute trading
        self.params = {
            'bb_period': 20,           # Bollinger Bands period
            'bb_std': 2.0,            # Standard deviations
            'rsi_period': 14,         # RSI period
            'rsi_oversold': 30,       # More aggressive for 5-min
            'rsi_overbought': 70,     
            'volume_spike': 1.5,      # Lower threshold for 5-min
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'extreme_move': 0.03,     # 3% move in 5 minutes is extreme
            'mean_reversion_threshold': 0.015,  # 1.5% from mean
            'min_holding_bars': 3,    # Hold at least 15 minutes
            'max_holding_bars': 24,   # Exit within 2 hours
            'stop_loss': 0.02,        # 2% stop loss
            'take_profit': 0.03       # 3% take profit
        }
        
    def download_data(self, symbol):
        """Download 5-minute data"""
        print(f"📊 Downloading 5-minute data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=self.timeframe
        )
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        print(f"✅ Downloaded {len(df)} bars of 5-minute data")
        print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def add_indicators(self, df):
        """Add technical indicators optimized for 5-minute trading"""
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (faster for 5-min)
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(self.params['bb_period']).mean()
        bb_std = df['close'].rolling(self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.params['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.params['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
        
        # MACD
        exp1 = df['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
        exp2 = df['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.params['macd_signal'], adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility (5-min adjusted)
        df['volatility_5m'] = df['returns'].rolling(20).std()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['open']
        
        # Mean reversion indicators
        df['distance_from_mean'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['mean_reversion_score'] = 0
        
        # Price patterns for 5-min
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        # Momentum for scalping
        df['momentum_5bar'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10bar'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, df):
        """Generate trading signals optimized for 5-minute timeframe"""
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0
        df['signal_reason'] = ''
        
        for i in range(50, len(df)):  # Need enough history
            
            # Skip if we're in the last few bars (can't hold position)
            if i >= len(df) - self.params['max_holding_bars']:
                continue
            
            # Mean Reversion Signals (primary for 5-min)
            mean_reversion_score = 0
            reasons = []
            
            # 1. Bollinger Band Squeeze + Breakout
            if df['bb_position'].iloc[i] < 0.1 and df['volume_ratio'].iloc[i] > 1.5:
                mean_reversion_score += 30
                reasons.append('BB_oversold_volume')
            
            # 2. RSI Oversold + Momentum shift
            if df['rsi'].iloc[i] < self.params['rsi_oversold']:
                if df['momentum_5bar'].iloc[i] > df['momentum_5bar'].iloc[i-1]:
                    mean_reversion_score += 25
                    reasons.append('RSI_oversold_momentum')
            
            # 3. Extreme 5-min move reversal
            if df['returns'].iloc[i] < -self.params['extreme_move']:
                if df['volume_ratio'].iloc[i] > 2:
                    mean_reversion_score += 35
                    reasons.append('Extreme_5min_reversal')
            
            # 4. Distance from mean reversion
            if abs(df['distance_from_mean'].iloc[i]) > self.params['mean_reversion_threshold']:
                if df['distance_from_mean'].iloc[i] < 0:  # Below mean
                    mean_reversion_score += 20
                    reasons.append('Below_mean_reversion')
            
            # 5. MACD convergence in oversold
            if df['macd_diff'].iloc[i] > df['macd_diff'].iloc[i-1] and df['rsi'].iloc[i] < 40:
                mean_reversion_score += 15
                reasons.append('MACD_convergence')
            
            # 6. Quick scalp opportunity
            if df['high_low_ratio'].iloc[i] > 0.02:  # 2% range in 5 min
                if df['close'].iloc[i] < df['sma_10'].iloc[i]:
                    mean_reversion_score += 15
                    reasons.append('Scalp_opportunity')
            
            # Generate signal if score is high enough
            if mean_reversion_score >= 50:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_strength'] = min(mean_reversion_score, 100)
                df.loc[df.index[i], 'signal_reason'] = ', '.join(reasons[:3])  # Top 3 reasons
        
        return df
    
    def backtest(self, df, initial_capital=10000):
        """Backtest the 5-minute strategy with realistic execution"""
        
        # Add indicators and signals
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        
        # Commission and slippage for 5-min trading
        commission = 0.001  # 0.1%
        slippage = 0.0005   # 0.05% for liquid stocks
        
        # Track positions
        in_position = False
        entry_price = 0
        entry_time = None
        entry_idx = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # Check exit conditions if in position
            if in_position:
                bars_held = i - entry_idx
                returns = (current_price - entry_price) / entry_price
                
                exit_signal = False
                exit_reason = ''
                
                # Stop loss
                if returns <= -self.params['stop_loss']:
                    exit_signal = True
                    exit_reason = 'Stop_loss'
                
                # Take profit
                elif returns >= self.params['take_profit']:
                    exit_signal = True
                    exit_reason = 'Take_profit'
                
                # Time exit (max holding period)
                elif bars_held >= self.params['max_holding_bars']:
                    exit_signal = True
                    exit_reason = 'Time_exit'
                
                # Mean reversion completed
                elif abs(df['distance_from_mean'].iloc[i]) < 0.005:  # Back to mean
                    if bars_held >= self.params['min_holding_bars']:
                        exit_signal = True
                        exit_reason = 'Mean_reversion_complete'
                
                # Exit signal
                if exit_signal:
                    # Sell with slippage
                    exit_price = current_price * (1 - slippage)
                    capital = position * exit_price * (1 - commission)
                    
                    # Record trade
                    trade_return = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    in_position = False
            
            # Check entry conditions if not in position
            elif df['signal'].iloc[i] == 1 and not in_position:
                # Buy with slippage
                entry_price = current_price * (1 + slippage)
                position = (capital * (1 - commission)) / entry_price
                entry_time = current_time
                entry_idx = i
                in_position = True
                capital = 0
        
        # Close any open position at end
        if in_position:
            exit_price = df['close'].iloc[-1]
            capital = position * exit_price * (1 - commission)
        else:
            capital = capital if capital > 0 else position * df['close'].iloc[-1]
        
        # Calculate performance metrics
        results = self.calculate_performance(trades, initial_capital, capital, df)
        
        return results, trades, df
    
    def calculate_performance(self, trades, initial_capital, final_capital, df):
        """Calculate comprehensive performance metrics for 5-minute trading"""
        
        if not trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'message': 'No trades executed'
            }
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital
        returns = [t['return'] for t in trades]
        
        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = (winning_trades / len(trades)) * 100
        
        # Average returns
        avg_return = np.mean(returns) * 100
        avg_win = np.mean([r for r in returns if r > 0]) * 100 if winning_trades > 0 else 0
        avg_loss = np.mean([r for r in returns if r < 0]) * 100 if winning_trades < len(trades) else 0
        
        # Sharpe ratio (annualized for 5-min bars)
        if len(returns) > 1 and np.std(returns) > 0:
            # 78 five-minute bars per day, 252 trading days
            bars_per_year = 78 * 252
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)
        else:
            sharpe = 0
        
        # Average holding time
        avg_bars = np.mean([t['bars_held'] for t in trades])
        avg_hold_minutes = avg_bars * 5
        
        # Buy and hold comparison
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        excess_return = total_return - buy_hold_return
        
        results = {
            'total_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': excess_return * 100,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'avg_hold_minutes': avg_hold_minutes,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'best_trade': max(returns) * 100,
            'worst_trade': min(returns) * 100,
            'final_capital': final_capital
        }
        
        return results
    
    def display_results(self, symbol, results, trades):
        """Display backtest results"""
        
        print(colored(f"\n📊 AEGS 5-MINUTE BACKTEST RESULTS FOR {symbol}", 'yellow', attrs=['bold']))
        print("="*60)
        
        print(f"\n💰 Returns:")
        print(f"   Strategy Return: {results['total_return']:+.2f}%")
        print(f"   Buy & Hold Return: {results['buy_hold_return']:+.2f}%")
        print(f"   Excess Return: {results['excess_return']:+.2f}%")
        
        print(f"\n📊 Trading Statistics:")
        print(f"   Total Trades: {results['num_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Avg Return per Trade: {results['avg_return']:+.2f}%")
        print(f"   Avg Holding Time: {results['avg_hold_minutes']:.0f} minutes")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Best Trade: {results['best_trade']:+.2f}%")
        print(f"   Worst Trade: {results['worst_trade']:+.2f}%")
        
        # Trading recommendation
        print(colored(f"\n💡 RECOMMENDATION:", 'green'))
        if results['excess_return'] > 20 and results['win_rate'] > 50:
            print(f"   ✅ STRONG BUY - Use 5-minute AEGS strategy")
            print(f"   Monitor every 5-15 minutes during market hours")
            print(f"   Position size: 10-15% of capital")
        elif results['excess_return'] > 0:
            print(f"   ⚠️ MODERATE - Consider for active trading")
            print(f"   Use smaller position sizes")
        else:
            print(f"   ❌ AVOID - Strategy underperforms")
            print(f"   Look for other opportunities")

def run_aegs_5min_backtest(symbol):
    """Main function to run AEGS 5-minute backtest"""
    
    strategy = AEGS5MinStrategy()
    
    try:
        # Download data
        df = strategy.download_data(symbol)
        
        # Run backtest
        results, trades, df_with_signals = strategy.backtest(df)
        
        # Display results
        strategy.display_results(symbol, results, trades)
        
        # Save results
        save_results(symbol, results, trades)
        
        return results, trades, df_with_signals
        
    except Exception as e:
        print(colored(f"❌ Error: {e}", 'red'))
        return None, None, None

def save_results(symbol, results, trades):
    """Save backtest results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary
    summary = {
        'symbol': symbol,
        'strategy': 'AEGS_5MIN',
        'timestamp': timestamp,
        'results': results,
        'num_trades': len(trades)
    }
    
    filename = f'aegs_5min_{symbol}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")

def main():
    """Run AEGS 5-minute strategy"""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python aegs_5min_strategy.py SYMBOL")
        print("Example: python aegs_5min_strategy.py AAPL")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    run_aegs_5min_backtest(symbol)

if __name__ == "__main__":
    main()