"""
🎯 VXX 1-Hour Bollinger Band Strategy - Optimized Version
Fixing the disconnect between alpha discovery and implementation
Focus on replicating the exact conditions that generated α=5.45

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

class VXX1HBBOptimized:
    """
    Optimized VXX 1-Hour BB strategy based on alpha source analysis
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        
    def prepare_data_exact_match(self, symbol: str = "VXX", period: str = "2y") -> pd.DataFrame:
        """Prepare data to exactly match alpha source analysis"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1h")
            
            if df.columns.nlevels > 1:
                df.columns = [col[0] for col in df.columns]
                
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
        
        if len(df) < 100:
            return None
        
        # === EXACT ALPHA SOURCE INDICATORS ===
        # Use identical parameters to alpha source analysis
        
        # SMA and distance (matching alpha source methodology)
        sma_period = 20  # Standard for 1h
        df['SMA'] = df['Close'].rolling(sma_period).mean()
        df['Distance_Pct'] = (df['Close'] - df['SMA']) / df['SMA'] * 100
        
        # Bollinger Bands (standard 20, 2.0)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values, 20, 2, 2)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI (14 period for 1h)
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        
        # Volume ratio
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        return df.dropna()
    
    def replicate_alpha_source_strategy(self, df: pd.DataFrame) -> dict:
        """Replicate the exact strategy that generated α=5.45"""
        
        trades = []
        current_capital = self.initial_capital
        position = None
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # === EXACT ALPHA SOURCE CONDITIONS ===
            # From alpha source analysis: BB_Reversion_1h
            # Entry: BB_Position < 0.2 (near lower band)
            # Exit: BB_Position > 0.5 (near middle band)
            
            if position is None:
                # Entry condition: Price near lower Bollinger Band
                bb_position = df['BB_Position'].iloc[i]
                
                if pd.notna(bb_position) and bb_position < 0.2:
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'entry_idx': i
                    }
            
            elif position is not None:
                # Exit conditions
                bb_position = df['BB_Position'].iloc[i]
                hours_held = i - position['entry_idx']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Primary exit: Mean reversion to middle band
                if pd.notna(bb_position) and bb_position > 0.5:
                    exit_reason = 'BB Mean Reversion'
                # Safety exits
                elif pnl_pct >= 10.0:  # Large profit target
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -10.0:  # Large stop loss
                    exit_reason = 'Stop Loss'
                elif hours_held >= 168:  # 1 week max
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    position_size = current_capital * 0.95
                    shares = position_size / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price'])
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'hours_held': hours_held,
                        'exit_reason': exit_reason,
                        'win': pnl_pct > 0
                    })
                    position = None
        
        if not trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_return_pct = (current_capital / self.initial_capital - 1) * 100
        
        # Time period
        start_time = trades_df['entry_time'].min()
        end_time = trades_df['exit_time'].max() 
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'total_days': total_days,
            'avg_hold_hours': trades_df['hours_held'].mean(),
            'trades_df': trades_df
        }
    
    def test_parameter_variations(self, df: pd.DataFrame) -> dict:
        """Test different parameter variations to find what works"""
        
        print("🔧 Testing parameter variations...")
        
        # Test different BB entry/exit thresholds
        test_configs = [
            # Original alpha source
            {'name': 'Alpha_Source_Exact', 'entry_thresh': 0.2, 'exit_thresh': 0.5, 'profit': 10, 'stop': 10},
            
            # Tighter entry
            {'name': 'Tighter_Entry', 'entry_thresh': 0.1, 'exit_thresh': 0.5, 'profit': 10, 'stop': 10},
            {'name': 'Very_Tight_Entry', 'entry_thresh': 0.05, 'exit_thresh': 0.5, 'profit': 10, 'stop': 10},
            
            # Different exit points
            {'name': 'Earlier_Exit', 'entry_thresh': 0.2, 'exit_thresh': 0.3, 'profit': 10, 'stop': 10},
            {'name': 'Later_Exit', 'entry_thresh': 0.2, 'exit_thresh': 0.7, 'profit': 10, 'stop': 10},
            
            # Smaller targets
            {'name': 'Small_Targets', 'entry_thresh': 0.2, 'exit_thresh': 0.5, 'profit': 5, 'stop': 5},
            {'name': 'Micro_Targets', 'entry_thresh': 0.2, 'exit_thresh': 0.5, 'profit': 3, 'stop': 3},
            
            # Conservative
            {'name': 'Conservative', 'entry_thresh': 0.15, 'exit_thresh': 0.4, 'profit': 7, 'stop': 5},
        ]
        
        results = []
        
        for config in test_configs:
            result = self.test_single_config(df, config)
            if 'error' not in result:
                results.append({
                    'config': config,
                    'result': result
                })
                print(f"   {config['name']}: {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win, {result['total_return_pct']:+.1f}% return")
        
        # Find best configuration
        if results:
            best = max(results, key=lambda x: x['result']['total_return_pct'])
            print(f"\n🏆 Best configuration: {best['config']['name']}")
            print(f"   Return: {best['result']['total_return_pct']:+.1f}%")
            print(f"   Win Rate: {best['result']['win_rate']:.1f}%")
            print(f"   Trades: {best['result']['total_trades']}")
            
            return best
        
        return None
    
    def test_single_config(self, df: pd.DataFrame, config: dict) -> dict:
        """Test a single parameter configuration"""
        
        trades = []
        current_capital = self.initial_capital
        position = None
        
        entry_thresh = config['entry_thresh']
        exit_thresh = config['exit_thresh']
        profit_target = config['profit']
        stop_loss = config['stop']
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            if position is None:
                bb_position = df['BB_Position'].iloc[i]
                
                if pd.notna(bb_position) and bb_position < entry_thresh:
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'entry_idx': i
                    }
            
            elif position is not None:
                bb_position = df['BB_Position'].iloc[i]
                hours_held = i - position['entry_idx']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                if pnl_pct >= profit_target:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -stop_loss:
                    exit_reason = 'Stop Loss'
                elif pd.notna(bb_position) and bb_position > exit_thresh:
                    exit_reason = 'BB Mean Reversion'
                elif hours_held >= 168:
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    position_size = current_capital * 0.95
                    shares = position_size / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price'])
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'hours_held': hours_held,
                        'exit_reason': exit_reason,
                        'win': pnl_pct > 0
                    })
                    position = None
        
        if not trades:
            return {'error': 'No trades'}
        
        trades_df = pd.DataFrame(trades)
        
        return {
            'total_trades': len(trades_df),
            'win_rate': trades_df['win'].mean() * 100,
            'total_return_pct': (current_capital / self.initial_capital - 1) * 100,
            'avg_hold_hours': trades_df['hours_held'].mean()
        }

def main():
    """Run optimized VXX 1h BB strategy analysis"""
    print("🎯 VXX 1-HOUR BOLLINGER BAND - OPTIMIZED")
    print("=" * 60)
    print("Goal: Replicate α=5.45 discovery with 100% win rate")
    print("Method: Exact parameter matching + optimization")
    
    # Initialize
    optimizer = VXX1HBBOptimized(initial_capital=10000)
    
    # Prepare data
    print(f"\n📊 Preparing data to match alpha source analysis...")
    df = optimizer.prepare_data_exact_match("VXX", "2y")
    
    if df is None:
        print("❌ Failed to prepare data")
        return
    
    print(f"✅ Prepared {len(df)} hours of data")
    
    # Test original alpha source conditions
    print(f"\n🔬 Testing exact alpha source conditions...")
    original_result = optimizer.replicate_alpha_source_strategy(df)
    
    if 'error' not in original_result:
        print(f"Original conditions:")
        print(f"  Trades: {original_result['total_trades']}")
        print(f"  Win Rate: {original_result['win_rate']:.1f}%")
        print(f"  Return: {original_result['total_return_pct']:+.1f}%")
        print(f"  Avg Hold: {original_result['avg_hold_hours']:.1f}h")
    else:
        print(f"❌ Original conditions failed: {original_result['error']}")
    
    # Parameter optimization
    print(f"\n🔧 Running parameter optimization...")
    best_config = optimizer.test_parameter_variations(df)
    
    if best_config:
        best_result = best_config['result']
        best_params = best_config['config']
        
        print(f"\n🏆 OPTIMIZED RESULTS:")
        print(f"Strategy: {best_params['name']}")
        print(f"Total Return: {best_result['total_return_pct']:+.1f}%")
        print(f"Win Rate: {best_result['win_rate']:.1f}%")
        print(f"Total Trades: {best_result['total_trades']}")
        print(f"Avg Hold: {best_result['avg_hold_hours']:.1f} hours")
        
        print(f"\nOptimal Parameters:")
        print(f"  BB Entry Threshold: {best_params['entry_thresh']:.2f}")
        print(f"  BB Exit Threshold: {best_params['exit_thresh']:.2f}")
        print(f"  Profit Target: {best_params['profit']:.1f}%")
        print(f"  Stop Loss: {best_params['stop']:.1f}%")
        
        # Compare to discoveries
        print(f"\n📊 COMPARISON TO ALPHA SOURCE:")
        print(f"Discovery: α=5.45, 100% win rate")
        print(f"Strategy: {best_result['total_return_pct']:+.1f}% return, {best_result['win_rate']:.1f}% win rate")
        
        if best_result['total_return_pct'] > 0 and best_result['win_rate'] > 50:
            print("✅ Successfully optimized BB strategy!")
        else:
            print("⚠️ Still needs work - may require different approach")
        
        return best_config
    
    else:
        print("❌ No profitable configurations found")
        
        print(f"\n💡 TROUBLESHOOTING INSIGHTS:")
        print("1. Alpha source may have used different timeframe resolution")
        print("2. Market regime may have changed since discovery")
        print("3. Need to investigate exact discovery methodology")
        print("4. Consider different entry/exit logic")
    
    return None

if __name__ == "__main__":
    result = main()