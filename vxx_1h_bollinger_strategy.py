"""
⚡ VXX 1-Hour Bollinger Band Alpha Strategy
Based on alpha source mapping discovery: BB 1h has highest alpha (5.45) with 100% win rate
Designed to capture the strongest identified inefficiency in VXX

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class VXX1HBBResult:
    strategy_name: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_return: float
    avg_hold_hours: float
    best_trade: float
    worst_trade: float
    total_days: float

class VXX1HBollingerStrategy:
    """
    VXX 1-Hour Bollinger Band Mean Reversion Strategy
    Exploits the highest alpha inefficiency identified in alpha source mapping
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 commission: float = 7.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
        
        # BB-specific parameters (optimized for 1h timeframe)
        self.bb_period = 20         # Standard BB period
        self.bb_std = 2.0           # Standard deviations
        self.bb_entry_threshold = 0.1  # How close to lower band for entry (0 = touching, 0.2 = 20% up from lower)
        self.bb_exit_threshold = 0.5   # Exit when price reaches middle/upper (0.5 = middle band)
        
        # Additional filters for quality
        self.min_bb_width = 2.0     # Minimum BB width as % of price (avoid low volatility periods)
        self.volume_filter = True   # Use volume confirmation
        self.rsi_filter = True      # Additional RSI confirmation
        
    def prepare_1h_data(self, symbol: str = "VXX", period: str = "2y") -> pd.DataFrame:
        """Prepare 1-hour VXX data with Bollinger Band indicators"""
        try:
            print(f"📊 Downloading {symbol} 1-hour data...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1h")
            
            if df.columns.nlevels > 1:
                df.columns = [col[0] for col in df.columns]
                
            print(f"   ✅ Downloaded {len(df)} hours of data")
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # === CORE BOLLINGER BAND INDICATORS ===
        print("   🔧 Calculating Bollinger Band indicators...")
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['Close'].values, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle  # This is the SMA
        df['BB_Lower'] = bb_lower
        
        # BB Position (0 = lower band, 0.5 = middle, 1 = upper band)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # BB Width as percentage of price (volatility measure)
        df['BB_Width_Pct'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        
        # === ADDITIONAL QUALITY FILTERS ===
        
        # RSI for momentum confirmation
        if self.rsi_filter:
            df['RSI'] = talib.RSI(df['Close'].values, 14)
        
        # Volume analysis
        if self.volume_filter and 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].rolling(3).mean().pct_change()
        
        # Market session (for entry timing optimization)
        df['Hour'] = df.index.hour
        df['Market_Session'] = 'Regular'
        df.loc[(df['Hour'] >= 9) & (df['Hour'] < 10), 'Market_Session'] = 'Open'
        df.loc[(df['Hour'] >= 15) & (df['Hour'] < 16), 'Market_Session'] = 'Close'
        df.loc[(df['Hour'] < 9) | (df['Hour'] >= 16), 'Market_Session'] = 'Extended'
        
        print(f"   ✅ Calculated indicators for {len(df.dropna())} valid periods")
        return df.dropna()
    
    def generate_bb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band trading signals"""
        df = df.copy()
        
        # Initialize signals
        df['Signal'] = 'HOLD'
        df['Entry_Quality'] = 0
        df['Signal_Strength'] = 0.0
        
        for i in range(len(df)):
            # === ENTRY SIGNAL ===
            # Long entry: Price near lower Bollinger Band
            bb_position = df['BB_Position'].iloc[i]
            bb_width = df['BB_Width_Pct'].iloc[i]
            
            # Primary condition: Close to lower band
            near_lower_band = bb_position <= self.bb_entry_threshold
            
            # Quality filters
            sufficient_volatility = bb_width >= self.min_bb_width
            
            volume_ok = True
            if self.volume_filter:
                volume_ok = df['Volume_Ratio'].iloc[i] >= 0.8  # Not unusually low volume
            
            rsi_ok = True
            if self.rsi_filter:
                rsi_ok = df['RSI'].iloc[i] < 50  # Oversold or neutral
            
            # Market hours filter (avoid extended hours for quality)
            market_hours_ok = df['Market_Session'].iloc[i] != 'Extended'
            
            if (near_lower_band and sufficient_volatility and volume_ok and rsi_ok and market_hours_ok):
                df.loc[df.index[i], 'Signal'] = 'BUY'
                
                # Calculate entry quality score (0-100)
                quality_score = 0
                
                # Proximity to lower band (0-30 points)
                band_proximity = (self.bb_entry_threshold - bb_position) / self.bb_entry_threshold
                quality_score += min(30, band_proximity * 30)
                
                # Volatility bonus (0-20 points)
                vol_score = min(20, (bb_width - self.min_bb_width) / 2.0 * 20)
                quality_score += vol_score
                
                # Volume bonus (0-20 points)
                if self.volume_filter:
                    vol_ratio = df['Volume_Ratio'].iloc[i]
                    if vol_ratio > 1.2:
                        quality_score += 20
                    elif vol_ratio > 1.0:
                        quality_score += 15
                    elif vol_ratio > 0.8:
                        quality_score += 10
                
                # RSI bonus (0-20 points)
                if self.rsi_filter:
                    rsi = df['RSI'].iloc[i]
                    if rsi < 30:
                        quality_score += 20
                    elif rsi < 40:
                        quality_score += 15
                    elif rsi < 50:
                        quality_score += 10
                
                # Time bonus (0-10 points)
                if df['Market_Session'].iloc[i] == 'Open':
                    quality_score += 10
                elif df['Market_Session'].iloc[i] == 'Close':
                    quality_score += 5
                
                df.loc[df.index[i], 'Entry_Quality'] = min(100, quality_score)
                df.loc[df.index[i], 'Signal_Strength'] = bb_width / 10.0  # Volatility as signal strength
        
        return df
    
    def backtest_bb_strategy(self, df: pd.DataFrame, min_quality_score: int = 50) -> Dict:
        """Backtest the 1-hour Bollinger Band strategy"""
        
        # Generate signals
        df = self.generate_bb_signals(df)
        
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check for entry
            if position is None and df['Signal'].iloc[i] == 'BUY':
                entry_quality = df['Entry_Quality'].iloc[i]
                
                if entry_quality >= min_quality_score:
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'entry_idx': i,
                        'entry_quality': entry_quality,
                        'bb_width_entry': df['BB_Width_Pct'].iloc[i]
                    }
            
            # Check for exit
            elif position is not None:
                hours_held = i - position['entry_idx']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                bb_position = df['BB_Position'].iloc[i]
                
                # Exit conditions (in order of priority)
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif bb_position >= self.bb_exit_threshold:
                    exit_reason = 'BB Mean Reversion'
                elif hours_held >= 24:  # 24 hours max hold (1 day)
                    exit_reason = 'Time Limit'
                
                if exit_reason:
                    # Calculate trade results
                    position_size = current_capital * 0.95  # 95% of capital
                    shares = position_size / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price']) - (2 * self.commission)
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
                        'entry_quality': position['entry_quality'],
                        'bb_width_entry': position['bb_width_entry'],
                        'win': pnl_pct > 0
                    })
                    position = None
            
            # Record equity
            equity_curve.append({
                'time': current_time,
                'equity': current_capital,
                'price': current_price
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Analyze results
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Basic metrics
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_return_pct = (current_capital / self.initial_capital - 1) * 100
        
        # Time-based metrics
        start_time = trades_df['entry_time'].min()
        end_time = trades_df['exit_time'].max()
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        annual_return_pct = total_return_pct * (365 / total_days) if total_days > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        wins = trades_df[trades_df['win']]['pnl_pct']
        losses = trades_df[~trades_df['win']]['pnl_pct']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
        
        # Drawdown
        equity = equity_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            trades_per_year = len(trades_df) * (365 / total_days)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'avg_trade_return': returns.mean(),
            'avg_hold_hours': trades_df['hours_held'].mean(),
            'best_trade': returns.max(),
            'worst_trade': returns.min(),
            'total_days': total_days,
            'trades_df': trades_df,
            'equity_df': equity_df,
            'signals_df': df
        }
    
    def optimize_bb_parameters(self, df: pd.DataFrame) -> Dict:
        """Optimize Bollinger Band parameters"""
        print("🔧 Optimizing BB parameters...")
        
        # Parameter ranges to test
        param_combinations = []
        
        # BB Period: 15, 20, 25
        # BB Std: 1.5, 2.0, 2.5  
        # Entry threshold: 0.05, 0.10, 0.15, 0.20
        # Exit threshold: 0.4, 0.5, 0.6
        
        for bb_period in [15, 20, 25]:
            for bb_std in [1.5, 2.0, 2.5]:
                for entry_thresh in [0.05, 0.10, 0.15, 0.20]:
                    for exit_thresh in [0.4, 0.5, 0.6]:
                        param_combinations.append({
                            'bb_period': bb_period,
                            'bb_std': bb_std, 
                            'entry_threshold': entry_thresh,
                            'exit_threshold': exit_thresh
                        })
        
        best_result = None
        best_score = -float('inf')
        optimization_results = []
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")
            
            # Update parameters temporarily
            original_bb_period = self.bb_period
            original_bb_std = self.bb_std
            original_entry_thresh = self.bb_entry_threshold
            original_exit_thresh = self.bb_exit_threshold
            
            self.bb_period = params['bb_period']
            self.bb_std = params['bb_std']
            self.bb_entry_threshold = params['entry_threshold'] 
            self.bb_exit_threshold = params['exit_threshold']
            
            try:
                # Recalculate indicators with new parameters
                test_df = self.prepare_1h_data("VXX", "2y")
                result = self.backtest_bb_strategy(test_df, min_quality_score=40)
                
                if 'error' not in result and result['total_trades'] > 5:
                    # Score based on risk-adjusted return
                    score = result['total_return_pct'] / max(abs(result['max_drawdown']), 1) * (result['win_rate'] / 100)
                    
                    optimization_results.append({
                        'params': params.copy(),
                        'total_return': result['total_return_pct'],
                        'win_rate': result['win_rate'],
                        'max_drawdown': result['max_drawdown'],
                        'trades': result['total_trades'],
                        'sharpe': result['sharpe_ratio'],
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'params': params.copy(),
                            'result': result,
                            'score': score
                        }
                        
            except Exception:
                pass  # Skip failed parameter combinations
            
            # Restore original parameters
            self.bb_period = original_bb_period
            self.bb_std = original_bb_std
            self.bb_entry_threshold = original_entry_thresh
            self.bb_exit_threshold = original_exit_thresh
        
        print(f"\n✅ Optimization complete!")
        
        if best_result:
            print(f"Best parameters:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")
            print(f"Score: {best_result['score']:.2f}")
            print(f"Return: {best_result['result']['total_return_pct']:+.1f}%")
            print(f"Win Rate: {best_result['result']['win_rate']:.1f}%")
        
        return {
            'best_result': best_result,
            'all_results': optimization_results
        }
    
    def create_strategy_visualization(self, result: Dict, symbol: str = "VXX"):
        """Create comprehensive strategy visualization"""
        
        if 'error' in result:
            print(f"Cannot visualize: {result['error']}")
            return None
        
        trades_df = result['trades_df']
        equity_df = result['equity_df']
        signals_df = result['signals_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{symbol} - 1-Hour Bollinger Band Strategy Results', fontsize=16, fontweight='bold')
        
        # 1. Price chart with Bollinger Bands and trades
        ax = axes[0, 0]
        
        # Plot price and BB
        price_data = signals_df.tail(500)  # Last 500 hours for clarity
        ax.plot(price_data.index, price_data['Close'], label='VXX Price', color='white', linewidth=1)
        ax.plot(price_data.index, price_data['BB_Upper'], label='BB Upper', color='red', alpha=0.7)
        ax.plot(price_data.index, price_data['BB_Middle'], label='BB Middle', color='yellow', alpha=0.7)
        ax.plot(price_data.index, price_data['BB_Lower'], label='BB Lower', color='green', alpha=0.7)
        ax.fill_between(price_data.index, price_data['BB_Lower'], price_data['BB_Upper'], alpha=0.1, color='blue')
        
        # Plot trades
        for _, trade in trades_df.tail(50).iterrows():  # Last 50 trades
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            
            if entry_time >= price_data.index[0]:
                color = 'lime' if trade['win'] else 'red'
                ax.scatter(entry_time, trade['entry_price'], color='cyan', s=60, marker='^', zorder=5)
                ax.scatter(exit_time, trade['exit_price'], color=color, s=60, marker='v', zorder=5)
                ax.plot([entry_time, exit_time], [trade['entry_price'], trade['exit_price']], 
                       color=color, linewidth=2, alpha=0.7)
        
        ax.set_title('Price Chart with BB and Recent Trades', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax = axes[0, 1]
        ax.plot(equity_df['time'], equity_df['equity'], color='lime', linewidth=2)
        ax.axhline(y=self.initial_capital, color='white', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Trade distribution
        ax = axes[1, 0]
        wins = trades_df[trades_df['win']]['pnl_pct']
        losses = trades_df[~trades_df['win']]['pnl_pct']
        
        ax.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})', density=True)
        ax.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})', density=True)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        ax.set_title('Trade P&L Distribution', fontweight='bold')
        ax.set_xlabel('P&L (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Strategy metrics table
        ax = axes[1, 1]
        ax.axis('off')
        
        metrics_text = f"""1-HOUR BOLLINGER BAND STRATEGY
        
🏆 PERFORMANCE METRICS:
Total Return: {result['total_return_pct']:+.1f}%
Annual Return: {result['annual_return_pct']:+.1f}%
Win Rate: {result['win_rate']:.1f}%
Total Trades: {result['total_trades']}

📊 RISK METRICS:
Max Drawdown: {result['max_drawdown']:.1f}%
Sharpe Ratio: {result['sharpe_ratio']:.2f}
Profit Factor: {result['profit_factor']:.2f}

📈 TRADE METRICS:
Avg Return/Trade: {result['avg_trade_return']:.2f}%
Avg Hold Time: {result['avg_hold_hours']:.1f} hours
Best Trade: {result['best_trade']:+.1f}%
Worst Trade: {result['worst_trade']:+.1f}%

⚙️ STRATEGY PARAMETERS:
BB Period: {self.bb_period}
BB Std Dev: {self.bb_std}
Entry Threshold: {self.bb_entry_threshold:.2f}
Exit Threshold: {self.bb_exit_threshold:.2f}
Min BB Width: {self.min_bb_width:.1f}%

💡 Based on Alpha Source Analysis:
• Highest alpha inefficiency (5.45)
• 100% win rate in discovery
• 580-minute average decay
• Optimized for 1-hour timeframe"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig

def main():
    """Run VXX 1-Hour Bollinger Band strategy analysis"""
    print("⚡ VXX 1-HOUR BOLLINGER BAND ALPHA STRATEGY")
    print("=" * 60)
    print("Based on alpha source discovery: BB 1h = highest alpha (5.45)")
    print("Targeting the strongest identified inefficiency in VXX")
    
    # Initialize strategy
    strategy = VXX1HBollingerStrategy(
        initial_capital=10000,
        profit_target_pct=5.0,   # Conservative profit target
        stop_loss_pct=7.5,       # Reasonable stop loss
        commission=7.0           # Realistic commission
    )
    
    print(f"\n{'='*20} DATA PREPARATION {'='*20}")
    
    # Prepare data  
    df = strategy.prepare_1h_data("VXX", "2y")
    
    if df is None:
        print("❌ Failed to prepare data")
        return
    
    print(f"\n{'='*20} STRATEGY BACKTEST {'='*20}")
    
    # Run backtest
    result = strategy.backtest_bb_strategy(df, min_quality_score=50)
    
    if 'error' in result:
        print(f"❌ Backtest failed: {result['error']}")
        return
    
    # Display results
    print(f"\n🏆 VXX 1-HOUR BOLLINGER BAND RESULTS:")
    print("-" * 50)
    print(f"Total Return: {result['total_return_pct']:+.1f}%")
    print(f"Annual Return: {result['annual_return_pct']:+.1f}%") 
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Avg Return/Trade: {result['avg_trade_return']:.2f}%")
    print(f"Max Drawdown: {result['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print(f"Avg Hold Time: {result['avg_hold_hours']:.1f} hours")
    
    # Create visualization
    print(f"\n📊 Creating strategy visualization...")
    fig = strategy.create_strategy_visualization(result, "VXX")
    
    if fig:
        filename = 'VXX_1h_bollinger_strategy.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        print(f"✅ Saved: {filename}")
        
        plt.close()
    
    # Compare to alpha source discovery
    print(f"\n🔬 ALPHA SOURCE VALIDATION:")
    print(f"Discovery stats: α=5.45, 100% win rate")
    print(f"Strategy stats: {result['total_return_pct']:+.1f}% return, {result['win_rate']:.1f}% win rate")
    
    validation_score = (result['win_rate'] / 100) * (result['total_return_pct'] / 10)
    print(f"Validation score: {validation_score:.2f}")
    
    if result['win_rate'] > 60 and result['total_return_pct'] > 10:
        print("✅ Strategy successfully exploits identified alpha source!")
    elif result['total_return_pct'] > 0:
        print("✅ Strategy profitable but may need optimization")
    else:
        print("⚠️ Strategy needs parameter optimization")
    
    print(f"\n💡 INTEGRATION OPTIONS:")
    print(f"1. Replace 15-minute strategy with 1-hour BB strategy")
    print(f"2. Run both strategies in parallel (different timeframes)")
    print(f"3. Use 1-hour BB as primary, 15-minute as backup")
    
    return strategy, result

if __name__ == "__main__":
    strategy, result = main()