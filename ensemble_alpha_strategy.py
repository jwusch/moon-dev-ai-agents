"""
🎯 Ensemble Alpha Strategy Builder
Combines top alpha sources into a unified trading strategy using signal weighting,
correlation analysis, and dynamic position sizing

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
import yfinance as yf
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our alpha scanner
from alpha_scanner_cli import AlphaSourceScanner, AlphaSourceResult

@dataclass
class EnsembleSignal:
    timestamp: pd.Timestamp
    signal_strength: float  # -1 to +1
    component_signals: Dict[str, float]
    confidence: float  # 0 to 1
    position_size: float  # 0 to 1

@dataclass
class EnsembleBacktestResult:
    total_return_pct: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    alpha_score: float
    avg_hold_minutes: float
    individual_performance: Dict[str, Dict]

class EnsembleAlphaStrategy:
    """
    Multi-signal ensemble strategy that combines top alpha sources
    """
    
    def __init__(self, symbol: str, top_n_signals: int = 5):
        self.symbol = symbol
        self.top_n_signals = top_n_signals
        self.alpha_sources = []
        self.correlation_matrix = None
        self.weights = {}
        self.scanner = AlphaSourceScanner(min_alpha_threshold=1.0)  # Lower threshold for more options
        
    def discover_alpha_sources(self, timeframes: List[str] = None) -> List[AlphaSourceResult]:
        """Discover alpha sources for the symbol"""
        
        print(f"🔍 Discovering alpha sources for {self.symbol}...")
        
        if timeframes is None:
            timeframes = ["5m", "15m", "1h", "1d"]
        
        # Use our alpha scanner
        alpha_sources = self.scanner.scan_symbol(self.symbol, timeframes)
        
        # Sort by alpha score and take top N
        alpha_sources.sort(key=lambda x: x.alpha_score, reverse=True)
        self.alpha_sources = alpha_sources[:self.top_n_signals]
        
        print(f"✅ Selected top {len(self.alpha_sources)} alpha sources:")
        for i, source in enumerate(self.alpha_sources, 1):
            print(f"   {i}. {source.strategy_name}: α={source.alpha_score:.2f} ({source.win_rate:.1f}% win)")
        
        return self.alpha_sources
    
    def prepare_ensemble_data(self, period: str = "60d") -> pd.DataFrame:
        """Prepare unified dataset with all alpha source indicators"""
        
        print(f"📊 Preparing ensemble data for {self.symbol}...")
        
        # Determine the finest timeframe for our dataset
        timeframes_in_sources = [source.timeframe for source in self.alpha_sources]
        
        # Use the most common timeframe or finest available
        if "5m" in timeframes_in_sources:
            base_timeframe = "5m"
        elif "15m" in timeframes_in_sources:
            base_timeframe = "15m" 
        elif "1h" in timeframes_in_sources:
            base_timeframe = "1h"
        else:
            base_timeframe = "1d"
        
        print(f"   Base timeframe: {base_timeframe}")
        
        # Download base data
        df = yf.download(self.symbol, period=period, interval=base_timeframe, progress=False)
        
        if df.columns.nlevels > 1:
            df.columns = [col[0] for col in df.columns]
        
        # Add comprehensive indicators
        df = self.scanner._add_comprehensive_indicators(df, base_timeframe)
        
        # Generate signals for each alpha source
        for source in self.alpha_sources:
            signal_name = f"signal_{source.strategy_name.replace(f'_{source.timeframe}', '')}"
            df[signal_name] = self._generate_signal_for_source(df, source, base_timeframe)
        
        print(f"✅ Prepared {len(df)} bars with {len(self.alpha_sources)} signal components")
        
        return df.dropna()
    
    def _generate_signal_for_source(self, df: pd.DataFrame, source: AlphaSourceResult, timeframe: str) -> pd.Series:
        """Generate signal for a specific alpha source"""
        
        # Extract strategy type and conditions from source
        strategy_name = source.strategy_name.replace(f"_{source.timeframe}", "")
        
        # Map strategy names to signal logic
        if "RSI_Reversion" in strategy_name:
            entry = (df['RSI'] < 30) & (df['Distance_Medium'] < -1.0)
            exit_signal = (df['RSI'] > 50) | (df['Distance_Medium'] > 0)
            
        elif "BB_Reversion" in strategy_name:
            entry = df['BB_Position'] < 0.2
            exit_signal = df['BB_Position'] > 0.5
            
        elif "Extreme_Reversion" in strategy_name:
            entry = df['Extreme_Move'] & (df['Distance_Short'] < -2.0)
            exit_signal = abs(df['Distance_Short']) < 0.5
            
        elif "Vol_Expansion" in strategy_name:
            entry = df['Vol_Regime'] == 'Low'
            exit_signal = df['Vol_Regime'] == 'High'
            
        elif "MACD_Momentum" in strategy_name:
            entry = (df['MACD'] > df['MACD_Signal']) & (df['MACD_Hist'] > 0) & (df['ADX'] > 25)
            exit_signal = (df['MACD'] < df['MACD_Signal']) | (df['MACD_Hist'] < 0)
            
        elif "Friday_Effect" in strategy_name:
            entry = (df['DayOfWeek'] == 4) & (df['Distance_Medium'] < -1)
            exit_signal = df['DayOfWeek'] == 0
            
        else:
            # Default to RSI reversion if unknown
            entry = (df['RSI'] < 30) & (df['Distance_Medium'] < -1.0)
            exit_signal = (df['RSI'] > 50) | (df['Distance_Medium'] > 0)
        
        # Convert to signal strength (-1 to +1)
        signal = np.zeros(len(df))
        signal[entry] = 1.0
        signal[exit_signal] = -1.0
        
        return pd.Series(signal, index=df.index)
    
    def calculate_signal_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations between different alpha signals"""
        
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        
        if len(signal_columns) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        self.correlation_matrix = df[signal_columns].corr()
        
        print(f"📊 Signal Correlation Matrix:")
        print(self.correlation_matrix.round(3))
        
        return self.correlation_matrix
    
    def calculate_dynamic_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic weights for each alpha source"""
        
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        
        # Base weights on alpha scores
        alpha_weights = {}
        total_alpha = sum(source.alpha_score for source in self.alpha_sources)
        
        for source in self.alpha_sources:
            signal_name = f"signal_{source.strategy_name.replace(f'_{source.timeframe}', '')}"
            if signal_name in signal_columns:
                # Weight by alpha score and win rate
                base_weight = source.alpha_score / total_alpha
                confidence_factor = source.win_rate / 100  # Adjust by win rate
                alpha_weights[signal_name] = base_weight * confidence_factor
        
        # Normalize weights
        total_weight = sum(alpha_weights.values())
        if total_weight > 0:
            for signal in alpha_weights:
                alpha_weights[signal] /= total_weight
        
        # Apply correlation penalty (reduce weights for highly correlated signals)
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            correlation_penalty = {}
            for signal in alpha_weights:
                if signal in self.correlation_matrix.columns:
                    # Penalize based on average correlation with other signals
                    avg_corr = abs(self.correlation_matrix.loc[signal, self.correlation_matrix.columns != signal]).mean()
                    correlation_penalty[signal] = 1 - (avg_corr * 0.3)  # Up to 30% penalty
                else:
                    correlation_penalty[signal] = 1.0
            
            # Apply penalties
            for signal in alpha_weights:
                alpha_weights[signal] *= correlation_penalty[signal]
        
        # Renormalize
        total_weight = sum(alpha_weights.values())
        if total_weight > 0:
            for signal in alpha_weights:
                alpha_weights[signal] /= total_weight
        
        self.weights = alpha_weights
        
        print(f"📊 Dynamic Signal Weights:")
        for signal, weight in self.weights.items():
            print(f"   {signal}: {weight:.3f}")
        
        return self.weights
    
    def generate_ensemble_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble trading signals"""
        
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        
        if not signal_columns:
            raise ValueError("No alpha signals found in data")
        
        # Calculate correlations and weights
        self.calculate_signal_correlations(df)
        self.calculate_dynamic_weights(df)
        
        # Create ensemble signal
        ensemble_signal = np.zeros(len(df))
        ensemble_confidence = np.zeros(len(df))
        
        for i in range(len(df)):
            weighted_signal = 0
            total_confidence = 0
            active_signals = 0
            
            for signal_col in signal_columns:
                if signal_col in self.weights and abs(df[signal_col].iloc[i]) > 0:
                    weight = self.weights[signal_col]
                    signal_value = df[signal_col].iloc[i]
                    
                    weighted_signal += signal_value * weight
                    total_confidence += weight
                    active_signals += 1
            
            ensemble_signal[i] = weighted_signal
            # Confidence increases with number of agreeing signals
            ensemble_confidence[i] = total_confidence * min(1.0, active_signals / len(signal_columns))
        
        df['ensemble_signal'] = ensemble_signal
        df['ensemble_confidence'] = ensemble_confidence
        
        # Generate trading signals with thresholds
        df['trade_signal'] = 0
        df['position_size'] = 0
        
        # Entry thresholds based on signal strength and confidence
        buy_threshold = 0.3  # Require 30% of max signal strength
        sell_threshold = -0.3
        min_confidence = 0.2  # Minimum confidence required
        
        buy_condition = (df['ensemble_signal'] > buy_threshold) & (df['ensemble_confidence'] > min_confidence)
        sell_condition = (df['ensemble_signal'] < sell_threshold) & (df['ensemble_confidence'] > min_confidence)
        
        df.loc[buy_condition, 'trade_signal'] = 1
        df.loc[sell_condition, 'trade_signal'] = -1
        
        # Dynamic position sizing based on confidence
        df.loc[df['trade_signal'] != 0, 'position_size'] = df.loc[df['trade_signal'] != 0, 'ensemble_confidence']
        
        print(f"✅ Generated ensemble signals:")
        print(f"   Buy signals: {(df['trade_signal'] == 1).sum()}")
        print(f"   Sell signals: {(df['trade_signal'] == -1).sum()}")
        print(f"   Avg confidence: {df['ensemble_confidence'].mean():.3f}")
        
        return df
    
    def backtest_ensemble_strategy(self, df: pd.DataFrame) -> EnsembleBacktestResult:
        """Backtest the ensemble strategy"""
        
        print(f"📈 Backtesting ensemble strategy for {self.symbol}...")
        
        trades = []
        position = 0
        entry_price = None
        entry_idx = None
        entry_signal_strength = None
        
        initial_capital = 10000
        current_capital = initial_capital
        
        # Also backtest individual components for comparison
        individual_performance = {}
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        
        for signal_col in signal_columns:
            individual_performance[signal_col] = self._backtest_individual_signal(df, signal_col)
        
        # Backtest ensemble
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            trade_signal = df['trade_signal'].iloc[i]
            position_size = df['position_size'].iloc[i]
            
            # Entry
            if position == 0 and trade_signal == 1:  # Buy signal
                position = 1
                entry_price = current_price
                entry_idx = i
                entry_signal_strength = df['ensemble_signal'].iloc[i]
            
            # Exit on sell signal or stop conditions
            elif position == 1 and (trade_signal == -1 or self._check_stop_conditions(df, i, entry_idx, entry_price, current_price)):
                if entry_price is not None:
                    hold_periods = i - entry_idx
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    # Apply position sizing to P&L
                    effective_pnl = pnl_pct * position_size
                    
                    # Calculate dollar P&L
                    position_value = current_capital * 0.95  # Use 95% of capital
                    shares = position_value / entry_price
                    pnl_dollars = shares * (current_price - entry_price)
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'hold_periods': hold_periods,
                        'pnl_pct': pnl_pct,
                        'effective_pnl_pct': effective_pnl,
                        'pnl_dollars': pnl_dollars,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'signal_strength': entry_signal_strength,
                        'position_size': position_size,
                        'win': pnl_pct > 0
                    })
                
                position = 0
                entry_price = None
                entry_idx = None
                entry_signal_strength = None
        
        if not trades:
            print("❌ No trades generated")
            return None
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        total_return_pct = (current_capital / initial_capital - 1) * 100
        win_rate = trades_df['win'].mean() * 100
        total_trades = len(trades_df)
        
        returns = trades_df['effective_pnl_pct'].values  # Use effective P&L for metrics
        avg_return = returns.mean()
        sharpe_ratio = avg_return / returns.std() if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative_returns = trades_df['effective_pnl_pct'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        wins = trades_df[trades_df['win']]['effective_pnl_pct']
        losses = trades_df[~trades_df['win']]['effective_pnl_pct']
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
        
        # Alpha score
        alpha_score = total_return_pct / total_trades * (win_rate / 100) if total_trades > 0 else 0
        
        # Average holding time
        avg_hold_periods = trades_df['hold_periods'].mean()
        timeframe_minutes = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}
        base_tf = "15m"  # Default assumption
        avg_hold_minutes = avg_hold_periods * timeframe_minutes.get(base_tf, 15)
        
        result = EnsembleBacktestResult(
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            alpha_score=alpha_score,
            avg_hold_minutes=avg_hold_minutes,
            individual_performance=individual_performance
        )
        
        print(f"✅ Ensemble Strategy Results:")
        print(f"   Total Return: {total_return_pct:+.1f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Trades: {total_trades}")
        print(f"   Alpha Score: {alpha_score:.2f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.1f}%")
        
        return result
    
    def _backtest_individual_signal(self, df: pd.DataFrame, signal_col: str) -> Dict:
        """Backtest individual signal component"""
        
        trades = []
        position = 0
        entry_price = None
        entry_idx = None
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            signal = df[signal_col].iloc[i]
            
            # Entry
            if position == 0 and signal > 0:
                position = 1
                entry_price = current_price
                entry_idx = i
            
            # Exit
            elif position == 1 and signal < 0:
                if entry_price is not None:
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'win': pnl_pct > 0
                    })
                
                position = 0
                entry_price = None
                entry_idx = None
        
        if not trades:
            return {'return': 0, 'win_rate': 0, 'trades': 0}
        
        trades_df = pd.DataFrame(trades)
        return {
            'return': trades_df['pnl_pct'].sum(),
            'win_rate': trades_df['win'].mean() * 100,
            'trades': len(trades_df)
        }
    
    def _check_stop_conditions(self, df: pd.DataFrame, current_idx: int, entry_idx: int, entry_price: float, current_price: float) -> bool:
        """Check stop loss and take profit conditions"""
        
        if entry_idx is None:
            return False
        
        # Stop loss: -3%
        stop_loss_pct = -3.0
        if (current_price - entry_price) / entry_price * 100 <= stop_loss_pct:
            return True
        
        # Take profit: +6%
        take_profit_pct = 6.0
        if (current_price - entry_price) / entry_price * 100 >= take_profit_pct:
            return True
        
        # Maximum holding period (prevent indefinite holds)
        max_hold_periods = 100  # Adjust based on timeframe
        if current_idx - entry_idx >= max_hold_periods:
            return True
        
        return False
    
    def create_strategy_visualization(self, df: pd.DataFrame, backtest_result: EnsembleBacktestResult) -> plt.Figure:
        """Create comprehensive visualization of the ensemble strategy"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle(f'{self.symbol} - Ensemble Alpha Strategy Analysis', fontsize=18, fontweight='bold')
        
        # 1. Price and Ensemble Signal
        ax = axes[0, 0]
        ax.plot(df.index, df['Close'], 'white', alpha=0.8, linewidth=1, label='Price')
        ax2 = ax.twinx()
        ax2.plot(df.index, df['ensemble_signal'], 'cyan', alpha=0.7, label='Ensemble Signal')
        ax2.axhline(y=0.3, color='green', alpha=0.5, linestyle='--', label='Buy Threshold')
        ax2.axhline(y=-0.3, color='red', alpha=0.5, linestyle='--', label='Sell Threshold')
        
        # Mark trade signals
        buy_signals = df[df['trade_signal'] == 1]
        sell_signals = df[df['trade_signal'] == -1]
        ax.scatter(buy_signals.index, buy_signals['Close'], color='green', s=50, alpha=0.8, label='Buy')
        ax.scatter(sell_signals.index, sell_signals['Close'], color='red', s=50, alpha=0.8, label='Sell')
        
        ax.set_title('Price & Ensemble Signal', fontweight='bold')
        ax.set_ylabel('Price')
        ax2.set_ylabel('Signal Strength')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. Signal Components Heatmap
        ax = axes[0, 1]
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        if signal_columns:
            signal_data = df[signal_columns].T
            im = ax.imshow(signal_data.values, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_yticks(range(len(signal_columns)))
            ax.set_yticklabels([col.replace('signal_', '') for col in signal_columns])
            ax.set_title('Individual Signal Components', fontweight='bold')
            ax.set_xlabel('Time')
            plt.colorbar(im, ax=ax, label='Signal Strength')
        
        # 3. Performance Comparison
        ax = axes[1, 0]
        if backtest_result.individual_performance:
            names = []
            returns = []
            win_rates = []
            
            for signal_name, performance in backtest_result.individual_performance.items():
                names.append(signal_name.replace('signal_', ''))
                returns.append(performance['return'])
                win_rates.append(performance['win_rate'])
            
            # Add ensemble results
            names.append('ENSEMBLE')
            returns.append(backtest_result.total_return_pct)
            win_rates.append(backtest_result.win_rate)
            
            x = np.arange(len(names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, returns, width, label='Total Return %', color='skyblue', alpha=0.8)
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, win_rates, width, label='Win Rate %', color='lightcoral', alpha=0.8)
            
            ax.set_xlabel('Strategy Component')
            ax.set_ylabel('Total Return %')
            ax2.set_ylabel('Win Rate %')
            ax.set_title('Individual vs Ensemble Performance', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Highlight ensemble bar
            bars1[-1].set_color('gold')
            bars2[-1].set_color('orange')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 4. Signal Correlation Matrix
        ax = axes[1, 1]
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
            sns.heatmap(self.correlation_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Signal Correlation Matrix', fontweight='bold')
        
        # 5. Confidence and Position Sizing
        ax = axes[2, 0]
        ax.plot(df.index, df['ensemble_confidence'], 'purple', alpha=0.7, label='Confidence')
        ax.fill_between(df.index, 0, df['position_size'], alpha=0.3, color='yellow', label='Position Size')
        ax.set_title('Ensemble Confidence & Position Sizing', fontweight='bold')
        ax.set_ylabel('Confidence / Position Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Strategy Summary
        ax = axes[2, 1]
        ax.axis('off')
        
        summary_text = f"""ENSEMBLE STRATEGY SUMMARY
        
🎯 ALPHA SOURCES COMBINED:
"""
        
        for i, source in enumerate(self.alpha_sources, 1):
            weight = self.weights.get(f"signal_{source.strategy_name.replace(f'_{source.timeframe}', '')}", 0)
            summary_text += f"{i}. {source.strategy_name}: α={source.alpha_score:.2f} (w={weight:.2f})\n"
        
        summary_text += f"""
📊 ENSEMBLE PERFORMANCE:
• Total Return: {backtest_result.total_return_pct:+.1f}%
• Win Rate: {backtest_result.win_rate:.1f}%
• Alpha Score: {backtest_result.alpha_score:.2f}
• Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}
• Max Drawdown: {backtest_result.max_drawdown:.1f}%
• Total Trades: {backtest_result.total_trades}

💡 KEY INSIGHTS:
• Signal Diversification: {len(self.alpha_sources)} alpha sources
• Avg Confidence: {df['ensemble_confidence'].mean():.2f}
• Correlation Penalty Applied: {'Yes' if self.correlation_matrix is not None else 'No'}
• Dynamic Position Sizing: Active
"""
        
        # Add comparison to best individual signal
        if backtest_result.individual_performance:
            best_individual = max(backtest_result.individual_performance.items(), 
                                key=lambda x: x[1]['return'])
            improvement = backtest_result.total_return_pct - best_individual[1]['return']
            summary_text += f"• Improvement over best single: {improvement:+.1f}%"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig

def main():
    """Run ensemble alpha strategy analysis"""
    
    print("🎯 ENSEMBLE ALPHA STRATEGY BUILDER")
    print("=" * 60)
    print("Combining top alpha sources into unified trading strategy")
    
    # Test symbols
    test_symbols = ["VXX", "SPY", "QQQ"]
    
    for symbol in test_symbols:
        print(f"\n{'='*20} {symbol} ENSEMBLE STRATEGY {'='*20}")
        
        try:
            # Build ensemble strategy
            ensemble = EnsembleAlphaStrategy(symbol, top_n_signals=5)
            
            # Discover alpha sources
            alpha_sources = ensemble.discover_alpha_sources()
            
            if not alpha_sources:
                print(f"❌ No alpha sources found for {symbol}")
                continue
            
            # Prepare ensemble data
            df = ensemble.prepare_ensemble_data(period="60d")
            
            if len(df) < 100:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            # Generate ensemble signals
            df = ensemble.generate_ensemble_signals(df)
            
            # Backtest strategy
            result = ensemble.backtest_ensemble_strategy(df)
            
            if result is None:
                print(f"❌ Backtest failed for {symbol}")
                continue
            
            # Create visualization
            print(f"📊 Creating ensemble strategy visualization...")
            fig = ensemble.create_strategy_visualization(df, result)
            
            filename = f'{symbol}_ensemble_strategy.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"✅ Saved: {filename}")
            
            # Export strategy details
            strategy_export = {
                'symbol': symbol,
                'alpha_sources': [asdict(source) for source in alpha_sources],
                'weights': ensemble.weights,
                'performance': asdict(result),
                'creation_date': datetime.now().isoformat()
            }
            
            export_filename = f'{symbol}_ensemble_strategy.json'
            with open(export_filename, 'w') as f:
                json.dump(strategy_export, f, indent=2, default=str)
            print(f"✅ Strategy exported: {export_filename}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error building ensemble for {symbol}: {e}")
    
    print(f"\n✅ Ensemble strategy analysis complete!")

if __name__ == "__main__":
    main()