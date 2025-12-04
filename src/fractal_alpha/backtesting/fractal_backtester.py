"""
📊 FRACTAL INDICATORS BACKTESTING FRAMEWORK
Comprehensive backtesting system for fractal alpha strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import all indicators
from ..indicators.microstructure.vpin import VPINIndicator
from ..indicators.microstructure.kyles_lambda import KylesLambdaIndicator
from ..indicators.multi_timeframe.hurst_exponent import HurstExponentIndicator
from ..indicators.multi_timeframe.williams_fractals import WilliamsFractalIndicator
from ..indicators.ml_features.entropy import EntropyIndicator
from ..indicators.ml_features.hmm import HMMIndicator
from ..base.types import SignalType


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    position_size: float = 0.02  # 2% per trade
    stop_loss: float = 0.02  # 2% stop loss
    take_profit: float = 0.05  # 5% take profit
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    max_positions: int = 5
    use_regime_filter: bool = True
    confidence_threshold: float = 60.0


@dataclass
class Trade:
    """Individual trade record"""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # 'long' or 'short'
    entry_signal: str
    exit_reason: Optional[str]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    is_open: bool = True


@dataclass
class BacktestResults:
    """Backtesting results"""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: timedelta
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    
    # Regime analysis
    regime_performance: Dict[str, float]
    indicator_performance: Dict[str, Dict[str, float]]
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades: List[Trade]
    
    # Metadata
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    
    def to_dict(self) -> dict:
        """Convert results to dictionary"""
        return {
            'performance': {
                'total_return': f"{self.total_return:.2%}",
                'annualized_return': f"{self.annualized_return:.2%}",
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'sortino_ratio': f"{self.sortino_ratio:.2f}",
                'max_drawdown': f"{self.max_drawdown:.2%}",
                'win_rate': f"{self.win_rate:.2%}",
                'profit_factor': f"{self.profit_factor:.2f}"
            },
            'trade_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_win': f"{self.avg_win:.2%}",
                'avg_loss': f"{self.avg_loss:.2%}",
                'avg_duration': str(self.avg_trade_duration)
            },
            'risk_metrics': {
                'volatility': f"{self.volatility:.2%}",
                'var_95': f"{self.var_95:.2%}",
                'cvar_95': f"{self.cvar_95:.2%}"
            },
            'regime_performance': self.regime_performance,
            'indicator_performance': self.indicator_performance
        }


class FractalBacktester:
    """Main backtesting engine for fractal indicators"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.indicators = {}
        self.regime_indicators = {}
        self.signal_indicators = {}
        
    def add_regime_indicator(self, name: str, indicator):
        """Add regime detection indicator"""
        self.regime_indicators[name] = indicator
        
    def add_signal_indicator(self, name: str, indicator):
        """Add signal generation indicator"""
        self.signal_indicators[name] = indicator
        
    def backtest(self, data: pd.DataFrame, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> BacktestResults:
        """Run backtest on historical data"""
        
        # Filter date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Initialize tracking
        capital = self.config.initial_capital
        positions = []
        trades = []
        equity_curve = []
        
        # Calculate all indicators
        print("🔄 Calculating indicators...")
        indicator_signals = self._calculate_all_indicators(data)
        
        # Detect regimes
        print("🔍 Detecting market regimes...")
        regimes = self._detect_regimes(data)
        
        # Simulate trading
        print("💹 Running backtest simulation...")
        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Update equity
            equity = capital + sum(self._calculate_position_value(p, current_price) 
                                 for p in positions if p.is_open)
            equity_curve.append(equity)
            
            # Check exits
            positions = self._check_exits(positions, current_price, current_date, trades)
            
            # Generate signals
            if len([p for p in positions if p.is_open]) < self.config.max_positions:
                signal = self._generate_ensemble_signal(
                    indicator_signals, regimes, i
                )
                
                if signal and signal['confidence'] >= self.config.confidence_threshold:
                    # Enter trade
                    trade = self._enter_trade(
                        current_date, current_price, signal, capital
                    )
                    positions.append(trade)
                    trades.append(trade)
                    capital -= trade.position_size * current_price * (1 + self.config.commission)
        
        # Close all open positions
        for position in positions:
            if position.is_open:
                self._close_position(position, data['Close'].iloc[-1], 
                                   data.index[-1], 'backtest_end')
        
        # Calculate results
        results = self._calculate_results(trades, equity_curve, data, regimes)
        
        print(f"✅ Backtest complete: {results.total_return:.2%} return")
        
        return results
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all indicator signals"""
        signals = {'regime': {}, 'signal': {}}
        
        # Calculate regime indicators
        for name, indicator in self.regime_indicators.items():
            try:
                results = []
                for i in range(indicator.lookback_periods, len(data)):
                    window_data = data.iloc[i-indicator.lookback_periods:i+1]
                    result = indicator.calculate(window_data, "BACKTEST")
                    results.append(result)
                signals['regime'][name] = results
            except Exception as e:
                print(f"Warning: {name} calculation failed: {e}")
        
        # Calculate signal indicators
        for name, indicator in self.signal_indicators.items():
            try:
                results = []
                for i in range(indicator.lookback_periods, len(data)):
                    window_data = data.iloc[i-indicator.lookback_periods:i+1]
                    result = indicator.calculate(window_data, "BACKTEST")
                    results.append(result)
                signals['signal'][name] = results
            except Exception as e:
                print(f"Warning: {name} calculation failed: {e}")
                
        return signals
    
    def _detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regimes using Hurst and Entropy"""
        regimes = []
        
        hurst_ind = HurstExponentIndicator()
        entropy_ind = EntropyIndicator()
        
        for i in range(100, len(data)):
            window = data.iloc[i-100:i+1]
            
            try:
                hurst_result = hurst_ind.calculate(window, "REGIME")
                entropy_result = entropy_ind.calculate(window, "REGIME")
                
                hurst_value = hurst_result.metadata.get('hurst_value', 0.5)
                entropy_value = entropy_result.value / 100
                
                # Classify regime
                if hurst_value > 0.55 and entropy_value < 0.3:
                    regime = 'strong_trend'
                elif hurst_value > 0.55:
                    regime = 'trending'
                elif hurst_value < 0.45:
                    regime = 'mean_reverting'
                elif entropy_value > 0.7:
                    regime = 'chaotic'
                else:
                    regime = 'neutral'
                    
            except:
                regime = 'unknown'
                
            regimes.append(regime)
            
        # Pad beginning
        regimes = ['unknown'] * 100 + regimes
        
        return pd.Series(regimes, index=data.index)
    
    def _generate_ensemble_signal(self, indicator_signals: Dict, 
                                regimes: pd.Series, index: int) -> Optional[Dict]:
        """Generate ensemble signal from multiple indicators"""
        
        if index < 100:  # Not enough data
            return None
            
        current_regime = regimes.iloc[index]
        
        # Regime-based indicator weights
        regime_weights = {
            'trending': {
                'williams_fractal': 0.3,
                'hurst': 0.3,
                'vpin': 0.2,
                'entropy': 0.2
            },
            'mean_reverting': {
                'vpin': 0.4,
                'entropy': 0.3,
                'williams_fractal': 0.2,
                'hurst': 0.1
            },
            'chaotic': {
                'entropy': 0.5,
                'vpin': 0.3,
                'hurst': 0.2,
                'williams_fractal': 0.0
            },
            'default': {
                'williams_fractal': 0.25,
                'hurst': 0.25,
                'vpin': 0.25,
                'entropy': 0.25
            }
        }
        
        weights = regime_weights.get(current_regime, regime_weights['default'])
        
        # Collect signals
        buy_score = 0
        sell_score = 0
        total_confidence = 0
        active_indicators = []
        
        for ind_type in ['regime', 'signal']:
            for name, results in indicator_signals[ind_type].items():
                if name in weights and len(results) > index - 100:
                    result = results[index - 100]
                    
                    if result.signal == SignalType.BUY:
                        buy_score += weights[name] * result.confidence
                    elif result.signal == SignalType.SELL:
                        sell_score += weights[name] * result.confidence
                        
                    total_confidence += result.confidence
                    active_indicators.append(name)
        
        if not active_indicators:
            return None
            
        # Determine signal
        if buy_score > sell_score and buy_score > 30:
            return {
                'type': 'BUY',
                'confidence': buy_score,
                'regime': current_regime,
                'indicators': active_indicators
            }
        elif sell_score > buy_score and sell_score > 30:
            return {
                'type': 'SELL',
                'confidence': sell_score,
                'regime': current_regime,
                'indicators': active_indicators
            }
            
        return None
    
    def _enter_trade(self, date: datetime, price: float, 
                    signal: Dict, capital: float) -> Trade:
        """Enter a new trade"""
        position_value = capital * self.config.position_size
        shares = position_value / (price * (1 + self.config.slippage))
        
        return Trade(
            entry_date=date,
            exit_date=None,
            entry_price=price * (1 + self.config.slippage),
            exit_price=None,
            position_size=shares,
            direction='long' if signal['type'] == 'BUY' else 'short',
            entry_signal=f"{signal['regime']}:{','.join(signal['indicators'][:2])}",
            exit_reason=None,
            pnl=None,
            pnl_pct=None,
            is_open=True
        )
    
    def _check_exits(self, positions: List[Trade], current_price: float,
                    current_date: datetime, all_trades: List[Trade]) -> List[Trade]:
        """Check and execute exits"""
        for position in positions:
            if not position.is_open:
                continue
                
            # Calculate current P&L
            if position.direction == 'long':
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price
                
            # Check stop loss
            if pnl_pct <= -self.config.stop_loss:
                self._close_position(position, current_price, current_date, 'stop_loss')
                
            # Check take profit
            elif pnl_pct >= self.config.take_profit:
                self._close_position(position, current_price, current_date, 'take_profit')
                
        return positions
    
    def _close_position(self, position: Trade, price: float, 
                       date: datetime, reason: str):
        """Close a position"""
        position.exit_date = date
        position.exit_price = price * (1 - self.config.slippage)
        position.exit_reason = reason
        position.is_open = False
        
        # Calculate P&L
        if position.direction == 'long':
            position.pnl = position.position_size * (position.exit_price - position.entry_price)
            position.pnl_pct = (position.exit_price - position.entry_price) / position.entry_price
        else:
            position.pnl = position.position_size * (position.entry_price - position.exit_price)
            position.pnl_pct = (position.entry_price - position.exit_price) / position.entry_price
            
        # Subtract commission
        position.pnl -= position.position_size * (position.entry_price + position.exit_price) * self.config.commission
        
    def _calculate_position_value(self, position: Trade, current_price: float) -> float:
        """Calculate current position value"""
        if not position.is_open:
            return position.pnl
            
        if position.direction == 'long':
            return position.position_size * (current_price - position.entry_price)
        else:
            return position.position_size * (position.entry_price - current_price)
    
    def _calculate_results(self, trades: List[Trade], equity_curve: List[float],
                          data: pd.DataFrame, regimes: pd.Series) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        equity_series = pd.Series(equity_curve, index=data.index)
        returns = equity_series.pct_change().dropna()
        
        # Calculate performance metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        days = (data.index[-1] - data.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        closed_trades = [t for t in trades if not t.is_open]
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            
            avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(t.pnl for t in winning_trades) / 
                              sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else 0
            
            durations = [(t.exit_date - t.entry_date) for t in closed_trades]
            avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
            
            largest_win = max([t.pnl_pct for t in closed_trades]) if closed_trades else 0
            largest_loss = min([t.pnl_pct for t in closed_trades]) if closed_trades else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            avg_duration = timedelta()
            largest_win = largest_loss = 0
            winning_trades = []
            losing_trades = []
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Regime performance
        regime_performance = {}
        for regime in regimes.unique():
            regime_returns = returns[regimes[1:] == regime]
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'trades': len([t for t in closed_trades if regime in t.entry_signal])
                }
        
        # Indicator performance
        indicator_performance = {}
        for indicator_name in set([ind for t in trades for ind in t.entry_signal.split(':')[1].split(',')]):
            ind_trades = [t for t in closed_trades if indicator_name in t.entry_signal]
            if ind_trades:
                indicator_performance[indicator_name] = {
                    'trades': len(ind_trades),
                    'win_rate': len([t for t in ind_trades if t.pnl > 0]) / len(ind_trades),
                    'avg_pnl': np.mean([t.pnl_pct for t in ind_trades])
                }
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95,
            regime_performance=regime_performance,
            indicator_performance=indicator_performance,
            equity_curve=equity_series,
            drawdown_series=drawdown,
            trades=trades,
            config=self.config,
            start_date=data.index[0],
            end_date=data.index[-1]
        )
    
    def plot_results(self, results: BacktestResults):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        ax = axes[0, 0]
        results.equity_curve.plot(ax=ax, title='Equity Curve')
        ax.set_ylabel('Portfolio Value ($)')
        
        # Drawdown
        ax = axes[0, 1]
        results.drawdown_series.plot(ax=ax, title='Drawdown', color='red')
        ax.fill_between(results.drawdown_series.index, 0, results.drawdown_series, alpha=0.3, color='red')
        ax.set_ylabel('Drawdown (%)')
        
        # Monthly returns heatmap
        ax = axes[1, 0]
        returns = results.equity_curve.pct_change()
        monthly_returns = returns.resample('M').apply(lambda x: (1+x).prod()-1)
        monthly_table = pd.pivot_table(
            pd.DataFrame({
                'returns': monthly_returns,
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month
            }),
            values='returns', index='year', columns='month'
        )
        sns.heatmap(monthly_table, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monthly Returns Heatmap')
        
        # Trade distribution
        ax = axes[1, 1]
        trade_pnls = [t.pnl_pct for t in results.trades if not t.is_open]
        ax.hist(trade_pnls, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(trade_pnls), color='red', linestyle='--', label='Mean')
        ax.set_title('Trade P&L Distribution')
        ax.set_xlabel('P&L (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n📊 BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annualized_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Total Trades: {results.total_trades}")
        
        print("\n📈 REGIME PERFORMANCE")
        for regime, stats in results.regime_performance.items():
            print(f"{regime}: Return={stats['return']:.2%}, Sharpe={stats['sharpe']:.2f}, Trades={stats['trades']}")


class PerformanceBenchmark:
    """Performance benchmarking for fractal indicators"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_indicator_suite(self, data: pd.DataFrame,
                                 indicators: List[Any],
                                 configs: List[BacktestConfig]):
        """Benchmark multiple indicators with different configs"""
        
        results = {}
        
        for indicator in indicators:
            for config in configs:
                # Create backtester
                backtester = FractalBacktester(config)
                
                # Add indicator
                if hasattr(indicator, 'is_regime_indicator'):
                    backtester.add_regime_indicator(indicator.name, indicator)
                else:
                    backtester.add_signal_indicator(indicator.name, indicator)
                
                # Run backtest
                result = backtester.backtest(data)
                
                # Store results
                key = f"{indicator.name}_{config.position_size}_{config.confidence_threshold}"
                results[key] = result
                
                print(f"✅ {key}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
        
        self.results = results
        return results
    
    def compare_results(self):
        """Compare benchmark results"""
        
        comparison = pd.DataFrame({
            name: {
                'return': r.total_return,
                'sharpe': r.sharpe_ratio,
                'drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'trades': r.total_trades
            }
            for name, r in self.results.items()
        }).T
        
        # Sort by Sharpe ratio
        comparison = comparison.sort_values('sharpe', ascending=False)
        
        print("\n🏆 BENCHMARK COMPARISON")
        print("=" * 80)
        print(comparison.to_string(float_format='%.2f'))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        comparison[['return']].plot(kind='bar', ax=axes[0,0], title='Total Return')
        comparison[['sharpe']].plot(kind='bar', ax=axes[0,1], title='Sharpe Ratio')
        comparison[['drawdown']].plot(kind='bar', ax=axes[1,0], title='Max Drawdown')
        comparison[['win_rate']].plot(kind='bar', ax=axes[1,1], title='Win Rate')
        
        plt.tight_layout()
        plt.show()
        
        return comparison


def run_fractal_backtest_suite():
    """Run comprehensive backtesting suite"""
    
    print("🚀 FRACTAL INDICATORS BACKTEST SUITE")
    print("=" * 80)
    
    # Load sample data
    import yfinance as yf
    data = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)
    
    # Initialize indicators
    indicators = [
        HurstExponentIndicator(),
        EntropyIndicator(),
        VPINIndicator(),
        WilliamsFractalIndicator()
    ]
    
    # Test configurations
    configs = [
        BacktestConfig(position_size=0.01, confidence_threshold=70),  # Conservative
        BacktestConfig(position_size=0.02, confidence_threshold=60),  # Moderate
        BacktestConfig(position_size=0.05, confidence_threshold=50),  # Aggressive
    ]
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_indicator_suite(data, indicators, configs)
    comparison = benchmark.compare_results()
    
    # Save results
    with open('backtest_results.json', 'w') as f:
        json.dump({
            name: result.to_dict() 
            for name, result in benchmark.results.items()
        }, f, indent=2)
    
    print("\n✅ Backtest suite complete! Results saved to backtest_results.json")
    
    return benchmark.results


if __name__ == "__main__":
    results = run_fractal_backtest_suite()