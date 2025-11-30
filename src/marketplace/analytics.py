"""
🌙 Moon Dev's Strategy Performance Analytics
Calculates and tracks comprehensive performance metrics for marketplace strategies
Built with love by Moon Dev 🚀
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class StrategyAnalytics:
    """Comprehensive performance analytics for trading strategies"""
    
    def __init__(self):
        self.metrics_path = os.path.join(project_root, "src", "data", "marketplace", "metrics")
        os.makedirs(self.metrics_path, exist_ok=True)
        
    def calculate_metrics(self, 
                         equity_curve: pd.Series,
                         trades: pd.DataFrame,
                         initial_capital: float = 10000.0,
                         risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from backtest results
        
        Args:
            equity_curve: Time series of portfolio value
            trades: DataFrame with trade history (columns: entry_time, exit_time, pnl, side)
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert equity curve to returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
        
        # Annualized metrics
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = ((equity_curve.iloc[-1] / initial_capital) ** (1/years) - 1) * 100
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino(returns, risk_free_rate)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_curve)
        
        # Trade statistics
        trade_stats = self._analyze_trades(trades)
        
        # Market correlation (if market data available)
        market_correlation = self._calculate_market_correlation(returns)
        
        # Compile all metrics
        metrics = {
            # Return metrics
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "monthly_return": round(total_return / (days / 30), 2),
            
            # Risk metrics
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "volatility_annual": round(returns.std() * np.sqrt(252) * 100, 2),
            
            # Trade metrics
            **trade_stats,
            
            # Additional metrics
            "calmar_ratio": round(annual_return / abs(max_drawdown) if max_drawdown != 0 else 0, 3),
            "profit_factor": trade_stats.get("profit_factor", 0),
            "expectancy": round(trade_stats.get("avg_trade", 0) * trade_stats.get("total_trades", 0), 2),
            "market_correlation": market_correlation,
            
            # Time metrics
            "backtest_days": days,
            "exposure_time": round(trade_stats.get("time_in_market", 0), 2),
            
            # Risk-adjusted metrics
            "risk_reward_ratio": round(trade_stats.get("avg_win", 0) / abs(trade_stats.get("avg_loss", 1)) if trade_stats.get("avg_loss", 0) != 0 else 0, 2)
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        max_dd = drawdown.min()
        
        # Calculate max drawdown duration
        dd_start = drawdown[drawdown == max_dd].index[0]
        dd_recovered = drawdown[drawdown.index > dd_start]
        dd_recovered = dd_recovered[dd_recovered >= -1]  # Nearly recovered
        
        if len(dd_recovered) > 0:
            dd_end = dd_recovered.index[0]
            duration = (dd_end - dd_start).days
        else:
            duration = (equity_curve.index[-1] - dd_start).days
            
        return max_dd, duration
    
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze individual trades"""
        if trades.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "avg_trade": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "avg_trade_duration_hours": 0,
                "profit_factor": 0,
                "time_in_market": 0
            }
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        # Calculate time in market
        total_time = timedelta(0)
        for _, trade in trades.iterrows():
            if pd.notna(trade['entry_time']) and pd.notna(trade['exit_time']):
                total_time += (trade['exit_time'] - trade['entry_time'])
        
        # Calculate total possible time
        if not trades.empty:
            first_trade = trades['entry_time'].min()
            last_trade = trades['exit_time'].max()
            total_possible_time = last_trade - first_trade
            time_in_market = (total_time.total_seconds() / total_possible_time.total_seconds() * 100) if total_possible_time.total_seconds() > 0 else 0
        else:
            time_in_market = 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 1
        profit_factor = total_wins / total_losses if total_losses != 0 else 0
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(len(winning_trades) / len(trades) * 100, 2) if len(trades) > 0 else 0,
            "avg_win": round(winning_trades['pnl'].mean(), 2) if not winning_trades.empty else 0,
            "avg_loss": round(losing_trades['pnl'].mean(), 2) if not losing_trades.empty else 0,
            "avg_trade": round(trades['pnl'].mean(), 2),
            "best_trade": round(trades['pnl'].max(), 2),
            "worst_trade": round(trades['pnl'].min(), 2),
            "avg_trade_duration_hours": round(total_time.total_seconds() / 3600 / len(trades), 2) if len(trades) > 0 else 0,
            "profit_factor": round(profit_factor, 2),
            "time_in_market": round(time_in_market, 2)
        }
    
    def _calculate_market_correlation(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> float:
        """Calculate correlation with market returns"""
        # TODO: Integrate with actual market data
        # For now, return 0 as placeholder
        return 0.0
    
    def compare_strategies(self, strategy_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple strategies side by side
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for strategy_id in strategy_ids:
            metrics_file = os.path.join(self.metrics_path, f"{strategy_id}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['strategy_id'] = strategy_id
                    comparison_data.append(metrics)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        # Sort by total return by default
        df = df.sort_values('total_return', ascending=False)
        
        return df
    
    def save_metrics(self, strategy_id: str, metrics: Dict[str, Any], backtest_config: Dict[str, Any] = None):
        """
        Save metrics to disk
        
        Args:
            strategy_id: Strategy identifier
            metrics: Calculated metrics
            backtest_config: Backtest configuration used
        """
        # Add metadata
        full_metrics = {
            "strategy_id": strategy_id,
            "calculated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "backtest_config": backtest_config or {}
        }
        
        # Save to file
        filename = os.path.join(self.metrics_path, f"{strategy_id}_metrics.json")
        with open(filename, 'w') as f:
            json.dump(full_metrics, f, indent=2)
    
    def generate_performance_report(self, strategy_id: str, metrics: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance report
        
        Args:
            strategy_id: Strategy identifier
            metrics: Performance metrics
            
        Returns:
            Formatted report string
        """
        report = f"""
📊 PERFORMANCE REPORT - Strategy: {strategy_id}
{'='*60}

RETURNS
-------
Total Return: {metrics.get('total_return', 0):.2f}%
Annual Return: {metrics.get('annual_return', 0):.2f}%
Monthly Return: {metrics.get('monthly_return', 0):.2f}%

RISK METRICS
------------
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
Max DD Duration: {metrics.get('max_drawdown_duration_days', 0)} days
Annual Volatility: {metrics.get('volatility_annual', 0):.2f}%

TRADE STATISTICS
----------------
Total Trades: {metrics.get('total_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.2f}%
Profit Factor: {metrics.get('profit_factor', 0):.2f}
Average Trade: {metrics.get('avg_trade', 0):.2f}
Best Trade: {metrics.get('best_trade', 0):.2f}
Worst Trade: {metrics.get('worst_trade', 0):.2f}

EFFICIENCY
----------
Time in Market: {metrics.get('exposure_time', 0):.2f}%
Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
Expectancy: ${metrics.get('expectancy', 0):.2f}

{'='*60}
"""
        return report
    
    def categorize_performance(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Categorize strategy performance into tiers
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance categories
        """
        categories = {}
        
        # Return category
        total_return = metrics.get('total_return', 0)
        if total_return > 100:
            categories['return_tier'] = 'excellent'
        elif total_return > 50:
            categories['return_tier'] = 'good'
        elif total_return > 20:
            categories['return_tier'] = 'moderate'
        elif total_return > 0:
            categories['return_tier'] = 'low'
        else:
            categories['return_tier'] = 'negative'
        
        # Risk category
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            categories['risk_adjusted'] = 'excellent'
        elif sharpe > 1:
            categories['risk_adjusted'] = 'good'
        elif sharpe > 0.5:
            categories['risk_adjusted'] = 'moderate'
        else:
            categories['risk_adjusted'] = 'poor'
        
        # Consistency category
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 60:
            categories['consistency'] = 'high'
        elif win_rate > 45:
            categories['consistency'] = 'moderate'
        else:
            categories['consistency'] = 'low'
        
        # Drawdown category
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd < 10:
            categories['drawdown_risk'] = 'low'
        elif max_dd < 20:
            categories['drawdown_risk'] = 'moderate'
        elif max_dd < 30:
            categories['drawdown_risk'] = 'high'
        else:
            categories['drawdown_risk'] = 'extreme'
        
        return categories


if __name__ == "__main__":
    # Example usage
    analytics = StrategyAnalytics()
    
    # Example: Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    equity_curve = pd.Series(
        10000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
        index=dates
    )
    
    # Sample trades
    trades = pd.DataFrame({
        'entry_time': dates[::10][:-1],
        'exit_time': dates[5::10],
        'pnl': np.random.normal(50, 100, len(dates[::10])-1),
        'side': ['long'] * (len(dates[::10])-1)
    })
    
    # Calculate metrics
    metrics = analytics.calculate_metrics(equity_curve, trades)
    
    # Generate report
    report = analytics.generate_performance_report("test_strategy", metrics)
    print(report)
    
    # Categorize performance
    categories = analytics.categorize_performance(metrics)
    print("\nPerformance Categories:", categories)