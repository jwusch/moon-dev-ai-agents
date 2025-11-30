"""
🌙 Moon Dev Portfolio Performance Tracker
Tracks real strategy performance from position history
Built with love by Moon Dev 🚀
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from termcolor import cprint
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.base_agent import BaseAgent
from src.config import address, EXCLUDED_TOKENS
from src import nice_funcs as n


class PortfolioPerformanceTracker(BaseAgent):
    """Track and calculate performance metrics for portfolio strategies"""
    
    def __init__(self):
        """Initialize performance tracker"""
        super().__init__(agent_type='performance_tracker', use_exchange_manager=False)
        
        # Data storage
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "src" / "data" / "portfolio" / "performance"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for performance data
        self.performance_cache = {}
        self.position_history = {}
        
        cprint("📊 Portfolio Performance Tracker initialized!", "green")
    
    def load_position_history(self, lookback_days: int = 30) -> pd.DataFrame:
        """Load position history from saved data or fetch from wallet"""
        history_file = self.data_dir / "position_history.csv"
        
        try:
            if history_file.exists():
                # Load existing history
                df = pd.read_csv(history_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter for recent data
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df['timestamp'] >= cutoff_date]
                
                cprint(f"📈 Loaded {len(df)} position records", "green")
                return df
            else:
                cprint("⚠️ No position history found, creating new file", "yellow")
                return pd.DataFrame(columns=['timestamp', 'token_address', 'strategy_id', 
                                           'amount', 'usd_value', 'price'])
                
        except Exception as e:
            cprint(f"❌ Error loading position history: {str(e)}", "red")
            return pd.DataFrame()
    
    def save_position_snapshot(self, strategy_positions: Dict[str, Dict]) -> None:
        """Save current position snapshot for historical tracking"""
        try:
            history_file = self.data_dir / "position_history.csv"
            
            # Load existing history
            if history_file.exists():
                df = pd.read_csv(history_file)
            else:
                df = pd.DataFrame()
            
            # Add new snapshot data
            timestamp = datetime.now()
            new_rows = []
            
            for strategy_id, position_data in strategy_positions.items():
                for token_address, token_data in position_data.get('tokens', {}).items():
                    new_row = {
                        'timestamp': timestamp,
                        'token_address': token_address,
                        'strategy_id': strategy_id,
                        'amount': token_data.get('amount', 0),
                        'usd_value': token_data.get('usd_value', 0),
                        'price': token_data.get('price', 0)
                    }
                    new_rows.append(new_row)
            
            # Append new data
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Save updated history
            df.to_csv(history_file, index=False)
            cprint(f"💾 Saved {len(new_rows)} position records", "green")
            
        except Exception as e:
            cprint(f"❌ Error saving position snapshot: {str(e)}", "red")
    
    def calculate_returns(self, strategy_id: str, lookback_days: int = 30) -> pd.Series:
        """Calculate daily returns for a strategy"""
        try:
            # Load position history
            history = self.load_position_history(lookback_days)
            
            if history.empty:
                cprint(f"⚠️ No history available for {strategy_id}", "yellow")
                return pd.Series()
            
            # Filter for strategy
            strategy_history = history[history['strategy_id'] == strategy_id].copy()
            
            if strategy_history.empty:
                return pd.Series()
            
            # Calculate daily portfolio values
            daily_values = strategy_history.groupby(strategy_history['timestamp'].dt.date)['usd_value'].sum()
            
            # Calculate returns
            returns = daily_values.pct_change().fillna(0)
            
            return returns
            
        except Exception as e:
            cprint(f"❌ Error calculating returns: {str(e)}", "red")
            return pd.Series()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns series"""
        if returns.empty or len(returns) < 2:
            return 0.0
        
        # Annualize returns and volatility
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_vol = returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return sharpe
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        if returns.empty:
            return 0.0
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running max
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate from returns series"""
        if returns.empty:
            return 0.0
        
        positive_returns = returns > 0
        win_rate = positive_returns.sum() / len(returns)
        
        return win_rate
    
    def get_strategy_performance(self, strategy_id: str, lookback_days: int = 30) -> Dict:
        """Get comprehensive performance metrics for a strategy"""
        try:
            # Check cache first
            cache_key = f"{strategy_id}_{lookback_days}"
            if cache_key in self.performance_cache:
                cached_data = self.performance_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 min cache
                    return cached_data['performance']
            
            # Calculate returns
            returns = self.calculate_returns(strategy_id, lookback_days)
            
            if returns.empty:
                # Return default metrics
                return {
                    'strategy_id': strategy_id,
                    'returns': pd.Series(),
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.5,
                    'volatility': 0.0,
                    'data_points': 0
                }
            
            # Calculate metrics
            total_return = (1 + returns).prod() - 1
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(returns)
            win_rate = self.calculate_win_rate(returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            performance = {
                'strategy_id': strategy_id,
                'returns': returns,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'data_points': len(returns)
            }
            
            # Cache results
            self.performance_cache[cache_key] = {
                'timestamp': datetime.now(),
                'performance': performance
            }
            
            return performance
            
        except Exception as e:
            cprint(f"❌ Error getting strategy performance: {str(e)}", "red")
            return {
                'strategy_id': strategy_id,
                'returns': pd.Series(),
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.5,
                'volatility': 0.0,
                'data_points': 0
            }
    
    def get_correlation_matrix(self, strategy_ids: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """Calculate correlation matrix between strategy returns"""
        try:
            returns_dict = {}
            
            # Get returns for each strategy
            for strategy_id in strategy_ids:
                returns = self.calculate_returns(strategy_id, lookback_days)
                if not returns.empty:
                    returns_dict[strategy_id] = returns
            
            if len(returns_dict) < 2:
                cprint("⚠️ Not enough data for correlation calculation", "yellow")
                return pd.DataFrame()
            
            # Create returns dataframe
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            cprint(f"❌ Error calculating correlations: {str(e)}", "red")
            return pd.DataFrame()
    
    def save_performance_report(self, performance_data: Dict[str, Dict]) -> None:
        """Save performance report to file"""
        try:
            report_file = self.data_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare data for JSON (convert Series to list)
            save_data = {}
            for strategy_id, perf in performance_data.items():
                save_data[strategy_id] = {
                    'total_return': perf['total_return'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'max_drawdown': perf['max_drawdown'],
                    'win_rate': perf['win_rate'],
                    'volatility': perf['volatility'],
                    'data_points': perf['data_points']
                }
            
            with open(report_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            cprint(f"💾 Performance report saved to: {report_file}", "green")
            
        except Exception as e:
            cprint(f"❌ Error saving performance report: {str(e)}", "red")


def main():
    """Demo performance tracking"""
    cprint("🌙 Moon Dev Portfolio Performance Tracker Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    tracker = PortfolioPerformanceTracker()
    
    # Example strategy IDs
    strategies = ["momentum_btc", "mean_reversion_eth", "arbitrage_sol"]
    
    # Get performance for each strategy
    cprint("\n📊 Strategy Performance Metrics:", "yellow")
    performance_data = {}
    
    for strategy_id in strategies:
        perf = tracker.get_strategy_performance(strategy_id, lookback_days=30)
        performance_data[strategy_id] = perf
        
        cprint(f"\n{strategy_id}:", "white")
        cprint(f"  Total Return: {perf['total_return']:.2%}", "white")
        cprint(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}", "white")
        cprint(f"  Max Drawdown: {perf['max_drawdown']:.2%}", "white")
        cprint(f"  Win Rate: {perf['win_rate']:.2%}", "white")
        cprint(f"  Volatility: {perf['volatility']:.2%}", "white")
    
    # Calculate correlation matrix
    cprint("\n🔍 Correlation Matrix:", "yellow")
    corr_matrix = tracker.get_correlation_matrix(strategies, lookback_days=30)
    if not corr_matrix.empty:
        print(corr_matrix)
    else:
        cprint("No correlation data available", "white")
    
    # Save report
    tracker.save_performance_report(performance_data)


if __name__ == "__main__":
    main()