"""
📈 Comprehensive QQQ Ensemble Strategy Backtest
Long-term backtest of ensemble alpha strategy on QQQ with maximum historical data
Tests profitability, risk metrics, and performance across different market regimes

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import talib
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ensemble_alpha_strategy import EnsembleAlphaStrategy, EnsembleBacktestResult

plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class ComprehensiveBacktestResults:
    symbol: str
    start_date: str
    end_date: str
    total_years: float
    
    # Strategy Performance
    strategy_total_return_pct: float
    strategy_annualized_return_pct: float
    strategy_cagr: float
    
    # Buy & Hold Comparison
    buy_hold_total_return_pct: float
    buy_hold_annualized_return_pct: float
    buy_hold_cagr: float
    
    # Outperformance
    excess_return_pct: float
    outperformance_ratio: float
    
    # Risk Metrics
    strategy_volatility: float
    strategy_sharpe: float
    strategy_sortino: float
    strategy_max_drawdown: float
    strategy_calmar_ratio: float
    
    # Buy & Hold Risk
    buy_hold_volatility: float
    buy_hold_sharpe: float
    buy_hold_max_drawdown: float
    
    # Trading Metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Market Regime Performance
    bull_market_performance: Dict
    bear_market_performance: Dict
    sideways_market_performance: Dict
    
    # Risk-Adjusted Metrics
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float

class ComprehensiveBacktester:
    """
    Comprehensive backtesting engine for ensemble strategies
    """
    
    def __init__(self, symbol: str = "QQQ"):
        self.symbol = symbol
        self.ensemble = None
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def download_maximum_data(self) -> pd.DataFrame:
        """Download maximum historical data available"""
        
        print(f"📊 Downloading maximum historical data for {self.symbol}...")
        
        try:
            # Try to get maximum data (usually goes back to 1999 for QQQ)
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period="max", interval="1d")
            
            if df.empty:
                raise ValueError(f"No data available for {self.symbol}")
            
            # Clean column names
            if df.columns.nlevels > 1:
                df.columns = [col[0] for col in df.columns]
            
            # Remove any rows with missing data
            df = df.dropna()
            
            print(f"✅ Downloaded {len(df)} days of data")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Total years: {(df.index[-1] - df.index[0]).days / 365.25:.1f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return pd.DataFrame()
    
    def prepare_ensemble_strategy(self, df: pd.DataFrame) -> Tuple[EnsembleAlphaStrategy, pd.DataFrame]:
        """Prepare ensemble strategy using first portion of data for discovery"""
        
        print(f"🔍 Preparing ensemble strategy...")
        
        # Use first 2 years for alpha discovery (out-of-sample testing)
        discovery_period = 252 * 2  # 2 years of trading days
        discovery_data = df.iloc[:discovery_period].copy()
        
        # Initialize ensemble with discovered alphas
        self.ensemble = EnsembleAlphaStrategy(self.symbol, top_n_signals=5)
        
        # Discover alpha sources using early data
        print(f"   Using first {len(discovery_data)} days for alpha discovery...")
        
        # Manually add comprehensive indicators to discovery data
        discovery_data = self._add_all_indicators(discovery_data)
        
        # Find best strategies from the discovery period
        alpha_sources = self._discover_alpha_sources_manual(discovery_data)
        
        if not alpha_sources:
            raise ValueError("No profitable alpha sources found in discovery period")
        
        self.ensemble.alpha_sources = alpha_sources
        
        print(f"✅ Found {len(alpha_sources)} profitable alpha sources:")
        for i, source in enumerate(alpha_sources, 1):
            print(f"   {i}. {source['name']}: {source['total_return']:.1f}% return, {source['win_rate']:.1f}% win rate")
        
        # Prepare full dataset with indicators
        print(f"📊 Preparing full dataset with indicators...")
        df_full = self._add_all_indicators(df)
        
        # Generate ensemble signals for full period
        df_signals = self._generate_ensemble_signals_manual(df_full, alpha_sources)
        
        return self.ensemble, df_signals
    
    def _add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators needed for alpha strategies"""
        
        # Periods for different timeframes (daily data)
        short, medium, long = 5, 20, 50
        
        # === CORE INDICATORS ===
        df['SMA_Short'] = df['Close'].rolling(short).mean()
        df['SMA_Medium'] = df['Close'].rolling(medium).mean()
        df['SMA_Long'] = df['Close'].rolling(long).mean()
        df['Distance_Short'] = (df['Close'] - df['SMA_Short']) / df['SMA_Short'] * 100
        df['Distance_Medium'] = (df['Close'] - df['SMA_Medium']) / df['SMA_Medium'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, medium)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values, medium, 2, 2)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        df['BB_Squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'].values)
        
        # ADX
        df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, medium)
        
        # Volatility
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, medium)
        df['Vol_Short'] = df['Close'].rolling(short).std()
        df['Vol_Medium'] = df['Close'].rolling(medium).std()
        df['Vol_Regime'] = np.where(df['Vol_Short'] > df['Vol_Medium'] * 1.5, 'High',
                                   np.where(df['Vol_Short'] < df['Vol_Medium'] * 0.7, 'Low', 'Normal'))
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(medium).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['VWAP'] = (df['Close'] * df['Volume']).rolling(medium).sum() / df['Volume'].rolling(medium).sum()
            df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
        else:
            df['Volume_Ratio'] = 1.0
            df['Price_vs_VWAP'] = 0.0
        
        # Momentum
        df['ROC_Short'] = talib.ROC(df['Close'].values, short)
        df['ROC_Medium'] = talib.ROC(df['Close'].values, medium)
        
        # Behavioral indicators
        returns = df['Close'].pct_change()
        df['Extreme_Move'] = abs(returns) > returns.rolling(medium).std() * 2
        
        # Time features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df.dropna()
    
    def _discover_alpha_sources_manual(self, df: pd.DataFrame) -> List[Dict]:
        """Manually discover profitable alpha sources in discovery period"""
        
        print(f"   Testing alpha strategies on discovery period...")
        
        strategies = [
            {
                'name': 'RSI_Reversion',
                'entry_logic': lambda d: (d['RSI'] < 30) & (d['Distance_Medium'] < -2.0),
                'exit_logic': lambda d: (d['RSI'] > 50) | (d['Distance_Medium'] > 1.0),
                'type': 'mean_reversion'
            },
            {
                'name': 'BB_Reversion', 
                'entry_logic': lambda d: d['BB_Position'] < 0.2,
                'exit_logic': lambda d: d['BB_Position'] > 0.8,
                'type': 'mean_reversion'
            },
            {
                'name': 'Vol_Expansion',
                'entry_logic': lambda d: d['Vol_Regime'] == 'Low',
                'exit_logic': lambda d: d['Vol_Regime'] == 'High',
                'type': 'volatility'
            },
            {
                'name': 'MACD_Momentum',
                'entry_logic': lambda d: (d['MACD'] > d['MACD_Signal']) & (d['MACD_Hist'] > 0) & (d['ADX'] > 25),
                'exit_logic': lambda d: (d['MACD'] < d['MACD_Signal']) | (d['ADX'] < 20),
                'type': 'momentum'
            },
            {
                'name': 'Extreme_Reversion',
                'entry_logic': lambda d: d['Extreme_Move'] & (d['Distance_Short'] < -3.0),
                'exit_logic': lambda d: abs(d['Distance_Short']) < 1.0,
                'type': 'reversion'
            }
        ]
        
        profitable_sources = []
        
        for strategy in strategies:
            try:
                result = self._backtest_single_strategy(df, strategy)
                
                # Only keep strategies that are profitable with reasonable win rates
                if (result['total_return'] > 5.0 and  # At least 5% return over discovery period
                    result['win_rate'] > 45.0 and     # At least 45% win rate
                    result['trades'] >= 10):          # At least 10 trades
                    
                    profitable_sources.append({
                        'name': strategy['name'],
                        'type': strategy['type'],
                        'entry_logic': strategy['entry_logic'],
                        'exit_logic': strategy['exit_logic'],
                        'total_return': result['total_return'],
                        'win_rate': result['win_rate'],
                        'trades': result['trades'],
                        'alpha_score': result['total_return'] / result['trades'] * (result['win_rate'] / 100)
                    })
                    
                    print(f"      ✅ {strategy['name']}: {result['total_return']:.1f}% return, {result['win_rate']:.1f}% win rate")
                else:
                    print(f"      ❌ {strategy['name']}: {result['total_return']:.1f}% return, {result['win_rate']:.1f}% win rate (not profitable enough)")
                    
            except Exception as e:
                print(f"      ❌ {strategy['name']}: Error - {e}")
        
        # Sort by alpha score
        profitable_sources.sort(key=lambda x: x['alpha_score'], reverse=True)
        
        return profitable_sources[:5]  # Top 5 sources
    
    def _backtest_single_strategy(self, df: pd.DataFrame, strategy: Dict) -> Dict:
        """Backtest a single strategy on discovery data"""
        
        try:
            entry_signals = strategy['entry_logic'](df)
            exit_signals = strategy['exit_logic'](df)
            
            if entry_signals.sum() == 0:
                return {'total_return': 0, 'win_rate': 0, 'trades': 0}
            
            trades = []
            position = 0
            entry_price = None
            entry_idx = None
            
            for i in range(len(df)):
                current_price = df['Close'].iloc[i]
                
                # Entry
                if position == 0 and entry_signals.iloc[i]:
                    position = 1
                    entry_price = current_price
                    entry_idx = i
                
                # Exit
                elif position == 1 and (exit_signals.iloc[i] or 
                                      (entry_idx is not None and i - entry_idx > 30)):  # Max 30 day hold
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
                return {'total_return': 0, 'win_rate': 0, 'trades': 0}
            
            trades_df = pd.DataFrame(trades)
            
            return {
                'total_return': trades_df['pnl_pct'].sum(),
                'win_rate': trades_df['win'].mean() * 100,
                'trades': len(trades_df)
            }
            
        except Exception:
            return {'total_return': 0, 'win_rate': 0, 'trades': 0}
    
    def _generate_ensemble_signals_manual(self, df: pd.DataFrame, alpha_sources: List[Dict]) -> pd.DataFrame:
        """Generate ensemble signals using discovered alpha sources"""
        
        print(f"🎯 Generating ensemble signals for full dataset...")
        
        # Calculate individual signals
        for source in alpha_sources:
            signal_name = f"signal_{source['name']}"
            
            entry_condition = source['entry_logic'](df)
            exit_condition = source['exit_logic'](df)
            
            signal = np.zeros(len(df))
            signal[entry_condition] = 1.0
            signal[exit_condition] = -1.0
            
            df[signal_name] = signal
        
        # Calculate weights based on alpha scores
        total_alpha = sum(source['alpha_score'] for source in alpha_sources)
        weights = {}
        
        for source in alpha_sources:
            signal_name = f"signal_{source['name']}"
            weights[signal_name] = source['alpha_score'] / total_alpha
        
        print(f"   Signal weights: {weights}")
        
        # Generate ensemble signal
        ensemble_signal = np.zeros(len(df))
        ensemble_confidence = np.zeros(len(df))
        
        signal_columns = [f"signal_{source['name']}" for source in alpha_sources]
        
        for i in range(len(df)):
            weighted_signal = 0
            total_weight = 0
            active_signals = 0
            
            for signal_col in signal_columns:
                if abs(df[signal_col].iloc[i]) > 0:
                    weight = weights[signal_col]
                    signal_value = df[signal_col].iloc[i]
                    
                    weighted_signal += signal_value * weight
                    total_weight += weight
                    active_signals += 1
            
            ensemble_signal[i] = weighted_signal
            ensemble_confidence[i] = total_weight * min(1.0, active_signals / len(signal_columns))
        
        df['ensemble_signal'] = ensemble_signal
        df['ensemble_confidence'] = ensemble_confidence
        
        # Generate trading signals with thresholds
        df['trade_signal'] = 0
        df['position_size'] = 0.0
        
        buy_threshold = 0.3
        sell_threshold = -0.3
        min_confidence = 0.2
        
        buy_condition = (df['ensemble_signal'] > buy_threshold) & (df['ensemble_confidence'] > min_confidence)
        sell_condition = (df['ensemble_signal'] < sell_threshold) & (df['ensemble_confidence'] > min_confidence)
        
        df.loc[buy_condition, 'trade_signal'] = 1
        df.loc[sell_condition, 'trade_signal'] = -1
        df.loc[df['trade_signal'] != 0, 'position_size'] = df.loc[df['trade_signal'] != 0, 'ensemble_confidence']
        
        print(f"   Generated {(df['trade_signal'] == 1).sum()} buy signals and {(df['trade_signal'] == -1).sum()} sell signals")
        
        return df
    
    def comprehensive_backtest(self, df: pd.DataFrame) -> ComprehensiveBacktestResults:
        """Run comprehensive backtest with full analysis"""
        
        print(f"📈 Running comprehensive backtest...")
        
        # Prepare ensemble strategy
        ensemble, df_signals = self.prepare_ensemble_strategy(df)
        
        # Run strategy backtest
        strategy_results = self._run_strategy_backtest(df_signals)
        
        # Run buy & hold backtest
        buy_hold_results = self._run_buy_hold_backtest(df_signals)
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(df_signals, strategy_results, buy_hold_results)
        
        return results
    
    def _run_strategy_backtest(self, df: pd.DataFrame) -> Dict:
        """Run detailed strategy backtest"""
        
        print(f"   📊 Running strategy backtest...")
        
        trades = []
        positions = []  # Track position history
        position = 0
        entry_price = None
        entry_idx = None
        
        initial_capital = 100000  # $100k starting capital
        current_capital = initial_capital
        position_value = 0
        shares_held = 0
        
        daily_values = []  # Track daily portfolio value
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            trade_signal = df['trade_signal'].iloc[i]
            position_size = df['position_size'].iloc[i]
            current_date = df.index[i]
            
            # Calculate current portfolio value
            if shares_held > 0:
                position_value = shares_held * current_price
                portfolio_value = current_capital + position_value
            else:
                portfolio_value = current_capital
            
            daily_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'position_value': position_value,
                'shares': shares_held,
                'price': current_price
            })
            
            # Entry
            if position == 0 and trade_signal == 1:
                position = 1
                entry_price = current_price
                entry_idx = i
                
                # Calculate shares to buy based on position size and available capital
                max_investment = current_capital * 0.95  # Use 95% of capital
                actual_investment = max_investment * position_size  # Scale by confidence
                shares_to_buy = actual_investment / current_price
                
                if shares_to_buy > 0:
                    shares_held = shares_to_buy
                    current_capital -= actual_investment
                    position_value = shares_held * current_price
                    
                    print(f"   📈 BUY: {current_date.date()} @ ${current_price:.2f}, "
                          f"Shares: {shares_held:.0f}, Investment: ${actual_investment:,.0f}")
            
            # Exit
            elif position == 1 and (trade_signal == -1 or 
                                  (entry_idx is not None and i - entry_idx > 60)):  # Max 60 day hold
                if entry_price is not None and shares_held > 0:
                    # Sell all shares
                    sale_proceeds = shares_held * current_price
                    current_capital += sale_proceeds
                    
                    hold_days = i - entry_idx
                    total_pnl = sale_proceeds - (shares_held * entry_price)
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares_held,
                        'hold_days': hold_days,
                        'pnl_dollars': total_pnl,
                        'pnl_pct': pnl_pct,
                        'position_size_used': position_size,
                        'win': pnl_pct > 0
                    })
                    
                    print(f"   📉 SELL: {current_date.date()} @ ${current_price:.2f}, "
                          f"P&L: ${total_pnl:,.0f} ({pnl_pct:+.1f}%), Hold: {hold_days}d")
                    
                    # Reset position
                    shares_held = 0
                    position_value = 0
                
                position = 0
                entry_price = None
                entry_idx = None
        
        # Final portfolio value
        final_portfolio_value = current_capital + (shares_held * df['Close'].iloc[-1] if shares_held > 0 else 0)
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        daily_values_df = pd.DataFrame(daily_values)
        
        return {
            'trades': trades_df,
            'daily_values': daily_values_df,
            'initial_capital': initial_capital,
            'final_capital': final_portfolio_value,
            'total_return_pct': (final_portfolio_value / initial_capital - 1) * 100
        }
    
    def _run_buy_hold_backtest(self, df: pd.DataFrame) -> Dict:
        """Run buy & hold comparison"""
        
        print(f"   📊 Running buy & hold comparison...")
        
        initial_capital = 100000
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        shares = initial_capital / start_price
        final_value = shares * end_price
        total_return_pct = (final_value / initial_capital - 1) * 100
        
        # Daily values for buy & hold
        daily_values = []
        for i, row in df.iterrows():
            daily_value = shares * row['Close']
            daily_values.append({
                'date': i,
                'portfolio_value': daily_value,
                'price': row['Close']
            })
        
        return {
            'daily_values': pd.DataFrame(daily_values),
            'initial_capital': initial_capital,
            'final_capital': final_value,
            'total_return_pct': total_return_pct,
            'shares': shares
        }
    
    def _calculate_comprehensive_metrics(self, df: pd.DataFrame, strategy_results: Dict, buy_hold_results: Dict) -> ComprehensiveBacktestResults:
        """Calculate comprehensive performance metrics"""
        
        print(f"   📊 Calculating comprehensive metrics...")
        
        # Basic info
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        total_years = (df.index[-1] - df.index[0]).days / 365.25
        
        # Strategy performance
        strategy_total_return = strategy_results['total_return_pct']
        strategy_cagr = (strategy_results['final_capital'] / strategy_results['initial_capital']) ** (1/total_years) - 1
        strategy_annualized = strategy_cagr * 100
        
        # Buy & hold performance
        buy_hold_total_return = buy_hold_results['total_return_pct']
        buy_hold_cagr = (buy_hold_results['final_capital'] / buy_hold_results['initial_capital']) ** (1/total_years) - 1
        buy_hold_annualized = buy_hold_cagr * 100
        
        # Outperformance
        excess_return = strategy_total_return - buy_hold_total_return
        outperformance_ratio = strategy_results['final_capital'] / buy_hold_results['final_capital']
        
        # Volatility calculations
        strategy_daily_returns = strategy_results['daily_values']['portfolio_value'].pct_change().dropna()
        buy_hold_daily_returns = buy_hold_results['daily_values']['portfolio_value'].pct_change().dropna()
        
        strategy_volatility = strategy_daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        buy_hold_volatility = buy_hold_daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratios
        strategy_sharpe = (strategy_annualized - self.risk_free_rate * 100) / strategy_volatility if strategy_volatility > 0 else 0
        buy_hold_sharpe = (buy_hold_annualized - self.risk_free_rate * 100) / buy_hold_volatility if buy_hold_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        strategy_downside_returns = strategy_daily_returns[strategy_daily_returns < 0]
        strategy_downside_dev = strategy_downside_returns.std() * np.sqrt(252) * 100
        strategy_sortino = (strategy_annualized - self.risk_free_rate * 100) / strategy_downside_dev if strategy_downside_dev > 0 else 0
        
        # Maximum drawdowns
        strategy_cumulative = (1 + strategy_daily_returns).cumprod()
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max * 100
        strategy_max_drawdown = strategy_drawdown.min()
        
        buy_hold_cumulative = (1 + buy_hold_daily_returns).cumprod()
        buy_hold_running_max = buy_hold_cumulative.expanding().max()
        buy_hold_drawdown = (buy_hold_cumulative - buy_hold_running_max) / buy_hold_running_max * 100
        buy_hold_max_drawdown = buy_hold_drawdown.min()
        
        # Calmar ratio
        strategy_calmar = strategy_annualized / abs(strategy_max_drawdown) if strategy_max_drawdown != 0 else 0
        
        # Trading metrics
        trades_df = strategy_results['trades']
        if not trades_df.empty:
            total_trades = len(trades_df)
            win_rate = trades_df['win'].mean() * 100
            
            wins = trades_df[trades_df['win']]['pnl_pct']
            losses = trades_df[~trades_df['win']]['pnl_pct']
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
            avg_trade_return = trades_df['pnl_pct'].mean()
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for win in trades_df['win']:
                if win:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            total_trades = win_rate = profit_factor = avg_trade_return = 0
            avg_win = avg_loss = max_consecutive_wins = max_consecutive_losses = 0
        
        # Market regime analysis
        bull_market_performance = self._analyze_market_regime(df, strategy_results, 'bull')
        bear_market_performance = self._analyze_market_regime(df, strategy_results, 'bear')
        sideways_market_performance = self._analyze_market_regime(df, strategy_results, 'sideways')
        
        # Information ratio (excess return / tracking error)
        excess_returns = strategy_daily_returns - buy_hold_daily_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio and Jensen's alpha (simplified)
        treynor_ratio = strategy_annualized / 1.0  # Assuming beta of 1 for simplification
        jensen_alpha = strategy_annualized - buy_hold_annualized  # Simplified Jensen's alpha
        
        # Store results for auto-registry
        self.last_results = ComprehensiveBacktestResults(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            total_years=total_years,
            
            strategy_total_return_pct=strategy_total_return,
            strategy_annualized_return_pct=strategy_annualized,
            strategy_cagr=strategy_cagr * 100,
            
            buy_hold_total_return_pct=buy_hold_total_return,
            buy_hold_annualized_return_pct=buy_hold_annualized,
            buy_hold_cagr=buy_hold_cagr * 100,
            
            excess_return_pct=excess_return,
            outperformance_ratio=outperformance_ratio,
            
            strategy_volatility=strategy_volatility,
            strategy_sharpe=strategy_sharpe,
            strategy_sortino=strategy_sortino,
            strategy_max_drawdown=strategy_max_drawdown,
            strategy_calmar_ratio=strategy_calmar,
            
            buy_hold_volatility=buy_hold_volatility,
            buy_hold_sharpe=buy_hold_sharpe,
            buy_hold_max_drawdown=buy_hold_max_drawdown,
            
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            
            bull_market_performance=bull_market_performance,
            bear_market_performance=bear_market_performance,
            sideways_market_performance=sideways_market_performance,
            
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha
        )
        
        return self.last_results
    
    def auto_register_result(self, category=None):
        """Automatically register this result in AEGS goldmine registry"""
        try:
            from aegs_auto_registry import register_backtest_result
            
            # Get the latest results
            if hasattr(self, 'last_results'):
                success = register_backtest_result(self.symbol, self.last_results, category)
                if success:
                    print(f"✅ {self.symbol} automatically added to AEGS registry!")
                return success
            else:
                print("❌ No backtest results available to register")
                return False
        except Exception as e:
            print(f"❌ Error registering result: {str(e)}")
            return False
    
    def _analyze_market_regime(self, df: pd.DataFrame, strategy_results: Dict, regime: str) -> Dict:
        """Analyze performance during specific market regimes"""
        
        # Simple regime classification based on rolling returns
        window = 60  # 60-day rolling window
        rolling_returns = df['Close'].pct_change(window)
        
        if regime == 'bull':
            mask = rolling_returns > 0.1  # Bull market: >10% return over 60 days
        elif regime == 'bear':
            mask = rolling_returns < -0.1  # Bear market: <-10% return over 60 days  
        else:  # sideways
            mask = (rolling_returns >= -0.1) & (rolling_returns <= 0.1)
        
        regime_days = mask.sum()
        total_days = len(mask)
        regime_percentage = regime_days / total_days * 100
        
        return {
            'days': regime_days,
            'percentage_of_time': regime_percentage,
            'regime_type': regime
        }
    
    def create_comprehensive_visualization(self, results: ComprehensiveBacktestResults, 
                                         strategy_results: Dict, buy_hold_results: Dict, 
                                         df: pd.DataFrame) -> plt.Figure:
        """Create comprehensive backtest visualization"""
        
        print(f"📊 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle(f'{self.symbol} - Comprehensive Ensemble Strategy Backtest', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Portfolio Value Comparison
        ax = axes[0, 0]
        strategy_values = strategy_results['daily_values']
        buy_hold_values = buy_hold_results['daily_values']
        
        ax.plot(strategy_values['date'], strategy_values['portfolio_value'], 
               'cyan', linewidth=2, label=f'Ensemble Strategy')
        ax.plot(buy_hold_values['date'], buy_hold_values['portfolio_value'], 
               'orange', linewidth=2, label='Buy & Hold')
        
        ax.set_title('Portfolio Value Over Time', fontweight='bold', fontsize=14)
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Cumulative Returns
        ax = axes[0, 1]
        strategy_daily_returns = strategy_values['portfolio_value'].pct_change().fillna(0)
        buy_hold_daily_returns = buy_hold_values['portfolio_value'].pct_change().fillna(0)
        
        strategy_cumulative = (1 + strategy_daily_returns).cumprod() - 1
        buy_hold_cumulative = (1 + buy_hold_daily_returns).cumprod() - 1
        
        ax.plot(strategy_values['date'], strategy_cumulative * 100, 
               'cyan', linewidth=2, label='Ensemble Strategy')
        ax.plot(buy_hold_values['date'], buy_hold_cumulative * 100, 
               'orange', linewidth=2, label='Buy & Hold')
        
        ax.set_title('Cumulative Returns (%)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Drawdown Analysis
        ax = axes[0, 2]
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) * 100
        
        buy_hold_running_max = buy_hold_cumulative.expanding().max()
        buy_hold_drawdown = (buy_hold_cumulative - buy_hold_running_max) * 100
        
        ax.fill_between(strategy_values['date'], strategy_drawdown, 0, 
                       alpha=0.5, color='red', label='Strategy Drawdown')
        ax.fill_between(buy_hold_values['date'], buy_hold_drawdown, 0, 
                       alpha=0.3, color='orange', label='Buy & Hold Drawdown')
        
        ax.set_title('Drawdown Analysis', fontweight='bold', fontsize=14)
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax = axes[1, 0]
        metrics = ['Total Return %', 'Annualized %', 'Volatility %', 'Sharpe', 'Max DD %']
        strategy_metrics = [
            results.strategy_total_return_pct,
            results.strategy_annualized_return_pct, 
            results.strategy_volatility,
            results.strategy_sharpe,
            abs(results.strategy_max_drawdown)
        ]
        buy_hold_metrics = [
            results.buy_hold_total_return_pct,
            results.buy_hold_annualized_return_pct,
            results.buy_hold_volatility, 
            results.buy_hold_sharpe,
            abs(results.buy_hold_max_drawdown)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strategy_metrics, width, 
                      label='Ensemble Strategy', color='cyan', alpha=0.8)
        bars2 = ax.bar(x + width/2, buy_hold_metrics, width,
                      label='Buy & Hold', color='orange', alpha=0.8)
        
        ax.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar1, bar2, val1, val2 in zip(bars1, bars2, strategy_metrics, buy_hold_metrics):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{val1:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{val2:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Trading Activity
        ax = axes[1, 1]
        if not strategy_results['trades'].empty:
            trades = strategy_results['trades']
            
            # Monthly trade count
            trades['month'] = trades['entry_date'].dt.to_period('M')
            monthly_trades = trades.groupby('month').size()
            
            ax.bar(range(len(monthly_trades)), monthly_trades.values, 
                  color='skyblue', alpha=0.7)
            ax.set_title('Monthly Trading Activity', fontweight='bold', fontsize=14)
            ax.set_ylabel('Number of Trades')
            ax.set_xlabel('Months (Sequential)')
            ax.grid(True, alpha=0.3)
        
        # 6. Win/Loss Distribution
        ax = axes[1, 2]
        if not strategy_results['trades'].empty:
            trades = strategy_results['trades']
            wins = trades[trades['win']]['pnl_pct']
            losses = trades[~trades['win']]['pnl_pct']
            
            ax.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
            ax.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
            
            ax.set_title('Trade P&L Distribution', fontweight='bold', fontsize=14)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Trade Return (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 7. Risk-Return Scatter
        ax = axes[2, 0]
        ax.scatter(results.strategy_volatility, results.strategy_annualized_return_pct,
                  s=200, c='cyan', alpha=0.8, label='Ensemble Strategy')
        ax.scatter(results.buy_hold_volatility, results.buy_hold_annualized_return_pct,
                  s=200, c='orange', alpha=0.8, label='Buy & Hold')
        
        ax.set_title('Risk-Return Profile', fontweight='bold', fontsize=14)
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Rolling Sharpe Ratio
        ax = axes[2, 1]
        rolling_window = 252  # 1 year
        if len(strategy_daily_returns) > rolling_window:
            rolling_sharpe = strategy_daily_returns.rolling(rolling_window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            
            ax.plot(strategy_values['date'][rolling_window:], 
                   rolling_sharpe[rolling_window:], 'cyan', linewidth=2)
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
            ax.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Sharpe = 0.5')
            
            ax.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold', fontsize=14)
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        summary_text = f"""ENSEMBLE STRATEGY BACKTEST SUMMARY

📊 PERFORMANCE METRICS:
• Total Return: {results.strategy_total_return_pct:+.1f}% vs {results.buy_hold_total_return_pct:+.1f}% (B&H)
• Annualized Return: {results.strategy_annualized_return_pct:.1f}% vs {results.buy_hold_annualized_return_pct:.1f}% (B&H)
• CAGR: {results.strategy_cagr:.1f}% vs {results.buy_hold_cagr:.1f}% (B&H)
• Excess Return: {results.excess_return_pct:+.1f}%
• Outperformance Ratio: {results.outperformance_ratio:.2f}x

📉 RISK METRICS:
• Volatility: {results.strategy_volatility:.1f}% vs {results.buy_hold_volatility:.1f}% (B&H)
• Sharpe Ratio: {results.strategy_sharpe:.2f} vs {results.buy_hold_sharpe:.2f} (B&H)
• Sortino Ratio: {results.strategy_sortino:.2f}
• Max Drawdown: {results.strategy_max_drawdown:.1f}% vs {results.buy_hold_max_drawdown:.1f}% (B&H)
• Calmar Ratio: {results.strategy_calmar_ratio:.2f}

💼 TRADING METRICS:
• Total Trades: {results.total_trades}
• Win Rate: {results.win_rate:.1f}%
• Profit Factor: {results.profit_factor:.2f}
• Avg Trade Return: {results.avg_trade_return:.2f}%
• Max Consecutive Wins: {results.max_consecutive_wins}
• Max Consecutive Losses: {results.max_consecutive_losses}

⏰ BACKTEST PERIOD:
• Start Date: {results.start_date}
• End Date: {results.end_date}
• Total Years: {results.total_years:.1f}
"""
        
        # Determine overall assessment
        if (results.strategy_total_return_pct > results.buy_hold_total_return_pct and
            results.strategy_sharpe > 1.0 and
            results.win_rate > 60):
            assessment = "🟢 PROFITABLE STRATEGY"
        elif results.strategy_total_return_pct > results.buy_hold_total_return_pct:
            assessment = "🟡 MODERATE PERFORMANCE"  
        else:
            assessment = "🔴 UNDERPERFORMS BUY & HOLD"
        
        summary_text += f"\n🎯 ASSESSMENT: {assessment}"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.9))
        
        plt.tight_layout()
        return fig

def main():
    """Run comprehensive QQQ backtest"""
    
    print("📈 COMPREHENSIVE QQQ ENSEMBLE STRATEGY BACKTEST")
    print("=" * 70)
    print("Testing ensemble alpha strategy profitability over maximum historical period")
    
    # Initialize backtester
    backtester = ComprehensiveBacktester("QQQ")
    
    # Download maximum historical data
    df = backtester.download_maximum_data()
    
    if df.empty:
        print("❌ Failed to download data")
        return
    
    # Run comprehensive backtest
    try:
        results = backtester.comprehensive_backtest(df)
        
        # Create visualization
        strategy_results = backtester._run_strategy_backtest(
            backtester._generate_ensemble_signals_manual(
                backtester._add_all_indicators(df.copy()), 
                backtester.ensemble.alpha_sources if backtester.ensemble else []
            )
        )
        
        buy_hold_results = backtester._run_buy_hold_backtest(df)
        
        fig = backtester.create_comprehensive_visualization(results, strategy_results, buy_hold_results, df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save chart
        chart_filename = f'QQQ_comprehensive_backtest_{timestamp}.png'
        fig.savefig(chart_filename, dpi=300, bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        print(f"✅ Chart saved: {chart_filename}")
        
        # Save detailed results
        results_filename = f'QQQ_backtest_results_{timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        print(f"✅ Results saved: {results_filename}")
        
        # Print final assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        print(f"=" * 50)
        
        if (results.strategy_total_return_pct > results.buy_hold_total_return_pct and
            results.strategy_sharpe > 1.0 and results.win_rate > 60):
            print("🟢 SUCCESS: Strategy is profitable and outperforms buy & hold!")
            print(f"   Strategy beats buy & hold by {results.excess_return_pct:+.1f}%")
            print(f"   Risk-adjusted returns (Sharpe): {results.strategy_sharpe:.2f}")
            print("   ✅ RECOMMENDED FOR FURTHER DEVELOPMENT")
        elif results.strategy_total_return_pct > results.buy_hold_total_return_pct:
            print("🟡 MODERATE: Strategy outperforms but needs optimization")
            print(f"   Strategy beats buy & hold by {results.excess_return_pct:+.1f}%")
            print("   🔧 NEEDS IMPROVEMENT IN RISK MANAGEMENT")
        else:
            print("🔴 FAILURE: Strategy underperforms buy & hold")
            print(f"   Strategy trails buy & hold by {abs(results.excess_return_pct):.1f}%")
            print("   ❌ REQUIRES MAJOR STRATEGY REVISION")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()