#!/usr/bin/env python
"""
📊 POSITION SIZING CALCULATOR
Smart position sizing recommendations based on risk management and portfolio theory
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

from src.data.position_tracker import PositionTracker


class PositionSizingCalculator:
    """Advanced position sizing calculator with multiple strategies"""
    
    def __init__(self, account_balance: float = 10000):
        self.account_balance = account_balance
        self.tracker = PositionTracker()
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
        # Risk management parameters
        self.max_position_risk = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.06  # 6% max total portfolio risk
        self.max_single_position_pct = 0.20  # 20% max of portfolio in single position
        self.correlation_threshold = 0.7  # Correlation limit for diversification
        
    def get_current_portfolio_value(self) -> Tuple[float, Dict]:
        """Get current portfolio value and allocation"""
        positions = self.tracker.get_open_positions()
        
        if positions.empty:
            return self.account_balance, {}
            
        total_value = self.account_balance
        allocations = {}
        
        try:
            for _, position in positions.iterrows():
                symbol = position['symbol']
                shares = position['shares']
                
                # Get current price
                ticker = yf.download(symbol, period='1d', progress=False)
                if not ticker.empty:
                    # Fix multi-level column issue if present
                    if isinstance(ticker.columns, pd.MultiIndex):
                        ticker.columns = ticker.columns.get_level_values(0)
                    current_price = ticker['Close'].iloc[-1]
                    position_value = shares * current_price
                    total_value += position_value
                    allocations[symbol] = {
                        'value': position_value,
                        'shares': shares,
                        'current_price': current_price,
                        'pct': position_value / total_value  # Will be recalculated
                    }
        except Exception as e:
            print(f"Warning: Error fetching portfolio data: {e}")
            
        # Recalculate percentages
        for symbol in allocations:
            allocations[symbol]['pct'] = allocations[symbol]['value'] / total_value
            
        return total_value, allocations
        
    def calculate_volatility(self, symbol: str, days: int = 252) -> float:
        """Calculate annualized volatility for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50)  # Extra buffer
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 30:
                return 0.25  # Default 25% volatility if no data
            
            # Fix multi-level column issue if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            return min(volatility, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.25  # Default volatility
            
    def calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix between symbols"""
        if len(symbols) <= 1:
            return pd.DataFrame()
            
        try:
            # Get data for all symbols
            data = {}
            for symbol in symbols:
                ticker_data = yf.download(symbol, period='252d', progress=False)
                if not ticker_data.empty:
                    # Fix multi-level column issue if present
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    data[symbol] = ticker_data['Close'].pct_change().dropna()
                    
            if len(data) <= 1:
                return pd.DataFrame()
                
            # Create correlation matrix
            df = pd.DataFrame(data)
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception:
            return pd.DataFrame()
            
    def kelly_criterion_sizing(self, symbol: str, win_rate: float = None, 
                              avg_win: float = None, avg_loss: float = None) -> float:
        """Calculate position size using Kelly Criterion"""
        
        # Get historical performance if not provided
        if any(x is None for x in [win_rate, avg_win, avg_loss]):
            history = self.tracker.get_symbol_history(symbol)
            closed_trades = history[history['status'] == 'CLOSED']
            
            if len(closed_trades) > 0:
                wins = closed_trades[closed_trades['profit_loss'] > 0]
                losses = closed_trades[closed_trades['profit_loss'] <= 0]
                
                win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0.5
                avg_win = wins['profit_loss_pct'].mean() / 100 if len(wins) > 0 else 0.05
                avg_loss = abs(losses['profit_loss_pct'].mean() / 100) if len(losses) > 0 else 0.03
            else:
                # Default values for new symbols
                win_rate = 0.55  # 55% win rate
                avg_win = 0.08   # 8% average win
                avg_loss = 0.04  # 4% average loss
                
        # Kelly formula: f = (bp - q) / b
        # f = fraction of capital to wager
        # b = odds (avg_win / avg_loss)
        # p = probability of win
        # q = probability of loss (1 - p)
        
        if avg_loss == 0:
            return 0
            
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_pct = (b * p - q) / b
        
        # Limit Kelly to reasonable bounds
        kelly_pct = max(0, min(kelly_pct, 0.25))  # 0% to 25%
        
        return kelly_pct
        
    def volatility_based_sizing(self, symbol: str, target_risk: float = 0.02) -> float:
        """Calculate position size based on volatility targeting"""
        volatility = self.calculate_volatility(symbol)
        
        if volatility == 0:
            return 0
            
        # Position size = Target Risk / Volatility
        position_pct = target_risk / volatility
        
        # Limit to max single position
        position_pct = min(position_pct, self.max_single_position_pct)
        
        return position_pct
        
    def equal_risk_contribution(self, new_symbol: str, 
                               current_symbols: List[str]) -> float:
        """Calculate position size for equal risk contribution"""
        
        if not current_symbols:
            return self.max_single_position_pct
            
        # Calculate risk contribution for each position
        new_vol = self.calculate_volatility(new_symbol)
        
        if new_vol == 0:
            return 0
            
        # Get correlations
        all_symbols = current_symbols + [new_symbol]
        corr_matrix = self.calculate_correlation_matrix(all_symbols)
        
        if corr_matrix.empty:
            # Fallback to volatility-based sizing
            return self.volatility_based_sizing(new_symbol)
            
        # Calculate target equal risk weight
        target_risk_weight = 1.0 / len(all_symbols)
        
        # Simple approximation: inverse volatility weighting
        volatilities = {symbol: self.calculate_volatility(symbol) for symbol in all_symbols}
        
        inv_vol_sum = sum(1/vol for vol in volatilities.values() if vol > 0)
        
        if inv_vol_sum == 0:
            return 0
            
        weight = (1 / new_vol) / inv_vol_sum if new_vol > 0 else 0
        
        return min(weight, self.max_single_position_pct)
        
    def max_diversification_sizing(self, new_symbol: str, 
                                  current_symbols: List[str]) -> float:
        """Calculate position size for maximum diversification"""
        
        if not current_symbols:
            return self.max_single_position_pct
            
        # Check correlations with existing positions
        correlations = []
        
        for existing_symbol in current_symbols:
            corr_matrix = self.calculate_correlation_matrix([new_symbol, existing_symbol])
            if not corr_matrix.empty and len(corr_matrix) > 1:
                correlation = corr_matrix.loc[new_symbol, existing_symbol]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
                    
        if not correlations:
            return self.volatility_based_sizing(new_symbol)
            
        avg_correlation = np.mean(correlations)
        max_correlation = max(correlations)
        
        # Reduce position size for highly correlated assets
        if max_correlation > self.correlation_threshold:
            correlation_penalty = max_correlation
        else:
            correlation_penalty = avg_correlation * 0.5
            
        base_size = self.volatility_based_sizing(new_symbol)
        adjusted_size = base_size * (1 - correlation_penalty)
        
        return max(adjusted_size, 0.01)  # Minimum 1%
        
    def get_comprehensive_sizing_recommendation(self, symbol: str, 
                                              entry_price: float = None,
                                              stop_loss_price: float = None) -> Dict:
        """Get comprehensive position sizing recommendation"""
        
        portfolio_value, current_allocations = self.get_current_portfolio_value()
        current_symbols = list(current_allocations.keys())
        
        # Get current price if not provided
        if entry_price is None:
            try:
                ticker = yf.download(symbol, period='1d', progress=False)
                if not ticker.empty:
                    # Fix multi-level column issue if present
                    if isinstance(ticker.columns, pd.MultiIndex):
                        ticker.columns = ticker.columns.get_level_values(0)
                    entry_price = ticker['Close'].iloc[-1]
                else:
                    return {'error': f'Cannot fetch price for {symbol}'}
            except:
                return {'error': f'Cannot fetch price for {symbol}'}
                
        # Calculate different sizing methods
        kelly_size = self.kelly_criterion_sizing(symbol)
        volatility_size = self.volatility_based_sizing(symbol)
        equal_risk_size = self.equal_risk_contribution(symbol, current_symbols)
        max_div_size = self.max_diversification_sizing(symbol, current_symbols)
        
        # Risk-based sizing if stop loss provided
        risk_based_size = None
        if stop_loss_price and entry_price:
            risk_per_share = abs(entry_price - stop_loss_price)
            risk_pct = risk_per_share / entry_price
            
            if risk_pct > 0:
                target_dollar_risk = portfolio_value * self.max_position_risk
                position_value = target_dollar_risk / risk_pct
                risk_based_size = position_value / portfolio_value
                risk_based_size = min(risk_based_size, self.max_single_position_pct)
                
        # Ensemble recommendation (weighted average)
        weights = {
            'kelly': 0.3,
            'volatility': 0.25,
            'equal_risk': 0.25,
            'max_diversification': 0.2
        }
        
        ensemble_size = (
            kelly_size * weights['kelly'] +
            volatility_size * weights['volatility'] +
            equal_risk_size * weights['equal_risk'] +
            max_div_size * weights['max_diversification']
        )
        
        # Portfolio constraint check
        total_allocation = sum(pos['pct'] for pos in current_allocations.values())
        available_allocation = 1.0 - total_allocation
        
        final_size = min(ensemble_size, available_allocation, self.max_single_position_pct)
        
        # Calculate position details
        position_value = portfolio_value * final_size
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        actual_position_value = shares * entry_price
        actual_position_pct = actual_position_value / portfolio_value
        
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'portfolio_value': portfolio_value,
            'sizing_methods': {
                'kelly_criterion': kelly_size,
                'volatility_based': volatility_size,
                'equal_risk_contribution': equal_risk_size,
                'max_diversification': max_div_size,
                'risk_based': risk_based_size
            },
            'recommendation': {
                'ensemble_pct': ensemble_size,
                'final_pct': final_size,
                'position_value': position_value,
                'recommended_shares': shares,
                'actual_position_value': actual_position_value,
                'actual_position_pct': actual_position_pct
            },
            'risk_metrics': {
                'symbol_volatility': self.calculate_volatility(symbol),
                'max_correlation': self._get_max_correlation(symbol, current_symbols),
                'portfolio_allocation': total_allocation,
                'available_allocation': available_allocation
            },
            'constraints': {
                'max_single_position': self.max_single_position_pct,
                'max_position_risk': self.max_position_risk,
                'max_portfolio_risk': self.max_portfolio_risk
            }
        }
        
    def _get_max_correlation(self, symbol: str, current_symbols: List[str]) -> float:
        """Get maximum correlation with existing positions"""
        if not current_symbols:
            return 0.0
            
        correlations = []
        for existing_symbol in current_symbols:
            corr_matrix = self.calculate_correlation_matrix([symbol, existing_symbol])
            if not corr_matrix.empty and len(corr_matrix) > 1:
                correlation = corr_matrix.loc[symbol, existing_symbol]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
                    
        return max(correlations) if correlations else 0.0
        
    def display_sizing_recommendation(self, recommendation: Dict):
        """Display position sizing recommendation in a nice format"""
        
        if 'error' in recommendation:
            print(colored(f"❌ {recommendation['error']}", 'red'))
            return
            
        symbol = recommendation['symbol']
        entry_price = recommendation['entry_price']
        
        print(colored(f"\n📊 POSITION SIZING RECOMMENDATION: {symbol}", 'cyan', attrs=['bold']))
        print("=" * 70)
        
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Portfolio Value: ${recommendation['portfolio_value']:,.2f}")
        
        # Sizing methods
        print(colored("\n🎯 SIZING METHODS:", 'yellow'))
        methods = recommendation['sizing_methods']
        for method, size in methods.items():
            if size is not None:
                print(f"  {method.replace('_', ' ').title()}: {size:.1%}")
                
        # Final recommendation
        rec = recommendation['recommendation']
        print(colored(f"\n💡 FINAL RECOMMENDATION:", 'green', attrs=['bold']))
        print(f"  Position Size: {rec['final_pct']:.1%} of portfolio")
        print(f"  Dollar Amount: ${rec['position_value']:,.2f}")
        print(f"  Recommended Shares: {rec['recommended_shares']:,}")
        print(f"  Actual Investment: ${rec['actual_position_value']:,.2f} ({rec['actual_position_pct']:.1%})")
        
        # Risk metrics
        risk = recommendation['risk_metrics']
        print(colored(f"\n⚠️ RISK METRICS:", 'yellow'))
        print(f"  Symbol Volatility: {risk['symbol_volatility']:.1%}")
        print(f"  Max Correlation: {risk['max_correlation']:.1%}")
        print(f"  Current Allocation: {risk['portfolio_allocation']:.1%}")
        print(f"  Available Allocation: {risk['available_allocation']:.1%}")
        
        # Warnings
        constraints = recommendation['constraints']
        print(colored(f"\n⚡ RISK WARNINGS:", 'red'))
        
        if rec['actual_position_pct'] > constraints['max_single_position']:
            print(f"  ⚠️ Position exceeds max single position limit ({constraints['max_single_position']:.1%})")
            
        if risk['max_correlation'] > self.correlation_threshold:
            print(f"  ⚠️ High correlation ({risk['max_correlation']:.1%}) with existing positions")
            
        if risk['portfolio_allocation'] > 0.8:
            print(f"  ⚠️ Portfolio highly allocated ({risk['portfolio_allocation']:.1%})")
            
        if risk['symbol_volatility'] > 0.5:
            print(f"  ⚠️ High volatility symbol ({risk['symbol_volatility']:.1%})")
            
        print("=" * 70)


def demo_position_sizing():
    """Demo position sizing calculator"""
    
    print(colored("📊 POSITION SIZING CALCULATOR DEMO", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    # Initialize calculator
    calculator = PositionSizingCalculator(account_balance=10000)
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'SPY']
    
    for symbol in test_symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        
        try:
            recommendation = calculator.get_comprehensive_sizing_recommendation(symbol)
            calculator.display_sizing_recommendation(recommendation)
        except Exception as e:
            print(colored(f"❌ Error analyzing {symbol}: {str(e)}", 'red'))
            
        print("\n" + "-" * 60)


def analyze_open_positions():
    """Analyze sizing recommendations for open positions"""
    
    print(colored("📊 POSITION SIZING ANALYSIS - OPEN POSITIONS", 'cyan', attrs=['bold']))
    print("=" * 70)
    
    try:
        # Get open positions
        tracker = PositionTracker()
        positions = tracker.get_open_positions()
        
        if positions.empty:
            print(colored("❌ No open positions to analyze", 'yellow'))
            return
            
        calculator = PositionSizingCalculator(account_balance=50000)  # Realistic balance
        
        print(f"Analyzing {len(positions)} open positions...\n")
        
        for _, position in positions.iterrows():
            symbol = position['symbol']
            current_shares = position['shares']
            entry_price = position['entry_price']
            
            print(f"{'='*70}")
            print(colored(f"📊 CURRENT POSITION: {symbol}", 'cyan', attrs=['bold']))
            print(f"Current Shares: {current_shares}")
            print(f"Entry Price: ${entry_price:.2f}")
            
            try:
                # Get sizing recommendation for this symbol
                recommendation = calculator.get_comprehensive_sizing_recommendation(symbol)
                
                if 'error' not in recommendation:
                    current_value = current_shares * entry_price
                    recommended_shares = recommendation['recommendation']['recommended_shares']
                    recommended_value = recommendation['recommendation']['actual_position_value']
                    
                    # Compare current vs recommended
                    size_difference = recommended_shares - current_shares
                    value_difference = recommended_value - current_value
                    
                    print(f"\n💡 SIZING ANALYSIS:")
                    print(f"  Current Position: ${current_value:,.2f} ({current_shares} shares)")
                    print(f"  Recommended Size: ${recommended_value:,.2f} ({recommended_shares} shares)")
                    
                    if abs(size_difference) > 10:  # Significant difference
                        action_color = 'green' if size_difference > 0 else 'red'
                        action = 'INCREASE' if size_difference > 0 else 'DECREASE'
                        print(colored(f"  Sizing Advice: {action} by {abs(size_difference)} shares (${abs(value_difference):,.2f})", action_color, attrs=['bold']))
                        
                        # Show reasoning
                        sizing_methods = recommendation['sizing_methods']
                        print(f"\n📋 SIZING BREAKDOWN:")
                        for method, pct in sizing_methods.items():
                            if pct is not None:
                                method_value = calculator.account_balance * pct
                                print(f"  {method.replace('_', ' ').title()}: {pct:.1%} (${method_value:,.0f})")
                        
                        # Risk metrics
                        risk_metrics = recommendation['risk_metrics']
                        print(f"\n⚠️ RISK PROFILE:")
                        print(f"  Volatility: {risk_metrics['symbol_volatility']:.1%}")
                        print(f"  Max Correlation: {risk_metrics['max_correlation']:.1%}")
                        print(f"  Portfolio Impact: {recommendation['recommendation']['actual_position_pct']:.1%}")
                    else:
                        print(colored("  ✅ Position size is appropriate", 'green'))
                        
                else:
                    print(colored(f"❌ {recommendation['error']}", 'red'))
                    
            except Exception as e:
                print(colored(f"❌ Error analyzing {symbol}: {str(e)}", 'red'))
                
            print()
            
        # Summary recommendations
        print(f"{'='*70}")
        print(colored("💼 PORTFOLIO SIZING SUMMARY", 'cyan', attrs=['bold']))
        
        portfolio_value, allocations = calculator.get_current_portfolio_value()
        print(f"Total Portfolio Value: ${portfolio_value:,.2f}")
        
        if allocations:
            print(f"\nCurrent Allocations:")
            for symbol, allocation in allocations.items():
                pct = allocation['pct']
                value = allocation['value']
                print(f"  {symbol}: {pct:.1%} (${value:,.2f})")
                
        print(f"\nCash Available: ${calculator.account_balance:,.2f}")
        
    except Exception as e:
        print(colored(f"❌ Error: {str(e)}", 'red'))


def interactive_position_sizing():
    """Interactive position sizing tool"""
    
    print(colored("💰 INTERACTIVE POSITION SIZING TOOL", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    try:
        balance = float(input("Enter your account balance: $") or "10000")
        calculator = PositionSizingCalculator(account_balance=balance)
        
        while True:
            symbol = input("\nEnter symbol to analyze (or 'quit'): ").upper()
            
            if symbol in ['QUIT', 'Q', 'EXIT']:
                break
                
            if not symbol:
                continue
                
            # Optional stop loss
            stop_loss_input = input(f"Enter stop loss price for {symbol} (optional): ")
            stop_loss = float(stop_loss_input) if stop_loss_input else None
            
            print(f"\n🔍 Analyzing {symbol}...")
            
            try:
                recommendation = calculator.get_comprehensive_sizing_recommendation(
                    symbol, stop_loss_price=stop_loss
                )
                calculator.display_sizing_recommendation(recommendation)
            except Exception as e:
                print(colored(f"❌ Error: {str(e)}", 'red'))
                
    except KeyboardInterrupt:
        print(colored("\n👋 Goodbye!", 'yellow'))
    except Exception as e:
        print(colored(f"❌ Error: {str(e)}", 'red'))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_position_sizing()
        elif sys.argv[1] == '--positions':
            analyze_open_positions()
        elif sys.argv[1] == '--demo':
            demo_position_sizing()
        else:
            print("Usage: python position_sizing_calculator.py [--interactive|--positions|--demo]")
    else:
        # Default: analyze open positions if any, otherwise demo
        try:
            tracker = PositionTracker()
            positions = tracker.get_open_positions()
            if not positions.empty:
                analyze_open_positions()
            else:
                demo_position_sizing()
        except:
            demo_position_sizing()