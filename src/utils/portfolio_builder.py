"""
🌙 Interactive Portfolio Builder for Moon Dev Trading System
Helps users create and configure portfolios with proper validation
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from termcolor import cprint

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.portfolio_models import (
    PortfolioConfig, RebalancingMethod, 
    DEFAULT_RISK_LIMITS, DEFAULT_REBALANCING_PARAMS
)
from src.agents.strategy_registry_agent import StrategyRegistryAgent


class PortfolioBuilder:
    """Interactive portfolio configuration builder"""
    
    def __init__(self):
        self.registry = StrategyRegistryAgent()
        self.portfolio_dir = Path(project_root) / "src" / "data" / "portfolios"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
    
    def create_portfolio(self) -> PortfolioConfig:
        """Interactive CLI for portfolio creation"""
        cprint("\n🌙 Moon Dev Portfolio Builder", "cyan", attrs=["bold"])
        cprint("=" * 50, "blue")
        
        # Step 1: Portfolio name
        name = self._get_portfolio_name()
        
        # Step 2: Select strategies
        strategies = self._select_strategies()
        
        # Step 3: Set allocations
        allocations = self._set_allocations(strategies)
        
        # Step 4: Choose rebalancing method
        method, params = self._choose_rebalancing_method()
        
        # Step 5: Set risk limits
        risk_limits = self._configure_risk_limits()
        
        # Create portfolio config
        config = PortfolioConfig(
            name=name,
            target_allocations=allocations,
            rebalancing_method=method,
            rebalancing_params=params,
            risk_limits=risk_limits
        )
        
        # Validate
        try:
            config.validate()
            cprint("\n✅ Portfolio configuration valid!", "green")
        except ValueError as e:
            cprint(f"\n❌ Invalid configuration: {e}", "red")
            return None
        
        # Save portfolio
        self._save_portfolio(config)
        
        # Display summary
        self._display_summary(config)
        
        return config
    
    def _get_portfolio_name(self) -> str:
        """Get portfolio name from user"""
        cprint("\n📝 Portfolio Name", "yellow")
        name = input("Enter portfolio name: ").strip()
        
        while not name:
            cprint("Portfolio name cannot be empty!", "red")
            name = input("Enter portfolio name: ").strip()
        
        return name
    
    def _select_strategies(self) -> List[str]:
        """Select strategies for portfolio"""
        cprint("\n📊 Strategy Selection", "yellow")
        
        # Get available strategies
        available = self.registry.list_strategies(status="active")
        
        if not available:
            cprint("No active strategies available. Using demo strategies.", "yellow")
            return ["rsi_mean_reversion", "macd_momentum", "bollinger_breakout"]
        
        cprint("\nAvailable strategies:", "white")
        for i, strategy in enumerate(available, 1):
            perf = strategy.get("performance_data", {})
            cprint(
                f"  {i}. {strategy['name']} - "
                f"Return: {perf.get('total_return', 0):.1%}, "
                f"Sharpe: {perf.get('sharpe_ratio', 0):.2f}",
                "white"
            )
        
        # Select strategies
        selected = []
        while True:
            choice = input("\nSelect strategy number (or 'done' to finish): ").strip()
            
            if choice.lower() == 'done':
                if len(selected) >= 2:
                    break
                else:
                    cprint("Please select at least 2 strategies", "red")
                    continue
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    strategy_id = available[idx]["strategy_id"]
                    if strategy_id not in selected:
                        selected.append(strategy_id)
                        cprint(f"✅ Added {available[idx]['name']}", "green")
                    else:
                        cprint("Strategy already selected", "yellow")
                else:
                    cprint("Invalid selection", "red")
            except ValueError:
                cprint("Please enter a number or 'done'", "red")
        
        return selected
    
    def _set_allocations(self, strategies: List[str]) -> Dict[str, float]:
        """Set allocation weights for strategies"""
        cprint("\n⚖️ Allocation Weights", "yellow")
        cprint("Weights must sum to 100%", "white")
        
        allocations = {}
        remaining = 100.0
        
        for i, strategy in enumerate(strategies):
            if i == len(strategies) - 1:
                # Last strategy gets remaining weight
                weight = remaining
            else:
                while True:
                    try:
                        weight = float(input(f"\nWeight for {strategy} (remaining: {remaining:.1f}%): "))
                        if 0 <= weight <= remaining:
                            break
                        else:
                            cprint(f"Weight must be between 0 and {remaining:.1f}", "red")
                    except ValueError:
                        cprint("Please enter a valid number", "red")
            
            allocations[strategy] = weight / 100.0  # Convert to decimal
            remaining -= weight
            cprint(f"✅ {strategy}: {weight:.1f}%", "green")
        
        return allocations
    
    def _choose_rebalancing_method(self) -> tuple:
        """Choose rebalancing method and parameters"""
        cprint("\n🔄 Rebalancing Method", "yellow")
        
        methods = {
            "1": RebalancingMethod.THRESHOLD,
            "2": RebalancingMethod.CALENDAR,
            "3": RebalancingMethod.ADAPTIVE,
            "4": RebalancingMethod.RISK_PARITY
        }
        
        cprint("\n1. Threshold - Rebalance when drift exceeds limit", "white")
        cprint("2. Calendar - Rebalance on fixed schedule", "white")
        cprint("3. Adaptive - Adjust based on performance", "white")
        cprint("4. Risk Parity - Equal risk contribution", "white")
        
        while True:
            choice = input("\nSelect method (1-4): ").strip()
            if choice in methods:
                method = methods[choice]
                break
            else:
                cprint("Invalid choice", "red")
        
        # Get method-specific parameters
        if method == RebalancingMethod.THRESHOLD:
            params = self._get_threshold_params()
        elif method == RebalancingMethod.CALENDAR:
            params = self._get_calendar_params()
        elif method == RebalancingMethod.ADAPTIVE:
            params = DEFAULT_REBALANCING_PARAMS["adaptive"].copy()
        else:  # RISK_PARITY
            params = DEFAULT_REBALANCING_PARAMS["risk_parity"].copy()
        
        return method, params
    
    def _get_threshold_params(self) -> Dict:
        """Get threshold rebalancing parameters"""
        params = DEFAULT_REBALANCING_PARAMS["threshold"].copy()
        
        cprint("\nThreshold Parameters:", "white")
        
        # Get drift threshold
        while True:
            try:
                drift = float(input(f"Drift threshold (default {params['min_drift']*100}%): ") or params['min_drift']*100)
                params['min_drift'] = drift / 100.0
                break
            except ValueError:
                cprint("Please enter a valid number", "red")
        
        # Get minimum trade size
        while True:
            try:
                min_trade = float(input(f"Minimum trade size (default ${params['min_trade_size']}): ") or params['min_trade_size'])
                params['min_trade_size'] = min_trade
                break
            except ValueError:
                cprint("Please enter a valid number", "red")
        
        return params
    
    def _get_calendar_params(self) -> Dict:
        """Get calendar rebalancing parameters"""
        params = DEFAULT_REBALANCING_PARAMS["calendar"].copy()
        
        cprint("\nCalendar Parameters:", "white")
        
        # Get frequency
        freqs = {"1": "daily", "2": "weekly", "3": "monthly", "4": "quarterly"}
        cprint("\n1. Daily", "white")
        cprint("2. Weekly", "white")
        cprint("3. Monthly", "white")
        cprint("4. Quarterly", "white")
        
        while True:
            choice = input("\nSelect frequency (1-4): ").strip()
            if choice in freqs:
                params['frequency'] = freqs[choice]
                break
            else:
                cprint("Invalid choice", "red")
        
        return params
    
    def _configure_risk_limits(self) -> Dict[str, float]:
        """Configure risk limits"""
        cprint("\n🛡️ Risk Limits", "yellow")
        
        use_defaults = input("\nUse default risk limits? (y/n): ").strip().lower() == 'y'
        
        if use_defaults:
            return DEFAULT_RISK_LIMITS.copy()
        
        limits = DEFAULT_RISK_LIMITS.copy()
        
        # Customize key limits
        cprint("\nCustomize risk limits:", "white")
        
        # Max single strategy
        while True:
            try:
                max_single = float(
                    input(f"Max single strategy allocation (default {limits['max_single_strategy']*100}%): ") 
                    or limits['max_single_strategy']*100
                )
                limits['max_single_strategy'] = max_single / 100.0
                break
            except ValueError:
                cprint("Please enter a valid number", "red")
        
        # Max drawdown
        while True:
            try:
                max_dd = float(
                    input(f"Max portfolio drawdown (default {abs(limits['max_portfolio_drawdown'])*100}%): ") 
                    or abs(limits['max_portfolio_drawdown'])*100
                )
                limits['max_portfolio_drawdown'] = -max_dd / 100.0
                break
            except ValueError:
                cprint("Please enter a valid number", "red")
        
        return limits
    
    def _save_portfolio(self, config: PortfolioConfig):
        """Save portfolio configuration"""
        filename = f"{config.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.portfolio_dir / filename
        
        data = {
            "name": config.name,
            "target_allocations": config.target_allocations,
            "rebalancing_method": config.rebalancing_method.value,
            "rebalancing_params": config.rebalancing_params,
            "risk_limits": config.risk_limits,
            "created_at": config.created_at.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        cprint(f"\n💾 Portfolio saved to: {filepath}", "green")
    
    def _display_summary(self, config: PortfolioConfig):
        """Display portfolio summary"""
        cprint("\n📊 Portfolio Summary", "cyan", attrs=["bold"])
        cprint("=" * 50, "blue")
        
        cprint(f"\nName: {config.name}", "white")
        cprint(f"Method: {config.rebalancing_method.value}", "white")
        
        cprint("\nAllocations:", "yellow")
        for strategy, weight in config.target_allocations.items():
            cprint(f"  • {strategy}: {weight:.1%}", "white")
        
        cprint("\nRebalancing Parameters:", "yellow")
        for key, value in config.rebalancing_params.items():
            cprint(f"  • {key}: {value}", "white")
        
        cprint("\nKey Risk Limits:", "yellow")
        cprint(f"  • Max single strategy: {config.risk_limits['max_single_strategy']:.1%}", "white")
        cprint(f"  • Max drawdown: {config.risk_limits['max_portfolio_drawdown']:.1%}", "white")
        cprint(f"  • Min Sharpe: {config.risk_limits['min_portfolio_sharpe']}", "white")
    
    def load_portfolio(self, name: str) -> Optional[PortfolioConfig]:
        """Load a saved portfolio"""
        # Find matching portfolio file
        for file in self.portfolio_dir.glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                if data['name'] == name:
                    return PortfolioConfig(
                        name=data['name'],
                        target_allocations=data['target_allocations'],
                        rebalancing_method=RebalancingMethod(data['rebalancing_method']),
                        rebalancing_params=data['rebalancing_params'],
                        risk_limits=data['risk_limits']
                    )
        
        return None
    
    def list_portfolios(self) -> List[Dict]:
        """List all saved portfolios"""
        portfolios = []
        
        for file in self.portfolio_dir.glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                portfolios.append({
                    "name": data['name'],
                    "method": data['rebalancing_method'],
                    "strategies": len(data['target_allocations']),
                    "created": data['created_at'],
                    "file": file.name
                })
        
        return portfolios


def main():
    """Demo portfolio builder"""
    builder = PortfolioBuilder()
    
    cprint("\n🌙 Moon Dev Portfolio Management", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    cprint("\n1. Create new portfolio", "white")
    cprint("2. List existing portfolios", "white")
    cprint("3. Load portfolio", "white")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        portfolio = builder.create_portfolio()
        if portfolio:
            cprint("\n✅ Portfolio created successfully!", "green")
    
    elif choice == "2":
        portfolios = builder.list_portfolios()
        if portfolios:
            cprint("\n📁 Existing Portfolios:", "yellow")
            for p in portfolios:
                cprint(
                    f"  • {p['name']} - {p['strategies']} strategies, "
                    f"{p['method']} rebalancing",
                    "white"
                )
        else:
            cprint("\nNo portfolios found", "yellow")
    
    elif choice == "3":
        name = input("\nEnter portfolio name to load: ").strip()
        portfolio = builder.load_portfolio(name)
        if portfolio:
            builder._display_summary(portfolio)
        else:
            cprint(f"\nPortfolio '{name}' not found", "red")


if __name__ == "__main__":
    main()