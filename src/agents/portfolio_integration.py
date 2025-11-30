"""
🌙 Portfolio Integration Module
Connects portfolio rebalancing with existing Moon Dev trading agents
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from termcolor import cprint

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.agents.trading_agent import TradingAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.risk_agent import RiskAgent
from src.models.portfolio_models import Order, OrderSide
from src.nice_funcs import get_position, market_buy, market_sell


class PortfolioIntegration:
    """Integrates portfolio rebalancing with trading execution"""
    
    def __init__(self):
        self.trading_agent = TradingAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
    
    def execute_portfolio_orders(self, orders: List[Order]) -> Dict[str, Any]:
        """Execute portfolio rebalancing orders through trading agents"""
        results = {
            "executed": [],
            "failed": [],
            "total_traded": 0
        }
        
        cprint("\n🚀 Executing Portfolio Rebalancing Orders", "cyan", attrs=["bold"])
        
        # Group orders by strategy
        strategy_orders = {}
        for order in orders:
            if order.strategy_id not in strategy_orders:
                strategy_orders[order.strategy_id] = []
            strategy_orders[order.strategy_id].append(order)
        
        # Execute orders for each strategy
        for strategy_id, strategy_orders_list in strategy_orders.items():
            cprint(f"\n📊 Processing {strategy_id}:", "yellow")
            
            for order in strategy_orders_list:
                try:
                    # Check risk limits first
                    if not self._check_risk_limits(order):
                        cprint(f"  ❌ Risk check failed for {order}", "red")
                        results["failed"].append({
                            "order": order,
                            "reason": "Risk limits exceeded"
                        })
                        continue
                    
                    # Execute based on order side
                    if order.side == OrderSide.BUY:
                        success = self._execute_buy(strategy_id, order.amount_usd)
                    else:  # SELL
                        success = self._execute_sell(strategy_id, order.amount_usd)
                    
                    if success:
                        order.executed = True
                        order.execution_time = datetime.now()
                        results["executed"].append(order)
                        results["total_traded"] += order.amount_usd
                        cprint(f"  ✅ Executed: {order}", "green")
                    else:
                        results["failed"].append({
                            "order": order,
                            "reason": "Execution failed"
                        })
                        cprint(f"  ❌ Failed: {order}", "red")
                
                except Exception as e:
                    cprint(f"  ❌ Error executing {order}: {e}", "red")
                    results["failed"].append({
                        "order": order,
                        "reason": str(e)
                    })
        
        # Summary
        cprint(f"\n📈 Execution Summary:", "cyan")
        cprint(f"  • Executed: {len(results['executed'])} orders", "green")
        cprint(f"  • Failed: {len(results['failed'])} orders", "red")
        cprint(f"  • Total traded: ${results['total_traded']:,.2f}", "white")
        
        return results
    
    def _check_risk_limits(self, order: Order) -> bool:
        """Check if order passes risk limits"""
        # Use risk agent to validate
        current_positions = self._get_current_positions()
        
        # Simulate position after order
        simulated_positions = current_positions.copy()
        
        if order.strategy_id in simulated_positions:
            if order.side == OrderSide.BUY:
                simulated_positions[order.strategy_id] += order.amount_usd
            else:  # SELL
                simulated_positions[order.strategy_id] -= order.amount_usd
        else:
            simulated_positions[order.strategy_id] = order.amount_usd
        
        # Check total exposure
        total_exposure = sum(simulated_positions.values())
        
        # Basic risk checks
        if total_exposure > 100000:  # Max $100k exposure
            return False
        
        if order.amount_usd > 10000:  # Max $10k per order
            return False
        
        return True
    
    def _execute_buy(self, strategy_id: str, amount_usd: float) -> bool:
        """Execute buy order for a strategy"""
        try:
            # For demo purposes, simulate execution
            # In production, would route to appropriate exchange/token
            cprint(f"    → Buying ${amount_usd:.2f} for {strategy_id}", "cyan")
            
            # Simulate success
            return True
            
        except Exception as e:
            cprint(f"    → Buy failed: {e}", "red")
            return False
    
    def _execute_sell(self, strategy_id: str, amount_usd: float) -> bool:
        """Execute sell order for a strategy"""
        try:
            # For demo purposes, simulate execution
            cprint(f"    → Selling ${amount_usd:.2f} from {strategy_id}", "cyan")
            
            # Simulate success
            return True
            
        except Exception as e:
            cprint(f"    → Sell failed: {e}", "red")
            return False
    
    def _get_current_positions(self) -> Dict[str, float]:
        """Get current positions for all strategies"""
        # In production, would query actual positions
        return {
            "btc_momentum": 5000,
            "eth_mean_reversion": 3000,
            "sol_breakout": 2000
        }
    
    def get_strategy_performance_live(self, strategy_id: str) -> Dict[str, float]:
        """Get live performance metrics for a strategy"""
        # Would integrate with actual strategy performance tracking
        # For now, return sample data
        return {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.10,
            "win_rate": 0.60,
            "volatility": 0.18,
            "total_trades": 45
        }
    
    def sync_portfolio_with_marketplace(self, portfolio_agent: PortfolioRebalancingAgent):
        """Sync portfolio strategies with marketplace"""
        cprint("\n🔄 Syncing portfolio with marketplace...", "cyan")
        
        # Get active strategies from registry
        active_strategies = portfolio_agent.registry.list_strategies(status="active")
        
        # Update portfolio allocations based on marketplace
        for strategy in active_strategies:
            strategy_id = strategy["strategy_id"]
            
            # Check if strategy is in portfolio
            if strategy_id in portfolio_agent.config.target_allocations:
                # Update performance data
                perf_data = strategy.get("performance_data", {})
                cprint(f"  • {strategy_id}: Sharpe {perf_data.get('sharpe_ratio', 0):.2f}", "white")
        
        cprint("✅ Sync complete", "green")


def integrate_portfolio_with_main():
    """Example of integrating portfolio rebalancing into main trading loop"""
    
    # This would be added to main.py
    code_snippet = '''
# In main.py, add to imports:
from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.agents.portfolio_integration import PortfolioIntegration

# Add to ACTIVE_AGENTS:
ACTIVE_AGENTS = {
    'risk': True,
    'trading': True,
    'strategy': True,
    'portfolio': True,  # New portfolio agent
    # ... other agents
}

# In run_agents(), add:
if ACTIVE_AGENTS['portfolio']:
    # Load portfolio configuration
    from src.utils.portfolio_builder import PortfolioBuilder
    builder = PortfolioBuilder()
    
    # Load default portfolio or create one
    portfolio_config = builder.load_portfolio("Moon Dev Balanced")
    if portfolio_config:
        portfolio_agent = PortfolioRebalancingAgent(portfolio_config)
        integrator = PortfolioIntegration()
        
        # Check and rebalance
        cprint("\\n💼 Running Portfolio Rebalancing...", "cyan")
        result = portfolio_agent.check_and_rebalance()
        
        if result["executed"]:
            # Execute through trading infrastructure
            orders = portfolio_agent.rebalancing_history[-1].orders
            integrator.execute_portfolio_orders(orders)
'''
    
    return code_snippet


def demo_integration():
    """Demonstrate portfolio integration"""
    from src.models.portfolio_models import PortfolioConfig, RebalancingMethod, DEFAULT_RISK_LIMITS
    
    cprint("\n🌙 Portfolio Integration Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create sample portfolio
    config = PortfolioConfig(
        name="Integration Demo",
        target_allocations={
            "btc_momentum": 0.50,
            "eth_mean_reversion": 0.30,
            "sol_breakout": 0.20
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params={"min_drift": 0.05},
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Create agents
    portfolio_agent = PortfolioRebalancingAgent(config)
    integrator = PortfolioIntegration()
    
    # Sync with marketplace
    integrator.sync_portfolio_with_marketplace(portfolio_agent)
    
    # Check for rebalancing
    cprint("\n📊 Checking portfolio balance...", "yellow")
    result = portfolio_agent.check_and_rebalance()
    
    if result["executed"]:
        # Get orders from last rebalancing event
        orders = portfolio_agent.rebalancing_history[-1].orders
        
        # Execute through integration
        execution_result = integrator.execute_portfolio_orders(orders)
        
        cprint(f"\n✅ Integration complete!", "green")
    else:
        cprint(f"\n⚠️ No rebalancing needed: {result['reason']}", "yellow")


if __name__ == "__main__":
    demo_integration()