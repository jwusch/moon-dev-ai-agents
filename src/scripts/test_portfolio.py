"""
Quick test of portfolio rebalancing functionality
"""
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.models.portfolio_models import PortfolioConfig, RebalancingMethod, DEFAULT_RISK_LIMITS
from termcolor import cprint

def test_basic_portfolio():
    cprint("\n🌙 Testing Portfolio Rebalancing", "cyan", attrs=["bold"])
    
    # Create simple config
    config = PortfolioConfig(
        name="Test Portfolio",
        target_allocations={
            "strategy_1": 0.50,
            "strategy_2": 0.30,
            "strategy_3": 0.20
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params={"min_drift": 0.05},
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Validate
    config.validate()
    cprint("✅ Config validated", "green")
    
    # Create agent
    agent = PortfolioRebalancingAgent(config)
    cprint("✅ Agent created", "green")
    
    # Display dashboard
    agent.display_portfolio_dashboard()
    
    # Check rebalancing
    result = agent.check_and_rebalance()
    cprint(f"\n📊 Rebalancing result: {result}", "yellow")

if __name__ == "__main__":
    test_basic_portfolio()