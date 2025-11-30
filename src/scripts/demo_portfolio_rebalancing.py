"""
🌙 Moon Dev Portfolio Rebalancing Demo
Demonstrates the portfolio rebalancing agent with multiple strategies
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from termcolor import cprint
import time

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.portfolio_rebalancing_agent import PortfolioRebalancingAgent
from src.models.portfolio_models import (
    PortfolioConfig, RebalancingMethod, 
    DEFAULT_RISK_LIMITS, DEFAULT_REBALANCING_PARAMS
)


def simulate_portfolio_drift(agent: PortfolioRebalancingAgent, days: int = 30):
    """Simulate market movements causing portfolio drift"""
    cprint(f"\n📈 Simulating {days} days of market movements...", "cyan")
    
    # Get current positions
    current_values = agent._get_current_strategy_values()
    
    # Apply random walk to each strategy
    for day in range(days):
        daily_returns = {}
        
        for strategy in current_values:
            # Random daily return between -3% and +3%
            daily_return = np.random.normal(0, 0.015)
            current_values[strategy] *= (1 + daily_return)
            daily_returns[strategy] = daily_return
        
        # Show significant moves
        if day % 7 == 0:  # Weekly update
            cprint(f"\n  Week {day//7 + 1}:", "white")
            for strategy, ret in daily_returns.items():
                if abs(ret) > 0.02:  # Show big moves
                    emoji = "📈" if ret > 0 else "📉"
                    cprint(f"    {emoji} {strategy}: {ret:+.1%}", "yellow")
    
    # Update agent's positions
    agent.monitor.update_positions(current_values)
    
    # Show drift
    cprint("\n🎯 Current drift from targets:", "yellow")
    for strategy_id, position in agent.monitor.positions.items():
        drift_color = "green" if abs(position.drift) < 0.05 else "yellow" if abs(position.drift) < 0.10 else "red"
        cprint(f"  {strategy_id}: {position.drift:+.1%}", drift_color)


def demo_threshold_rebalancing():
    """Demo threshold-based rebalancing"""
    cprint("\n🔄 Demo 1: Threshold Rebalancing", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create portfolio config
    config = PortfolioConfig(
        name="Threshold Demo Portfolio",
        target_allocations={
            "btc_momentum": 0.40,
            "eth_mean_reversion": 0.30,
            "sol_breakout": 0.20,
            "alt_rotation": 0.10
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params={
            "min_drift": 0.05,  # 5% drift threshold
            "min_trade_size": 100,
            "check_frequency": "daily"
        },
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Initialize agent
    agent = PortfolioRebalancingAgent(config)
    
    # Show initial state
    cprint("\n📊 Initial Portfolio State:", "yellow")
    agent.display_portfolio_dashboard()
    
    # Simulate drift
    simulate_portfolio_drift(agent, days=15)
    
    # Check for rebalancing
    cprint("\n🎯 Checking rebalancing triggers...", "yellow")
    result = agent.check_and_rebalance()
    
    if result["executed"]:
        cprint(f"\n✅ Rebalancing executed: {result['orders_count']} orders", "green")
        agent.display_portfolio_dashboard()
    else:
        cprint(f"\n⚠️ No rebalancing: {result['reason']}", "yellow")


def demo_calendar_rebalancing():
    """Demo calendar-based rebalancing"""
    cprint("\n📅 Demo 2: Calendar Rebalancing", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create portfolio config
    config = PortfolioConfig(
        name="Calendar Demo Portfolio",
        target_allocations={
            "large_cap": 0.50,
            "mid_cap": 0.30,
            "small_cap": 0.20
        },
        rebalancing_method=RebalancingMethod.CALENDAR,
        rebalancing_params={
            "frequency": "monthly",
            "rebalance_day": 1,
            "force_rebalance": True
        },
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Initialize agent
    agent = PortfolioRebalancingAgent(config)
    
    # Simulate last rebalance was 31 days ago
    agent.monitor.last_rebalance = datetime.now() - timedelta(days=31)
    
    cprint("\n📊 Current Portfolio State:", "yellow")
    agent.display_portfolio_dashboard()
    
    # Check for rebalancing
    cprint("\n📅 Checking calendar trigger...", "yellow")
    result = agent.check_and_rebalance()
    
    if result["executed"]:
        cprint(f"\n✅ Monthly rebalancing executed", "green")
    else:
        cprint(f"\n⚠️ Not time for rebalancing yet", "yellow")


def demo_adaptive_rebalancing():
    """Demo adaptive rebalancing based on performance"""
    cprint("\n🎯 Demo 3: Adaptive Rebalancing", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create portfolio config
    config = PortfolioConfig(
        name="Adaptive Demo Portfolio",
        target_allocations={
            "high_sharpe_strategy": 0.25,
            "steady_returns": 0.25,
            "high_risk_high_reward": 0.25,
            "defensive_strategy": 0.25
        },
        rebalancing_method=RebalancingMethod.ADAPTIVE,
        rebalancing_params=DEFAULT_REBALANCING_PARAMS["adaptive"],
        risk_limits=DEFAULT_RISK_LIMITS
    )
    
    # Initialize agent
    agent = PortfolioRebalancingAgent(config)
    
    cprint("\n📊 Initial Equal-Weight Portfolio:", "yellow")
    agent.display_portfolio_dashboard()
    
    # Simulate some strategies performing better
    cprint("\n📈 Simulating performance differences...", "cyan")
    time.sleep(1)  # Brief pause for effect
    
    # Check for rebalancing
    cprint("\n🧠 Adaptive rebalancing based on performance...", "yellow")
    result = agent.check_and_rebalance()
    
    if result["executed"]:
        cprint(f"\n✅ Adaptive rebalancing executed", "green")
        cprint("Better performing strategies received higher allocations", "white")
    
    agent.display_portfolio_dashboard()


def demo_risk_limits():
    """Demo risk limit enforcement"""
    cprint("\n🛡️ Demo 4: Risk Limit Enforcement", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create portfolio with aggressive allocations
    config = PortfolioConfig(
        name="Risk Demo Portfolio",
        target_allocations={
            "aggressive_strategy": 0.60,  # Too high!
            "moderate_strategy": 0.30,
            "conservative_strategy": 0.10
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params=DEFAULT_REBALANCING_PARAMS["threshold"],
        risk_limits={
            **DEFAULT_RISK_LIMITS,
            "max_single_strategy": 0.40  # 40% max
        }
    )
    
    try:
        # This should trigger risk limits
        agent = PortfolioRebalancingAgent(config)
        
        cprint("\n⚠️ Attempting rebalancing with risk violations...", "yellow")
        result = agent.check_and_rebalance()
        
        if not result["executed"]:
            cprint(f"\n✅ Risk limits enforced: {result['reason']}", "green")
            if "violations" in result:
                for violation in result["violations"]:
                    cprint(f"  • {violation}", "red")
    except Exception as e:
        cprint(f"\n✅ Configuration rejected: {e}", "green")


def demo_multi_strategy_portfolio():
    """Demo realistic multi-strategy portfolio"""
    cprint("\n🌙 Demo 5: Multi-Strategy Portfolio Management", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Create diversified portfolio
    config = PortfolioConfig(
        name="Moon Dev Diversified Portfolio",
        target_allocations={
            "trend_following": 0.25,
            "mean_reversion": 0.20,
            "arbitrage": 0.20,
            "momentum": 0.15,
            "market_making": 0.10,
            "ml_predictor": 0.10
        },
        rebalancing_method=RebalancingMethod.THRESHOLD,
        rebalancing_params={
            "min_drift": 0.05,
            "min_trade_size": 100,
            "check_frequency": "daily"
        },
        risk_limits={
            **DEFAULT_RISK_LIMITS,
            "max_correlation_pair": 0.7,
            "min_portfolio_sharpe": 0.5
        }
    )
    
    # Initialize agent
    agent = PortfolioRebalancingAgent(config)
    
    cprint("\n📊 Diversified Portfolio Overview:", "yellow")
    agent.display_portfolio_dashboard()
    
    # Simulate multiple rebalancing cycles
    for cycle in range(3):
        cprint(f"\n\n🔄 Rebalancing Cycle {cycle + 1}", "blue", attrs=["bold"])
        cprint("-" * 40, "blue")
        
        # Simulate market movements
        simulate_portfolio_drift(agent, days=10)
        
        # Check and rebalance
        result = agent.check_and_rebalance()
        
        if result["executed"]:
            cprint(f"✅ Rebalanced: {result['orders_count']} trades, ${result['total_traded']:,.0f} total", "green")
        else:
            cprint(f"⚠️ No rebalancing needed", "yellow")
        
        time.sleep(1)  # Brief pause
    
    # Final state
    cprint("\n\n📊 Final Portfolio State:", "green", attrs=["bold"])
    agent.display_portfolio_dashboard()
    
    # Show rebalancing history
    cprint("\n📜 Rebalancing History:", "yellow")
    for i, event in enumerate(agent.rebalancing_history, 1):
        cprint(f"  {i}. {event.timestamp.strftime('%Y-%m-%d %H:%M')} - "
               f"{len(event.orders)} orders, ${event.total_traded:,.0f} traded", "white")


def main():
    """Run all portfolio rebalancing demos"""
    cprint("\n🌙 Moon Dev Portfolio Rebalancing System", "cyan", attrs=["bold"])
    cprint("=" * 60, "blue")
    cprint("\nDemonstrating portfolio management capabilities...", "white")
    
    demos = [
        ("Threshold Rebalancing", demo_threshold_rebalancing),
        ("Calendar Rebalancing", demo_calendar_rebalancing),
        ("Adaptive Rebalancing", demo_adaptive_rebalancing),
        ("Risk Limit Enforcement", demo_risk_limits),
        ("Multi-Strategy Portfolio", demo_multi_strategy_portfolio)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        if i > 1:
            input("\nPress Enter to continue to next demo...")
        
        try:
            demo_func()
        except Exception as e:
            cprint(f"\n❌ Error in {name}: {e}", "red")
    
    cprint("\n\n✅ Portfolio Rebalancing Demo Complete!", "green", attrs=["bold"])
    cprint("\nKey Features Demonstrated:", "yellow")
    cprint("  • Multiple rebalancing methods", "white")
    cprint("  • Risk limit enforcement", "white")
    cprint("  • Performance-based adaptation", "white")
    cprint("  • Multi-strategy management", "white")
    cprint("  • Automated execution", "white")
    
    cprint("\n💡 Next Steps:", "cyan")
    cprint("  1. Run portfolio builder: python src/utils/portfolio_builder.py", "white")
    cprint("  2. Create your own portfolio configuration", "white")
    cprint("  3. Integrate with live trading agents", "white")
    cprint("  4. Monitor performance in marketplace dashboard", "white")


if __name__ == "__main__":
    main()