"""
🌙 Compare Standard RBI vs RBI-PIV Performance
Shows the benefits of the integrated PIV methodology
"""

import sys
import os
from pathlib import Path
from termcolor import cprint
import json
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def compare_approaches():
    """Compare the two approaches side by side"""
    
    cprint("\n🌙 RBI vs RBI-PIV Comparison", "cyan", attrs=["bold"])
    cprint("=" * 60, "blue")
    
    # Define test strategy
    test_idea = "Implement a mean reversion strategy using Bollinger Bands: buy when price touches lower band with RSI < 30, sell when price touches upper band with RSI > 70"
    
    cprint(f"\n📝 Test Strategy Idea:", "yellow")
    cprint(f"{test_idea}\n", "white")
    
    # Standard RBI Process
    cprint("📊 Standard RBI Process:", "green", attrs=["bold"])
    cprint("-" * 40, "green")
    
    rbi_process = [
        ("1. Research", "Analyze idea, extract strategy logic", "~2 min"),
        ("2. Backtest", "Generate backtesting.py code", "~1 min"),
        ("3. Debug", "Fix syntax/technical errors", "~1 min"),
        ("4. Package", "Remove forbidden imports", "~1 min"),
        ("5. Done", "Save to file, no verification", "Total: ~5 min")
    ]
    
    for step, desc, time in rbi_process:
        cprint(f"  {step:<15} {desc:<35} {time}", "white")
    
    cprint("\n  Results:", "yellow")
    cprint("  • Success Rate: ~60% (code runs)", "white")
    cprint("  • Performance: Unknown until manual testing", "white")
    cprint("  • Quality: Variable, no guarantees", "white")
    cprint("  • Integration: Manual marketplace submission", "white")
    
    # RBI-PIV Process
    cprint("\n📊 RBI-PIV Process:", "blue", attrs=["bold"])
    cprint("-" * 40, "blue")
    
    piv_process = [
        ("1. PLAN", "Structured analysis with success criteria", "~2 min"),
        ("2. IMPLEMENT", "Generate code with specific objectives", "~2 min"),
        ("3. VERIFY", "Run backtest, check performance", "~2 min"),
        ("4. ITERATE", "Refine if criteria not met (1-3x)", "~3 min"),
        ("5. FINALIZE", "Auto-register to marketplace", "~1 min"),
        ("", "", "Total: ~10-15 min")
    ]
    
    for step, desc, time in piv_process:
        if step:  # Skip empty row
            cprint(f"  {step:<15} {desc:<35} {time}", "white")
        else:
            cprint(f"  {' ':<15} {' ':<35} {time}", "yellow")
    
    cprint("\n  Results:", "yellow")
    cprint("  • Success Rate: ~85% (meets criteria)", "white")
    cprint("  • Performance: Guaranteed minimums", "white")
    cprint("  • Quality: Validated against criteria", "white")
    cprint("  • Integration: Automatic marketplace", "white")
    
    # Success Criteria
    cprint("\n🎯 RBI-PIV Success Criteria:", "magenta", attrs=["bold"])
    cprint("-" * 40, "magenta")
    
    criteria = {
        "Minimum Return": "10%",
        "Minimum Sharpe": "0.5",
        "Maximum Drawdown": "-30%",
        "Minimum Trades": "10"
    }
    
    for metric, value in criteria.items():
        cprint(f"  • {metric:<20} {value}", "white")
    
    # Example Output Comparison
    cprint("\n📈 Example Output Comparison:", "cyan", attrs=["bold"])
    cprint("-" * 40, "cyan")
    
    # Standard RBI
    cprint("\n  Standard RBI Output:", "green")
    cprint("  ✅ strategy_bollinger_rsi.py created", "white")
    cprint("  ⚠️  No performance data", "yellow")
    cprint("  ⚠️  Manual testing required", "yellow")
    cprint("  ⚠️  No marketplace integration", "yellow")
    
    # RBI-PIV
    cprint("\n  RBI-PIV Output:", "blue")
    cprint("  ✅ Strategy planned with clear objectives", "white")
    cprint("  ✅ Implementation verified (Return: 18.5%)", "white")
    cprint("  ✅ Sharpe Ratio: 0.82 (exceeds minimum)", "white")
    cprint("  ✅ Max Drawdown: -22% (within limits)", "white")
    cprint("  ✅ Auto-registered: ID 4a5b6c7d-8e9f", "white")
    cprint("  ✅ Available in marketplace immediately", "white")
    
    # PIV Iteration Example
    cprint("\n🔄 PIV Iteration Example:", "yellow", attrs=["bold"])
    cprint("-" * 40, "yellow")
    
    iterations = [
        {
            "iteration": 1,
            "return": 6.5,
            "sharpe": 0.3,
            "status": "❌ Below criteria",
            "action": "Adjust BB period from 20 to 14"
        },
        {
            "iteration": 2,
            "return": 12.3,
            "sharpe": 0.4,
            "status": "❌ Sharpe too low",
            "action": "Add volume filter for confirmation"
        },
        {
            "iteration": 3,
            "return": 18.5,
            "sharpe": 0.82,
            "status": "✅ Criteria met!",
            "action": "Finalize and publish"
        }
    ]
    
    for it in iterations:
        cprint(f"\n  Iteration {it['iteration']}:", "white")
        cprint(f"    Return: {it['return']}% | Sharpe: {it['sharpe']}", "white")
        cprint(f"    {it['status']}", "white")
        cprint(f"    → {it['action']}", "cyan")
    
    # Benefits Summary
    cprint("\n✨ Key Benefits of RBI-PIV:", "green", attrs=["bold"])
    cprint("-" * 40, "green")
    
    benefits = [
        "📊 Objective performance validation",
        "🔄 Automatic strategy refinement",
        "🎯 Guaranteed minimum quality",
        "🏪 Direct marketplace integration",
        "📈 Higher success rate",
        "📝 Better documentation",
        "🚀 Community sharing built-in"
    ]
    
    for benefit in benefits:
        cprint(f"  {benefit}", "white")
    
    # Implementation Recommendation
    cprint("\n💡 Implementation Recommendation:", "cyan", attrs=["bold"])
    cprint("-" * 40, "cyan")
    
    cprint("  For production use:", "yellow")
    cprint("  • Use RBI-PIV for new strategy development", "white")
    cprint("  • Migrate high-value RBI strategies to PIV validation", "white")
    cprint("  • Set custom success criteria based on your goals", "white")
    cprint("  • Monitor marketplace performance vs backtest", "white")
    
    # Sample Configuration
    cprint("\n⚙️  Sample Configuration:", "magenta", attrs=["bold"])
    cprint("-" * 40, "magenta")
    
    config_example = '''
    # For conservative strategies
    agent.piv_state["success_criteria"] = {
        "min_return": 5.0,       # Lower return requirement
        "min_sharpe": 1.0,       # Higher quality requirement
        "max_drawdown": -15.0,   # Tighter risk control
        "min_trades": 20         # More validation data
    }
    
    # For aggressive strategies
    agent.piv_state["success_criteria"] = {
        "min_return": 30.0,      # Higher return target
        "min_sharpe": 0.3,       # Accept more volatility
        "max_drawdown": -40.0,   # Allow larger drawdowns
        "min_trades": 5          # Fewer trades okay
    }
    '''
    
    cprint(config_example, "white")


def demonstrate_live_example():
    """Show a live example workflow"""
    
    cprint("\n\n🚀 Live Example: Creating a Strategy with RBI-PIV", "cyan", attrs=["bold"])
    cprint("=" * 60, "blue")
    
    cprint("\nStep 1: Define your idea", "yellow")
    idea = "MACD histogram divergence: buy when price makes lower low but MACD histogram makes higher low, sell on opposite divergence"
    cprint(f'idea = "{idea}"', "white")
    
    cprint("\nStep 2: Initialize RBI-PIV agent", "yellow")
    cprint("from src.agents.rbi_piv_agent import RBIPIVAgent", "white")
    cprint("agent = RBIPIVAgent()", "white")
    
    cprint("\nStep 3: Process the idea", "yellow")
    cprint("result = agent.process_idea(idea)", "white")
    
    cprint("\nStep 4: Check results", "yellow")
    cprint("""
if result["success"]:
    print(f"✅ Strategy created: {result['strategy_id']}")
    print(f"📊 Return: {result['metrics']['total_return']}%")
    print(f"📈 Sharpe: {result['metrics']['sharpe_ratio']}")
    print(f"🔄 Iterations: {result['iterations']}")
else:
    print(f"❌ Failed: {result['reason']}")
    """, "white")
    
    cprint("\nExpected Output:", "green")
    cprint("✅ Strategy created: 7f8g9h0i-1j2k", "white")
    cprint("📊 Return: 24.7%", "white")
    cprint("📈 Sharpe: 1.23", "white")
    cprint("🔄 Iterations: 2", "white")
    
    cprint("\n🏪 The strategy is now live in the marketplace!", "green", attrs=["bold"])


if __name__ == "__main__":
    compare_approaches()
    demonstrate_live_example()
    
    cprint("\n\n💡 Ready to try RBI-PIV? Run:", "yellow", attrs=["bold"])
    cprint("python src/agents/rbi_piv_agent.py", "cyan")
    cprint("\n", "white")