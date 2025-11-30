"""
🌙 Moon Dev's RBI-PIV Agent (Research-Backtest-Implement with Plan-Implement-Verify)
An enhanced version of RBI that integrates the PIV iterative process
Built with love by Moon Dev 🚀

This agent combines:
- RBI's proven strategy generation from YouTube/PDFs/text
- PIV's iterative refinement and verification process
- Enhanced planning and feedback loops
- Automatic performance validation

The PIV enhancement adds:
1. Structured planning phase with clear objectives
2. Implementation with checkpoints
3. Verification against success criteria
4. Iterative refinement based on performance
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from termcolor import cprint
import pandas as pd
import numpy as np
from io import StringIO
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.model_factory import ModelFactory
from src.agents.strategy_registry_agent import StrategyRegistryAgent
from src.marketplace.analytics import StrategyAnalytics


class RBIPIVAgent:
    """Enhanced RBI agent with PIV (Plan-Implement-Verify) methodology"""
    
    def __init__(self):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "src" / "data" / "rbi_piv"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models for different phases
        self.plan_model = ModelFactory.create_model("openai", "gpt-4o")  # Planning
        self.implement_model = ModelFactory.create_model("deepseek", "deepseek-coder")  # Coding
        self.verify_model = ModelFactory.create_model("anthropic", "claude-3-5-sonnet-latest")  # Verification
        
        # Initialize marketplace integration
        self.registry = StrategyRegistryAgent()
        self.analytics = StrategyAnalytics()
        
        # PIV tracking
        self.piv_state = {
            "current_iteration": 0,
            "max_iterations": 3,
            "success_criteria": {
                "min_return": 10.0,  # Minimum 10% return
                "min_sharpe": 0.5,   # Minimum Sharpe ratio
                "max_drawdown": -30.0,  # Maximum 30% drawdown
                "min_trades": 10      # Minimum number of trades
            }
        }
    
    def process_idea(self, idea: str) -> Dict[str, Any]:
        """
        Process a trading idea through the enhanced RBI-PIV pipeline
        
        Args:
            idea: Trading strategy idea (URL, PDF, or text)
            
        Returns:
            Dictionary with strategy details and results
        """
        cprint(f"\n🚀 Starting RBI-PIV Process for: {idea[:100]}...", "cyan", attrs=["bold"])
        
        # Create session directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.data_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: PLAN (Enhanced Research)
        cprint("\n📋 Phase 1: PLANNING", "yellow", attrs=["bold"])
        plan = self._plan_phase(idea, session_dir)
        
        if not plan["viable"]:
            cprint("❌ Strategy deemed not viable during planning", "red")
            return {"success": False, "reason": "Not viable", "plan": plan}
        
        # PIV Loop
        best_result = None
        for iteration in range(self.piv_state["max_iterations"]):
            self.piv_state["current_iteration"] = iteration + 1
            cprint(f"\n🔄 PIV Iteration {iteration + 1}/{self.piv_state['max_iterations']}", "blue", attrs=["bold"])
            
            # Phase 2: IMPLEMENT (Generate Backtest)
            cprint("\n💻 Phase 2: IMPLEMENTING", "yellow", attrs=["bold"])
            implementation = self._implement_phase(plan, session_dir, iteration)
            
            if not implementation["success"]:
                cprint(f"❌ Implementation failed: {implementation.get('error', 'Unknown error')}", "red")
                continue
            
            # Phase 3: VERIFY (Test and Validate)
            cprint("\n✅ Phase 3: VERIFYING", "yellow", attrs=["bold"])
            verification = self._verify_phase(implementation, session_dir)
            
            # Check if success criteria met
            if self._check_success_criteria(verification):
                cprint("🎉 Success criteria met!", "green", attrs=["bold"])
                best_result = {
                    "success": True,
                    "plan": plan,
                    "implementation": implementation,
                    "verification": verification,
                    "iteration": iteration + 1
                }
                break
            
            # Prepare feedback for next iteration
            if iteration < self.piv_state["max_iterations"] - 1:
                plan = self._refine_plan(plan, verification)
        
        # Phase 4: FINALIZE (Package and Register)
        if best_result:
            cprint("\n📦 Phase 4: FINALIZING", "yellow", attrs=["bold"])
            final_result = self._finalize_phase(best_result, session_dir)
            return final_result
        
        return {
            "success": False,
            "reason": "Failed to meet success criteria after all iterations",
            "last_verification": verification if 'verification' in locals() else None
        }
    
    def _plan_phase(self, idea: str, session_dir: Path) -> Dict[str, Any]:
        """
        Enhanced planning phase that structures the strategy development
        """
        prompt = f"""
        You are a expert trading strategy architect. Analyze this trading idea and create a structured plan:
        
        Trading Idea: {idea}
        
        Create a comprehensive plan that includes:
        
        1. **Strategy Overview**
           - Core concept and edge
           - Market conditions it exploits
           - Expected performance characteristics
        
        2. **Technical Specifications**
           - Required indicators and parameters
           - Entry conditions (be very specific)
           - Exit conditions (including stops and targets)
           - Position sizing approach
           - Risk management rules
        
        3. **Implementation Requirements**
           - Data requirements (timeframe, lookback period)
           - Computational complexity
           - Special considerations
        
        4. **Success Metrics**
           - Expected return range
           - Expected Sharpe ratio
           - Maximum acceptable drawdown
           - Minimum trade frequency
        
        5. **Potential Challenges**
           - Implementation difficulties
           - Market regime dependencies
           - Overfitting risks
        
        Return as JSON with these sections. Also include a "viable" boolean indicating if this strategy is worth implementing.
        """
        
        response = self.plan_model.generate_response(
            system_prompt="You are a expert quantitative trading strategist with deep knowledge of technical analysis and market microstructure.",
            user_content=prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = {"viable": False, "reason": "Failed to parse plan"}
        except Exception as e:
            plan = {"viable": False, "reason": f"JSON parsing error: {str(e)}"}
        
        # Save plan
        plan_file = session_dir / "strategy_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        
        if plan.get("viable", False):
            cprint("✅ Strategy plan created and deemed viable", "green")
        else:
            cprint(f"❌ Strategy not viable: {plan.get('reason', 'Unknown')}", "red")
        
        return plan
    
    def _implement_phase(self, plan: Dict[str, Any], session_dir: Path, iteration: int) -> Dict[str, Any]:
        """
        Implementation phase with iterative improvements
        """
        # Prepare implementation prompt with PIV context
        refinements = ""
        if iteration > 0 and "refinements" in plan:
            refinements = f"\nRefinements from previous iteration:\n{plan['refinements']}\n"
        
        prompt = f"""
        Implement a backtest for this trading strategy using backtesting.py library.
        
        Strategy Plan:
        {json.dumps(plan, indent=2)}
        {refinements}
        
        Requirements:
        1. Use backtesting.py library structure
        2. Implement ALL indicators using self.I() wrapper with talib or pandas_ta
        3. Follow the exact entry/exit conditions from the plan
        4. Include proper position sizing (use fractions like 0.95 for 95% of capital)
        5. Add debug prints with emojis for key events
        6. Use this data: {self.project_root}/src/data/rbi/BTC-USD-15m.csv
        
        Return ONLY the Python code, no explanations.
        """
        
        response = self.implement_model.generate_response(
            system_prompt="You are an expert Python programmer specializing in backtesting.py implementations.",
            user_content=prompt,
            temperature=0.3,
            max_tokens=3000
        )
        
        # Extract code
        code = self._extract_code(response)
        
        # Save implementation
        impl_file = session_dir / f"strategy_v{iteration + 1}.py"
        with open(impl_file, 'w') as f:
            f.write(code)
        
        return {
            "success": True,
            "code_path": str(impl_file),
            "code": code
        }
    
    def _verify_phase(self, implementation: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        """
        Comprehensive verification phase with performance analysis
        """
        # Run backtest
        cprint("🧪 Running backtest...", "cyan")
        
        try:
            # Execute backtest
            result = subprocess.run(
                [sys.executable, implementation["code_path"]],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "metrics": {}
                }
            
            # Parse output for stats
            output = result.stdout
            
            # Extract key metrics (simplified - in reality would parse properly)
            metrics = self._parse_backtest_output(output)
            
            # Save results
            results_file = session_dir / f"results_v{self.piv_state['current_iteration']}.json"
            with open(results_file, 'w') as f:
                json.dump({"output": output, "metrics": metrics}, f, indent=2)
            
            # Display metrics
            cprint("\n📊 Backtest Results:", "green")
            for key, value in metrics.items():
                cprint(f"  {key}: {value}", "white")
            
            return {
                "success": True,
                "metrics": metrics,
                "output": output
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Backtest timed out",
                "metrics": {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metrics": {}
            }
    
    def _check_success_criteria(self, verification: Dict[str, Any]) -> bool:
        """
        Check if the verification results meet success criteria
        """
        if not verification.get("success", False):
            return False
        
        metrics = verification.get("metrics", {})
        criteria = self.piv_state["success_criteria"]
        
        checks = {
            "return": metrics.get("total_return", 0) >= criteria["min_return"],
            "sharpe": metrics.get("sharpe_ratio", 0) >= criteria["min_sharpe"],
            "drawdown": metrics.get("max_drawdown", -100) >= criteria["max_drawdown"],
            "trades": metrics.get("total_trades", 0) >= criteria["min_trades"]
        }
        
        cprint("\n🎯 Success Criteria Check:", "yellow")
        for criterion, passed in checks.items():
            emoji = "✅" if passed else "❌"
            cprint(f"  {emoji} {criterion}: {passed}", "white")
        
        return all(checks.values())
    
    def _refine_plan(self, plan: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the plan based on verification results
        """
        metrics = verification.get("metrics", {})
        
        prompt = f"""
        The strategy implementation needs refinement. Here are the results:
        
        Current Metrics:
        {json.dumps(metrics, indent=2)}
        
        Success Criteria:
        {json.dumps(self.piv_state["success_criteria"], indent=2)}
        
        Original Plan:
        {json.dumps(plan, indent=2)}
        
        Suggest specific refinements to improve performance:
        1. Parameter adjustments
        2. Entry/exit condition modifications
        3. Risk management improvements
        
        Focus on the metrics that failed to meet criteria.
        """
        
        response = self.verify_model.generate_response(
            system_prompt="You are a trading strategy optimization expert.",
            user_content=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        plan["refinements"] = response
        cprint("🔧 Plan refined for next iteration", "cyan")
        
        return plan
    
    def _finalize_phase(self, result: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        """
        Finalize successful strategy and register with marketplace
        """
        # Generate strategy metadata
        plan = result["plan"]
        metrics = result["verification"]["metrics"]
        
        # Register with marketplace
        strategy_name = plan.get("strategy_overview", {}).get("name", "RBI-PIV Strategy")
        
        metadata = self.registry.register_strategy(
            name=strategy_name,
            description=plan.get("strategy_overview", {}).get("description", "AI-generated strategy"),
            author="rbi_piv_agent",
            code_path=result["implementation"]["code_path"],
            category=["ai_generated", "technical"],
            timeframes=["15m"],  # From our backtest data
            instruments=["BTC"],
            min_capital=10000,
            risk_level="medium"
        )
        
        # Update performance
        self.registry.update_performance(metadata["strategy_id"], metrics)
        
        # Mark as approved
        self.registry.approve_strategy(metadata["strategy_id"])
        
        cprint(f"\n🎊 Strategy registered in marketplace: {metadata['strategy_id']}", "green", attrs=["bold"])
        
        return {
            "success": True,
            "strategy_id": metadata["strategy_id"],
            "name": strategy_name,
            "metrics": metrics,
            "iterations": result["iteration"],
            "session_dir": str(session_dir)
        }
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from model response"""
        # Try to find code blocks
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # If no code blocks, assume entire response is code
        return response.strip()
    
    def _parse_backtest_output(self, output: str) -> Dict[str, Any]:
        """Parse backtest output for metrics"""
        metrics = {
            "total_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "win_rate": 0
        }
        
        # Simple parsing - in production would be more robust
        lines = output.split('\n')
        for line in lines:
            if "Return [%]" in line:
                try:
                    metrics["total_return"] = float(line.split()[-1])
                except:
                    pass
            elif "Sharpe Ratio" in line:
                try:
                    metrics["sharpe_ratio"] = float(line.split()[-1])
                except:
                    pass
            elif "Max. Drawdown [%]" in line:
                try:
                    metrics["max_drawdown"] = float(line.split()[-1])
                except:
                    pass
            elif "# Trades" in line:
                try:
                    metrics["total_trades"] = int(line.split()[-1])
                except:
                    pass
            elif "Win Rate [%]" in line:
                try:
                    metrics["win_rate"] = float(line.split()[-1])
                except:
                    pass
        
        return metrics


def main():
    """Demo the RBI-PIV agent"""
    cprint("\n🌙 Moon Dev's RBI-PIV Agent Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "blue")
    
    # Initialize agent
    agent = RBIPIVAgent()
    
    # Test ideas
    test_ideas = [
        "Buy when price crosses above 20 SMA and RSI is oversold, sell when RSI is overbought",
        "Bollinger Band squeeze breakout strategy with volume confirmation",
        "MACD divergence strategy with support/resistance levels"
    ]
    
    # Process first idea as demo
    idea = test_ideas[0]
    cprint(f"\n🎯 Testing idea: {idea}", "yellow")
    
    result = agent.process_idea(idea)
    
    if result["success"]:
        cprint("\n✅ Strategy successfully created and registered!", "green", attrs=["bold"])
        cprint(f"Strategy ID: {result['strategy_id']}", "white")
        cprint(f"Total Return: {result['metrics']['total_return']}%", "white")
        cprint(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']}", "white")
        cprint(f"Iterations needed: {result['iterations']}", "white")
    else:
        cprint(f"\n❌ Strategy creation failed: {result.get('reason', 'Unknown')}", "red")


if __name__ == "__main__":
    import re  # Add this import
    main()