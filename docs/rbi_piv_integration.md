# Product Requirements Document: RBI-PIV Integration

## 1. Executive Summary

The RBI-PIV Integration represents a significant evolution in automated trading strategy generation, combining Moon Dev's proven RBI (Research-Backtest-Implement) agent with the structured PIV (Plan-Implement-Verify) methodology. This enhanced system addresses the critical gap between strategy ideation and validated implementation by introducing iterative refinement loops, objective performance criteria, and automatic marketplace integration.

The core value proposition is transforming the success rate of AI-generated trading strategies from approximately 60% to 85% by ensuring each strategy meets predefined performance criteria before publication. This integration eliminates the manual testing burden, reduces time-to-market for viable strategies, and guarantees a minimum quality standard for all published strategies in the Moon Dev ecosystem.

The MVP goal is to deliver a fully functional RBI-PIV agent that can process trading ideas from multiple sources (YouTube, PDFs, text), iteratively develop and refine backtesting code, validate performance against configurable success criteria, and automatically publish successful strategies to the Moon Dev Strategy Marketplace.

## 2. Mission

**Product Mission Statement:**
To democratize quantitative trading strategy development by combining AI-powered research capabilities with rigorous iterative validation, ensuring every published strategy meets objective performance standards while maintaining full transparency and community accessibility.

**Core Principles:**
- ✅ **Objective Validation**: Every strategy must meet measurable performance criteria before publication
- ✅ **Iterative Excellence**: Continuous refinement based on backtesting results ensures optimal strategy configuration
- ✅ **Community First**: Automatic marketplace integration shares successful strategies with the entire Moon Dev ecosystem
- ✅ **Transparency**: Full visibility into the planning, implementation, and verification process
- ✅ **Flexibility**: Configurable success criteria accommodate different risk profiles and trading objectives

## 3. Target Users

**Primary User Personas:**

1. **Algorithmic Trading Developers**
   - Technical comfort: Advanced Python developers with backtesting experience
   - Needs: Rapid strategy prototyping with guaranteed minimum performance
   - Pain points: Time-consuming manual backtesting and optimization cycles

2. **Quantitative Researchers**
   - Technical comfort: Expert-level understanding of financial markets and statistics
   - Needs: Automated strategy generation from research papers and videos
   - Pain points: Translation of theoretical concepts to working code

3. **Trading Strategy Enthusiasts**
   - Technical comfort: Intermediate programming skills, strong trading knowledge
   - Needs: Convert trading ideas to validated, shareable strategies
   - Pain points: Lack of coding expertise to implement complex strategies

4. **Moon Dev Community Members**
   - Technical comfort: Varies from beginner to expert
   - Needs: Access to high-quality, pre-validated trading strategies
   - Pain points: Uncertainty about strategy quality in marketplace

## 4. MVP Scope

### In Scope (Core Functionality)
**Strategy Generation:**
- ✅ Multi-source input processing (YouTube URLs, PDFs, plain text)
- ✅ Structured planning phase with strategy specifications
- ✅ Automated backtesting code generation
- ✅ Iterative refinement (up to 3 iterations)
- ✅ Performance verification against criteria

**Integration Features:**
- ✅ Automatic marketplace registration for successful strategies
- ✅ Performance metrics calculation and storage
- ✅ Strategy metadata management
- ✅ Export/import functionality

**Quality Assurance:**
- ✅ Configurable success criteria
- ✅ Code validation and debugging
- ✅ Backtesting execution and result parsing
- ✅ Performance tracking across iterations

### Out of Scope (Future Phases)
**Technical:**
- ❌ Real-time strategy execution
- ❌ Multi-asset simultaneous testing
- ❌ Machine learning model integration
- ❌ Cloud-based backtesting infrastructure

**Integration:**
- ❌ Direct broker connectivity
- ❌ Live performance tracking
- ❌ Strategy combination/ensemble methods
- ❌ Cross-platform deployment

## 5. User Stories

1. **"As a trading developer, I want to input a strategy idea and receive validated backtesting code, so that I can quickly test new concepts without manual coding"**
   - Example: Input "RSI divergence with Bollinger Bands" → Receive working strategy with 15% return

2. **"As a quantitative researcher, I want to convert academic papers into executable strategies, so that I can validate theoretical models with real data"**
   - Example: Upload PDF on mean reversion → Get implemented strategy meeting Sharpe > 0.5

3. **"As a strategy enthusiast, I want the system to automatically improve my ideas, so that even basic concepts can become profitable strategies"**
   - Example: "Buy low, sell high" → Refined to specific indicator-based rules with positive returns

4. **"As a marketplace user, I want to trust that all strategies meet quality standards, so that I can confidently use community strategies"**
   - Example: Browse marketplace knowing all strategies have >10% returns and controlled drawdowns

5. **"As an AI developer, I want to see how different models contribute to strategy development, so that I can understand the multi-model approach"**
   - Example: View which model handled planning vs implementation vs verification

6. **"As a risk-conscious trader, I want to set custom success criteria, so that strategies match my risk tolerance"**
   - Example: Configure max drawdown of 15% for conservative portfolio

7. **"As a strategy author, I want my successful strategies automatically published, so that I can contribute to the community without extra steps"**
   - Example: Strategy achieving targets auto-appears in marketplace with full attribution

## 6. Core Architecture & Patterns

**High-Level Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    RBI-PIV Agent System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ Input Layer  │   │Process Layer │   │ Output Layer │  │
│  │              │   │              │   │              │  │
│  │ • YouTube    │──▶│ • Planning   │──▶│ • Strategy   │  │
│  │ • PDF        │   │ • Implement  │   │   Code      │  │
│  │ • Text       │   │ • Verify     │   │ • Metrics   │  │
│  │              │   │ • Iterate    │   │ • Registry  │  │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
│                             ▲                               │
│                             │                               │
│                    ┌────────┴────────┐                     │
│                    │ Model Factory   │                     │
│                    │                 │                     │
│                    │ • GPT-4        │                     │
│                    │ • DeepSeek     │                     │
│                    │ • Claude       │                     │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

**Directory Structure:**
```
src/
├── agents/
│   ├── rbi_agent.py           # Original RBI implementation
│   ├── rbi_piv_agent.py       # Enhanced PIV integration
│   └── strategy_registry_agent.py
├── data/
│   ├── rbi_piv/              # PIV-specific data
│   │   ├── YYYYMMDD_HHMMSS/  # Session directories
│   │   │   ├── strategy_plan.json
│   │   │   ├── strategy_v1.py
│   │   │   ├── results_v1.json
│   │   │   └── final_metrics.json
│   └── marketplace/          # Published strategies
└── models/                   # LLM integrations
```

**Key Design Patterns:**
- **Factory Pattern**: ModelFactory for multi-LLM support
- **Strategy Pattern**: Pluggable success criteria configurations
- **Iterator Pattern**: PIV loop for refinement cycles
- **Observer Pattern**: Performance tracking across iterations
- **Template Method**: Structured phase execution (Plan→Implement→Verify)

## 7. Tools/Features

### Core Agent Capabilities

**1. Multi-Model Planning Tool**
- Purpose: Analyze ideas and create structured strategy specifications
- Operations: Research extraction, strategy formulation, success criteria definition
- Key Features:
  - JSON-structured planning documents
  - Viability assessment
  - Technical requirement identification

**2. Iterative Implementation Tool**
- Purpose: Generate and refine backtesting code
- Operations: Code generation, debugging, optimization
- Key Features:
  - Multi-iteration support (up to 3)
  - Feedback incorporation
  - Code validation

**3. Performance Verification Tool**
- Purpose: Execute backtests and validate results
- Operations: Backtest execution, metric extraction, criteria checking
- Key Features:
  - Automated backtest running
  - Comprehensive metric calculation
  - Success criteria validation

**4. Marketplace Integration Tool**
- Purpose: Publish validated strategies
- Operations: Strategy registration, metadata management, performance tracking
- Key Features:
  - Automatic publishing
  - Performance data inclusion
  - Version control

### Supporting Features

**Configuration Management:**
- Customizable success criteria
- Model selection per phase
- Iteration limits

**Session Management:**
- Organized directory structure
- Full audit trail
- Result persistence

**Error Handling:**
- Graceful degradation
- Detailed error reporting
- Automatic retry logic

## 8. Technology Stack

**Core Technologies:**
- Python 3.10.9
- backtesting.py (strategy implementation)
- pandas/numpy (data manipulation)

**AI/ML Libraries:**
- anthropic (Claude models)
- openai (GPT models)
- Custom DeepSeek integration
- ModelFactory abstraction layer

**Dependencies:**
```python
# Required
pandas>=1.3.0
numpy>=1.21.0
backtesting>=0.3.3
termcolor>=1.1.0
python-dotenv>=0.19.0

# AI Model SDKs
anthropic>=0.3.0
openai>=1.0.0
# Custom implementations for other providers

# Technical Indicators
ta-lib>=0.4.0
pandas-ta>=0.3.0
```

**Third-party Integrations:**
- BirdEye API (market data)
- Moon Dev API (custom signals)
- Strategy Marketplace (internal)

## 9. Security & Configuration

**Configuration Management:**
```python
# Environment Variables (.env)
ANTHROPIC_KEY=xxx
OPENAI_KEY=xxx
DEEPSEEK_KEY=xxx

# Success Criteria Configuration
DEFAULT_SUCCESS_CRITERIA = {
    "min_return": 10.0,
    "min_sharpe": 0.5,
    "max_drawdown": -30.0,
    "min_trades": 10
}

# Model Configuration
MODEL_ASSIGNMENTS = {
    "planning": "gpt-4o",
    "implementation": "deepseek-coder",
    "verification": "claude-3-5-sonnet"
}
```

**Security Scope:**
- ✅ API key protection via environment variables
- ✅ Sandboxed code execution for backtests
- ✅ Input validation for strategy ideas
- ❌ Code signing/verification (future)
- ❌ Rate limiting for API calls (future)

## 10. API Specification

### Internal API Structure

```python
class RBIPIVAgent:
    def process_idea(self, idea: str) -> Dict[str, Any]:
        """
        Process a trading idea through PIV pipeline
        
        Args:
            idea: Strategy description/URL/PDF path
            
        Returns:
            {
                "success": bool,
                "strategy_id": str (if successful),
                "metrics": Dict[str, float],
                "iterations": int,
                "session_dir": str,
                "reason": str (if failed)
            }
        """
    
    def set_success_criteria(self, criteria: Dict[str, float]) -> None:
        """Update success criteria for validation"""
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Retrieve full session details including all iterations"""
```

### Integration Points

**Marketplace Registration:**
```python
# Automatic call after successful validation
registry.register_strategy(
    name=strategy_name,
    code_path=implementation_path,
    author="rbi_piv_agent",
    performance_data=metrics
)
```

## 11. Success Criteria

**MVP Success Definition:**
Successfully process diverse trading ideas through automated planning, implementation, and verification cycles, achieving an 85% success rate for strategies meeting predefined performance criteria.

**Functional Requirements:**
- ✅ Process at least 3 different input types (YouTube, PDF, text)
- ✅ Generate syntactically correct backtesting code
- ✅ Execute backtests without manual intervention
- ✅ Achieve 85% success rate for viable strategies
- ✅ Complete iteration cycles in under 15 minutes
- ✅ Automatically publish to marketplace

**Quality Indicators:**
- Code generation accuracy > 90%
- Iteration convergence within 3 cycles
- Performance improvement per iteration > 20%
- Zero manual intervention required

**User Experience Goals:**
- Single function call to process ideas
- Clear progress indication during iterations
- Comprehensive result reporting
- Seamless marketplace integration

## 12. Implementation Phases

### Phase 1: Core PIV Framework (Week 1-2)
**Goal:** Establish basic PIV loop with planning, implementation, and verification
**Deliverables:**
- ✅ RBIPIVAgent class structure
- ✅ Phase execution methods
- ✅ Basic iteration logic
- ✅ Success criteria checking
**Validation:** Successfully process simple moving average strategy

### Phase 2: Multi-Model Integration (Week 3-4)
**Goal:** Integrate multiple AI models for optimal phase performance
**Deliverables:**
- ✅ Model assignment per phase
- ✅ Unified response handling
- ✅ Error recovery mechanisms
- ✅ Performance tracking
**Validation:** Demonstrate model-specific advantages in each phase

### Phase 3: Marketplace Integration (Week 5)
**Goal:** Connect validated strategies to marketplace ecosystem
**Deliverables:**
- ✅ Automatic strategy registration
- ✅ Metadata generation
- ✅ Performance data inclusion
- ✅ Export/import compatibility
**Validation:** End-to-end flow from idea to published strategy

### Phase 4: Optimization & Polish (Week 6)
**Goal:** Enhance performance and user experience
**Deliverables:**
- ✅ Parallel processing options
- ✅ Advanced error handling
- ✅ Comprehensive logging
- ✅ Documentation and examples
**Validation:** Process 10 strategies with 85% success rate

## 13. Future Considerations

**Post-MVP Enhancements:**
- Multi-timeframe strategy testing
- Cross-asset validation
- Strategy combination algorithms
- Real-time performance tracking
- Community feedback integration

**Integration Opportunities:**
- Direct broker execution
- TradingView webhook support
- Discord/Telegram notifications
- CI/CD for strategy updates
- A/B testing framework

**Advanced Features:**
- Neural architecture search for ML strategies
- Genetic algorithms for parameter optimization
- Ensemble strategy creation
- Market regime adaptation
- Automated strategy deprecation

## 14. Risks & Mitigations

1. **Risk: AI Model Hallucinations**
   - Mitigation: Multi-model verification, strict output parsing, backtesting validation ensures code actually works

2. **Risk: Overfitting to Historical Data**
   - Mitigation: Multiple timeframe testing, out-of-sample validation periods, conservative default success criteria

3. **Risk: API Rate Limiting**
   - Mitigation: Request pooling, configurable delays, fallback to alternative models

4. **Risk: Infinite Iteration Loops**
   - Mitigation: Hard limit of 3 iterations, timeout controls, diminishing returns detection

5. **Risk: Poor Strategy Quality Despite Metrics**
   - Mitigation: Community rating system, real-world performance tracking, periodic criteria adjustment

## 15. Appendix

**Related Documents:**
- `/docs/strategy_marketplace_guide.md` - Marketplace user guide
- `/src/agents/rbi_agent.py` - Original RBI implementation
- `/CLAUDE.md` - AI assistant integration guide

**Key Dependencies:**
- [backtesting.py](https://github.com/kernc/backtesting.py) - Core backtesting engine
- [ModelFactory](/src/models/model_factory.py) - Multi-LLM abstraction
- [Strategy Registry](/src/agents/strategy_registry_agent.py) - Marketplace integration

**Performance Benchmarks:**
| Metric | Standard RBI | RBI-PIV |
|--------|-------------|---------|
| Success Rate | ~60% | ~85% |
| Avg Time | 5 min | 12 min |
| Quality Guarantee | No | Yes |
| Auto-publish | No | Yes |

**Configuration Examples:**
```python
# Conservative Configuration
agent.piv_state["success_criteria"] = {
    "min_return": 5.0,
    "min_sharpe": 1.0,
    "max_drawdown": -15.0,
    "min_trades": 20
}

# Aggressive Configuration
agent.piv_state["success_criteria"] = {
    "min_return": 30.0,
    "min_sharpe": 0.3,
    "max_drawdown": -40.0,
    "min_trades": 5
}
```