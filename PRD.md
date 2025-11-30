# Product Requirements Document: Moon Dev AI Agents for Trading

## 1. Executive Summary

Moon Dev AI Agents for Trading is an experimental, open-source AI trading system that orchestrates 48+ specialized AI agents to analyze cryptocurrency markets, execute trading strategies, and manage risk. The platform demonstrates practical multi-agent AI patterns through algorithmic trading, supporting multiple exchanges (primarily Solana DEX and HyperLiquid) with a unified LLM abstraction layer that integrates 8+ AI providers.

The core value proposition is democratizing AI agent development by providing a modular, educational framework where each agent can operate independently or as part of an orchestrated system. The platform emphasizes risk management, real-data backtesting, and community-driven development through weekly YouTube updates and Discord support.

The MVP goal is to provide a fully functional multi-agent trading system that allows users to research strategies through AI (via YouTube videos, PDFs, or text), automatically generate and backtest trading algorithms, and optionally execute trades with comprehensive risk management controls.

## 2. Mission

**Product Mission Statement:**  
To democratize AI agent development and demonstrate practical multi-agent orchestration patterns that can be applied beyond trading, while providing an educational, open-source platform for algorithmic trading experimentation.

**Core Principles:**
- ✅ **Education First**: All code is open-source and free for learning
- ✅ **Risk Management**: Risk agent always runs first before any trading decisions
- ✅ **Real Data Only**: No synthetic or fake data - use actual market data or fail
- ✅ **Modular Architecture**: Each agent must be independently executable and under 800 lines
- ✅ **Community Driven**: Weekly YouTube updates and active Discord support

## 3. Target Users

**Primary User Personas:**

1. **Algorithmic Trading Learners**
   - Technical comfort: Intermediate Python developers
   - Needs: Learn AI agent patterns, understand algorithmic trading
   - Pain points: Lack of practical examples, complex trading systems

2. **Quantitative Researchers**
   - Technical comfort: Advanced developers/researchers
   - Needs: Rapid strategy prototyping, multi-model backtesting
   - Pain points: Time-consuming backtest development, limited AI integration

3. **AI Developer Community**
   - Technical comfort: Experienced programmers
   - Needs: Real-world multi-agent examples, LLM integration patterns
   - Pain points: Lack of production-ready agent orchestration examples

## 4. MVP Scope

### In Scope (Core Functionality)
**Trading & Execution:**
- ✅ Multi-agent orchestration with 48+ specialized agents
- ✅ Unified LLM abstraction supporting 8+ AI providers
- ✅ Automated backtesting from YouTube/PDF/text sources
- ✅ Live trading on Solana DEX
- ✅ HyperLiquid perpetuals integration
- ✅ Risk management with configurable circuit breakers

**Research & Analysis:**
- ✅ RBI Agent for AI-driven strategy generation
- ✅ Parallel backtesting across 20+ data sources
- ✅ Web dashboard for backtest management
- ✅ Swarm consensus trading (6-model voting)
- ✅ Technical indicator integration (pandas_ta/talib)

**Data & Integration:**
- ✅ Real-time market data from BirdEye, CoinGecko, Moon Dev APIs
- ✅ CSV/JSON output storage per agent
- ✅ Voice alerts for important events
- ✅ Configurable via environment variables

### Out of Scope (Future Phases)
**Technical:**
- ❌ Mobile application
- ❌ Cloud deployment infrastructure
- ❌ Automated portfolio rebalancing
- ❌ Machine learning model training

**Integration:**
- ❌ Traditional stock market integration
- ❌ Forex trading support
- ❌ Options/derivatives beyond perps
- ❌ Centralized exchange integration (beyond HyperLiquid)

## 5. User Stories

1. **"As a trading learner, I want to provide a YouTube video URL of a trading strategy, so that the system can automatically code and backtest it for me"**
   - Example: User pastes TradingView strategy video, RBI agent generates Python backtest

2. **"As a quant researcher, I want to test my strategy across multiple timeframes and assets simultaneously, so that I can validate robustness"**
   - Example: Single strategy tested on BTC, ETH, SOL across 15m, 1H, 4H timeframes

3. **"As a risk-conscious trader, I want the system to automatically stop trading when I hit my loss limit, so that I protect my capital"**
   - Example: Risk agent halts all trading when $25 USD loss limit reached

4. **"As an AI developer, I want to query multiple LLMs for consensus on trades, so that I reduce single-model bias"**
   - Example: Swarm agent queries Claude, GPT-5, Gemini for majority vote

5. **"As a strategy developer, I want to see which tokens have high whale activity, so that I can follow smart money"**
   - Example: Whale agent monitors and announces when whales enter positions

6. **"As a content creator, I want AI to analyze my live stream chat and respond to questions, so that I can focus on streaming"**
   - Example: Chat agent moderates YouTube live chat with context-aware responses

## 6. Core Architecture & Patterns

**High-Level Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   Main Orchestrator                     │
│                    (src/main.py)                       │
└─────────────────┬───────────────────────────┬──────────┘
                  │                           │
        ┌─────────▼──────────┐      ┌────────▼─────────┐
        │   Agent Ecosystem  │      │  Model Factory   │
        │  (48+ agents)      │      │  (8+ providers)  │
        └─────────┬──────────┘      └────────┬─────────┘
                  │                           │
        ┌─────────▼──────────────────────────▼─────────┐
        │           Shared Utilities                    │
        │    (nice_funcs.py, exchange_manager.py)      │
        └─────────────────┬─────────────────────────────┘
                          │
                ┌─────────▼─────────┐
                │   External APIs   │
                │ BirdEye,CoinGecko │
                └───────────────────┘
```

**Directory Structure:**
```
src/
├── agents/              # Individual AI agents (<800 lines each)
├── models/              # LLM provider implementations
├── strategies/          # User trading strategies
├── data/               # Agent outputs and cache
├── scripts/            # Utility scripts
├── config.py           # Global configuration
└── main.py             # Orchestrator entry point
```

**Key Design Patterns:**
- **Factory Pattern**: ModelFactory for unified LLM access
- **Agent Pattern**: Base agent class for consistent interface
- **Strategy Pattern**: Pluggable trading strategies
- **Observer Pattern**: Event-driven agent communication

## 7. Tools/Features

### Core Agent Categories

**Trading Execution Agents:**
- **Trading Agent**: DUAL-MODE system with single model (10s) or swarm consensus (45-60s)
- **Strategy Agent**: Executes user-defined strategies from strategies folder
- **Risk Agent**: Enforces position limits, PnL thresholds, minimum balance
- **CopyBot Agent**: Monitors and potentially copies successful traders

**Market Analysis Agents:**
- **Sentiment Agent**: Twitter sentiment analysis with voice alerts
- **Whale Agent**: Tracks large wallet movements
- **Funding Agent**: Monitors funding rates for arbitrage opportunities
- **Liquidation Agent**: Tracks liquidation spikes with configurable windows
- **Chart Analysis Agent**: AI-powered technical analysis

**Research & Backtesting:**
- **RBI Agent**: YouTube/PDF → Trading strategy → Backtest code
- **RBI Parallel**: 18-thread parallel testing across 20+ data sources
- **Research Agent**: Generates strategy ideas for continuous testing
- **WebSearch Agent**: Finds trading strategies online

**Content Creation:**
- **Chat Agent**: YouTube live stream moderation and response
- **Tweet Agent**: AI-generated trading commentary
- **Video Agent**: Sora 2 API integration for video generation
- **Clips Agent**: Long video → Short clips automation

## 8. Technology Stack

**Backend Technologies:**
- Python 3.10.9
- Conda environment (tflow)
- Async/await for concurrent operations

**AI/ML Libraries:**
- anthropic (Claude API)
- openai (GPT models)
- google-generativeai (Gemini)
- groq (Fast inference)
- deepseek-api
- ollama (Local models)

**Trading Libraries:**
- backtesting.py (core backtesting engine)
- pandas_ta (technical indicators)
- talib (additional indicators)
- ccxt (exchange connectivity)

**Data & APIs:**
- requests (HTTP client)
- websockets (Real-time data)
- pandas (Data manipulation)
- numpy (Numerical operations)

**Utilities:**
- python-dotenv (Environment management)
- termcolor (Console output)
- pyyaml (Configuration)
- Pillow (Image processing)

## 9. Security & Configuration

**Authentication/Authorization:**
- API keys stored in `.env` file (never committed)
- Each agent validates required keys on initialization
- Graceful degradation if APIs unavailable

**Configuration Management:**
```python
# Environment Variables (.env)
ANTHROPIC_KEY=xxx          # Claude API
OPENAI_KEY=xxx            # OpenAI GPT
DEEPSEEK_KEY=xxx          # DeepSeek
GROQ_API_KEY=xxx          # Groq
SOLANA_PRIVATE_KEY=xxx    # Wallet key
BIRDEYE_API_KEY=xxx       # Market data

# config.py settings
EXCHANGE = 'solana'       # or 'hyperliquid'
MAX_LOSS_USD = 25         # Risk limits
MINIMUM_BALANCE_USD = 50  # Safety buffer
```

**Security Scope:**
- ✅ API key protection via environment variables
- ✅ No hardcoded credentials
- ✅ Input validation for trading amounts
- ❌ Multi-signature wallets (future)
- ❌ Hardware wallet integration (future)

## 10. API Specification

### Internal Agent API Pattern
```python
class BaseAgent:
    def __init__(self):
        """Initialize agent with required APIs"""
        
    def run(self):
        """Main execution loop"""
        
    def analyze(self, data):
        """Process data and return decisions"""
```

### External API Integration
**BirdEye API:**
- Endpoint: `https://api.birdeye.so/`
- Purpose: Solana token prices, volume, liquidity
- Auth: API key in headers

**Moon Dev API:**
- Custom endpoints for liquidations, funding, OI data
- WebSocket support for real-time updates

## 11. Success Criteria

**MVP Success Definition:**
Successfully demonstrate a working multi-agent trading system that can research strategies, backtest them automatically, and optionally execute trades with proper risk management.

**Functional Requirements:**
- ✅ At least 10 agents operational and tested
- ✅ Successful strategy generation from YouTube URL
- ✅ Parallel backtesting completing in <10 minutes
- ✅ Risk agent preventing losses beyond configured limits
- ✅ Clean agent orchestration without race conditions

**Quality Indicators:**
- Code coverage >80% for critical paths
- Agent response time <5 seconds for decisions
- Backtest accuracy matching manual calculations
- Zero API key exposures in logs

**User Experience Goals:**
- Single command to start system
- Clear console output with color coding
- Intuitive web dashboard for backtests
- Voice alerts for important events

## 12. Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal:** Establish core architecture and basic agents
**Deliverables:**
- ✅ Model factory with 3+ LLM providers
- ✅ Base agent class implementation
- ✅ Risk and trading agent prototypes
- ✅ Basic backtesting integration
**Validation:** Successfully backtest simple RSI strategy

### Phase 2: Agent Ecosystem (Weeks 5-8)
**Goal:** Build out specialized agents
**Deliverables:**
- ✅ 15+ operational agents
- ✅ RBI agent with YouTube support
- ✅ Parallel processing implementation
- ✅ Web dashboard for monitoring
**Validation:** Generate profitable strategy from video

### Phase 3: Production Features (Weeks 9-12)
**Goal:** Add production-ready features
**Deliverables:**
- ✅ Swarm consensus trading
- ✅ Voice alerts and notifications
- ✅ Advanced risk management
- ✅ Performance optimization
**Validation:** Run live for 1 week without manual intervention

### Phase 4: Community & Scale (Weeks 13-16)
**Goal:** Open source release and documentation
**Deliverables:**
- ✅ Comprehensive documentation
- ✅ Video tutorials
- ✅ Discord community setup
- ✅ Contribution guidelines
**Validation:** 100+ GitHub stars, active community

## 13. Future Considerations

**Post-MVP Enhancements:**
- Machine learning for strategy optimization
- Cloud deployment with Kubernetes
- Mobile app for monitoring
- Traditional market integration
- Advanced portfolio optimization

**Integration Opportunities:**
- TradingView webhook support
- MetaTrader bridge
- Telegram bot interface
- API service for third-party access
- Strategy marketplace

**Advanced Features:**
- Reinforcement learning agents
- Natural language strategy definition
- Automated strategy discovery
- Cross-market arbitrage
- Social trading features

## 14. Risks & Mitigations

1. **Risk: API Rate Limiting**
   - Mitigation: Implement exponential backoff, cache responses, use multiple API keys

2. **Risk: Trading Losses**
   - Mitigation: Strict risk limits, paper trading mode, gradual position sizing

3. **Risk: Code Complexity**
   - Mitigation: 800-line limit per agent, modular design, comprehensive testing

4. **Risk: LLM Hallucinations**
   - Mitigation: Swarm consensus, validation layers, real-data constraints

5. **Risk: Security Breaches**
   - Mitigation: Environment variables, no credential logging, regular audits

## 15. Appendix

**Related Documents:**
- `/docs/` - Individual agent documentation
- `/CLAUDE.md` - AI assistant integration guide
- `/README.md` - Quick start guide
- `/.env_example` - Environment template

**Key Dependencies:**
- [backtesting.py](https://github.com/kernc/backtesting.py) - Core backtesting engine
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical indicators
- [Anthropic API](https://docs.anthropic.com) - Claude documentation
- [BirdEye API](https://docs.birdeye.so) - Solana market data

**Repository Structure:**
```
moon-dev-ai-agents/
├── src/               # Source code
├── docs/              # Documentation
├── .claude/           # AI assistant configs
├── requirements.txt   # Python dependencies
├── .env_example      # Environment template
└── PRD.md            # This document
```