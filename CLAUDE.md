# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental AI trading system that orchestrates 48+ specialized AI agents to analyze markets, execute strategies, and manage risk across cryptocurrency markets (primarily Solana). The project uses a modular agent architecture with unified LLM provider abstraction supporting Claude, GPT-4, DeepSeek, Groq, Gemini, and local Ollama models.

## MCP (Model Context Protocol) Integration

This project uses Archon as an MCP server to provide access to a curated knowledge base and project management tools for enhanced AI assistance.

### MCP Configuration

Add this to your Claude Code configuration:

```bash
claude mcp add --transport sse archon http://localhost:8051/sse
```

Or for other MCP clients (Cursor, Windsurf), add to your MCP settings:

```json
{
  "mcpServers": {
    "archon": {
      "uri": "http://localhost:8051/sse"
    }
  }
}
```

### Available MCP Tools

**Knowledge & RAG (7 tools):**
- `perform_rag_query` - Search the knowledge base semantically for trading strategies, agent patterns, API documentation
- `search_code_examples` - Find relevant code snippets and implementation patterns
- `get_available_sources` - List available knowledge sources (trading docs, AI frameworks, etc.)
- `smart_crawl_url` - Add new documentation/websites to knowledge base
- `upload_document` - Add trading PDFs, research papers, strategy documents
- `crawl_single_page` - Index specific documentation pages
- `delete_source` - Remove outdated sources

**Project Management (5 tools):**
- `manage_project` - Create, list, get, delete projects for trading strategies
- `manage_task` - Track agent development, backtesting tasks, optimization goals
- `manage_document` - Version-controlled strategy documentation and research notes
- `manage_versions` - Version control for trading strategies and agent configurations
- `get_project_features` - Retrieve project capabilities and feature status

**System (2 tools):**
- `health_check` - Archon system diagnostics
- `session_info` - MCP session management

### Usage Guidelines for Trading Development

1. **Research trading strategies**: Use `perform_rag_query` to search for algorithmic trading patterns, risk management techniques, and market analysis methods
2. **Find agent implementation patterns**: Use `search_code_examples` to locate similar agent architectures, trading bot implementations, and backtesting frameworks
3. **Add trading knowledge**: Use `smart_crawl_url` to add quantitative trading documentation, API references, and research papers
4. **Track development progress**: Create projects for strategy development, agent optimization, and backtesting campaigns
5. **Document strategies**: Use document management to maintain strategy documentation, performance analysis, and research notes
6. **Keep knowledge current**: Regularly add new trading resources, market analysis tools, and algorithmic trading research

### Prerequisites

- Archon must be running locally (http://localhost:8051/sse should be accessible)
- MCP client properly configured with the SSE transport

## Key Development Commands

### Environment Setup
```bash
# Use existing conda environment (DO NOT create new virtual environments)
conda activate tflow

# Install/update dependencies
pip install -r requirements.txt

# IMPORTANT: Update requirements.txt every time you add a new package
pip freeze > requirements.txt
```

### Running the System
```bash
# Run main orchestrator (controls multiple agents)
python src/main.py

# Run individual agents standalone
python src/agents/trading_agent.py
python src/agents/risk_agent.py
python src/agents/rbi_agent.py
python src/agents/chat_agent.py
# ... any agent in src/agents/ can run independently
```

### Backtesting
```bash
# Use backtesting.py library with pandas_ta or talib for indicators
# Sample OHLCV data available at:
# /Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv
```

## Architecture Overview

### Core Structure
```
src/
├── agents/              # 48+ specialized AI agents (each <800 lines)
├── models/              # LLM provider abstraction (ModelFactory pattern)
├── strategies/          # User-defined trading strategies
├── scripts/             # Standalone utility scripts
├── data/                # Agent outputs, memory, analysis results
├── config.py            # Global configuration (positions, risk limits, API settings)
├── main.py              # Main orchestrator for multi-agent loop
├── nice_funcs.py        # ~1,200 lines of shared trading utilities
├── nice_funcs_hl.py     # Hyperliquid-specific utilities
└── ezbot.py             # Legacy trading controller
```

### Agent Ecosystem

**Trading Agents**: `trading_agent`, `strategy_agent`, `risk_agent`, `copybot_agent`
**Market Analysis**: `sentiment_agent`, `whale_agent`, `funding_agent`, `liquidation_agent`, `chartanalysis_agent`
**Content Creation**: `chat_agent`, `clips_agent`, `tweet_agent`, `video_agent`, `phone_agent`
**Strategy Development**: `rbi_agent` (Research-Based Inference - codes backtests from videos/PDFs), `research_agent`
**Specialized**: `sniper_agent`, `solana_agent`, `tx_agent`, `million_agent`, `tiktok_agent`, `compliance_agent`

Each agent can run independently or as part of the main orchestrator loop.

### LLM Integration (Model Factory)

Located at `src/models/model_factory.py` and `src/models/README.md`

**Unified Interface**: All agents use `ModelFactory.create_model()` for consistent LLM access
**Supported Providers**: Anthropic Claude (default), OpenAI, DeepSeek, Groq, Google Gemini, Ollama (local)
**Key Pattern**:
```python
from src.models.model_factory import ModelFactory

model = ModelFactory.create_model('anthropic')  # or 'openai', 'deepseek', 'groq', etc.
response = model.generate_response(system_prompt, user_content, temperature, max_tokens)
```

### LangFuse Observability

**Automatic LLM Tracing**: All LLM calls are automatically traced with LangFuse when enabled
**Zero Code Changes**: Observability works transparently via decorators - no agent modifications needed
**Rich Metadata**: Captures prompts, responses, latency, tokens, costs, and custom metadata

**Configuration** (in `.env`):
```bash
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
ENABLE_LANGFUSE=true  # Set to false to disable
```

**Adding Context** (optional):
```python
from src.observability import ObservabilityContext

# Add agent metadata
ObservabilityContext.add_agent_metadata(
    agent_type="trading",
    agent_name="MyAgent",
    exchange="solana"
)

# Track trading signals
ObservabilityContext.add_trading_signal(
    action="BUY",
    confidence=85.0,
    reasoning="Bullish pattern detected",
    symbol="BTC"
)
```

**Testing**: Run `python example_langfuse_demo.py` to verify integration

### Configuration Management

**Primary Config**: `src/config.py`
- Trading settings: `MONITORED_TOKENS`, `EXCLUDED_TOKENS`, position sizing (`usd_size`, `max_usd_order_size`)
- Risk management: `CASH_PERCENTAGE`, `MAX_POSITION_PERCENTAGE`, `MAX_LOSS_USD`, `MAX_GAIN_USD`, `MINIMUM_BALANCE_USD`
- Agent behavior: `SLEEP_BETWEEN_RUNS_MINUTES`, `ACTIVE_AGENTS` dict in `main.py`
- AI settings: `AI_MODEL`, `AI_MAX_TOKENS`, `AI_TEMPERATURE`

**Environment Variables**: `.env` (see `.env_example`)
- Trading APIs: `BIRDEYE_API_KEY`, `MOONDEV_API_KEY`, `COINGECKO_API_KEY`
- AI Services: `ANTHROPIC_KEY`, `OPENAI_KEY`, `DEEPSEEK_KEY`, `GROQ_API_KEY`, `GEMINI_KEY`
- Blockchain: `SOLANA_PRIVATE_KEY`, `HYPER_LIQUID_ETH_PRIVATE_KEY`, `RPC_ENDPOINT`

### Shared Utilities

**`src/nice_funcs.py`** (~1,200 lines): Core trading functions
- Data: `token_overview()`, `token_price()`, `get_position()`, `get_ohlcv_data()`
- Trading: `market_buy()`, `market_sell()`, `chunk_kill()`, `open_position()`
- Analysis: Technical indicators, PnL calculations, rug pull detection

**`src/agents/api.py`**: `MoonDevAPI` class for custom Moon Dev API endpoints
- `get_liquidation_data()`, `get_funding_data()`, `get_oi_data()`, `get_copybot_follow_list()`

### Data Flow Pattern

```
Config/Input → Agent Init → API Data Fetch → Data Parsing →
LLM Analysis (via ModelFactory) → Decision Output →
Result Storage (CSV/JSON in src/data/) → Optional Trade Execution
```

## Development Rules

### File Management
- **Keep files under 800 lines** - if longer, split into new files and update README
- **DO NOT move files without asking** - you can create new files but no moving
- **NEVER create new virtual environments** - use existing `conda activate tflow`
- **Update requirements.txt** after adding any new package

### Backtesting
- Use `backtesting.py` library (NOT their built-in indicators)
- Use `pandas_ta` or `talib` for technical indicators instead
- Sample data available at `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv`

### Code Style
- **No fake/synthetic data** - always use real data or fail the script
- **Minimal error handling** - user wants to see errors, not over-engineered try/except blocks
- **No API key exposure** - never show keys from `.env` in output

### Agent Development Pattern

When creating new agents:
1. Inherit from base patterns in existing agents
2. Use `ModelFactory` for LLM access
3. Store outputs in `src/data/[agent_name]/`
4. Make agent independently executable (standalone script)
5. Add configuration to `config.py` if needed
6. Follow naming: `[purpose]_agent.py`

### Testing Strategies

Place strategy definitions in `src/strategies/` folder:
```python
class YourStrategy(BaseStrategy):
    name = "strategy_name"
    description = "what it does"

    def generate_signals(self, token_address, market_data):
        return {
            "action": "BUY"|"SELL"|"NOTHING",
            "confidence": 0-100,
            "reasoning": "explanation"
        }
```

## Important Context

### Risk-First Philosophy
- Risk Agent runs first in main loop before any trading decisions
- Configurable circuit breakers (`MAX_LOSS_USD`, `MINIMUM_BALANCE_USD`)
- AI confirmation for position-closing decisions (configurable via `USE_AI_CONFIRMATION`)

### Data Sources
1. **BirdEye API** - Solana token data (price, volume, liquidity, OHLCV)
2. **Moon Dev API** - Custom signals (liquidations, funding rates, OI, copybot data)
3. **CoinGecko API** - 15,000+ token metadata, market caps, sentiment
4. **Helius RPC** - Solana blockchain interaction

### Autonomous Execution
- Main loop runs every 15 minutes by default (`SLEEP_BETWEEN_RUNS_MINUTES`)
- Agents handle errors gracefully and continue execution
- Keyboard interrupt for graceful shutdown
- All agents log to console with color-coded output (termcolor)

### AI-Driven Strategy Generation (RBI Agent)
1. User provides: YouTube video URL / PDF / trading idea text
2. DeepSeek-R1 analyzes and extracts strategy logic
3. Generates backtesting.py compatible code
4. Executes backtest and returns performance metrics
5. Cost: ~$0.027 per backtest execution (~6 minutes)

## Common Patterns

### Adding New Agent
1. Create `src/agents/your_agent.py`
2. Implement standalone execution logic
3. Add to `ACTIVE_AGENTS` in `main.py` if needed for orchestration
4. Use `ModelFactory` for LLM calls
5. Store results in `src/data/your_agent/`

### Switching AI Models
Edit `config.py`:
```python
AI_MODEL = "claude-3-haiku-20240307"  # Fast, cheap
# AI_MODEL = "claude-3-sonnet-20240229"  # Balanced
# AI_MODEL = "claude-3-opus-20240229"  # Most powerful
```

Or use different models per agent via ModelFactory:
```python
model = ModelFactory.create_model('deepseek')  # Reasoning tasks
model = ModelFactory.create_model('groq')      # Fast inference
```

### Reading Market Data
```python
from src.nice_funcs import token_overview, get_ohlcv_data, token_price

# Get comprehensive token data
overview = token_overview(token_address)

# Get price history
ohlcv = get_ohlcv_data(token_address, timeframe='1H', days_back=3)

# Get current price
price = token_price(token_address)
```

## Project Philosophy

This is an **experimental, educational project** demonstrating AI agent patterns through algorithmic trading:
- No guarantees of profitability (substantial risk of loss)
- Open source and free for learning
- YouTube-driven development with weekly updates
- Community-supported via Discord
- No token associated with project (avoid scams)

The goal is to democratize AI agent development and show practical multi-agent orchestration patterns that can be applied beyond trading.

### Fundamental Trading Principles

**Fractal Market Efficiency Principle**: Markets are fractally efficient - different alpha sources exist at different timescales, and the optimal strategy depends on finding the right temporal resolution where signal exceeds noise while maintaining tradeable alpha. Key insights:

- **1-minute data**: Ultra-fast reversions (11-min avg) but extremely noisy - requires nanosecond precision
- **5-minute data**: Contains real inefficiencies but fragile to filtering - "noise" is often tradeable alpha  
- **15-minute data**: Sweet spot - captures inefficiencies without overwhelming noise
- **1-hour+ data**: Clean signals but fewer opportunities

**Strategic Implications**:
- Never assume shorter timeframes are inherently better
- Avoid over-filtering signals - you may destroy the very inefficiencies you're trying to exploit
- Match strategy complexity to timeframe characteristics
- Test across multiple timeframes to find optimal temporal resolution
- Remember: Market microstructure creates real, tradeable inefficiencies at sub-minute scales
