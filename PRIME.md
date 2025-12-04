# PRIME.md - AI Context Documentation for Moon Dev AI Agents

This document provides comprehensive context for AI assistants working with the Moon Dev AI Agents for Trading codebase. It is designed to be the ultimate reference for understanding project structure, conventions, and workflows.

## 🌙 Project Overview

**Moon Dev AI Agents for Trading** is an experimental open-source project that orchestrates 48+ specialized AI agents to analyze markets, execute trading strategies, and manage risk across cryptocurrency markets (primarily Solana and HyperLiquid). The project demonstrates practical multi-agent orchestration patterns applicable beyond trading.

### Core Philosophy
- **Experimental & Educational**: This is a research project, not a guaranteed trading system
- **Open Source**: 100% free and open-source to democratize AI agent development
- **Modular Architecture**: Each agent is independent (<800 lines) and can run standalone
- **Risk-First**: Risk management is prioritized before any trading decisions
- **AI-Driven**: Leverages multiple LLM providers through a unified interface

### Key Principles
- **No Token**: This project has NO associated token and never will
- **No Guarantees**: No promises of profitability - success depends on user's strategy
- **Educational Focus**: Teaching AI agent patterns through practical trading examples
- **Community-Driven**: Weekly YouTube updates and active Discord community

## 🏗️ Architecture & Design Patterns

### Directory Structure
```
moon-dev-ai-agents/
├── src/                      # Source code
│   ├── agents/              # 48+ specialized AI agents
│   ├── models/              # LLM provider abstraction
│   ├── strategies/          # Trading strategy implementations
│   ├── data/                # Agent outputs and historical data
│   ├── observability/       # LangFuse integration
│   ├── config.py            # Global configuration
│   ├── main.py              # Main orchestrator
│   ├── nice_funcs.py        # Core trading utilities (~1,200 lines)
│   └── nice_funcs_*.py      # Exchange-specific utilities
├── docs/                     # Documentation for each agent
├── tests/                    # Test files
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── CLAUDE.md                # AI assistant guidance
```

### Core Design Patterns

#### 1. Model Factory Pattern
All LLM interactions use a unified interface through `ModelFactory`:
```python
from src.models.model_factory import ModelFactory

# Create model instance
model = ModelFactory.create_model('anthropic')  # or 'openai', 'deepseek', etc.

# Generate response
response = model.generate_response(system_prompt, user_content, temperature, max_tokens)
```

Supported providers: Anthropic Claude, OpenAI GPT, DeepSeek, Groq, Google Gemini, Ollama (local), xAI Grok, OpenRouter (200+ models)

#### 2. Agent Architecture
Each agent follows a consistent pattern:
- Inherits from base agent patterns
- Uses ModelFactory for LLM access
- Stores outputs in `src/data/[agent_name]/`
- Can run independently as a standalone script
- Follows naming convention: `[purpose]_agent.py`

#### 3. Data Flow Pattern
```
Config/Input → Agent Init → API Data Fetch → Data Parsing →
LLM Analysis (via ModelFactory) → Decision Output →
Result Storage (CSV/JSON in src/data/) → Optional Trade Execution
```

#### 4. Observability Pattern
LangFuse integration provides automatic LLM tracing:
- Zero code changes required - works via decorators
- Captures prompts, responses, latency, tokens, costs
- Configurable via environment variables

## 🔧 Key Components

### 1. Configuration (`src/config.py`)
Central configuration for all agents:
- **Exchange Settings**: `EXCHANGE` (solana/hyperliquid/aster)
- **Token Lists**: `MONITORED_TOKENS`, `EXCLUDED_TOKENS`
- **Position Sizing**: `usd_size`, `max_usd_order_size`
- **Risk Management**: `MAX_LOSS_USD`, `MAX_GAIN_USD`, `MINIMUM_BALANCE_USD`
- **AI Settings**: `AI_MODEL`, `AI_MAX_TOKENS`, `AI_TEMPERATURE`
- **Agent Behavior**: `SLEEP_BETWEEN_RUNS_MINUTES`

### 2. Main Orchestrator (`src/main.py`)
Controls multi-agent execution:
- Configurable agent activation via `ACTIVE_AGENTS` dict
- Sequential agent execution with error handling
- Continuous loop with configurable sleep intervals
- Graceful shutdown on keyboard interrupt

### 3. Shared Utilities (`src/nice_funcs.py`)
Core trading functions (~1,200 lines):
- **Data Functions**: `token_overview()`, `token_price()`, `get_ohlcv_data()`
- **Trading Functions**: `market_buy()`, `market_sell()`, `chunk_kill()`, `open_position()`
- **Analysis Functions**: Technical indicators, PnL calculations, rug pull detection
- **API Integration**: BirdEye, CoinGecko, Moon Dev custom APIs

### 4. Model Factory (`src/models/model_factory.py`)
Unified LLM interface supporting:
- Dynamic model switching without code changes
- Automatic availability detection based on API keys
- Consistent response format across all providers
- Built-in retry logic and error handling

### 5. Agent Categories

#### Backtesting & Research
- **RBI Agent**: Converts trading ideas (YouTube/PDF/text) to backtests using DeepSeek
- **Research Agent**: Generates strategy ideas for continuous backtesting
- **Websearch Agent**: Discovers trading strategies from web resources

#### Live Trading
- **Trading Agent**: AI-driven trading decisions with optional swarm consensus
- **Strategy Agent**: Executes pre-defined trading strategies
- **Risk Agent**: Portfolio risk management and circuit breakers
- **Swarm Agent**: Multi-model consensus for high-confidence decisions

#### Market Analysis
- **Whale Agent**: Monitors large trader activity
- **Sentiment Agent**: Twitter sentiment analysis
- **Funding Agent**: Funding rate arbitrage opportunities
- **Liquidation Agent**: Tracks liquidation events

#### Content Creation
- **Chat Agent**: YouTube live stream moderation
- **Video Agent**: AI video generation with Sora 2
- **Clips Agent**: Video clipping and editing
- **Tweet Agent**: AI-powered tweet generation

## 📋 Important Conventions

### File Management Rules
1. **Keep files under 800 lines** - split larger files
2. **Never move files without permission** - create new files instead
3. **Always use existing conda environment** - `conda activate tflow`
4. **Update requirements.txt** after adding packages

### Code Style Guidelines
1. **No synthetic/fake data** - always use real data or fail
2. **Minimal error handling** - let errors surface for debugging
3. **Never expose API keys** - use environment variables
4. **Color-coded console output** - use termcolor for clarity

### Agent Development Pattern
1. Create `src/agents/[purpose]_agent.py`
2. Implement standalone execution logic
3. Use ModelFactory for all LLM calls
4. Store outputs in `src/data/[agent_name]/`
5. Add to `ACTIVE_AGENTS` in main.py if needed

### Backtesting Requirements
- Use `backtesting.py` library (NOT built-in indicators)
- Use `pandas_ta` or `talib` for technical indicators
- Sample data: `/src/data/rbi/BTC-USD-15m.csv`
- Only save strategies with >1% return

## 🚀 Common Workflows

### 1. Setting Up Environment
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/moon-dev-ai-agents-for-trading.git
cd moon-dev-ai-agents-for-trading

# Create environment file
cp .env.example .env
# Add your API keys to .env

# Use existing conda environment
conda activate tflow
pip install -r requirements.txt
```

### 2. Running Backtests
```bash
# Single backtest
echo "Buy when RSI < 30, sell when RSI > 70" > src/data/rbi_pp_multi/ideas.txt
python src/agents/rbi_agent_pp_multi.py

# Web dashboard
cd src/data/rbi_pp_multi
python app.py
# Open http://localhost:8001
```

### 3. Running Live Agents
```bash
# Run main orchestrator
python src/main.py

# Run individual agents
python src/agents/trading_agent.py
python src/agents/risk_agent.py
python src/agents/sentiment_agent.py
```

### 4. Switching AI Models
```python
# In config.py
AI_MODEL = "claude-3-haiku-20240307"  # Fast
# AI_MODEL = "claude-3-sonnet-20240229"  # Balanced
# AI_MODEL = "claude-3-opus-20240229"  # Powerful

# Or per-agent
model = ModelFactory.create_model('deepseek')  # Reasoning
model = ModelFactory.create_model('groq')      # Fast
```

## 🔌 Technical Stack

### Core Dependencies
- **Python 3.10.9**: Primary development version
- **backtesting.py**: Backtesting framework
- **pandas/numpy**: Data manipulation
- **pandas_ta/talib**: Technical indicators
- **termcolor**: Console output formatting
- **python-dotenv**: Environment management
- **langfuse**: Observability and tracing

### API Integrations
- **Trading Data**: BirdEye (Solana), CoinGecko, Moon Dev Custom API
- **LLM Providers**: Anthropic, OpenAI, DeepSeek, Groq, Gemini, xAI, OpenRouter
- **Blockchain**: Solana (via Helius RPC), HyperLiquid
- **Social**: Twitter API (sentiment analysis)
- **Content**: Sora 2 (video generation)

### Exchange Support
- **Solana**: Primary DEX trading via BirdEye API
- **HyperLiquid**: Perpetual futures trading
- **Aster**: Alternative exchange (in development)

## ⚙️ Configuration Details

### Environment Variables (.env)
```bash
# AI Model APIs (need at least one)
ANTHROPIC_KEY=your_key
OPENAI_KEY=your_key
DEEPSEEK_KEY=your_key
GROQ_API_KEY=your_key
GEMINI_KEY=your_key
GROK_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# Market Data APIs
BIRDEYE_API_KEY=your_key
COINGECKO_API_KEY=your_key
MOONDEV_API_KEY=your_key

# Blockchain
SOLANA_PRIVATE_KEY=your_key
HYPER_LIQUID_ETH_PRIVATE_KEY=your_key
RPC_ENDPOINT=your_endpoint

# Observability (optional)
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_HOST=https://cloud.langfuse.com
ENABLE_LANGFUSE=true
```

### Key Configuration Parameters
- **Position Sizing**: `usd_size=25`, `max_usd_order_size=3`
- **Risk Limits**: `MAX_LOSS_USD=25`, `MAX_GAIN_USD=25`
- **Safety Buffers**: `CASH_PERCENTAGE=20`, `MAX_POSITION_PERCENTAGE=30`
- **Timing**: `SLEEP_BETWEEN_RUNS_MINUTES=15`, `tx_sleep=30`
- **Market Making**: `buy_under`, `sell_over` thresholds

## 🧪 Testing & Deployment

### Backtesting Best Practices
1. Always backtest before live trading
2. Test across multiple timeframes and assets
3. Verify strategy logic by reading generated code
4. Use realistic position sizes in tests
5. Account for slippage and fees

### Deployment Considerations
1. Start with paper trading or small positions
2. Monitor logs for errors and performance
3. Use risk agent to enforce limits
4. Keep API keys secure and rotate regularly
5. Monitor API rate limits and costs

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify keys in .env file
   - Check API key permissions
   - Ensure no trailing spaces

2. **Import Errors**
   - Run from project root directory
   - Verify conda environment: `conda activate tflow`
   - Update requirements: `pip install -r requirements.txt`

3. **Data Errors**
   - Check internet connection
   - Verify API endpoints are accessible
   - Clear temp_data folder if corrupted

4. **Memory Issues**
   - Reduce MAX_WORKERS in parallel agents
   - Limit backtesting timeframe
   - Monitor system resources

### Debug Mode
Most agents support debug flags:
```python
DEBUG_BACKTEST_ERRORS = True  # Auto-fix coding errors
DEBUG_MODE = True  # Verbose logging
```

## 🌟 Advanced Topics

### Multi-Model Consensus (Swarm Mode)
Trading agent supports querying multiple models:
```python
USE_SWARM_MODE = True  # In config.py
# Queries: Claude, GPT, Gemini, Grok, DeepSeek, Local Ollama
# Returns majority vote decision
```

### Custom Strategy Development
1. Create strategy in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` method
4. Return action, confidence, reasoning

### Adding New Agents
1. Study existing agent patterns
2. Create new file in `src/agents/`
3. Use ModelFactory for LLM calls
4. Implement standalone execution
5. Document in `docs/` folder

### Performance Optimization
- Use Groq for fast inference
- Use DeepSeek for cost-effective reasoning
- Cache API responses when appropriate
- Batch operations where possible

## 📚 Resources & Support

### Official Resources
- **Documentation**: This file and `/docs` folder
- **YouTube**: Weekly updates and tutorials
- **Discord**: Community support and discussions
- **GitHub**: Issue tracking and contributions

### Learning Path
1. Start with RBI backtesting agent
2. Understand the data flow pattern
3. Run individual agents standalone
4. Experiment with different AI models
5. Develop custom strategies
6. Contribute improvements

### Best Practices Summary
- Always backtest first
- Start small with real money
- Monitor performance continuously
- Keep learning and adapting
- Share knowledge with community

## ⚠️ Final Reminders

1. **This is experimental software** - no guarantees of profitability
2. **Trading involves substantial risk** - only risk what you can afford to lose
3. **Success depends on your strategy** - the tools are just enablers
4. **Keep your API keys secure** - never commit .env files
5. **Contribute back** - share improvements with the community

---

Built with love by Moon Dev 🌙 - Pioneering the future of AI-powered trading through open-source education