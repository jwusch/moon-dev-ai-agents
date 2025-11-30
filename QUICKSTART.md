# 🚀 Moon Dev AI Trading Agents - Quick Start Guide

## Prerequisites
- Python 3.10.9
- Conda installed
- Git

## 1️⃣ Environment Setup

```bash
# Navigate to project directory
cd /mnt/c/Users/jwusc/moon-dev-ai-agents

# Create/activate conda environment
conda create -n tflow python=3.10.9  # If not already created
conda activate tflow

# Install dependencies
pip install -r requirements.txt
pip install flask  # For marketplace dashboard
```

## 2️⃣ Configure API Keys

```bash
# Copy example environment file
cp .env_example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Required API keys (at least one AI provider):
- `ANTHROPIC_KEY` - For Claude models
- `OPENAI_KEY` - For GPT models  
- `DEEPSEEK_KEY` - For DeepSeek (recommended for RBI agent)
- `BIRDEYE_API_KEY` - For Solana market data
- `COINGECKO_API_KEY` - For crypto market data

## 3️⃣ Test the Strategy Marketplace

```bash
# Run the demo to populate marketplace with sample strategy
python src/scripts/demo_marketplace.py

# Start the marketplace dashboard
python src/scripts/run_marketplace_dashboard.py
```

### Access the Dashboard:
- **Option 1**: Direct WSL IP - http://172.18.154.77:8002
- **Option 2**: Windows port forward (run in PowerShell as Admin):
  ```powershell
  netsh interface portproxy add v4tov4 listenport=8002 listenaddress=0.0.0.0 connectport=8002 connectaddress=172.18.154.77
  ```
  Then visit: http://localhost:8002

- **Option 3**: Open static preview:
  ```bash
  # In Windows Explorer, open:
  C:\Users\jwusc\moon-dev-ai-agents\src\data\marketplace\marketplace_viewer.html
  ```

## 4️⃣ Run Individual Agents

### Strategy Marketplace Agent
```bash
python src/agents/strategy_registry_agent.py
```

### RBI Agent (Strategy Generator from YouTube/PDFs)
```bash
# Single strategy
python src/agents/rbi_agent.py

# Parallel processing (faster)
python src/agents/rbi_agent_pp_multi.py
```

### Risk Management Agent
```bash
python src/agents/risk_agent.py
```

### Other Popular Agents
```bash
# Sentiment analysis
python src/agents/sentiment_agent.py

# Whale tracking
python src/agents/whale_agent.py

# Chart analysis
python src/agents/chartanalysis_agent.py
```

## 5️⃣ Run the Main Orchestrator

To run multiple agents in a coordinated loop:

```bash
# Edit src/main.py to enable desired agents
# Set ACTIVE_AGENTS dictionary values to True

# Run the orchestrator
python src/main.py
```

## 6️⃣ Create Your First Strategy

### Option A: Use RBI Agent
Create a file `src/data/rbi_pp_multi/ideas.txt`:
```
Buy when RSI < 30 and MACD crosses up, sell when RSI > 70
```

Then run:
```bash
python src/agents/rbi_agent_pp_multi.py
```

### Option B: Manual Creation
1. Create strategy file in `src/strategies/`
2. Register with marketplace:
```python
from src.agents.strategy_registry_agent import StrategyRegistryAgent

registry = StrategyRegistryAgent()
registry.register_strategy(
    name="My Strategy",
    description="Description here",
    author="your_name",
    code_path="src/strategies/my_strategy.py",
    category=["momentum"],
    timeframes=["1H"],
    instruments=["BTC"],
    min_capital=100.0,
    risk_level="medium"
)
```

## 7️⃣ View Results

- **Backtest Results**: `src/data/rbi_pp_multi/backtest_stats.csv`
- **Strategy Files**: `src/data/marketplace/strategies/`
- **Performance Metrics**: `src/data/marketplace/metrics/`
- **Agent Outputs**: `src/data/[agent_name]/`

## 🛠️ Troubleshooting

### Port 8002 Already in Use
```bash
# Find process using port
lsof -i :8002
# Kill process if needed
kill -9 <PID>
```

### Module Import Errors
```bash
# Make sure you're in project root
cd /mnt/c/Users/jwusc/moon-dev-ai-agents
# Reinstall requirements
pip install -r requirements.txt
```

### WSL2 Connection Issues
1. Check Windows Firewall settings
2. Allow Python through firewall when prompted
3. Use the static HTML viewer as fallback

## 📚 Next Steps

1. **Read Documentation**:
   - `docs/strategy_marketplace_guide.md` - Full marketplace guide
   - `docs/strategy_marketplace_integration.md` - Integration with other agents
   - `README.md` - Project overview

2. **Watch Tutorials**:
   - YouTube playlist with weekly updates
   - Discord community for support

3. **Contribute**:
   - Share your strategies on the marketplace
   - Rate and review community strategies
   - Report issues on GitHub

## 🆘 Getting Help

- **Discord**: https://discord.gg/8UPuVZ53bh
- **YouTube**: Moon Dev channel
- **GitHub Issues**: Report bugs or request features

---

Built with ❤️ by Moon Dev - Democratizing AI-powered trading