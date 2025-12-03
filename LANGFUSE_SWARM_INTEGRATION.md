# 🔍 LangFuse Observability for SwarmAgent Consensus Building

## Overview

This document describes the LangFuse observability integration for the SwarmAgent multi-model consensus system. All SwarmAgent operations are now automatically tracked and traced for monitoring, debugging, and optimization.

## What's Tracked

### 1. Main Swarm Query (`swarm_consensus_query`)
- **Decorator**: `@observe_llm(name="swarm_consensus_query")`
- **Location**: `SwarmAgent.query()` method
- **Metadata Captured**:
  - `agent_type`: 'swarm'
  - `agent_name`: 'SwarmAgent'
  - `models_count`: Number of active models
  - `models_list`: List of all participating models
  - `consensus_enabled`: True

### 2. Individual Model Queries (`swarm_model_query`)
- **Decorator**: `@observe_llm(name="swarm_model_query")`
- **Location**: `SwarmAgent._query_single_model()` method
- **Metadata Captured**:
  - `agent_type`: 'swarm_model'
  - `agent_name`: 'SwarmModel_{provider}'
  - `model_provider`: Provider name (claude, openai, etc.)
  - `model_name`: Specific model version
  - `is_consensus_member`: True
- **Trading-Specific Tracking**:
  - Detects symbol validation queries
  - Extracts stock symbols from queries
  - Creates `SYMBOL_VALIDATION` trading signals

### 3. Consensus Generation (`swarm_consensus_generation`)
- **Decorator**: `@observe_llm(name="swarm_consensus_generation")`
- **Location**: `SwarmAgent._generate_consensus_review()` method
- **Metadata Captured**:
  - `agent_type`: 'swarm_consensus'
  - `agent_name`: 'ConsensusGenerator'
  - `successful_models`: Count of models that responded
  - `total_models`: Total models queried
  - `consensus_model`: Model used for consensus (e.g., 'deepseek-chat')

## Integration Points

### 1. Base Model Integration
All models inherit from `BaseModel` which has LangFuse observability built-in:
```python
@observe_llm(name="llm_generation")
def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None):
    # Automatic tracing of all LLM calls
```

### 2. Trading Signal Detection
For trading-related queries, additional metadata is captured:
```python
if is_trading_query:
    ObservabilityContext.add_trading_signal(
        action='SYMBOL_VALIDATION',
        confidence=0.0,
        reasoning='SwarmAgent symbol validation query',
        symbol=symbol_match.group(1)
    )
```

## Configuration

### Environment Variables (.env)
```bash
# LangFuse Configuration
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
ENABLE_LANGFUSE=true
```

### Config Settings (src/config.py)
```python
# LangFuse Configuration
ENABLE_LANGFUSE = os.getenv('ENABLE_LANGFUSE', 'true').lower() == 'true'
LANGFUSE_CAPTURE_INPUT = True  # Capture prompts
LANGFUSE_CAPTURE_OUTPUT = True  # Capture responses
LANGFUSE_FLUSH_INTERVAL = 1  # Flush traces every 1 second
```

## Usage in AEGS Discovery

When AEGS Discovery Agent uses SwarmAgent for symbol validation, the following trace hierarchy is created:

```
aegs_discovery_agent.run()
└── swarm_consensus_query (main orchestration)
    ├── swarm_model_query (claude_haiku)
    ├── swarm_model_query (claude_opus)
    ├── swarm_model_query (deepseek)
    ├── swarm_model_query (groq)
    ├── swarm_model_query (openrouter_gemini)
    ├── swarm_model_query (openrouter_llama)
    ├── swarm_model_query (xai)
    ├── swarm_model_query (ollama_gpt)
    ├── swarm_model_query (ollama_llama)
    └── swarm_consensus_generation (summary)
```

## Testing LangFuse Integration

Run the test script to verify integration:
```bash
python test_swarm_langfuse.py
```

This will:
1. Check if LangFuse is enabled
2. Run symbol validation queries
3. Run trading strategy queries
4. Show expected trace counts

## Viewing Traces

1. Go to your LangFuse dashboard
2. Look for traces with names:
   - `swarm_consensus_query`
   - `swarm_model_query`
   - `swarm_consensus_generation`
3. Check metadata for:
   - Model participation
   - Symbol validation signals
   - Consensus scores
   - Response times

## Benefits

1. **Performance Monitoring**: Track response times for each model
2. **Cost Tracking**: Monitor token usage across all models
3. **Quality Assurance**: Review consensus formation process
4. **Debugging**: Trace failures and timeouts
5. **Trading Insights**: Track symbol validation patterns
6. **Model Comparison**: Compare model performance in consensus

## Trace Examples

### Symbol Validation Trace
```json
{
  "name": "swarm_model_query",
  "metadata": {
    "agent_type": "swarm_model",
    "agent_name": "SwarmModel_claude_opus",
    "model_provider": "claude",
    "model_name": "claude-opus-4-5-20251101",
    "is_consensus_member": true,
    "trading_signal": {
      "action": "SYMBOL_VALIDATION",
      "symbol": "GME",
      "reasoning": "SwarmAgent symbol validation query"
    }
  }
}
```

### Consensus Generation Trace
```json
{
  "name": "swarm_consensus_generation",
  "metadata": {
    "agent_type": "swarm_consensus",
    "agent_name": "ConsensusGenerator",
    "successful_models": 8,
    "total_models": 9,
    "consensus_model": "deepseek-chat"
  }
}
```

## Best Practices

1. **Regular Monitoring**: Check LangFuse dashboard daily for anomalies
2. **Cost Optimization**: Use trace data to identify expensive models
3. **Performance Tuning**: Adjust timeouts based on model response times
4. **Quality Improvement**: Review consensus patterns for better prompts
5. **Error Tracking**: Monitor failed queries and timeout patterns

## Troubleshooting

### No Traces Appearing
1. Check `ENABLE_LANGFUSE=true` in config
2. Verify API keys in .env
3. Check network connectivity to LangFuse host
4. Run `python test_swarm_langfuse.py` to diagnose

### Missing Metadata
1. Ensure latest version of SwarmAgent with observability
2. Check ObservabilityContext imports
3. Verify decorators are properly applied

### Performance Impact
1. LangFuse adds minimal overhead (<50ms per trace)
2. Async flushing prevents blocking
3. Disable with `ENABLE_LANGFUSE=false` if needed

## Future Enhancements

1. **Consensus Quality Metrics**: Track agreement scores
2. **Model Reliability Scoring**: Based on timeout/error rates  
3. **Trading Performance Correlation**: Link validation accuracy to trading results
4. **Automated Alerting**: Trigger alerts on consensus failures
5. **A/B Testing**: Compare different consensus strategies

---

With this integration, all SwarmAgent consensus building is now fully observable through LangFuse, providing deep insights into the multi-model decision-making process used in AEGS discovery and symbol validation.