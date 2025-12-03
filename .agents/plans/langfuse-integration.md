# LangFuse Integration Plan for Moon Dev AI Agents

## Overview
This document outlines the plan to integrate LangFuse observability into the Moon Dev AI Trading System, which consists of 48+ specialized AI agents using multiple LLM providers.

## What is LangFuse?
LangFuse is an open-source LLM observability platform that provides:
- **Tracing**: Track all LLM calls with full context
- **Metrics**: Monitor latency, costs, token usage
- **Debugging**: Inspect prompts, responses, and errors
- **Analytics**: Understand agent behavior and performance
- **Cost Tracking**: Monitor spending across different models/providers

## Integration Scope

### Current Architecture
1. **ModelFactory Pattern**: Centralized LLM access via `src/models/model_factory.py`
2. **Multiple Providers**: Claude, OpenAI, DeepSeek, Groq, Gemini, xAI, Ollama
3. **48+ Agents**: Each making independent LLM calls
4. **Base Agent Class**: Common functionality in `src/agents/base_agent.py`

### Integration Points
1. **Primary**: `BaseModel.generate_response()` method
2. **Secondary**: Direct API calls in some agents (needs refactoring)
3. **Swarm Mode**: Multi-model consensus in trading_agent.py

## Implementation Strategy

### Phase 1: Core Infrastructure
1. **LangFuse Client Setup**
   - Add langfuse package to requirements.txt
   - Create `src/observability/langfuse_client.py`
   - Environment configuration (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST)

2. **BaseModel Integration**
   - Wrap `generate_response()` with LangFuse tracing
   - Track: model, prompt, response, tokens, latency, cost
   - Add trace metadata: agent_name, agent_type, trading_pair

3. **Configuration Management**
   - Add to `.env_example`: LANGFUSE_* variables
   - Add to `config.py`: ENABLE_LANGFUSE flag
   - Support both cloud and self-hosted LangFuse

### Phase 2: Enhanced Tracking
1. **Agent-Level Tracing**
   - Add session/run tracking per agent execution
   - Link related LLM calls in a single trace
   - Track agent decisions and outcomes

2. **Custom Metadata**
   - Trading signals (BUY/SELL/NOTHING)
   - Confidence scores
   - Market conditions
   - PnL impact (when applicable)

3. **Error Tracking**
   - Capture and categorize LLM failures
   - Track rate limits and retries
   - Monitor timeout issues

### Phase 3: Advanced Features
1. **Performance Monitoring**
   - Agent response time dashboards
   - Model comparison metrics
   - Cost optimization insights

2. **Quality Assurance**
   - Prompt effectiveness scoring
   - Response quality metrics
   - Agent decision accuracy

## Technical Design

### 1. LangFuse Client Wrapper
```python
# src/observability/langfuse_client.py
class LangFuseTracker:
    def __init__(self):
        self.client = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
    
    def trace_llm_call(self, func):
        """Decorator for tracing LLM calls"""
        def wrapper(*args, **kwargs):
            # Start trace
            # Execute function
            # End trace with metadata
            pass
        return wrapper
```

### 2. BaseModel Integration
```python
# Modify src/models/base_model.py
from src.observability.langfuse_client import LangFuseTracker

class BaseModel:
    def __init__(self):
        self.tracker = LangFuseTracker() if config.ENABLE_LANGFUSE else None
    
    @trace_llm_call  # Decorator handles all tracking
    def generate_response(self, system_prompt, user_content, **kwargs):
        # Existing implementation
        pass
```

### 3. Agent Metadata
```python
# In each agent's LLM call
metadata = {
    'agent_type': self.agent_type,
    'agent_name': self.__class__.__name__,
    'trading_pair': symbol,
    'market_data': {...},
    'decision': 'BUY/SELL/NOTHING',
    'confidence': 85
}
```

## Migration Path

### Step 1: Non-Breaking Addition
- Add LangFuse without modifying existing behavior
- Use feature flag to enable/disable
- Test with single agent first

### Step 2: Gradual Rollout
- Enable for risk_agent (critical path)
- Add to trading_agent swarm mode
- Expand to all agents

### Step 3: Refactor Direct Calls
- Identify agents using direct API calls
- Migrate to ModelFactory pattern
- Ensure all calls are traced

## Benefits

1. **Debugging**: See exactly what prompts led to trading decisions
2. **Cost Control**: Track spending per agent and optimize
3. **Performance**: Identify slow models/agents
4. **Quality**: Monitor decision accuracy over time
5. **Compliance**: Full audit trail of AI decisions

## Risks & Mitigation

1. **Performance Impact**: 
   - Mitigation: Async logging, batching
   
2. **Data Privacy**:
   - Mitigation: Sanitize sensitive data, self-hosted option
   
3. **Complexity**:
   - Mitigation: Clean abstraction, minimal agent changes

## Success Metrics

1. 100% of LLM calls tracked
2. <50ms latency overhead
3. Zero impact on trading performance
4. Actionable insights within first week

## Timeline Estimate

- **Phase 1**: 2-3 days (core infrastructure)
- **Phase 2**: 3-4 days (enhanced tracking)
- **Phase 3**: 1 week (advanced features)
- **Total**: ~2 weeks for full implementation

## Next Steps

1. Confirm LangFuse account setup (cloud vs self-hosted)
2. Review and approve implementation plan
3. Create feature branch for development
4. Implement Phase 1 with risk_agent as pilot