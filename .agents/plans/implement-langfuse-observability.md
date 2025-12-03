# Feature: LangFuse Observability Integration

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Integrate LangFuse observability platform into the Moon Dev AI Trading System to provide comprehensive tracing, monitoring, and analytics for all LLM interactions across 48+ specialized trading agents. This will enable debugging, cost tracking, performance optimization, and decision auditability for all AI-driven trading operations.

## User Story

As a trading system operator
I want to observe and analyze all LLM interactions across my agents
So that I can optimize performance, control costs, and debug trading decisions

## Problem Statement

The Moon Dev AI Trading System currently lacks centralized observability for its 48+ agents using multiple LLM providers (Claude, OpenAI, DeepSeek, etc.). This makes it difficult to:
- Debug why agents made specific trading decisions
- Track costs across different models and agents
- Identify performance bottlenecks
- Maintain audit trails for compliance
- Optimize prompt effectiveness

## Solution Statement

Implement LangFuse observability using a decorator-based approach that integrates with the existing ModelFactory pattern. This provides automatic tracing of all LLM calls with minimal code changes, capturing prompts, responses, latency, tokens, costs, and custom metadata like trading signals and confidence scores.

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: ModelFactory, BaseModel, all AI agents
**Dependencies**: langfuse Python SDK

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/models/base_model.py` (lines 36-62) - Why: Core generate_response method that needs LangFuse decorator
- `src/models/model_factory.py` (lines 1-50) - Why: Factory pattern for model creation, needs initialization logic
- `src/agents/base_agent.py` (lines 13-58) - Why: Base agent pattern, understand agent initialization
- `src/config.py` (lines 98-104) - Why: Configuration patterns for new settings
- `.env_example` (lines 30-38) - Why: Environment variable patterns for API keys
- `src/agents/liquidation_agent.py` (lines 100-120, 367-376) - Why: Example of direct LLM usage that needs migration
- `tests/test_portfolio_optimization.py` (lines 1-50) - Why: Testing patterns and pytest structure

### New Files to Create

- `src/observability/__init__.py` - Package initialization
- `src/observability/langfuse_tracker.py` - Core LangFuse integration and decorators
- `src/observability/context.py` - Context management for trace metadata
- `tests/test_langfuse_integration.py` - Unit tests for observability

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [LangFuse Python Decorators](https://langfuse.com/docs/sdk/python/decorators)
  - Specific section: @observe() decorator usage
  - Why: Core implementation pattern for automatic tracing
- [LangFuse OpenAI Integration](https://langfuse.com/guides/cookbook/integration_openai_sdk)
  - Specific section: Interoperability with manual tracing
  - Why: Shows how to integrate with existing OpenAI calls
- [LangFuse Context Management](https://langfuse.com/docs/observability/sdk/python/decorators#update-current-observation)
  - Specific section: langfuse_context.update_current_observation
  - Why: Required for adding custom metadata like trading signals

### Patterns to Follow

**Naming Conventions:**
```python
# From base_model.py - Moon Dev's style
"""
🌙 Moon Dev's Observability Module
Built with love by Moon Dev 🚀
"""

# Class names: PascalCase
class LangFuseTracker:
    pass

# Methods: snake_case
def initialize_client(self):
    pass
```

**Error Handling:**
```python
# From base_model.py line 58-62
try:
    # operation
except Exception as e:
    cprint(f"❌ Error message: {str(e)}", "red")
    return None
```

**Logging Pattern:**
```python
# From various agents - colorful terminal output
from termcolor import cprint
cprint("✅ Success message", "green")
cprint("⚠️ Warning message", "yellow")
cprint("❌ Error message", "red")
cprint("🤖 AI operation", "cyan")
```

**Configuration Pattern:**
```python
# From config.py - uppercase constants
ENABLE_LANGFUSE = True  # Feature flag
LANGFUSE_FLUSH_INTERVAL = 30  # Seconds between flushes
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Set up LangFuse infrastructure and core decorators

**Tasks:**
- Install langfuse dependency
- Create observability package structure
- Implement LangFuseTracker class with initialization
- Add environment configuration

### Phase 2: Core Implementation

Integrate LangFuse with BaseModel and ModelFactory

**Tasks:**
- Wrap BaseModel.generate_response with @observe decorator
- Add trace metadata context management
- Implement error handling and fallback behavior
- Add feature flag for enabling/disabling

### Phase 3: Integration

Connect observability to existing agent infrastructure

**Tasks:**
- Update model initialization in ModelFactory
- Add agent metadata to traces
- Migrate direct API calls to use ModelFactory
- Test with multiple model providers

### Phase 4: Testing & Validation

Comprehensive testing of observability features

**Tasks:**
- Unit tests for decorator functionality
- Integration tests with mock LangFuse
- Verify no performance regression
- Test feature flag on/off behavior

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task Format Guidelines

Use information-dense keywords for clarity:

- **CREATE**: New files or components
- **UPDATE**: Modify existing files
- **ADD**: Insert new functionality into existing code
- **REMOVE**: Delete deprecated code
- **REFACTOR**: Restructure without changing behavior
- **MIRROR**: Copy pattern from elsewhere in codebase

### UPDATE requirements.txt

- **IMPLEMENT**: Add langfuse==2.42.0 to requirements (latest stable version)
- **PATTERN**: Append to end of file like existing dependencies
- **IMPORTS**: langfuse
- **GOTCHA**: Ensure no version conflicts with existing packages
- **VALIDATE**: `pip install -r requirements.txt`

### UPDATE .env_example

- **IMPLEMENT**: Add LangFuse configuration variables after AI service keys (line 38)
```
# LangFuse Observability
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
ENABLE_LANGFUSE=true  # Set to false to disable observability
```
- **PATTERN**: Follow existing format with comments like lines 30-38
- **IMPORTS**: None
- **GOTCHA**: Keep consistent comment style with security reminders
- **VALIDATE**: `grep -n "LANGFUSE" .env_example`

### UPDATE src/config.py

- **IMPLEMENT**: Add LangFuse configuration section after AI Model Settings (line 104)
```python
# LangFuse Observability Settings 🔍
ENABLE_LANGFUSE = os.getenv('ENABLE_LANGFUSE', 'true').lower() == 'true'
LANGFUSE_FLUSH_INTERVAL = 30  # Seconds between batch flushes
LANGFUSE_CAPTURE_INPUT = True  # Capture prompts
LANGFUSE_CAPTURE_OUTPUT = True  # Capture responses
```
- **PATTERN**: Match existing config style with comments and emoji
- **IMPORTS**: Add `import os` if not present
- **GOTCHA**: Use string comparison for boolean env vars
- **VALIDATE**: `python -c "from src import config; print(config.ENABLE_LANGFUSE)"`

### CREATE src/observability/__init__.py

- **IMPLEMENT**: Package initialization with exports
```python
"""
🌙 Moon Dev's Observability Module
Built with love by Moon Dev 🚀
"""

from .langfuse_tracker import LangFuseTracker, observe_llm
from .context import ObservabilityContext

__all__ = ['LangFuseTracker', 'observe_llm', 'ObservabilityContext']
```
- **PATTERN**: Mirror style from src/models/__init__.py
- **IMPORTS**: Local modules
- **GOTCHA**: Use relative imports for package modules
- **VALIDATE**: `python -c "from src.observability import LangFuseTracker"`

### CREATE src/observability/langfuse_tracker.py

- **IMPLEMENT**: Core LangFuse integration with singleton pattern
```python
"""
🌙 Moon Dev's LangFuse Tracker
Built with love by Moon Dev 🚀

Provides observability for all AI model interactions
"""

import os
import functools
from typing import Optional, Any, Dict, Callable
from termcolor import cprint
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
from src import config

load_dotenv()

class LangFuseTracker:
    """Singleton LangFuse client for observability"""
    
    _instance: Optional['LangFuseTracker'] = None
    _client: Optional[Langfuse] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LangFuse client if not already done"""
        if self._client is None and config.ENABLE_LANGFUSE:
            try:
                secret_key = os.getenv('LANGFUSE_SECRET_KEY')
                public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
                host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                
                if not secret_key or not public_key:
                    cprint("⚠️ LangFuse keys not found in .env - observability disabled", "yellow")
                    return
                
                self._client = Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=host,
                    flush_interval=config.LANGFUSE_FLUSH_INTERVAL
                )
                
                cprint("✅ LangFuse observability initialized!", "green")
                
            except Exception as e:
                cprint(f"❌ Failed to initialize LangFuse: {str(e)}", "red")
                self._client = None
    
    @property
    def enabled(self) -> bool:
        """Check if LangFuse is enabled and initialized"""
        return config.ENABLE_LANGFUSE and self._client is not None
    
    def flush(self):
        """Manually flush pending traces"""
        if self._client:
            self._client.flush()

def observe_llm(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    as_type: str = "generation"
) -> Callable:
    """
    Decorator for observing LLM calls with LangFuse
    
    Args:
        name: Custom name for the trace
        capture_input: Whether to capture inputs
        capture_output: Whether to capture outputs
        as_type: Type of observation ("generation" for LLM calls)
    """
    tracker = LangFuseTracker()
    
    def decorator(func: Callable) -> Callable:
        # If LangFuse is disabled, return original function
        if not tracker.enabled:
            return func
        
        # Apply LangFuse observe decorator
        observed_func = observe(
            name=name or func.__name__,
            capture_input=capture_input and config.LANGFUSE_CAPTURE_INPUT,
            capture_output=capture_output and config.LANGFUSE_CAPTURE_OUTPUT,
            as_type=as_type
        )(func)
        
        return observed_func
    
    return decorator
```
- **PATTERN**: Singleton pattern like MoonDevAPI, decorator style from LangFuse docs
- **IMPORTS**: langfuse, config, os, dotenv
- **GOTCHA**: Check ENABLE_LANGFUSE before initializing to support feature flag
- **VALIDATE**: `python -c "from src.observability import LangFuseTracker; t = LangFuseTracker()"`

### CREATE src/observability/context.py

- **IMPLEMENT**: Context management for adding metadata to traces
```python
"""
🌙 Moon Dev's Observability Context
Built with love by Moon Dev 🚀

Manages trace metadata for agent operations
"""

from typing import Dict, Any, Optional
from langfuse.decorators import langfuse_context
from termcolor import cprint

class ObservabilityContext:
    """Helper for adding context to LangFuse traces"""
    
    @staticmethod
    def add_agent_metadata(
        agent_type: str,
        agent_name: str,
        **kwargs: Any
    ) -> None:
        """
        Add agent-specific metadata to current trace
        
        Args:
            agent_type: Type of agent (trading, risk, etc.)
            agent_name: Name of the agent class
            **kwargs: Additional metadata (trading_pair, exchange, etc.)
        """
        try:
            metadata = {
                'agent_type': agent_type,
                'agent_name': agent_name,
                **kwargs
            }
            
            langfuse_context.update_current_observation(
                metadata=metadata
            )
            
        except Exception as e:
            # Silently fail if no active trace
            pass
    
    @staticmethod
    def add_trading_signal(
        action: str,
        confidence: float,
        reasoning: str,
        symbol: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Add trading signal metadata to trace
        
        Args:
            action: BUY/SELL/NOTHING
            confidence: Confidence percentage (0-100)
            reasoning: AI's reasoning for the signal
            symbol: Trading symbol if applicable
            **kwargs: Additional signal data
        """
        try:
            signal_data = {
                'signal_action': action,
                'signal_confidence': confidence,
                'signal_reasoning': reasoning
            }
            
            if symbol:
                signal_data['trading_symbol'] = symbol
            
            signal_data.update(kwargs)
            
            langfuse_context.update_current_observation(
                metadata=signal_data,
                tags=[f"signal:{action.lower()}", f"confidence:{int(confidence)}"]
            )
            
        except Exception:
            pass
    
    @staticmethod
    def add_error(error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Add error information to trace
        
        Args:
            error: The exception that occurred
            context: Additional error context
        """
        try:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_context': context or {}
            }
            
            langfuse_context.update_current_observation(
                metadata=error_data,
                level="ERROR"
            )
            
        except Exception:
            pass
```
- **PATTERN**: Static methods like existing utility classes
- **IMPORTS**: langfuse_context from langfuse.decorators
- **GOTCHA**: Silently fail if no active trace to avoid disrupting operations
- **VALIDATE**: `python -c "from src.observability import ObservabilityContext; ObservabilityContext.add_agent_metadata('test', 'TestAgent')"`

### UPDATE src/models/base_model.py

- **IMPLEMENT**: Add observe_llm decorator to generate_response method
- **PATTERN**: Import at top, decorate method without changing logic
```python
# Add import after line 13
from src.observability import observe_llm, ObservabilityContext

# Replace generate_response method (lines 36-62) with decorated version
@observe_llm(name="llm_generation")
def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None):
    """Generate a response from the model with no caching"""
    try:
        # Add model metadata to trace
        ObservabilityContext.add_agent_metadata(
            agent_type='model',
            agent_name=self.model_type,
            model_name=getattr(self, 'model_name', 'unknown'),
            temperature=temperature,
            max_tokens=max_tokens if max_tokens else self.max_tokens
        )
        
        # Add random nonce to prevent caching
        nonce = f"_{random.randint(1, 1000000)}"
        current_time = int(time.time())
        
        # Each request will be unique
        unique_content = f"{user_content}_{nonce}_{current_time}"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"{system_prompt}_{current_time}"},
                {"role": "user", "content": unique_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens if max_tokens else self.max_tokens
        )
        
        return response.choices[0].message
        
    except Exception as e:
        if "503" in str(e):
            ObservabilityContext.add_error(e, {'error_type': 'rate_limit'})
            raise e  # Let the retry logic handle 503s
        ObservabilityContext.add_error(e)
        cprint(f"❌ Model error: {str(e)}", "red")
        return None
```
- **IMPORTS**: observe_llm, ObservabilityContext from src.observability
- **GOTCHA**: Preserve exact error handling behavior, especially 503 retry logic
- **VALIDATE**: `python -c "from src.models.claude_model import ClaudeModel; print('Import successful')"`

### UPDATE src/models/model_factory.py

- **IMPLEMENT**: Initialize LangFuse on factory import
```python
# Add after line 22
from src.observability import LangFuseTracker

# Add after line 23 (after imports)
# Initialize LangFuse tracker on module load
_langfuse_tracker = LangFuseTracker()
```
- **PATTERN**: Module-level initialization like existing patterns
- **IMPORTS**: LangFuseTracker
- **GOTCHA**: Use underscore prefix for module-private variable
- **VALIDATE**: `python -c "from src.models.model_factory import ModelFactory; print('LangFuse initialized')"`

### CREATE tests/test_langfuse_integration.py

- **IMPLEMENT**: Unit tests for LangFuse integration
```python
"""
Unit tests for LangFuse observability integration
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from src.observability import LangFuseTracker, observe_llm, ObservabilityContext
from src import config

class TestLangFuseIntegration:
    
    def test_tracker_singleton(self):
        """Test that LangFuseTracker is a singleton"""
        tracker1 = LangFuseTracker()
        tracker2 = LangFuseTracker()
        assert tracker1 is tracker2
    
    @patch.dict(os.environ, {'ENABLE_LANGFUSE': 'false'})
    def test_disabled_by_config(self):
        """Test that tracker respects feature flag"""
        # Force reload config
        import importlib
        importlib.reload(config)
        
        tracker = LangFuseTracker()
        assert not tracker.enabled
    
    def test_observe_llm_decorator_preserves_function(self):
        """Test that decorator doesn't break function"""
        @observe_llm(name="test_function")
        def dummy_function(x, y):
            return x + y
        
        result = dummy_function(1, 2)
        assert result == 3
    
    def test_context_methods_dont_raise(self):
        """Test that context methods handle errors gracefully"""
        # Should not raise even without active trace
        ObservabilityContext.add_agent_metadata("test", "TestAgent")
        ObservabilityContext.add_trading_signal("BUY", 85.5, "Test signal")
        ObservabilityContext.add_error(ValueError("Test error"))
    
    @patch('src.observability.langfuse_tracker.Langfuse')
    def test_initialization_with_valid_keys(self, mock_langfuse):
        """Test successful initialization with API keys"""
        with patch.dict(os.environ, {
            'LANGFUSE_SECRET_KEY': 'test_secret',
            'LANGFUSE_PUBLIC_KEY': 'test_public',
            'ENABLE_LANGFUSE': 'true'
        }):
            # Force reinitialization
            LangFuseTracker._instance = None
            LangFuseTracker._client = None
            
            tracker = LangFuseTracker()
            mock_langfuse.assert_called_once_with(
                secret_key='test_secret',
                public_key='test_public',
                host='https://cloud.langfuse.com',
                flush_interval=30
            )
```
- **PATTERN**: pytest style matching test_portfolio_optimization.py
- **IMPORTS**: pytest, unittest.mock, observability modules
- **GOTCHA**: Test both enabled and disabled states
- **VALIDATE**: `pytest tests/test_langfuse_integration.py -v`

### UPDATE src/agents/base_agent.py

- **IMPLEMENT**: Add observability context to base agent
```python
# Add import after line 11
from src.observability import ObservabilityContext

# Add method after line 57
def _set_observability_context(self):
    """Set observability context for this agent"""
    try:
        ObservabilityContext.add_agent_metadata(
            agent_type=self.type,
            agent_name=self.__class__.__name__,
            exchange=getattr(self, 'exchange', 'unknown')
        )
    except Exception:
        # Silently fail if observability not available
        pass
```
- **PATTERN**: Protected method with underscore prefix
- **IMPORTS**: ObservabilityContext
- **GOTCHA**: Use getattr for optional attributes
- **VALIDATE**: `python -c "from src.agents.base_agent import BaseAgent; print('Updated successfully')"`

### REFACTOR src/agents/liquidation_agent.py

- **IMPLEMENT**: Migrate from direct Anthropic client to ModelFactory (optional but recommended)
```python
# Replace lines 95-105 with ModelFactory initialization
from src.models.model_factory import ModelFactory

# In __init__ method
self.model = ModelFactory.create_model(
    model_type='anthropic',
    model_name=self.ai_model
)

# Replace direct client.messages.create (lines 367-376) with
response = self.model.generate_response(
    system_prompt="You are a liquidation analyst. You must respond in exactly 3 lines: BUY/SELL/NOTHING, reason, and confidence.",
    user_content=context,
    temperature=self.ai_temperature,
    max_tokens=self.ai_max_tokens
)
response_text = response.content if response else ""
```
- **PATTERN**: Use ModelFactory.create_model pattern
- **IMPORTS**: ModelFactory from src.models.model_factory
- **GOTCHA**: This is optional - can be done in separate PR
- **VALIDATE**: `python src/agents/liquidation_agent.py` (should initialize without errors)

---

## TESTING STRATEGY

Based on pytest patterns found in the project

### Unit Tests

Test observability components in isolation:
- Singleton pattern for LangFuseTracker
- Decorator preserves function behavior
- Feature flag enables/disables properly
- Context methods handle missing traces gracefully

### Integration Tests

Test with actual model calls:
- BaseModel.generate_response creates traces
- Metadata is properly attached
- Multiple providers work correctly
- Error conditions are traced

### Edge Cases

- LangFuse API keys missing
- Network failures to LangFuse
- Feature flag disabled mid-operation
- Concurrent agent execution

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/observability/*.py

# Verify imports work
python -c "from src.observability import LangFuseTracker, ObservabilityContext"
```

### Level 2: Unit Tests

```bash
# Run new observability tests
pytest tests/test_langfuse_integration.py -v

# Run all tests to ensure no regression
pytest tests/ -v
```

### Level 3: Integration Tests

```bash
# Test with feature enabled
ENABLE_LANGFUSE=true python -c "
from src.models.model_factory import ModelFactory
model = ModelFactory.create_model('anthropic')
print('✅ Model created with LangFuse enabled')
"

# Test with feature disabled
ENABLE_LANGFUSE=false python -c "
from src.models.model_factory import ModelFactory
model = ModelFactory.create_model('anthropic')
print('✅ Model created with LangFuse disabled')
"
```

### Level 4: Manual Validation

```bash
# Run a simple agent with observability
ENABLE_LANGFUSE=true python -c "
from src.agents.whale_agent import WhaleAgent
agent = WhaleAgent()
print('✅ Agent initialized with observability')
"

# Check LangFuse dashboard for traces (if API keys configured)
```

### Level 5: Additional Validation (Optional)

```bash
# Verify no performance regression
time python -c "
from src.models.model_factory import ModelFactory
for i in range(10):
    model = ModelFactory.create_model('anthropic')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] LangFuse integration works with feature flag
- [ ] All model providers create traces when enabled
- [ ] No errors when ENABLE_LANGFUSE=false
- [ ] Traces include agent metadata and context
- [ ] Trading signals are properly tagged
- [ ] No performance regression (< 50ms overhead)
- [ ] All existing tests still pass
- [ ] Can view traces in LangFuse dashboard
- [ ] Error conditions are properly traced

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms feature works
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability
- [ ] No regression in existing functionality
- [ ] Documentation updated in CLAUDE.md

---

## NOTES

### Design Decisions

1. **Singleton Pattern**: LangFuseTracker uses singleton to ensure single client instance and efficient batching
2. **Decorator Approach**: Using @observe_llm wrapper minimizes code changes across 48+ agents
3. **Feature Flag**: ENABLE_LANGFUSE allows quick enable/disable without code changes
4. **Silent Failures**: Observability errors don't disrupt trading operations
5. **Lazy Migration**: Direct API calls can be migrated to ModelFactory over time

### Trade-offs

1. **Performance**: ~30-50ms overhead per LLM call (acceptable for non-HFT operations)
2. **Storage**: LangFuse will store all prompts/responses (consider data retention policies)
3. **Cost**: LangFuse cloud has usage limits, may need self-hosting for scale

### Security Considerations

1. Never log private keys or sensitive wallet data
2. Consider self-hosted LangFuse for maximum data control
3. Use read-only API keys where possible
4. Implement data sanitization for sensitive prompts

### Future Enhancements

1. Custom dashboards for trading-specific metrics
2. Alerts for anomalous AI behavior
3. A/B testing different prompts
4. Cost optimization recommendations
5. Integration with backtesting for strategy evaluation