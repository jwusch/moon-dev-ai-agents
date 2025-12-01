"""
🌙 Moon Dev's Observability Context
Built with love by Moon Dev 🚀

Manages trace metadata for agent operations
"""

from typing import Dict, Any, Optional
try:
    from langfuse.decorators import langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    # LangFuse not installed
    LANGFUSE_AVAILABLE = False
    langfuse_context = None
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
            
            if LANGFUSE_AVAILABLE and langfuse_context:
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
            
            if LANGFUSE_AVAILABLE and langfuse_context:
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
            
            if LANGFUSE_AVAILABLE and langfuse_context:
                langfuse_context.update_current_observation(
                    metadata=error_data,
                    level="ERROR"
                )
            
        except Exception:
            pass