"""
🌙 Moon Dev's LangFuse Tracker
Built with love by Moon Dev 🚀

Provides observability for all AI model interactions
"""

import os
import functools
from typing import Optional, Any, Dict, Callable
from termcolor import cprint
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    # LangFuse not installed
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None
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
        if self._client is None and config.ENABLE_LANGFUSE and LANGFUSE_AVAILABLE:
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
        return config.ENABLE_LANGFUSE and LANGFUSE_AVAILABLE and self._client is not None
    
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
        
        # Apply LangFuse observe decorator if available
        if LANGFUSE_AVAILABLE and observe:
            observed_func = observe(
                name=name or func.__name__,
                capture_input=capture_input and config.LANGFUSE_CAPTURE_INPUT,
                capture_output=capture_output and config.LANGFUSE_CAPTURE_OUTPUT,
                as_type=as_type
            )(func)
        else:
            observed_func = func
        
        return observed_func
    
    return decorator