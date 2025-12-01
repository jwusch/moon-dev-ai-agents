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