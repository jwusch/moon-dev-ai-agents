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
    @patch('src.observability.langfuse_tracker.config.ENABLE_LANGFUSE', True)
    def test_initialization_with_valid_keys(self, mock_langfuse):
        """Test successful initialization with API keys"""
        with patch.dict(os.environ, {
            'LANGFUSE_SECRET_KEY': 'test_secret',
            'LANGFUSE_PUBLIC_KEY': 'test_public',
            'LANGFUSE_HOST': 'https://cloud.langfuse.com',  # Explicitly set host
            'ENABLE_LANGFUSE': 'true'
        }, clear=False):
            # Force complete reinitialization by creating new class instance
            class TestLangFuseTracker(LangFuseTracker):
                _instance = None
                _client = None
            
            tracker = TestLangFuseTracker()
            # Just verify that Langfuse was called with the right secret/public keys
            assert mock_langfuse.called
            call_kwargs = mock_langfuse.call_args[1]
            assert call_kwargs['secret_key'] == 'test_secret'
            assert call_kwargs['public_key'] == 'test_public'
            assert call_kwargs['flush_interval'] == 30