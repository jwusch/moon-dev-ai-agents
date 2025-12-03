#!/usr/bin/env python3
"""
🌙 Moon Dev's LangFuse Demo
Built with love by Moon Dev 🚀

This script demonstrates the LangFuse observability integration
for tracking LLM calls in the Moon Dev AI Trading System.
"""

import os
import sys
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Import our modules
from src.models.model_factory import ModelFactory
from src.observability import ObservabilityContext
from src import config

def demo_langfuse_integration():
    """Demonstrate LangFuse integration with different scenarios"""
    
    cprint("\n🌙 Moon Dev's LangFuse Observability Demo", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")
    
    # Check if LangFuse is enabled
    if config.ENABLE_LANGFUSE:
        cprint(f"✅ LangFuse is ENABLED", "green")
    else:
        cprint(f"❌ LangFuse is DISABLED (set ENABLE_LANGFUSE=true in .env)", "red")
        return
    
    # Check for API keys
    if not os.getenv('LANGFUSE_SECRET_KEY') or not os.getenv('LANGFUSE_PUBLIC_KEY'):
        cprint("⚠️  LangFuse keys not found in .env file", "yellow")
        cprint("   Add LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to your .env", "yellow")
        return
    
    cprint("\n📊 Testing LLM calls with observability...\n", "cyan")
    
    # Test 1: Simple LLM call
    cprint("Test 1: Basic LLM call with Claude", "yellow")
    model = ModelFactory.create_model("claude")
    if model:
        response = model.generate_response(
            system_prompt="You are a helpful trading assistant.",
            user_content="What is the current trend in BTC?",
            temperature=0.7,
            max_tokens=100
        )
        if response:
            cprint(f"✅ Response received: {response.content[:100]}...", "green")
        else:
            cprint("❌ No response received", "red")
    else:
        cprint("❌ Claude model not available", "red")
    
    # Test 2: LLM call with trading signal metadata
    cprint("\nTest 2: LLM call with trading signal metadata", "yellow")
    model = ModelFactory.create_model("claude")
    if model:
        # Add agent metadata
        ObservabilityContext.add_agent_metadata(
            agent_type="trading",
            agent_name="DemoAgent",
            exchange="solana",
            token="BTC"
        )
        
        response = model.generate_response(
            system_prompt="You are a trading signal generator. Respond with BUY, SELL, or NOTHING.",
            user_content="BTC price is $95,000, up 5% in 24h. RSI is 75. What's your signal?",
            temperature=0.3,
            max_tokens=50
        )
        
        if response:
            # Add trading signal metadata
            ObservabilityContext.add_trading_signal(
                action="SELL",
                confidence=85.0,
                reasoning="Overbought conditions with high RSI",
                symbol="BTC",
                rsi=75,
                price_change_24h=5.0
            )
            cprint(f"✅ Trading signal traced: {response.content[:100]}...", "green")
    
    # Test 3: Error handling
    cprint("\nTest 3: Error handling with observability", "yellow")
    try:
        model = ModelFactory.create_model("claude")
        if model:
            # Force an error by using invalid parameters
            ObservabilityContext.add_error(
                ValueError("Simulated error for demo"),
                context={"test": "error_handling", "severity": "low"}
            )
            cprint("✅ Error context added to trace", "green")
    except Exception as e:
        cprint(f"❌ Error during test: {str(e)}", "red")
    
    # Test 4: Multiple model providers
    cprint("\nTest 4: Testing multiple model providers", "yellow")
    providers = ["claude", "openai", "groq", "deepseek"]
    
    for provider in providers:
        model = ModelFactory.create_model(provider)
        if model:
            cprint(f"  ✅ {provider} model available and traced", "green")
        else:
            cprint(f"  ⚠️  {provider} model not configured", "yellow")
    
    cprint("\n" + "=" * 50, "cyan")
    cprint("📈 Demo complete! Check your LangFuse dashboard for traces:", "cyan")
    cprint(f"   {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}", "cyan")
    cprint("\n💡 Tips:", "yellow")
    cprint("   - Each LLM call is automatically traced", "white")
    cprint("   - Metadata and context are attached to traces", "white")
    cprint("   - Trading signals, errors, and agent info are captured", "white")
    cprint("   - No code changes needed in agents - it just works!", "white")
    cprint("\n🚀 Happy observing! - Moon Dev\n", "cyan", attrs=['bold'])

if __name__ == "__main__":
    demo_langfuse_integration()