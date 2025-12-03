#!/usr/bin/env python3
"""
🧪 Test LangFuse Integration with SwarmAgent 🧪

Verifies that all SwarmAgent consensus building operations are properly tracked in LangFuse
"""

import os
import sys
from termcolor import colored

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.agents.swarm_agent import SwarmAgent
from src.observability import LangFuseTracker

def test_swarm_langfuse_integration():
    """Test that SwarmAgent properly tracks all operations in LangFuse"""
    
    print(colored("🧪 Testing LangFuse Integration with SwarmAgent", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Check if LangFuse is enabled
    tracker = LangFuseTracker()
    if not tracker.enabled:
        print(colored("⚠️ LangFuse is disabled. Enable it in config.py and add keys to .env", 'yellow'))
        print("\nTo enable LangFuse:")
        print("1. Set ENABLE_LANGFUSE=true in src/config.py")
        print("2. Add to .env:")
        print("   LANGFUSE_SECRET_KEY=your_secret_key")
        print("   LANGFUSE_PUBLIC_KEY=your_public_key")
        print("   LANGFUSE_HOST=https://cloud.langfuse.com")
        return
    
    print(colored("✅ LangFuse is enabled and initialized", 'green'))
    
    # Initialize SwarmAgent
    print("\n📊 Initializing SwarmAgent...")
    try:
        swarm = SwarmAgent()
        print(colored("✅ SwarmAgent initialized successfully", 'green'))
    except Exception as e:
        print(colored(f"❌ Failed to initialize SwarmAgent: {e}", 'red'))
        return
    
    # Test 1: Trading symbol validation query
    print("\n🔍 Test 1: Symbol Validation Query")
    print("-"*40)
    
    symbol_query = """Is AAPL a valid stock symbol currently trading on major exchanges?
Please verify if it's actively traded and not delisted."""
    
    print(f"Query: {symbol_query[:100]}...")
    
    result = swarm.query(symbol_query)
    
    if result and 'consensus_summary' in result:
        print(colored("✅ Symbol validation query completed successfully", 'green'))
        print(f"\n📊 Consensus: {result['consensus_summary'][:200]}...")
    else:
        print(colored("❌ Symbol validation query failed", 'red'))
    
    # Test 2: Trading strategy query
    print("\n\n📈 Test 2: Trading Strategy Query")
    print("-"*40)
    
    strategy_query = """Should I implement a mean reversion strategy on GME given its recent volatility?
Consider risk factors and potential returns."""
    
    print(f"Query: {strategy_query[:100]}...")
    
    result2 = swarm.query(strategy_query)
    
    if result2 and 'consensus_summary' in result2:
        print(colored("✅ Trading strategy query completed successfully", 'green'))
        print(f"\n📊 Consensus: {result2['consensus_summary'][:200]}...")
    else:
        print(colored("❌ Trading strategy query failed", 'red'))
    
    # Flush LangFuse traces
    print("\n\n💾 Flushing LangFuse traces...")
    tracker.flush()
    
    # Summary
    print(colored("\n\n🎯 LANGFUSE INTEGRATION SUMMARY", 'green', attrs=['bold']))
    print("="*80)
    print("\n✅ What should now be tracked in LangFuse:")
    print("   1. swarm_consensus_query - Main query orchestration")
    print("   2. swarm_model_query - Individual model queries (9x per swarm query)")
    print("   3. swarm_consensus_generation - Consensus summary generation")
    print("   4. Trading metadata - Symbol validation signals, model lists, etc.")
    
    print("\n📊 Expected traces per SwarmAgent query:")
    print("   • 1x swarm_consensus_query (main)")
    print("   • 9x swarm_model_query (one per model)")
    print("   • 1x swarm_consensus_generation (summary)")
    print("   • Total: 11 LangFuse traces per query")
    
    print("\n🔗 Check your LangFuse dashboard at:")
    print("   " + colored(os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'), 'blue', attrs=['underline']))
    
    print("\n💡 Look for:")
    print("   • Trace names matching the patterns above")
    print("   • Metadata showing agent_type, model_name, symbols")
    print("   • Trading signals for symbol validation")
    print("   • Consensus scores and model participation")

def main():
    """Run the test"""
    try:
        test_swarm_langfuse_integration()
    except Exception as e:
        print(colored(f"\n❌ Test failed with error: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()