#!/usr/bin/env python3
"""
🌙 Test API Conversion Script
Verifies that agents work with the new APIAdapter
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from termcolor import cprint
import time

def test_api_adapter():
    """Test the API adapter directly"""
    cprint("\n🧪 Testing API Adapter...", "cyan", attrs=['bold'])
    
    try:
        from src.agents.api_adapter import APIAdapter
        api = APIAdapter()
        cprint("✅ API Adapter initialized successfully", "green")
        
        # Test liquidation data
        cprint("\n📊 Testing liquidation data fetch...", "yellow")
        liq_data = api.get_liquidation_data(limit=10)
        if liq_data is not None and len(liq_data) > 0:
            cprint(f"✅ Got {len(liq_data)} liquidation records", "green")
            print(liq_data.head(3))
        else:
            cprint("⚠️ No liquidation data returned", "yellow")
        
        # Test funding data
        cprint("\n📊 Testing funding rate fetch...", "yellow")
        funding = api.get_funding_data()
        if funding is not None and len(funding) > 0:
            cprint(f"✅ Got {len(funding)} funding rates", "green")
            print(funding.head(3))
        else:
            cprint("⚠️ No funding data returned", "yellow")
            
        # Test OI data
        cprint("\n📊 Testing open interest fetch...", "yellow")
        oi_data = api.get_oi_data()
        if oi_data is not None and len(oi_data) > 0:
            cprint(f"✅ Got {len(oi_data)} OI records", "green")
            print(oi_data.head(3))
        else:
            cprint("⚠️ No OI data returned", "yellow")
            
    except Exception as e:
        cprint(f"❌ Error testing API adapter: {e}", "red")
        import traceback
        traceback.print_exc()

def test_agent_imports():
    """Test that agents can be imported with new API"""
    cprint("\n🧪 Testing Agent Imports...", "cyan", attrs=['bold'])
    
    agents = [
        ("Liquidation Agent", "src.agents.liquidation_agent", "LiquidationAgent"),
        ("Funding Agent", "src.agents.funding_agent", "FundingAgent"),
        ("Whale Agent", "src.agents.whale_agent", "WhaleAgent")
    ]
    
    for name, module_path, class_name in agents:
        try:
            cprint(f"\n📦 Testing {name}...", "yellow")
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            cprint(f"✅ {name} imported successfully", "green")
            
            # Try to initialize (might fail due to missing API keys, but import should work)
            try:
                agent = agent_class()
                cprint(f"✅ {name} initialized successfully", "green")
            except Exception as init_error:
                if "API" in str(init_error) or "key" in str(init_error).lower():
                    cprint(f"⚠️ {name} needs API keys (expected)", "yellow")
                else:
                    cprint(f"⚠️ {name} initialization warning: {init_error}", "yellow")
                    
        except ImportError as e:
            cprint(f"❌ Failed to import {name}: {e}", "red")
        except Exception as e:
            cprint(f"❌ Error with {name}: {e}", "red")
            import traceback
            traceback.print_exc()

def main():
    cprint("🌙 Moon Dev API Conversion Test", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")
    
    # Test API adapter functionality
    test_api_adapter()
    
    # Test agent imports
    test_agent_imports()
    
    cprint("\n✨ Testing complete!", "green", attrs=['bold'])
    cprint("\n💡 Next steps:", "yellow")
    cprint("1. If you see warnings about API keys, that's normal", "white")
    cprint("2. The agents should now work with public data sources", "white")
    cprint("3. Run individual agents to test full functionality", "white")

if __name__ == "__main__":
    main()