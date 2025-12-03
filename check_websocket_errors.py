"""
🔍 Check WebSocket Connection Errors
Debug what happens when TradingView tries to connect

Author: Claude (Anthropic)
"""

import requests
import json
import time

def trigger_chart_request_and_wait():
    """Make a chart request and wait to see server console output"""
    
    print("🔍 Triggering chart request to see server-side errors...")
    print("📋 Watch the TradingView server console for error messages!")
    print("=" * 60)
    
    # Make a simple request
    payload = {
        "symbol": "BTCUSDT",
        "timeframe": "D",
        "exchange": "BINANCE"
    }
    
    print(f"📡 Sending request: {json.dumps(payload, indent=2)}")
    print("⏱️ Making request now - check server console...")
    
    try:
        response = requests.post(
            "http://localhost:8891/chart",
            json=payload,
            timeout=15
        )
        
        print(f"📊 Response: {response.status_code}")
        print(f"📝 Data: {response.json()}")
        
    except Exception as e:
        print(f"❌ Request error: {e}")
    
    print("\n💡 What to look for in server console:")
    print("- Chart error: [error message]")
    print("- WebSocket connection failed")
    print("- Session validation errors") 
    print("- Network timeouts")

def test_session_token_validity():
    """Test if session tokens are actually valid"""
    
    print(f"\n🔍 Session Token Validity Test")
    print("=" * 40)
    
    # Get current session info
    try:
        response = requests.get("http://localhost:8891/session")
        data = response.json()
        
        print(f"Session ID Length: {data.get('sessionIdLength')} chars")
        print(f"Signature Length: {data.get('signatureLength')} chars")
        print(f"Client Active: {data.get('clientActive')}")
        
        # Check if these look like valid TradingView session tokens
        if data.get('sessionIdLength') == 32 and data.get('signatureLength') == 47:
            print("✅ Token format looks correct")
        else:
            print("❌ Token format looks wrong")
            
        # Test if tokens are expired
        print(f"\n💡 These tokens were extracted when?")
        print("- If more than 24 hours ago, they might be expired")
        print("- If you logged out of TradingView, they're invalid")
        print("- If TradingView detected multiple sessions, they're blocked")
        
    except Exception as e:
        print(f"❌ Session check failed: {e}")

def check_common_issues():
    """Check for common session issues"""
    
    print(f"\n🔧 Common Session Issues:")
    print("=" * 30)
    
    issues = [
        "🕐 Tokens expired (24+ hours old)",
        "🚪 Logged out of TradingView in browser", 
        "👥 Multiple sessions detected",
        "🌐 Network/firewall blocking WebSocket",
        "🔒 TradingView API rate limiting",
        "📍 Wrong session token format",
        "⚠️ TradingView API library bug"
    ]
    
    for issue in issues:
        print(f"  {issue}")
    
    print(f"\n🔍 Next debugging steps:")
    print("1. Check server console output during requests")
    print("2. Try fresh session tokens")
    print("3. Verify you're logged out of TradingView browser")
    print("4. Test with different symbols/exchanges")

def main():
    trigger_chart_request_and_wait()
    test_session_token_validity() 
    check_common_issues()
    
    print(f"\n📋 Action Items:")
    print("1. Paste any error messages from server console")
    print("2. Try extracting fresh session tokens")
    print("3. Ensure complete browser logout from TradingView")

if __name__ == "__main__":
    main()