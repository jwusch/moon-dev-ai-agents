"""
🔍 Debug TradingView Session Server Responses
See exactly what the server is returning

Author: Claude (Anthropic)
"""

import requests
import json
from datetime import datetime

def debug_server_endpoint(endpoint, method="GET", payload=None):
    """Debug a specific server endpoint with detailed logging"""
    
    server_url = "http://localhost:8891"
    url = f"{server_url}{endpoint}"
    
    print(f"\n🔍 Testing: {method} {endpoint}")
    print("-" * 60)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
        
        print(f"📡 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        # Try to parse as JSON
        try:
            json_data = response.json()
            print(f"📊 JSON Response:")
            print(json.dumps(json_data, indent=2))
            return json_data
        except:
            print(f"📝 Raw Response: {response.text}")
            return response.text
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return None

def debug_session_server():
    """Debug all session server endpoints"""
    
    print("🔍 TradingView Session Server Debug")
    print("=" * 60)
    
    # 1. Health check
    health = debug_server_endpoint("/health")
    
    # 2. Session info
    session = debug_server_endpoint("/session")
    
    # 3. Simple chart request
    chart_payload = {
        "symbol": "AAPL",
        "timeframe": "D", 
        "exchange": "NASDAQ"
    }
    chart = debug_server_endpoint("/chart", "POST", chart_payload)
    
    # 4. History request
    history_payload = {
        "symbol": "TSLA",
        "timeframe": "1D",
        "exchange": "NASDAQ", 
        "bars": 5  # Just 5 bars for testing
    }
    history = debug_server_endpoint("/history", "POST", history_payload)
    
    # 5. Different symbols/exchanges
    print(f"\n🧪 Testing Different Symbols:")
    print("=" * 40)
    
    test_symbols = [
        {"symbol": "BTCUSDT", "exchange": "BINANCE"},
        {"symbol": "EURUSD", "exchange": "FX_IDC"},
        {"symbol": "SPX", "exchange": "SP"},
        {"symbol": "TSLA", "exchange": "NASDAQ"},
        {"symbol": "AAPL", "exchange": "NASDAQ"}
    ]
    
    for test in test_symbols:
        payload = {**test, "timeframe": "D"}
        print(f"\n📊 Testing {test['exchange']}:{test['symbol']}")
        result = debug_server_endpoint("/chart", "POST", payload)
        
        if result and isinstance(result, dict):
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
            elif 'close' in result:
                print(f"   ✅ Success: Close = {result['close']}")
            else:
                print(f"   ⚠️ Unexpected: {result}")

def check_server_logs():
    """Check what the server console is showing"""
    print(f"\n📋 Server Console Check:")
    print("=" * 30)
    print("💡 Look at the TradingView server console output for:")
    print("   - 'Chart error:' messages")
    print("   - 'Test timeout' messages") 
    print("   - WebSocket connection errors")
    print("   - Session validation errors")
    print("\n💡 If you see any error messages, paste them here!")

def test_raw_websocket():
    """Test if we can check WebSocket connection"""
    print(f"\n🌐 WebSocket Connection Test:")
    print("=" * 30)
    
    # Try to hit any WebSocket endpoint to see what happens
    try:
        # This should fail but might give us info
        response = requests.get("http://localhost:8891/ws", timeout=5)
        print(f"WebSocket test response: {response.status_code}")
    except:
        print("WebSocket endpoint not accessible (expected)")

def main():
    debug_session_server()
    check_server_logs()
    test_raw_websocket()
    
    print(f"\n🔧 Analysis Questions:")
    print("=" * 30)
    print("1. Are we getting any data at all from any symbol?")
    print("2. Is the error consistent across all symbols?")
    print("3. What does the server console show during requests?")
    print("4. Are the session tokens being used correctly?")
    print("\n💭 Next steps based on what we see...")

if __name__ == "__main__":
    main()