"""
Test TradingView server routes
"""

import requests

base_url = 'http://localhost:8888'

print("🔍 Testing TradingView Server Routes")
print("=" * 60)

routes = [
    ('GET', '/'),
    ('GET', '/index.html'),
    ('GET', '/health'),
    ('GET', '/login-status'),
]

for method, route in routes:
    try:
        url = f'{base_url}{route}'
        print(f"\n📡 {method} {route}")
        
        if method == 'GET':
            response = requests.get(url, timeout=2)
        else:
            response = requests.post(url, json={}, timeout=2)
            
        print(f"   Status: {response.status_code}")
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        print(f"   Content-Type: {content_type}")
        
        # Show preview of response
        if 'html' in content_type:
            print(f"   Response: HTML page ({len(response.text)} bytes)")
        elif 'json' in content_type:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Response: {response.text[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")