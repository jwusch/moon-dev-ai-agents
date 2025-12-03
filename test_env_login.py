"""
Test login with .env password
"""

import requests
import os

server_url = os.getenv('TV_SERVER_URL', 'http://localhost:8888')

print("🔍 Testing Login with .env Password")
print("=" * 60)

# Check current status
try:
    # Get config
    config = requests.get(f'{server_url}/config').json()
    print(f"Username from .env: {config.get('username', 'Not set')}")
    print(f"Password configured: {config.get('passwordConfigured', False)}")
    
    # Check auth status
    status = requests.get(f'{server_url}/login-status').json()
    print(f"Currently authenticated: {status.get('authenticated', False)}")
    
    if not status.get('authenticated'):
        print("\n🔐 Testing login with env credentials...")
        
        # Method 1: Send only username (server should use env password)
        print("\nMethod 1: Username only (server uses env password)")
        response = requests.post(
            f'{server_url}/login',
            json={'username': config.get('username')}
        )
        print(f"Response: {response.status_code}")
        print(f"Result: {response.json()}")
        
        # Method 2: Send empty body (server uses all env vars)
        print("\nMethod 2: Empty body (server uses all env vars)")
        response = requests.post(f'{server_url}/login', json={})
        print(f"Response: {response.status_code}")
        print(f"Result: {response.json()}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    
print("\n📝 Tips:")
print("- Make sure TV_USERNAME and TV_PASSWORD are in .env")
print("- The checkbox 'Use password from .env' should be checked")
print("- Don't enter a password in the form when using env password")