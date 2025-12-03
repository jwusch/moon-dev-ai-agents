"""
🌙 Simple PIA Connectivity Test
"""

import socket
import subprocess
import os

def test_dns(hostname):
    """Simple DNS test"""
    try:
        ip = socket.gethostbyname(hostname)
        print(f"✅ {hostname} resolves to {ip}")
        return ip
    except:
        print(f"❌ Cannot resolve {hostname}")
        return None

def test_port(host, port):
    """Simple port test"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

print("🌙 PIA Proxy Server Connectivity Test")
print("="*60)

# Test PIA proxy servers
servers = [
    ('proxy-nl.privateinternetaccess.com', 'Netherlands'),
    ('proxy-uk.privateinternetaccess.com', 'UK'),
    ('proxy-ca.privateinternetaccess.com', 'Canada'),
    ('proxy-ca-toronto.privateinternetaccess.com', 'Canada Toronto'),
    ('proxy-ca-vancouver.privateinternetaccess.com', 'Canada Vancouver'),
]

print("\n📍 Testing PIA proxy servers...")
for hostname, region in servers:
    print(f"\n{region}: {hostname}")
    ip = test_dns(hostname)
    
    if ip:
        # Test common proxy ports
        http_8080 = test_port(ip, 8080)
        socks_1080 = test_port(ip, 1080)
        
        if http_8080:
            print(f"  ✅ HTTP proxy port 8080 is reachable")
        else:
            print(f"  ❌ HTTP proxy port 8080 is not reachable")
            
        if socks_1080:
            print(f"  ✅ SOCKS5 port 1080 is reachable")
        else:
            print(f"  ❌ SOCKS5 port 1080 is not reachable")

# Test if we're behind a restrictive firewall
print("\n\n📍 Testing outbound connections...")
test_sites = [
    ('google.com', 443, 'HTTPS'),
    ('cloudflare.com', 443, 'HTTPS'),
    ('1.1.1.1', 443, 'Cloudflare DNS'),
]

for host, port, service in test_sites:
    if test_port(host, port):
        print(f"✅ Can connect to {service} ({host}:{port})")
    else:
        print(f"❌ Cannot connect to {service} ({host}:{port})")

print("\n\n💡 If PIA servers are not reachable:")
print("1. Check if your credentials are correct in .env")
print("2. Try using a VPS in a different region")
print("3. Consider using Binance.US API (no proxy needed)")
print("4. Use a different proxy service")

# Show current environment
print("\n\n🔧 Current proxy environment:")
pia_user = os.getenv('PIA_USERNAME')
if pia_user:
    print(f"PIA_USERNAME: {pia_user}")
else:
    print("PIA_USERNAME: Not set")
    
if os.getenv('PIA_PASSWORD'):
    print("PIA_PASSWORD: *** (set)")
else:
    print("PIA_PASSWORD: Not set")