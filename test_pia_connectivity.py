"""
🌙 Test PIA Server Connectivity
Check if we can reach PIA proxy servers
"""

import socket
import subprocess
import dns.resolver
import requests

def test_dns_lookup(hostname):
    """Test if we can resolve the hostname"""
    print(f"\n🔍 DNS lookup for {hostname}...")
    try:
        # Using subprocess for nslookup
        result = subprocess.run(['nslookup', hostname], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            print("✅ DNS resolution successful:")
            # Parse output for IPs
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Address' in line and '#' not in line:
                    print(f"   IP: {line.strip()}")
            return True
        else:
            print(f"❌ DNS resolution failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during DNS lookup: {e}")
        
        # Try alternative method
        try:
            import socket
            ips = socket.gethostbyname_ex(hostname)[2]
            if ips:
                print(f"✅ Resolved to IPs: {', '.join(ips)}")
                return True
        except:
            pass
            
        return False

def test_port_connectivity(host, port, timeout=3):
    """Test if a specific port is reachable"""
    print(f"\n🔍 Testing connectivity to {host}:{port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        # First resolve hostname
        try:
            ip = socket.gethostbyname(host)
            print(f"📍 Resolved {host} to {ip}")
        except:
            print(f"❌ Cannot resolve {host}")
            return False
            
        # Try to connect
        result = sock.connect_ex((ip, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is reachable")
            return True
        else:
            print(f"❌ Port {port} is not reachable (error code: {result})")
            return False
            
    except socket.timeout:
        print(f"❌ Connection timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_http_connectivity(url):
    """Test HTTP/HTTPS connectivity"""
    print(f"\n🔍 Testing HTTP connectivity to {url}...")
    
    try:
        # Test without proxy
        response = requests.head(url, timeout=5, allow_redirects=True)
        print(f"✅ HTTP response: {response.status_code}")
        print(f"   Server: {response.headers.get('Server', 'Unknown')}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - server might be down or blocked")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pia_servers():
    """Test various PIA proxy servers"""
    servers = {
        'Netherlands': ('proxy-nl.privateinternetaccess.com', [8080, 1080]),
        'UK': ('proxy-uk.privateinternetaccess.com', [8080, 1080]),
        'US East': ('proxy-us-east.privateinternetaccess.com', [8080, 1080]),
        'Canada': ('proxy-ca.privateinternetaccess.com', [8080, 1080]),
    }
    
    print("\n🌍 Testing PIA proxy servers...")
    print("="*60)
    
    reachable = []
    
    for region, (hostname, ports) in servers.items():
        print(f"\n📍 {region}: {hostname}")
        
        # DNS lookup
        if test_dns_lookup(hostname):
            # Test ports
            for port in ports:
                if test_port_connectivity(hostname, port):
                    reachable.append((region, hostname, port))
    
    return reachable

def check_firewall():
    """Check for potential firewall issues"""
    print("\n🔍 Checking for potential firewall issues...")
    
    # Test common ports
    test_ports = {
        80: "HTTP",
        443: "HTTPS", 
        8080: "HTTP Proxy",
        1080: "SOCKS5",
        53: "DNS"
    }
    
    test_host = "google.com"
    
    for port, service in test_ports.items():
        if test_port_connectivity(test_host, port, timeout=2):
            print(f"✅ Outbound {service} ({port}) is allowed")
        else:
            print(f"⚠️ Outbound {service} ({port}) might be blocked")

if __name__ == "__main__":
    print("🌙 Moon Dev PIA Connectivity Test")
    print("="*60)
    
    # Test PIA servers
    reachable = test_pia_servers()
    
    # Check firewall
    check_firewall()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    if reachable:
        print(f"✅ Found {len(reachable)} reachable PIA endpoints:")
        for region, host, port in reachable:
            print(f"   - {region}: {host}:{port}")
    else:
        print("❌ No PIA proxy servers were reachable")
        print("\n💡 Possible issues:")
        print("1. Your network might be blocking outbound proxy connections")
        print("2. PIA servers might be experiencing issues")
        print("3. Your ISP might be blocking PIA proxy servers")
        print("\n💡 Alternative solutions:")
        print("1. Use a different VPN service with API-friendly proxies")
        print("2. Use a cloud server in a non-restricted region")
        print("3. Use Binance.US API if you're in the US")
    
    # Test Binance.US as alternative
    print("\n🔍 Alternative: Testing Binance.US API...")
    if test_http_connectivity("https://api.binance.us"):
        print("✅ Binance.US is accessible - consider using it instead")