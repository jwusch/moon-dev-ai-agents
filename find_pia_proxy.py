"""
🌙 Find PIA Proxy Ports
Scans for common proxy ports that PIA might use
"""

import socket
import subprocess
import platform
import os

def check_port(host='127.0.0.1', port=1080, timeout=1):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def scan_common_proxy_ports():
    """Scan common proxy ports"""
    print("🔍 Scanning for open proxy ports on localhost...")
    
    # Common proxy ports
    proxy_ports = {
        # SOCKS5 ports
        1080: "SOCKS5 (default)",
        1081: "SOCKS5 (alternate)",
        9050: "SOCKS5 (Tor default)",
        9150: "SOCKS5 (Tor Browser)",
        
        # HTTP/HTTPS proxy ports
        8080: "HTTP Proxy (common)",
        8888: "HTTP Proxy (alternate)",
        3128: "HTTP Proxy (Squid default)",
        8118: "HTTP Proxy (Privoxy)",
        
        # PIA specific ports
        8889: "PIA HTTP Proxy",
        8890: "PIA HTTPS Proxy",
        1198: "PIA OpenVPN",
    }
    
    open_ports = []
    
    for port, description in proxy_ports.items():
        if check_port(port=port):
            print(f"✅ Port {port} is OPEN - {description}")
            open_ports.append((port, description))
        else:
            print(f"❌ Port {port} is closed - {description}")
    
    return open_ports

def check_network_connections():
    """Check active network connections for PIA"""
    print("\n🔍 Checking for PIA-related network connections...")
    
    system = platform.system()
    
    try:
        if system == "Linux" or "Microsoft" in platform.release():  # WSL
            # Try netstat first
            try:
                output = subprocess.check_output(['netstat', '-tuln'], text=True)
                print("\n📊 Listening ports:")
                for line in output.split('\n'):
                    if 'LISTEN' in line and ('127.0.0.1' in line or '0.0.0.0' in line):
                        print(f"  {line.strip()}")
            except:
                # Try ss if netstat not available
                try:
                    output = subprocess.check_output(['ss', '-tuln'], text=True)
                    print("\n📊 Listening ports (via ss):")
                    for line in output.split('\n'):
                        if 'LISTEN' in line:
                            print(f"  {line.strip()}")
                except:
                    print("❌ Neither netstat nor ss available")
        
        elif system == "Windows":
            output = subprocess.check_output(['netstat', '-an'], text=True)
            print("\n📊 Listening ports:")
            for line in output.split('\n'):
                if 'LISTENING' in line and ('127.0.0.1' in line or '0.0.0.0' in line):
                    print(f"  {line.strip()}")
                    
    except Exception as e:
        print(f"❌ Error checking network connections: {e}")

def check_pia_processes():
    """Check if PIA processes are running"""
    print("\n🔍 Checking for PIA processes...")
    
    try:
        if platform.system() == "Linux" or "Microsoft" in platform.release():
            # Linux/WSL
            output = subprocess.check_output(['ps', 'aux'], text=True)
        else:
            # Windows
            output = subprocess.check_output(['tasklist'], text=True)
        
        pia_keywords = ['pia', 'privateinternet', 'openvpn', 'vpn']
        found_processes = []
        
        for line in output.split('\n'):
            for keyword in pia_keywords:
                if keyword.lower() in line.lower():
                    found_processes.append(line.strip())
                    break
        
        if found_processes:
            print("✅ Found PIA-related processes:")
            for proc in found_processes[:5]:  # Limit output
                print(f"  {proc[:100]}...")
        else:
            print("❌ No PIA processes found")
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

def suggest_env_config(open_ports):
    """Suggest environment configuration based on found ports"""
    if open_ports:
        print("\n📝 Based on open ports, try these configurations in your .env file:")
        print("="*60)
        
        for port, description in open_ports:
            if "SOCKS5" in description:
                print(f"# For {description} on port {port}")
                print(f"SOCKS5_PROXY=socks5://127.0.0.1:{port}")
                print()
            elif "HTTP" in description:
                print(f"# For {description} on port {port}")
                print(f"HTTP_PROXY=http://127.0.0.1:{port}")
                print(f"HTTPS_PROXY=http://127.0.0.1:{port}")
                print()
        print("="*60)

if __name__ == "__main__":
    print("🌙 Moon Dev PIA Proxy Port Scanner")
    print("="*60)
    
    # Scan for open proxy ports
    open_ports = scan_common_proxy_ports()
    
    # Check network connections
    check_network_connections()
    
    # Check PIA processes
    check_pia_processes()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    if open_ports:
        print(f"✅ Found {len(open_ports)} open proxy port(s)")
        suggest_env_config(open_ports)
    else:
        print("❌ No proxy ports found")
        print("\n💡 Suggestions:")
        print("1. Make sure PIA desktop app is installed and running")
        print("2. Check PIA settings for proxy configuration")
        print("3. Try using PIA's dedicated proxy credentials instead:")
        print("   - Go to PIA website → Control Panel → Proxy")
        print("   - Generate SOCKS5 credentials")
        print("   - Use those in your .env file")