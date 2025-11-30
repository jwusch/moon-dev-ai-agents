#!/usr/bin/env python
"""
🌙 Moon Dev's Strategy Marketplace Dashboard Launcher
Handles WSL2 networking and provides clear instructions
"""

import os
import sys
import socket
import subprocess
from termcolor import cprint

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def get_wsl_ip():
    """Get the WSL2 IP address"""
    try:
        # Get WSL IP address
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            # Return first IP address
            ips = result.stdout.strip().split()
            if ips:
                return ips[0]
    except:
        pass
    return None

def main():
    cprint("\n🌙 Moon Dev Strategy Marketplace Dashboard", "white", "on_blue")
    cprint("=" * 60, "blue")
    
    # Check if we're in WSL
    is_wsl = 'microsoft' in os.uname().release.lower()
    
    if is_wsl:
        wsl_ip = get_wsl_ip()
        cprint("\n🖥️  Running in WSL2 Environment", "yellow")
        cprint(f"WSL IP Address: {wsl_ip}", "cyan")
        
        cprint("\n📋 To access the dashboard from Windows:", "green")
        cprint("1. Open Windows Terminal/PowerShell as Administrator", "white")
        cprint("2. Run this command to open the port:", "white")
        cprint(f"   netsh interface portproxy add v4tov4 listenport=8002 listenaddress=0.0.0.0 connectport=8002 connectaddress={wsl_ip}", "cyan")
        cprint("3. Open your browser to: http://localhost:8002", "white")
        
        cprint("\n🔧 Alternative Quick Access:", "green")
        cprint(f"Try accessing directly at: http://{wsl_ip}:8002", "cyan")
        
        cprint("\n⚠️  If Windows Firewall blocks access:", "yellow")
        cprint("Allow Python through Windows Firewall when prompted", "white")
    
    # Import and run the dashboard
    try:
        cprint("\n🚀 Starting Dashboard Server...", "green")
        from marketplace_dashboard import app
        
        # For WSL, we need to be explicit about the host
        host = '0.0.0.0' if is_wsl else '127.0.0.1'
        
        cprint(f"\n✅ Dashboard starting on {host}:8002", "green")
        cprint("\nPress CTRL+C to stop the server", "yellow")
        
        # Run with host that allows external connections in WSL
        app.run(host=host, port=8002, debug=True)
        
    except KeyboardInterrupt:
        cprint("\n\n👋 Dashboard stopped", "yellow")
    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red")
        
        # Additional troubleshooting
        cprint("\n🔍 Troubleshooting:", "yellow")
        cprint("1. Make sure Flask is installed: pip install flask", "white")
        cprint("2. Check if port 8002 is already in use: lsof -i :8002", "white")
        cprint("3. Try a different port by editing the script", "white")

if __name__ == "__main__":
    main()