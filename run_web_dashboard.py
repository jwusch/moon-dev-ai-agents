"""
🚀 Quick Start Web Dashboard
Simple script to run the web-based quality monitor

Author: Claude (Anthropic)
"""

import subprocess
import webbrowser
import time
import os

def check_requirements():
    """Check and install required packages"""
    required = ['flask', 'flask-cors']
    
    print("🔍 Checking requirements...")
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.run(['pip', 'install', package], check=True)
    
    print("\n✅ All requirements satisfied!")

def main():
    print("=" * 60)
    print("🌐 VXX MEAN REVERSION 15 - WEB DASHBOARD")
    print("=" * 60)
    
    # Check requirements
    check_requirements()
    
    print("\n🚀 Starting web dashboard...")
    print("\nThe dashboard will:")
    print("  • Show real-time quality scores")
    print("  • Update every 10 seconds")
    print("  • Display active signals with visual indicators")
    print("  • Track 15+ symbols including crypto")
    
    print("\n📊 Opening in browser...")
    
    # Start the web server
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        from threading import Thread
        browser_thread = Thread(target=open_browser)
        browser_thread.start()
        
        # Run the dashboard
        print("\n✅ Dashboard running at: http://localhost:5000")
        print("\nPress Ctrl+C to stop")
        
        from web_quality_dashboard import main as run_dashboard
        run_dashboard()
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running directly: python web_quality_dashboard.py")

if __name__ == "__main__":
    main()