"""
🔥💎 RUN AEGS SWARM - AUTOMATIC GOLDMINE DISCOVERY 💎🔥

This is the main entry point for the AEGS AI Swarm system.
It will automatically discover, test, and register new goldmine symbols.

Usage:
    python run_aegs_swarm.py              # Run once
    python run_aegs_swarm.py --continuous  # Run continuously
    python run_aegs_swarm.py --test        # Test mode with small dataset
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from agents.aegs_swarm_coordinator import AEGSSwarmCoordinator
from termcolor import colored
import argparse

def print_banner():
    """Print the AEGS Swarm banner"""
    
    banner = """
    🔥💎 AEGS AI SWARM SYSTEM 💎🔥
    ================================
    Autonomous Goldmine Discovery
    
    Agents:
    1. 🔍 Discovery Agent - Finding volatility explosions
    2. 🧪 Backtest Agent - Testing with AEGS strategy  
    3. 📊 Analysis Agent - Ranking opportunities
    4. 🚨 Alert Agent - Notifying on goldmines
    
    Target: Find symbols with >1000% excess return potential
    ================================
    """
    
    print(colored(banner, 'cyan', attrs=['bold']))

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='AEGS AI Swarm System')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuously')
    parser.add_argument('--interval', type=float, default=4,
                       help='Hours between discovery cycles in continuous mode (default: 4)')
    parser.add_argument('--schedule', action='store_true',
                       help='Use daily schedule instead of interval mode')
    parser.add_argument('--time', default='06:00', 
                       help='Discovery time for schedule mode (default: 06:00)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - limited candidates')
    parser.add_argument('--use-validated', action='store_true',
                       help='Use validated symbols from smart validator')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    try:
        import yfinance
        import pandas
        import numpy
        print("✅ Dependencies verified")
    except ImportError as e:
        print(colored(f"❌ Missing dependency: {e}", 'red'))
        print("Run: pip install -r requirements.txt")
        return
    
    # Initialize coordinator
    print("\n🚀 Initializing AEGS Swarm...")
    coordinator = AEGSSwarmCoordinator()
    
    # Test mode configuration
    if args.test:
        print(colored("\n🧪 TEST MODE - Limited discovery", 'yellow'))
        coordinator.config['max_candidates_per_run'] = 5
    
    # Configure based on arguments
    if args.use_validated:
        print(colored("\n✅ Using validated symbols from smart validator", 'green'))
        # This will be handled by discovery agent
    
    # Run the swarm
    try:
        if args.continuous:
            coordinator.config['continuous_mode'] = True
            
            if args.schedule:
                print(colored(f"\n🔄 Starting continuous mode (daily at {args.time})", 'green'))
                coordinator.config['use_schedule'] = True
                coordinator.config['discovery_time'] = args.time
            else:
                print(colored(f"\n🔄 Starting continuous mode (every {args.interval} hours)", 'green'))
                coordinator.config['use_schedule'] = False
                coordinator.config['continuous_interval_hours'] = args.interval
            
            coordinator.run_continuous()
        else:
            print(colored("\n▶️  Running single discovery cycle", 'green'))
            coordinator.run_once()
            
            print(colored("\n✅ Discovery cycle complete!", 'green'))
            print("\n💡 Next steps:")
            print("   1. Check aegs_backtest_results_*.json for detailed results")
            print("   2. Run 'python aegs_enhanced_scanner.py' to check signals")
            print("   3. New goldmines are auto-registered and monitored")
            
    except KeyboardInterrupt:
        print(colored("\n\n🛑 Swarm stopped by user", 'yellow'))
    except Exception as e:
        print(colored(f"\n❌ Error: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()