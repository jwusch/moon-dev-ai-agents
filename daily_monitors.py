#!/usr/bin/env python
"""
🚀 DAILY TRADING MONITORS LAUNCHER
Runs all essential monitoring programs with error handling
"""

import subprocess
import sys
import time
from datetime import datetime
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')


def run_command(command, description):
    """Run a command with error handling"""
    print(colored(f"\n{'='*60}", 'blue'))
    print(colored(f"🔄 {description}", 'cyan', attrs=['bold']))
    print(colored(f"Command: {command}", 'white'))
    print(colored(f"{'='*60}", 'blue'))
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(colored(f"✅ {description} - SUCCESS", 'green', attrs=['bold']))
        else:
            print(colored(f"❌ {description} - FAILED", 'red'))
            if result.stderr:
                print(colored("Error output:", 'red'))
                print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(colored(f"❌ Failed to run: {str(e)}", 'red'))
        return False


def main():
    """Run all daily monitors"""
    print(colored("""
    ╔════════════════════════════════════════╗
    ║     🌅 DAILY TRADING MONITORS 🌅       ║
    ╚════════════════════════════════════════╝
    """, 'yellow', attrs=['bold']))
    
    start_time = datetime.now()
    monitors = [
        {
            'command': 'python all_positions_monitor.py',
            'description': 'Universal Position Monitor (All Open Positions)',
            'enabled': True
        },
        {
            'command': 'python aegs_live_scanner.py',
            'description': 'AEGS Oversold Bounce Scanner',
            'enabled': True
        },
        {
            'command': 'python position_portal.py --summary',
            'description': 'Position Portal Summary',
            'enabled': True
        },
        {
            'command': 'python src/fractal_alpha/monitoring/data_quality_monitor.py --positions',
            'description': 'Data Quality Monitor for Open Positions',
            'enabled': False  # Disabled - shows too many technical warnings for daily use
        },
        {
            'command': 'python src/agents/aegs_swarm_coordinator.py --quick',
            'description': 'AEGS Swarm Quick Analysis',
            'enabled': False  # Disabled by default as it takes longer
        },
        {
            'command': 'python position_sizing_calculator.py --positions',
            'description': 'Position Sizing Analysis for Open Positions',
            'enabled': True
        },
        {
            'command': 'python trading_analytics_dashboard.py --report',
            'description': 'Trading Performance Analytics Report',
            'enabled': True
        },
        {
            'command': 'python aegs_volatility_scanner.py --quick',
            'description': 'AEGS Enhanced Volatility Scanner (Auto-Discovery)',
            'enabled': False  # Optional - can be resource intensive
        }
    ]
    
    # Run enabled monitors
    successful = 0
    failed = 0
    
    for monitor in monitors:
        if monitor['enabled']:
            success = run_command(monitor['command'], monitor['description'])
            if success:
                successful += 1
            else:
                failed += 1
            
            # Small delay between monitors
            time.sleep(2)
    
    # Summary
    elapsed = datetime.now() - start_time
    print(colored(f"\n{'='*60}", 'blue'))
    print(colored("📊 DAILY MONITORS SUMMARY", 'cyan', attrs=['bold']))
    print(colored(f"{'='*60}", 'blue'))
    print(f"Total monitors run: {successful + failed}")
    print(colored(f"✅ Successful: {successful}", 'green'))
    if failed > 0:
        print(colored(f"❌ Failed: {failed}", 'red'))
    print(f"⏱️  Total time: {elapsed}")
    print(colored(f"{'='*60}\n", 'blue'))
    
    # Trading tips based on time of day
    hour = datetime.now().hour
    if 9 <= hour < 10:
        print(colored("💡 TIP: Market just opened - wait 30 mins for volatility to settle", 'yellow'))
    elif 10 <= hour < 15:
        print(colored("💡 TIP: Prime trading hours - watch for setups", 'yellow'))
    elif 15 <= hour < 16:
        print(colored("💡 TIP: Power hour - watch for closing moves", 'yellow'))
    else:
        print(colored("💡 TIP: Market closed - good time for analysis and planning", 'yellow'))


if __name__ == "__main__":
    main()