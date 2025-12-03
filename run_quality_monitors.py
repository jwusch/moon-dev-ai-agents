"""
🎯 How to Run Quality Score Monitors
Demo and instructions for running the real-time dashboards

Author: Claude (Anthropic)
"""

import subprocess
import time
from termcolor import colored

def show_instructions():
    """Show how to run the monitors"""
    
    print("=" * 80)
    print(colored("🎯 QUALITY SCORE MONITOR - USAGE GUIDE", 'cyan', attrs=['bold']))
    print(colored("VXX Mean Reversion 15 Strategy", 'white', attrs=['bold']))
    print("=" * 80)
    
    print("\n📊 TWO MONITORING OPTIONS:\n")
    
    # Option 1: Single Symbol Monitor
    print(colored("1. SINGLE SYMBOL MONITOR (quality_score_live.py)", 'yellow'))
    print("   Best for: Focused monitoring of one symbol")
    print("   Features: Detailed score breakdown, history tracking")
    print("\n   To run:")
    print(colored("   python quality_score_live.py", 'green'))
    print("   • Enter symbol (e.g., VXX, AMD, SQQQ)")
    print("   • Enter refresh rate (e.g., 5 for 5 seconds)")
    print("   • Press Ctrl+C to stop\n")
    
    # Option 2: Multi Symbol Dashboard  
    print(colored("2. MULTI-SYMBOL DASHBOARD (realtime_quality_monitor.py)", 'yellow'))
    print("   Best for: Scanning multiple symbols for opportunities")
    print("   Features: 15+ symbols, side-by-side comparison")
    print("\n   To run:")
    print(colored("   python realtime_quality_monitor.py", 'green'))
    print("   • Choose default symbols or enter custom list")
    print("   • Enter refresh rate (e.g., 15 for 15 seconds)")
    print("   • Press Ctrl+C to stop\n")
    
    print("=" * 80)
    print("\n📋 QUICK START EXAMPLES:\n")
    
    # Example commands
    examples = [
        ("Monitor VXX with 5-second updates:", 
         "python quality_score_live.py\n   → Enter: VXX\n   → Enter: 5"),
        
        ("Monitor all volatile symbols:", 
         "python realtime_quality_monitor.py\n   → Press Enter (use defaults)\n   → Enter: 10"),
        
        ("Monitor custom symbols:",
         "python realtime_quality_monitor.py\n   → Enter: y\n   → Enter: VXX,AMD,NVDA,TSLA\n   → Enter: 15")
    ]
    
    for desc, cmd in examples:
        print(colored(f"• {desc}", 'white'))
        print(colored(f"  {cmd}", 'green'))
        print()
    
    print("=" * 80)
    print("\n🎯 UNDERSTANDING THE QUALITY SCORE:\n")
    
    print("Score Range    Meaning              Action")
    print("-" * 45)
    print(colored("70-100", 'green') + "        Strong Signal        Consider Entry")
    print(colored("50-69", 'yellow') + "         Moderate Signal      Wait/Watch")
    print(colored("0-49", 'red') + "          Weak Signal          Avoid Entry")
    
    print("\n📊 SCORE COMPONENTS (Max Points):")
    print("• Distance from SMA (20): How far price is from mean")
    print("• RSI Level (20): Oversold/overbought reading")
    print("• Volume Ratio (15): Above-average volume confirmation")
    print("• Momentum (15): Price deceleration/reversal signs")
    print("• Higher TF (10): 60-min timeframe alignment")
    print("• Microstructure (10): 5-min momentum")
    print("• Time of Day (10): Avoid lunch, prefer open/close")
    
    print("\n" + "=" * 80)
    print(colored("\n🚀 READY TO START?", 'cyan'))
    
    choice = input("\nRun demo? (1=Single Symbol, 2=Multi Symbol, N=Exit): ").strip()
    
    if choice == '1':
        print("\nStarting Single Symbol Monitor...")
        time.sleep(1)
        subprocess.run(["python", "quality_score_live.py"])
    elif choice == '2':
        print("\nStarting Multi-Symbol Dashboard...")  
        time.sleep(1)
        subprocess.run(["python", "realtime_quality_monitor.py"])
    else:
        print("\nTo run later, use the commands shown above!")

if __name__ == "__main__":
    show_instructions()