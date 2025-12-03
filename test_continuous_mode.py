"""
🧪 Test AEGS Continuous Mode
"""

import subprocess
import sys
import time
from termcolor import colored

def test_continuous_mode():
    """Test the continuous mode with short interval"""
    
    print(colored("🧪 TESTING AEGS CONTINUOUS MODE", 'cyan', attrs=['bold']))
    print("=" * 60)
    print("This will run the swarm with a 5-minute interval")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run with 5 minute interval
    cmd = [
        sys.executable,
        "run_aegs_swarm.py",
        "--continuous",
        "--interval", "0.0833",  # 5 minutes = 0.0833 hours
        "--test"  # Test mode with limited candidates
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(colored("\n\nTest stopped by user", 'yellow'))

if __name__ == "__main__":
    test_continuous_mode()