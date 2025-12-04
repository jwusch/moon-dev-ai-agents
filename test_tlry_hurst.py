#!/usr/bin/env python3
"""Test TLRY Exit Tracker with Hurst Exponent"""

from tlry_exit_tracker import TLRYExitTracker

def main():
    tracker = TLRYExitTracker()
    
    # Run the analysis with example entry price
    print("Testing TLRY Exit Tracker with Hurst Exponent Integration\n")
    tracker.run_exit_analysis(entry_price=10.23)  # Example entry

if __name__ == "__main__":
    main()