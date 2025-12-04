#!/usr/bin/env python3
"""Test Bid-Ask Dynamics Analyzer"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.bid_ask_dynamics import demonstrate_bid_ask_dynamics

if __name__ == "__main__":
    demonstrate_bid_ask_dynamics()