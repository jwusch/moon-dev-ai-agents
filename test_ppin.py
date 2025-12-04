#!/usr/bin/env python3
"""Test PPIN indicator"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.ppin import demonstrate_ppin

if __name__ == "__main__":
    demonstrate_ppin()