#!/usr/bin/env python3
"""Test VPIN indicator"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.vpin import demonstrate_vpin

if __name__ == "__main__":
    demonstrate_vpin()