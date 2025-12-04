#!/usr/bin/env python3
"""Test Amihud Illiquidity indicator"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.amihud_illiquidity import demonstrate_amihud_illiquidity

if __name__ == "__main__":
    demonstrate_amihud_illiquidity()