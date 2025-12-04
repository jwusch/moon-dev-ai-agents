#!/usr/bin/env python3
"""Test Effective Spread indicator"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.effective_spread import demonstrate_effective_spread

if __name__ == "__main__":
    demonstrate_effective_spread()