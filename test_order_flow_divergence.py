#!/usr/bin/env python3
"""Test Order Flow Divergence indicator"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fractal_alpha.indicators.microstructure.order_flow_divergence import demonstrate_order_flow_divergence

if __name__ == "__main__":
    demonstrate_order_flow_divergence()