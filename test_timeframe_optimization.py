#!/usr/bin/env python3
"""
Test timeframe optimization on specific symbols
"""

from aegs_timeframe_optimizer import TimeframeOptimizer
from termcolor import colored

def test_specific_symbols():
    """Test timeframe optimization on your current positions"""
    
    # Initialize optimizer
    optimizer = TimeframeOptimizer()
    
    # Test your current positions
    symbols_to_test = ['SLAI', 'TLRY']
    
    # Add some known successful symbols from previous tests
    symbols_to_test.extend(['RIOT', 'GME', 'AMC', 'CLSK', 'HUT'])
    
    print(colored("🚀 TIMEFRAME OPTIMIZATION TEST", 'cyan', attrs=['bold']))
    print(f"Testing symbols: {', '.join(symbols_to_test)}")
    print("="*70)
    
    # Test each symbol
    for symbol in symbols_to_test:
        # For faster testing, only test shorter timeframes initially
        timeframes = ['5m', '15m', '1h', '1d']
        optimizer.optimize_symbol(symbol, timeframes_to_test=timeframes)
    
    # Generate report
    optimizer.generate_optimization_report()

if __name__ == "__main__":
    test_specific_symbols()