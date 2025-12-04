"""Test synthetic tick generation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fractal_alpha.utils.synthetic_ticks import SyntheticTickGenerator, demonstrate_generator
from src.fractal_alpha.base.types import BarData
from datetime import datetime
import pandas as pd
import numpy as np

def test_basic_generation():
    """Test basic tick generation"""
    
    print("🧪 Testing Synthetic Tick Generation\n")
    
    # Create a sample bar
    bar = BarData(
        timestamp=int(datetime.now().timestamp() * 1000),
        open=100.0,
        high=102.0,
        low=99.5,
        close=101.5,
        volume=10000
    )
    
    generator = SyntheticTickGenerator()
    
    # Test each method
    for method in ['simple', 'brownian', 'vwap', 'adaptive']:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} method")
        print(f"{'='*50}")
        
        generator.method = method
        ticks = generator.generate_ticks(bar, n_ticks=20)
        
        print(f"Generated {len(ticks)} ticks")
        
        # Analyze ticks
        prices = [t.price for t in ticks]
        volumes = [t.volume for t in ticks]
        
        print(f"\nPrice Statistics:")
        print(f"  Min: ${min(prices):.2f} (Bar Low: ${bar.low:.2f})")
        print(f"  Max: ${max(prices):.2f} (Bar High: ${bar.high:.2f})")
        print(f"  First: ${ticks[0].price:.2f} (Bar Open: ${bar.open:.2f})")
        print(f"  Last: ${ticks[-1].price:.2f} (Bar Close: ${bar.close:.2f})")
        
        print(f"\nVolume Statistics:")
        print(f"  Total: {sum(volumes):,} (Bar Volume: {bar.volume:,})")
        print(f"  Min tick: {min(volumes)}")
        print(f"  Max tick: {max(volumes)}")
        
        # Buy/Sell analysis
        buy_volume = sum(t.volume for t in ticks if t.side == 1)
        sell_volume = sum(t.volume for t in ticks if t.side == -1)
        neutral_volume = sum(t.volume for t in ticks if t.side == 0)
        
        print(f"\nOrder Flow:")
        print(f"  Buy Volume: {buy_volume:,} ({buy_volume/sum(volumes)*100:.1f}%)")
        print(f"  Sell Volume: {sell_volume:,} ({sell_volume/sum(volumes)*100:.1f}%)")
        print(f"  Neutral: {neutral_volume:,}")
        
        # Validate
        is_valid, tests = generator.validate_ticks(bar, ticks)
        print(f"\nValidation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        for test, passed in tests.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test}")

def test_different_bar_types():
    """Test generation for different bar types"""
    
    print("\n\n🎯 Testing Different Bar Types\n")
    
    generator = SyntheticTickGenerator(method='adaptive')
    
    # Define different bar scenarios
    scenarios = [
        {
            'name': 'Strong Uptrend',
            'bar': BarData(
                timestamp=int(datetime.now().timestamp() * 1000),
                open=100.0,
                high=105.0,
                low=99.8,
                close=104.5,
                volume=50000
            )
        },
        {
            'name': 'Strong Downtrend', 
            'bar': BarData(
                timestamp=int(datetime.now().timestamp() * 1000),
                open=100.0,
                high=100.2,
                low=95.0,
                close=95.5,
                volume=50000
            )
        },
        {
            'name': 'Doji/Indecision',
            'bar': BarData(
                timestamp=int(datetime.now().timestamp() * 1000),
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.05,
                volume=20000
            )
        },
        {
            'name': 'Reversal (Hammer)',
            'bar': BarData(
                timestamp=int(datetime.now().timestamp() * 1000),
                open=100.0,
                high=100.2,
                low=95.0,
                close=99.8,
                volume=35000
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Bar: O={scenario['bar'].open}, H={scenario['bar'].high}, L={scenario['bar'].low}, C={scenario['bar'].close}")
        
        ticks = generator.generate_ticks(scenario['bar'], n_ticks=30)
        
        # Analyze path
        prices = [t.price for t in ticks]
        
        # Find when extremes were hit
        high_idx = prices.index(max(prices))
        low_idx = prices.index(min(prices))
        
        print(f"  Path: High reached at tick {high_idx+1}/{len(ticks)}, Low at tick {low_idx+1}/{len(ticks)}")
        
        # Order flow
        buy_volume = sum(t.volume for t in ticks if t.side == 1)
        sell_volume = sum(t.volume for t in ticks if t.side == -1)
        
        if buy_volume + sell_volume > 0:
            buy_pct = buy_volume / (buy_volume + sell_volume) * 100
            print(f"  Order Flow: {buy_pct:.1f}% buying pressure")

def test_with_real_data():
    """Test with real market data if available"""
    
    try:
        import yfinance as yf
        
        print("\n\n📈 Testing with Real Market Data\n")
        
        # Get recent data
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="1d", interval="1m")
        
        if not df.empty:
            # Use last complete bar
            last_bar = df.iloc[-2]  # -2 to avoid incomplete current bar
            
            print(f"Using real SPY bar from {df.index[-2]}")
            
            generator = SyntheticTickGenerator(method='adaptive')
            ticks = generator.generate_ticks(last_bar, n_ticks=50)
            
            print(f"Generated {len(ticks)} synthetic ticks")
            
            # Show tick distribution
            print("\nFirst 10 ticks:")
            for i, tick in enumerate(ticks[:10]):
                side_str = "BUY " if tick.side == 1 else "SELL" if tick.side == -1 else "NEUT"
                print(f"  {i+1:2d}: ${tick.price:.2f} | Vol: {tick.volume:6d} | {side_str}")
            
            print("  ...")
            print(f"  Last: ${ticks[-1].price:.2f} | Vol: {ticks[-1].volume:6d}")
            
    except ImportError:
        print("\n⚠️ YFinance not available, skipping real data test")

if __name__ == "__main__":
    test_basic_generation()
    test_different_bar_types()
    test_with_real_data()
    
    print("\n\n✅ All tests completed!")