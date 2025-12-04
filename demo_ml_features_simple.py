"""
🤖 ML FEATURES SIMPLE DEMONSTRATION
Quick test of ML-based fractal indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import our ML indicators
from src.fractal_alpha.indicators.ml_features.entropy import EntropyIndicator
from src.fractal_alpha.indicators.ml_features.wavelet import WaveletIndicator
from src.fractal_alpha.indicators.ml_features.hmm import HMMIndicator


def test_ml_indicators():
    """Simple test of ML indicators"""
    
    print("🤖 ML FEATURES SIMPLE TEST")
    print("=" * 60)
    
    # Generate synthetic data to avoid data issues
    np.random.seed(42)
    n_points = 200
    
    # Generate synthetic price data with regimes
    prices = []
    price = 100
    
    for i in range(n_points):
        if i < 50:
            # Uptrend regime
            change = np.random.normal(0.001, 0.01)
        elif i < 100:
            # High volatility regime  
            change = np.random.normal(0, 0.03)
        elif i < 150:
            # Downtrend regime
            change = np.random.normal(-0.0005, 0.015)
        else:
            # Recovery regime
            change = np.random.normal(0.0008, 0.01)
            
        price *= (1 + change)
        prices.append(price)
    
    # Create DataFrame
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='D')
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': np.random.uniform(1000000, 2000000, n_points)
    }, index=dates)
    
    print(f"✅ Generated {len(data)} data points")
    
    # Test Entropy Indicator
    print("\n" + "=" * 60)
    print("📊 TESTING ENTROPY INDICATOR")
    print("=" * 60)
    
    try:
        entropy_ind = EntropyIndicator(
            lookback_periods=30,
            entropy_window=20,
            n_bins=8
        )
        
        entropy_result = entropy_ind.calculate(data, "TEST")
        
        print(f"✅ Signal: {entropy_result.signal.value}")
        print(f"✅ Confidence: {entropy_result.confidence:.1f}%")
        print(f"✅ Entropy Value: {entropy_result.value:.2f}/100")
        
        if 'current_state' in entropy_result.metadata:
            state = entropy_result.metadata['current_state']
            print(f"✅ Current Entropy: {state['entropy']:.3f}")
    except Exception as e:
        print(f"❌ Entropy Error: {str(e)[:100]}")
    
    # Test Wavelet Indicator
    print("\n" + "=" * 60)
    print("🌊 TESTING WAVELET INDICATOR")
    print("=" * 60)
    
    try:
        # Use fewer decomposition levels to avoid issues
        wavelet_ind = WaveletIndicator(
            wavelet_type='db4',
            decomposition_levels=3,
            denoising_threshold='soft',
            analyze_coherence=False  # Disable to avoid issues
        )
        
        wavelet_result = wavelet_ind.calculate(data, "TEST")
        
        print(f"✅ Signal: {wavelet_result.signal.value}")
        print(f"✅ Confidence: {wavelet_result.confidence:.1f}%")
        print(f"✅ Signal-to-Noise: {wavelet_result.value:.2f}/100")
        
        if 'patterns' in wavelet_result.metadata:
            patterns = wavelet_result.metadata['patterns']
            print(f"✅ Noise Level: {patterns.get('noise_level', 0):.1f}%")
    except Exception as e:
        print(f"❌ Wavelet Error: {str(e)[:100]}")
    
    # Test HMM Indicator
    print("\n" + "=" * 60)
    print("🔮 TESTING HMM INDICATOR")
    print("=" * 60)
    
    try:
        hmm_ind = HMMIndicator(
            n_states=3,
            lookback_periods=100,
            predict_transitions=True,
            use_volume=False  # Disable to simplify
        )
        
        hmm_result = hmm_ind.calculate(data, "TEST")
        
        print(f"✅ Signal: {hmm_result.signal.value}")
        print(f"✅ Confidence: {hmm_result.confidence:.1f}%") 
        print(f"✅ Regime Clarity: {hmm_result.value:.2f}/100")
        
        if 'current_state' in hmm_result.metadata:
            state = hmm_result.metadata['current_state']
            print(f"✅ Current State: {state['name']}")
            print(f"✅ State Confidence: {state['probability']:.1%}")
    except Exception as e:
        print(f"❌ HMM Error: {str(e)[:100]}")
    
    # Test with real data
    print("\n" + "=" * 60)
    print("📈 TESTING WITH REAL SPY DATA")
    print("=" * 60)
    
    try:
        spy_data = yf.download('SPY', period='200d', progress=False)
        
        if not spy_data.empty:
            print(f"✅ Loaded {len(spy_data)} days of SPY data")
            
            # Quick entropy test
            entropy_result = entropy_ind.calculate(spy_data, "SPY")
            print(f"\nEntropy: {entropy_result.signal.value} ({entropy_result.confidence:.0f}%)")
            
            # Quick wavelet test (simplified)
            wavelet_ind_simple = WaveletIndicator(
                decomposition_levels=2,
                analyze_coherence=False
            )
            wavelet_result = wavelet_ind_simple.calculate(spy_data, "SPY")
            print(f"Wavelet: {wavelet_result.signal.value} ({wavelet_result.confidence:.0f}%)")
            
            # Quick HMM test
            hmm_result = hmm_ind.calculate(spy_data, "SPY")
            print(f"HMM: {hmm_result.signal.value} ({hmm_result.confidence:.0f}%)")
            
            # Consensus
            signals = [entropy_result.signal.value, wavelet_result.signal.value, hmm_result.signal.value]
            buy_count = sum(1 for s in signals if s == 'BUY')
            sell_count = sum(1 for s in signals if s == 'SELL')
            
            if buy_count >= 2:
                print(f"\n✅ ML CONSENSUS: BUY ({buy_count}/3 indicators)")
            elif sell_count >= 2:
                print(f"\n❌ ML CONSENSUS: SELL ({sell_count}/3 indicators)")
            else:
                print(f"\n⏸️ ML CONSENSUS: HOLD (mixed signals)")
                
    except Exception as e:
        print(f"❌ Real data error: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("✅ ML FEATURES TEST COMPLETE")
    print("=" * 60)


def test_aegs_ml_integration():
    """Test ML indicators with AEGS goldmine symbols"""
    
    print("\n\n🔥 TESTING ML INDICATORS WITH AEGS GOLDMINES")
    print("=" * 60)
    
    symbols = ['MARA', 'WULF', 'EQT', 'RIOT']
    
    # Initialize indicators
    entropy_ind = EntropyIndicator(lookback_periods=30)
    wavelet_ind = WaveletIndicator(decomposition_levels=2, analyze_coherence=False)
    hmm_ind = HMMIndicator(n_states=3, use_volume=False)
    
    for symbol in symbols:
        print(f"\n📊 {symbol}:")
        
        try:
            data = yf.download(symbol, period='100d', progress=False)
            
            if len(data) < 50:
                print("   ❌ Insufficient data")
                continue
                
            # Calculate all three
            e_result = entropy_ind.calculate(data, symbol)
            w_result = wavelet_ind.calculate(data, symbol)
            h_result = hmm_ind.calculate(data, symbol)
            
            # Show results
            print(f"   Entropy: {e_result.signal.value} ({e_result.confidence:.0f}%)")
            print(f"   Wavelet: {w_result.signal.value} ({w_result.confidence:.0f}%)")
            print(f"   HMM: {h_result.signal.value} ({h_result.confidence:.0f}%)")
            
            # Special insights
            if h_result.metadata.get('current_state', {}).get('name'):
                print(f"   Regime: {h_result.metadata['current_state']['name']}")
                
            # Check for high volatility (good for AEGS)
            if 'patterns' in w_result.metadata:
                noise = w_result.metadata['patterns'].get('noise_level', 0)
                if noise > 30:
                    print(f"   🔥 High volatility detected: {noise:.0f}% noise")
                    
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
    
    print("\n✅ AEGS ML integration test complete")


if __name__ == "__main__":
    # Run simple tests
    test_ml_indicators()
    
    # Test AEGS integration
    test_aegs_ml_integration()