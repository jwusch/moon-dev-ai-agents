"""
🤖 ML FEATURES DEMONSTRATION - ENTROPY, WAVELETS, HMM
Complete showcase of machine learning-based fractal indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Import our ML indicators
from src.fractal_alpha.indicators.ml_features.entropy import EntropyIndicator
from src.fractal_alpha.indicators.ml_features.wavelet import WaveletIndicator
from src.fractal_alpha.indicators.ml_features.hmm import HMMIndicator


def demonstrate_ml_features(symbol: str = "SPY", days: int = 100):
    """
    Comprehensive demonstration of all ML-based fractal indicators
    
    Args:
        symbol: Stock symbol to analyze
        days: Number of days of data to fetch
    """
    
    print(f"🤖 ML FEATURES ANALYSIS FOR {symbol}")
    print("=" * 80)
    
    # Fetch real market data
    print(f"\n📊 Fetching {days} days of data for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        print("❌ Failed to fetch data")
        return
        
    print(f"✅ Loaded {len(data)} data points")
    
    # Initialize indicators
    entropy_ind = EntropyIndicator(
        lookback_periods=50,
        entropy_window=20,
        n_bins=10
    )
    
    wavelet_ind = WaveletIndicator(
        wavelet_type='db4',
        decomposition_levels=4,
        denoising_threshold='soft'
    )
    
    hmm_ind = HMMIndicator(
        n_states=3,
        lookback_periods=100,
        predict_transitions=True
    )
    
    # Calculate all indicators
    print("\n🔮 Calculating ML indicators...")
    
    entropy_result = entropy_ind.calculate(data, symbol)
    wavelet_result = wavelet_ind.calculate(data, symbol)
    hmm_result = hmm_ind.calculate(data, symbol)
    
    # Display Entropy Analysis
    print("\n" + "=" * 80)
    print("📊 ENTROPY ANALYSIS - Information Flow & Uncertainty")
    print("=" * 80)
    
    print(f"\n🎯 Trading Signal: {entropy_result.signal.value}")
    print(f"📈 Confidence: {entropy_result.confidence:.1f}%")
    print(f"🌡️ Information Entropy: {entropy_result.value:.2f}/100")
    
    meta = entropy_result.metadata
    if 'current_state' in meta:
        print(f"\n📍 Current Market State:")
        print(f"   Entropy: {meta['current_state']['entropy']:.3f}")
        print(f"   Normalized: {meta['current_state']['normalized_entropy']:.1%}")
        print(f"   Change from baseline: {meta['current_state']['entropy_change']:+.3f}")
    
    if 'regime_transitions' in meta:
        print(f"\n🔄 Regime Transitions:")
        print(f"   Recent change confidence: {meta['regime_transitions']['recent_change_confidence']:.1%}")
        print(f"   Transition detected: {meta['regime_transitions']['transition_detected']}")
    
    if 'distribution_analysis' in meta:
        print(f"\n📊 Distribution Analysis:")
        dist = meta['distribution_analysis']
        print(f"   Skewness: {dist['skewness']:.3f}")
        print(f"   Kurtosis: {dist['kurtosis']:.3f}")
        print(f"   Distribution type: {dist['distribution_type']}")
    
    print(f"\n💡 Insight: {meta.get('insight', 'No specific insight')}")
    
    # Display Wavelet Analysis
    print("\n" + "=" * 80)
    print("🌊 WAVELET ANALYSIS - Multi-Scale Signal Decomposition")
    print("=" * 80)
    
    print(f"\n🎯 Trading Signal: {wavelet_result.signal.value}")
    print(f"📈 Confidence: {wavelet_result.confidence:.1f}%")
    print(f"📡 Signal-to-Noise Ratio: {wavelet_result.value:.1f}/100")
    
    meta = wavelet_result.metadata
    if 'energy_distribution' in meta:
        energy = meta['energy_distribution']
        print(f"\n⚡ Energy Distribution:")
        print(f"   Dominant Scale: {energy.get('dominant_scale', 'Unknown')}")
        print(f"   Energy Entropy: {energy.get('energy_entropy', 0):.2f}")
        print(f"   Energy Concentration: {energy.get('energy_concentration', 0):.1f}%")
    
    if 'patterns' in meta:
        patterns = meta['patterns']
        print(f"\n🔍 Pattern Detection:")
        print(f"   Trend Strength: {patterns.get('trend_strength', 0):.2f}")
        print(f"   Cycle Detected: {patterns.get('cycle_detected', False)}")
        if patterns.get('dominant_period', 0) > 0:
            print(f"   Dominant Period: {patterns['dominant_period']} days")
        print(f"   Noise Level: {patterns.get('noise_level', 0):.1f}%")
    
    if 'regime_analysis' in meta:
        regime = meta['regime_analysis']
        print(f"\n🏷️ Regime Analysis:")
        print(f"   Change Detected: {regime.get('regime_change_detected', False)}")
        if regime.get('regime_change_detected'):
            print(f"   Regime Type: {regime.get('regime_type', 'Unknown')}")
            print(f"   Confidence: {regime.get('confidence', 0):.0f}%")
    
    print(f"\n💡 Insight: {meta.get('insight', 'No specific insight')}")
    
    # Display HMM Analysis
    print("\n" + "=" * 80)
    print("🔮 HIDDEN MARKOV MODEL - Probabilistic Regime Detection")
    print("=" * 80)
    
    print(f"\n🎯 Trading Signal: {hmm_result.signal.value}")
    print(f"📈 Confidence: {hmm_result.confidence:.1f}%")
    print(f"🎪 Regime Clarity: {hmm_result.value:.1f}/100")
    
    meta = hmm_result.metadata
    if 'current_state' in meta:
        current = meta['current_state']
        print(f"\n📍 Current Market State:")
        print(f"   State ID: {current['id']}")
        print(f"   State Name: {current['name']}")
        print(f"   Confidence: {current['probability']:.1%}")
    
    if 'state_analysis' in meta:
        print(f"\n📊 State Characteristics:")
        for state_name, info in meta['state_analysis'].items():
            print(f"\n   {state_name.upper()}:")
            print(f"      Interpretation: {info['interpretation']}")
            print(f"      Annual Return: {info['mean_return_annual']}")
            print(f"      Annual Volatility: {info['volatility_annual']}")
            print(f"      Frequency: {info['frequency']}")
    
    if 'next_state_prediction' in meta:
        pred = meta['next_state_prediction']
        print(f"\n🔮 Next State Prediction:")
        print(f"   Most Likely: State {pred['next_state']}")
        print(f"   Probability: {pred['next_state_probability']:.1%}")
        print(f"   Change Probability: {pred['state_change_probability']:.1%}")
    
    if 'regime_stability' in meta:
        stability = meta['regime_stability']
        print(f"\n🏛️ Regime Stability:")
        print(f"   Current Duration: {stability['current_regime_duration']} days")
        print(f"   Switch Frequency: {stability['switch_frequency']:.1%} per day")
        print(f"   Most Stable State: {stability['most_stable_state']}")
    
    print(f"\n💡 Insight: {meta.get('insight', 'No specific insight')}")
    
    # Combined Analysis
    print("\n" + "=" * 80)
    print("🤝 COMBINED ML ANALYSIS - Consensus View")
    print("=" * 80)
    
    # Calculate consensus
    signals = {
        'Entropy': entropy_result.signal.value,
        'Wavelet': wavelet_result.signal.value,
        'HMM': hmm_result.signal.value
    }
    
    confidences = {
        'Entropy': entropy_result.confidence,
        'Wavelet': wavelet_result.confidence,
        'HMM': hmm_result.confidence
    }
    
    # Count signals
    buy_count = sum(1 for s in signals.values() if s == 'BUY')
    sell_count = sum(1 for s in signals.values() if s == 'SELL')
    hold_count = sum(1 for s in signals.values() if s == 'HOLD')
    
    print(f"\n📊 Signal Distribution:")
    print(f"   BUY signals: {buy_count}/3")
    print(f"   SELL signals: {sell_count}/3")
    print(f"   HOLD signals: {hold_count}/3")
    
    print(f"\n🎯 Individual Signals:")
    for name, signal in signals.items():
        conf = confidences[name]
        print(f"   {name}: {signal} (confidence: {conf:.1f}%)")
    
    # Determine consensus
    if buy_count >= 2:
        consensus = "BUY"
        consensus_strength = "Strong" if buy_count == 3 else "Moderate"
    elif sell_count >= 2:
        consensus = "SELL"
        consensus_strength = "Strong" if sell_count == 3 else "Moderate"
    else:
        consensus = "HOLD"
        consensus_strength = "Mixed signals"
    
    avg_confidence = sum(confidences.values()) / len(confidences)
    
    print(f"\n🏆 CONSENSUS: {consensus}")
    print(f"💪 Strength: {consensus_strength}")
    print(f"📊 Average Confidence: {avg_confidence:.1f}%")
    
    # Key insights combination
    print(f"\n🔍 Combined Insights:")
    
    # Information flow
    if entropy_result.value < 30:
        print("   📉 Low information entropy suggests predictable patterns")
    elif entropy_result.value > 70:
        print("   📈 High information entropy indicates uncertainty/transitions")
    
    # Signal quality
    if wavelet_result.value > 70:
        print("   📡 High signal-to-noise ratio - clear patterns present")
    elif wavelet_result.value < 30:
        print("   🌫️ Low signal-to-noise ratio - noisy market conditions")
    
    # Regime stability
    if hmm_result.metadata.get('regime_stability', {}).get('switch_frequency', 0) > 0.2:
        print("   🔄 Frequent regime switching - unstable market")
    else:
        print("   🏛️ Stable regime conditions")
    
    # Trading recommendation
    print(f"\n💡 TRADING RECOMMENDATION:")
    
    if consensus == "BUY" and avg_confidence > 60:
        print("   ✅ Strong BUY opportunity detected by ML consensus")
        print("   🎯 Multiple indicators confirm oversold conditions")
        print("   ⏱️ Consider entering position with 2-3% allocation")
    elif consensus == "SELL" and avg_confidence > 60:
        print("   ❌ Strong SELL signal from ML indicators")
        print("   ⚠️ Consider reducing exposure or hedging")
        print("   🛡️ Wait for regime stabilization before new entries")
    elif consensus == "HOLD":
        print("   ⏸️ Mixed signals - maintain current positions")
        print("   👀 Monitor for clearer directional bias")
        print("   📊 Wait for indicator convergence")
    else:
        print("   ⚠️ Low confidence signals - exercise caution")
        print("   🔍 Require additional confirmation before trading")
    
    # Advanced pattern detection
    print(f"\n🧠 Advanced Pattern Recognition:")
    
    # Check for divergences
    if entropy_result.signal.value != hmm_result.signal.value:
        print("   ⚡ Information flow diverges from regime state")
        print("   🔄 Potential regime transition imminent")
    
    if wavelet_result.metadata.get('patterns', {}).get('cycle_detected', False):
        period = wavelet_result.metadata['patterns']['dominant_period']
        print(f"   🔄 Cyclic behavior detected: {period}-day cycle")
    
    if hmm_result.metadata.get('next_state_prediction', {}).get('state_change_probability', 0) > 0.7:
        print("   🚨 High probability of regime change")
        print("   📊 Adjust position sizing accordingly")
    
    print("\n" + "=" * 80)
    print("✅ ML FEATURES ANALYSIS COMPLETE")
    print("=" * 80)


def compare_ml_indicators(symbols: list = ['SPY', 'QQQ', 'IWM', 'VXX']):
    """Compare ML indicators across multiple symbols"""
    
    print("🤖 ML INDICATORS COMPARISON ACROSS MARKETS")
    print("=" * 80)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        # Fetch data
        data = yf.download(symbol, period='100d', progress=False)
        
        if data.empty:
            print(f"   ❌ No data for {symbol}")
            continue
        
        # Calculate indicators
        entropy_ind = EntropyIndicator()
        wavelet_ind = WaveletIndicator()
        hmm_ind = HMMIndicator()
        
        entropy_result = entropy_ind.calculate(data, symbol)
        wavelet_result = wavelet_ind.calculate(data, symbol)
        hmm_result = hmm_ind.calculate(data, symbol)
        
        results[symbol] = {
            'entropy': {
                'signal': entropy_result.signal.value,
                'confidence': entropy_result.confidence,
                'value': entropy_result.value
            },
            'wavelet': {
                'signal': wavelet_result.signal.value,
                'confidence': wavelet_result.confidence,
                'noise': wavelet_result.metadata.get('patterns', {}).get('noise_level', 0)
            },
            'hmm': {
                'signal': hmm_result.signal.value,
                'confidence': hmm_result.confidence,
                'state': hmm_result.metadata.get('current_state', {}).get('name', 'Unknown')
            }
        }
        
        print(f"   ✅ Analysis complete")
    
    # Display comparison
    print("\n" + "=" * 80)
    print("📊 CROSS-MARKET COMPARISON")
    print("=" * 80)
    
    # Signal alignment
    print("\n🎯 Signal Alignment:")
    for symbol, data in results.items():
        signals = [data['entropy']['signal'], data['wavelet']['signal'], data['hmm']['signal']]
        unique_signals = len(set(signals))
        alignment = "Strong" if unique_signals == 1 else "Moderate" if unique_signals == 2 else "Weak"
        
        print(f"\n{symbol}:")
        print(f"   Entropy: {data['entropy']['signal']} ({data['entropy']['confidence']:.0f}%)")
        print(f"   Wavelet: {data['wavelet']['signal']} ({data['wavelet']['confidence']:.0f}%)")
        print(f"   HMM: {data['hmm']['signal']} ({data['hmm']['confidence']:.0f}%)")
        print(f"   Alignment: {alignment}")
    
    # Market regimes
    print("\n🏛️ Market Regimes:")
    for symbol, data in results.items():
        print(f"   {symbol}: {data['hmm']['state']}")
    
    # Information entropy
    print("\n📊 Information Entropy Ranking:")
    entropy_sorted = sorted(results.items(), key=lambda x: x[1]['entropy']['value'], reverse=True)
    for symbol, data in entropy_sorted:
        print(f"   {symbol}: {data['entropy']['value']:.1f}/100")
    
    # Noise levels
    print("\n🌫️ Signal Noise Levels:")
    for symbol, data in results.items():
        noise = data['wavelet']['noise']
        quality = "High" if noise < 20 else "Medium" if noise < 40 else "Low"
        print(f"   {symbol}: {noise:.1f}% (Signal Quality: {quality})")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    # Single symbol deep analysis
    print("=" * 80)
    print("1️⃣ SINGLE SYMBOL DEEP ANALYSIS")
    print("=" * 80)
    demonstrate_ml_features("SPY", days=100)
    
    # Multi-symbol comparison
    print("\n\n" + "=" * 80)
    print("2️⃣ MULTI-SYMBOL COMPARISON")
    print("=" * 80)
    compare_ml_indicators(['SPY', 'QQQ', 'IWM', 'VXX'])
    
    # AEGS integration example
    print("\n\n" + "=" * 80)
    print("3️⃣ AEGS INTEGRATION EXAMPLE")
    print("=" * 80)
    print("\n🔥 Analyzing volatile symbol for AEGS strategy...")
    demonstrate_ml_features("MARA", days=100)