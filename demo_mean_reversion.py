#!/usr/bin/env python3
"""
📐 Mean Reversion Suite Demo
Demonstrates OU Process, Dynamic Z-Score, and Pairs Trading indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Import our mean reversion indicators
from src.fractal_alpha.indicators.mean_reversion.ou_process import OUProcessIndicator
from src.fractal_alpha.indicators.mean_reversion.dynamic_zscore import DynamicZScoreIndicator
from src.fractal_alpha.indicators.mean_reversion.pairs_trading import PairsTradingIndicator


def fetch_data(symbols: list, days: int = 100) -> dict:
    """Fetch market data for symbols"""
    
    data = {}
    
    print(f"Fetching {days} days of data...")
    
    for symbol in symbols:
        print(f"  Downloading {symbol}...", end="", flush=True)
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d", interval="1d")
            
            if not hist.empty:
                data[symbol] = hist
                print(" ✓")
            else:
                print(" ✗ (no data)")
        except Exception as e:
            print(f" ✗ ({str(e)[:30]})")
    
    return data


def analyze_mean_reversion(symbol: str = "SPY", pair_symbol: Optional[str] = None):
    """Comprehensive mean reversion analysis"""
    
    print(colored(f"\n📐 MEAN REVERSION ANALYSIS FOR {symbol}", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Fetch data
    symbols_to_fetch = [symbol]
    if pair_symbol:
        symbols_to_fetch.append(pair_symbol)
    
    market_data = fetch_data(symbols_to_fetch, days=100)
    
    if symbol not in market_data:
        print(colored(f"❌ Unable to fetch data for {symbol}", "red"))
        return
    
    data = market_data[symbol]
    
    # 1. Ornstein-Uhlenbeck Process Analysis
    print(colored("\n📊 1. ORNSTEIN-UHLENBECK PROCESS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    ou_indicator = OUProcessIndicator(
        lookback_periods=60,
        estimation_method='mle',
        z_score_threshold=2.0
    )
    
    ou_result = ou_indicator.calculate(data, symbol)
    
    # Display OU results
    ou_params = ou_result.metadata['ou_parameters']
    print(f"OU Parameters:")
    print(f"  θ (Mean Reversion Speed): {ou_params['theta']:.3f}")
    print(f"  μ (Long-term Mean): ${np.exp(ou_params['mu']):.2f}")
    print(f"  σ (Volatility): {ou_params['sigma']:.3f}")
    print(f"  Half-life: {ou_params['half_life']:.1f} days")
    
    current_state = ou_result.metadata['current_state']
    print(f"\nCurrent State:")
    print(f"  Z-score: {current_state['z_score']:.2f}")
    print(f"  Percentile: {current_state['percentile']:.1f}%")
    
    mr_analysis = ou_result.metadata['mean_reversion']
    print(f"\nMean Reversion Quality:")
    print(f"  Is Mean-Reverting: {colored('YES' if mr_analysis['is_mean_reverting'] else 'NO', 'green' if mr_analysis['is_mean_reverting'] else 'red')}")
    print(f"  Strength: {colored(mr_analysis['strength'].upper(), 'green' if mr_analysis['strength'] == 'strong' else 'yellow')}")
    print(f"  Expected Return Time: {mr_analysis['expected_return_time']:.1f} days")
    
    print(f"\nOU Signal: {colored(ou_result.signal.value, 'green' if 'BUY' in ou_result.signal.value else 'red' if 'SELL' in ou_result.signal.value else 'white')}")
    print(f"OU Confidence: {ou_result.confidence:.1f}%")
    
    # 2. Dynamic Z-Score Analysis
    print(colored("\n📈 2. DYNAMIC Z-SCORE BANDS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    zscore_indicator = DynamicZScoreIndicator(
        lookback_window=20,
        volatility_window=10,
        use_robust_stats=True,
        adapt_to_regime=True
    )
    
    zscore_result = zscore_indicator.calculate(data, symbol)
    
    # Display Z-Score results
    z_scores = zscore_result.metadata['z_scores']
    print(f"Z-Score Values:")
    print(f"  Raw Z-Score: {z_scores['raw']:.2f}")
    print(f"  Adjusted Z-Score: {z_scores['adjusted']:.2f}")
    print(f"  Composite Score: {z_scores['composite']:.2f}")
    
    vol_regime = zscore_result.metadata['volatility_regime']
    print(f"\nVolatility Regime:")
    print(f"  Current: {colored(vol_regime['regime'].replace('_', ' ').upper(), 'red' if 'high' in vol_regime['regime'] else 'green')}")
    print(f"  Volatility Percentile: {vol_regime['volatility_percentile']:.1f}%")
    print(f"  Trend: {vol_regime['volatility_trend'].upper()}")
    
    bands = zscore_result.metadata['bands']
    current_price = data['Close'].iloc[-1]
    print(f"\nDynamic Bands:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Center: ${bands['center']:.2f}")
    print(f"  Entry Bands: ${bands['lower_entry']:.2f} - ${bands['upper_entry']:.2f}")
    print(f"  Exit Bands: ${bands['lower_exit']:.2f} - ${bands['upper_exit']:.2f}")
    
    print(f"\nZ-Score Signal: {colored(zscore_result.signal.value, 'green' if 'BUY' in zscore_result.signal.value else 'red' if 'SELL' in zscore_result.signal.value else 'white')}")
    print(f"Z-Score Confidence: {zscore_result.confidence:.1f}%")
    
    # 3. Pairs Trading Analysis (if pair provided)
    if pair_symbol and pair_symbol in market_data:
        print(colored(f"\n🤝 3. PAIRS TRADING ({symbol}-{pair_symbol})", "yellow", attrs=["bold"]))
        print("-" * 60)
        
        pairs_indicator = PairsTradingIndicator(
            lookback_periods=60,
            zscore_window=20,
            entry_zscore=2.0,
            test_cointegration=True
        )
        
        pairs_data = {
            'asset1': data,
            'asset2': market_data[pair_symbol]
        }
        
        pairs_result = pairs_indicator.calculate(pairs_data, f"{symbol}-{pair_symbol}")
        
        # Display Pairs results
        print(f"Pair Statistics:")
        print(f"  Correlation: {pairs_result.metadata['correlation']:.3f}")
        
        coint = pairs_result.metadata.get('cointegration', {})
        if 'is_cointegrated' in coint:
            print(f"  Cointegrated: {colored('YES' if coint['is_cointegrated'] else 'NO', 'green' if coint['is_cointegrated'] else 'red')}")
            if 'p_value' in coint:
                print(f"  P-value: {coint['p_value']:.4f}")
        
        if 'hedge_ratio' in pairs_result.metadata:
            hedge = pairs_result.metadata['hedge_ratio']
            print(f"\nHedge Ratio: {hedge['hedge_ratio']:.3f}")
            
            spread = pairs_result.metadata.get('spread', {})
            if 'current_zscore' in spread:
                print(f"\nSpread Z-Score: {spread['current_zscore']:.2f}")
                print(f"Spread Percentile: {spread.get('percentile', 50):.1f}%")
            
            quality = pairs_result.metadata.get('spread_quality', {})
            if 'half_life' in quality:
                print(f"\nSpread Half-life: {quality['half_life']:.1f} days")
                print(f"Quality Spread: {colored('YES' if quality.get('is_quality_spread', False) else 'NO', 'green' if quality.get('is_quality_spread', False) else 'red')}")
        
        print(f"\nPairs Signal: {colored(pairs_result.signal.value, 'green' if 'BUY' in pairs_result.signal.value else 'red' if 'SELL' in pairs_result.signal.value else 'white')}")
        print(f"Pairs Confidence: {pairs_result.confidence:.1f}%")
    
    # Combined Analysis
    print(colored("\n🎯 COMBINED MEAN REVERSION ANALYSIS", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Aggregate signals
    signals = [
        (ou_result.signal.value, ou_result.confidence, "OU Process"),
        (zscore_result.signal.value, zscore_result.confidence, "Dynamic Z-Score")
    ]
    
    if pair_symbol and pair_symbol in market_data:
        signals.append((pairs_result.signal.value, pairs_result.confidence, "Pairs Trading"))
    
    # Calculate consensus
    buy_votes = sum(1 for s, _, _ in signals if s == "BUY")
    sell_votes = sum(1 for s, _, _ in signals if s == "SELL")
    
    avg_buy_conf = np.mean([c for s, c, _ in signals if s == "BUY"]) if buy_votes > 0 else 0
    avg_sell_conf = np.mean([c for s, c, _ in signals if s == "SELL"]) if sell_votes > 0 else 0
    
    if buy_votes > sell_votes:
        consensus = "BUY"
        consensus_confidence = avg_buy_conf
    elif sell_votes > buy_votes:
        consensus = "SELL"
        consensus_confidence = avg_sell_conf
    else:
        consensus = "HOLD"
        consensus_confidence = 0
    
    print(f"Consensus Signal: {colored(consensus, 'green' if consensus == 'BUY' else 'red' if consensus == 'SELL' else 'yellow', attrs=['bold'])}")
    print(f"Consensus Confidence: {consensus_confidence:.1f}%")
    print(f"Signal Agreement: {max(buy_votes, sell_votes)}/{len(signals)}")
    
    print("\nIndividual Signals:")
    for sig, conf, name in signals:
        print(f"  {name}: {colored(sig, 'white')} ({conf:.1f}%)")
    
    # Mean Reversion Conditions
    print(colored("\n📊 MEAN REVERSION CONDITIONS", "yellow"))
    print("-" * 60)
    
    conditions = []
    
    # OU conditions
    if ou_result.metadata['mean_reversion']['is_mean_reverting']:
        conditions.append(f"✓ Asset exhibits mean reversion (Half-life: {ou_params['half_life']:.1f} days)")
    else:
        conditions.append("✗ Asset not mean-reverting in acceptable timeframe")
    
    # Volatility conditions
    if vol_regime['regime'] == 'high_volatility':
        conditions.append("⚠️ High volatility regime - wider bands active")
    elif vol_regime['regime'] == 'low_volatility':
        conditions.append("✓ Low volatility regime - tighter bands")
    
    # Extreme conditions
    if abs(z_scores['composite']) > 3:
        conditions.append(f"🚨 Extreme deviation detected ({z_scores['composite']:.1f}σ)")
    elif abs(z_scores['composite']) > 2:
        conditions.append(f"📍 Significant deviation ({z_scores['composite']:.1f}σ)")
    
    for condition in conditions:
        print(f"• {condition}")
    
    # Trading Recommendations
    print(colored("\n💡 MEAN REVERSION TRADING RECOMMENDATIONS", "green"))
    print("-" * 60)
    
    recommendations = []
    
    # Based on consensus and conditions
    if consensus == "BUY" and consensus_confidence > 65:
        recommendations.append("📈 Strong mean reversion BUY opportunity")
        if abs(current_state['z_score']) > 2:
            recommendations.append("📍 Oversold condition supports entry")
    elif consensus == "SELL" and consensus_confidence > 65:
        recommendations.append("📉 Strong mean reversion SELL opportunity")
        if abs(current_state['z_score']) > 2:
            recommendations.append("📍 Overbought condition supports entry")
    
    # Risk warnings
    if not ou_result.metadata['mean_reversion']['is_mean_reverting']:
        recommendations.append("⚠️ Caution: Asset may trend rather than revert")
    
    if vol_regime['regime'] == 'high_volatility':
        recommendations.append("⚠️ Use wider stops in high volatility regime")
    
    # Position sizing
    if consensus != "HOLD":
        half_life = ou_params['half_life']
        if half_life < 5:
            recommendations.append("💰 Short holding period expected (< 5 days)")
        elif half_life < 15:
            recommendations.append("💰 Medium holding period expected (5-15 days)")
        else:
            recommendations.append("💰 Longer holding period expected (> 15 days)")
    
    for rec in recommendations:
        print(f"• {rec}")
    
    # Key Insights Summary
    print(colored("\n🔍 KEY INSIGHTS", "yellow"))
    print("-" * 60)
    
    print(f"• Current Price: ${current_price:.2f}")
    print(f"• OU Long-term Mean: ${np.exp(ou_params['mu']):.2f}")
    print(f"• Distance from Mean: {((current_price/np.exp(ou_params['mu'])) - 1)*100:.1f}%")
    print(f"• Mean Reversion Speed: {ou_params['theta']:.3f} (Half-life: {ou_params['half_life']:.1f} days)")
    print(f"• Current Market Regime: {vol_regime['regime'].replace('_', ' ').title()}")
    
    if pair_symbol and 'spread' in locals():
        print(f"• Pair Spread Z-Score: {spread.get('current_zscore', 0):.2f}")


if __name__ == "__main__":
    import sys
    from typing import Optional
    
    # Default to SPY if no symbol provided
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    pair_symbol = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Some natural pairs for testing
    if symbol == "XOM" and not pair_symbol:
        pair_symbol = "CVX"  # Oil companies
    elif symbol == "GLD" and not pair_symbol:
        pair_symbol = "SLV"  # Precious metals
    elif symbol == "MA" and not pair_symbol:
        pair_symbol = "V"    # Payment processors
    
    try:
        analyze_mean_reversion(symbol, pair_symbol)
    except Exception as e:
        print(colored(f"\n❌ Error: {e}", "red"))
        print("Make sure you have yfinance installed: pip install yfinance")